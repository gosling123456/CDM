import torch
import gc
import numpy as np
import time

from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
from src.datasets import SARImageLabelDataset

from model_diffusion_seg_cssl import diffusion_classification_cssl

from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
from improved_diffusion.dist_util import dev

from utils.label_color import sar_path, label_color
from utils.tools import read_img_information
from utils.calculate import cal_hist, OverallAccuracy, Kappa, Frequency_Weighted_Intersection_over_Union

def get_args_parser():
    parser = argparse.ArgumentParser('diffusion_model_finetune', add_help=False)

    # Model parameters
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--step', default=25, type=int,
                        help='step of time')
    # [4,5,6,7,8,9,10]--2304  [4,6,8,10]--1280
    # Dataset parameters
    parser.add_argument('--index', default='sar15', type=str,
                        help='the index of train sar images')
    parser.add_argument('--slidesize', default=0, type=int)
    parser.add_argument('--result_save_dir', default='./cls_model_result/SAR15/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8 ',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int) 
    parser.add_argument('--cls_model',
                        default='./cls_model/SAR15/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8/model.pth',
                        help='resume from checkpoint')

    return parser


def create_model(
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.0,
        learn_sigma=True,
        use_checkpoint=True,
        use_scale_shift_norm=True
):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return dict(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm
    )


def create_gaussian_diffusion(
        *,
        steps=1000,
        learn_sigma=True,
        sigma_small=False,
        noise_schedule="cosine",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=False,
        timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def get_test_dataload(index):
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: 2 * x - 1
    ])
    root, _, _, test_list = sar_path(index)
    test_data = SARImageLabelDataset(root=root, list_path=test_list, transform=transform)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=False)

    return test_loader

def colorize_mask(args, mask):
    palette = label_color(args.index)
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def Mean_Intersection_over_Union(args, confusion_matrix):
    IoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) +
                np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
    if args.index not in ['sar13', 'sar14', 'sar15']:
        IoU = IoU[:-1]
    else:
        IoU = IoU[1:]

    MIoU = np.nanmean(IoU)
    return IoU, MIoU

def Precision(args, confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    if args.index not in ['sar13', 'sar14', 'sar15']:
        precision = precision[:-1]
    else:
        precision = precision[1:]
    return precision

def Recall(args, confusionMatrix):
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    if args.index not in ['sar13', 'sar14', 'sar15']:
        recall = recall[:-1]
    else:
        recall = recall[1:]
    return recall

def F1Score(args, confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    f1score = 2 * precision * recall / (precision + recall)
    if args.index not in ['sar13', 'sar14', 'sar15']:
        f1score = f1score[:-1]
    else:
        f1score = f1score[1:]
    return f1score

def Record_result_evaluation(args, hist, target_names, precision, recall, f1ccore, OA, kappa, MIoU, FWIoU):
    with open(args.result_save_dir + '/Record_test_parameters_and_pred_result.txt', 'a') as f:
        f.write('# ========================================================================================== \n')
        f.write('Confusion matrix:\n')
        f.write(str(hist) + '\n')
        f.write('target_names:    \n' + str(target_names) + '\n')
        f.write('precision:       \n' + str(precision) + '\n')
        f.write('recall:          \n' + str(recall)    + '\n' )
        f.write('f1ccore:         \n' + str(f1ccore)   + '\n')
        f.write("OA:           " + str(OA) + '\n')
        f.write("kappa:        " + str(kappa) + '\n')
        f.write("MIoU:         " + str(MIoU) + '\n')
        f.write("FWIoU:        " + str(FWIoU) + '\n')

def calculate_eval(args, hist, target_names):
    precision = Precision(args, hist)
    recall = Recall(args, hist)
    f1ccore = F1Score(args, hist)
    OA = OverallAccuracy(hist)
    kappa = Kappa(hist)
    IoU, MIoU = Mean_Intersection_over_Union(args, hist)
    FWIoU = Frequency_Weighted_Intersection_over_Union(hist)
    Record_result_evaluation(args, hist, target_names, precision, recall, f1ccore, OA, kappa, MIoU, FWIoU)

# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434
def test(args):

    test_loader = get_test_dataload(args.index)
    gc.collect()
    para = create_model()
    diffusion = create_gaussian_diffusion()

    if args.index in ['sar1', 'sar2', 'sar3', 'sar4', 'sar5', 'sar8', 'sar9', 'sar10', 'sar11', 'sar19']:
        num_class = 5
    elif args.index in ['sar12']:
        num_class = 4
    elif args.index in ['sar13', 'sar14', 'sar15']:
        num_class = 6
    elif args.index in ['sar6']:
        num_class = 7
    elif args.index in ['sar7']:
        num_class = 9

    classifier = diffusion_classification_cssl(num_class, **para)
    classifier.init_weights()

    state_dict = torch.load(args.cls_model)['model_state_dict']
    msg = classifier.load_state_dict(state_dict)
    print(msg)

    classifier = classifier.to(dev())

    t = torch.tensor([args.step]).to(dev())
    rnd_gen = torch.Generator(device=dev()).manual_seed(args.seed)
    noise = torch.randn(1, 3, args.input_size, args.input_size,
                        generator=rnd_gen, device=dev())
    classifier.eval()

    # set para
    single_size, gt_shape, img_name, target_names, aim, image_num = read_img_information(args.index)
    h_ipt = gt_shape[0]
    w_ipt = gt_shape[1]

    rowheight = args.input_size - 2 * args.slidesize
    colwidth = args.input_size - 2 * args.slidesize
    rownum = h_ipt // rowheight
    colnum = w_ipt // colwidth

    result_pixel_img = np.zeros((rowheight * rownum, colwidth * colnum), np.uint8)

    hist = np.zeros((num_class, num_class))

    image_index = 0
    start_time = time.time()
    for img, label in tqdm(test_loader):
        img = img.to(dev())
        label = label.to(dev())

        noisy_img = diffusion.q_sample(img, t, noise=noise)
        y_pred = classifier(noisy_img, diffusion._scale_timesteps(t))
        pred = y_pred.argmax(dim=1)


        label = np.asarray(label[0].data.cpu(), dtype=np.uint8)
        pred = np.asarray(pred.squeeze().data.cpu(), dtype=np.uint8)
        # print(pred.shape,label.shape)

        label_small = label[args.slidesize: label.shape[0] - args.slidesize,
                      args.slidesize:label.shape[1] - args.slidesize]
        pre_label_samll = pred[args.slidesize: pred.shape[0] - args.slidesize,
                          args.slidesize:pred.shape[1] - args.slidesize]
        assert label_small.shape == pre_label_samll.shape
        pre_label_samll[label_small == aim] = aim

        img_index = image_index
        r = img_index // colnum
        c = img_index - r * colnum

        result_pixel_img[r * rowheight:(r + 1) * rowheight, c * colwidth:(c + 1) * colwidth] = pre_label_samll
        label_small = label_small.flatten()
        pre_label_samll = pre_label_samll.flatten()
        unknown_index = np.where(label_small == aim)  # 去除真实标签中 未知类的像素，未知类像素的预测情况不参与计算
        label_small = np.delete(label_small, unknown_index)
        pre_label_samll = np.delete(pre_label_samll, unknown_index)

        hist += cal_hist(np.asarray(label_small), pre_label_samll, num_class)

        image_index += 1

    u_time = time.time()
    img_time = u_time - start_time
    print("time: {} ".format(img_time))

    result_all = colorize_mask(args, result_pixel_img)
    result_all.save('%s/%s_color.png' % (args.result_save_dir, str(img_name)))

    calculate_eval(args, hist, target_names)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.result_save_dir:
        Path(args.result_save_dir).mkdir(parents=True, exist_ok=True)
    test(args)

