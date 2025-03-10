import torch
import torch.nn as nn
import os
import gc
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

import argparse

from src.datasets import SARImageLabelDataset

from model_diffusion_seg_cssl import diffusion_classification_cssl

from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
from improved_diffusion.dist_util import dev

from utils.label_color import sar_path



def get_args_parser():
    parser = argparse.ArgumentParser('diffusion_model_finetune', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1000, type=int)

    # Model parameters
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--step', default=25, type=int,
                        help='step of time')
    # [4,5,6,7,8,9,10]--2304  [4,6,8,10]--1280
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--index', default='sar15', type=str,
                        help='the index of train sar images')
    parser.add_argument('--aim', default=0, type=int)
    parser.add_argument('--nb_classes', default=6, type=int)
    parser.add_argument('--output_dir', default='./cls_model/SAR15/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--pretrain_model', default='/data/dz/diffusion_model/IDDPM_CSSL/model_log/SAR1-18_fb_bs8/ema_0.9999_250000.pt',
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


def get_dataload(index, batch_size):
    transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: 2 * x - 1
        ])
    
    root, train_list, val_list, _ = sar_path(index)
    if index != 'sar9':
        train_data = SARImageLabelDataset(root = root, list_path = train_list, transform=transform)
        val_data = SARImageLabelDataset(root = root, list_path = val_list, transform=transform)
    else:
        train_data = SARImageLabelDataset(root = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/256size_sar9/test_slide_86_new/', list_path = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/256size_sar9/test_slide_86_new/ImageSets/train9.txt', transform=transform)
        val_data = SARImageLabelDataset(root = root, list_path = val_list, transform=transform)    

    # if 'share_noise' in args and args['share_noise']:
    #     rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
    #     noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
    #                         generator=rnd_gen, device=dev())
    # else:
    #     noise = None 
        
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader

# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434
def train(args):
    
    train_loader, val_loader = get_dataload(args.index, args.batch_size)
    print(" *********************** Current train dataloader length " +  str(len(train_loader)) + " ***********************")
    print(" *********************** Current val dataloader length " +  str(len(val_loader)) + " ***********************")
    gc.collect()
    para = create_model()
    diffusion = create_gaussian_diffusion()

    classifier = diffusion_classification_cssl(args.nb_classes, **para)
    classifier.init_weights()
    check_point = torch.load(args.pretrain_model, map_location="cpu")
    msg = classifier.load_state_dict(check_point, strict=False)
    print(msg)
    classifier = classifier.to(dev())

    t = torch.tensor([args.step]).to(dev())
    
    rnd_gen = torch.Generator(device=dev()).manual_seed(args.seed)
    noise = torch.randn(args.batch_size, 3, args.input_size, args.input_size,
                        generator=rnd_gen, device=dev())
        
        # 增加类别权重
    if args.index in ['sar1', 'sar2', 'sar3', 'sar4', 'sar5', 'sar8', 'sar9', 'sar10', 'sar11', 'sar19']:
        weight = torch.from_numpy(np.array([1.0, 1.0, 1.0, 1.0, 0.0])).float().cuda()
    elif args.index in ['sar12']:
        weight = torch.from_numpy(np.array([1.0, 1.0, 1.0, 0.0])).float().cuda()
    elif args.index in ['sar13', 'sar14', 'sar15']:
        weight = torch.from_numpy(np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])).float().cuda()
    elif args.index in ['sar6']:
        weight = torch.from_numpy(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])).float().cuda()
    elif args.index in ['sar7']:
        weight = torch.from_numpy(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])).float().cuda()
        
    criterion = nn.CrossEntropyLoss(weight = weight)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    classifier.train()

    iteration = 0
    break_count = 0
    best_loss = 10000000
    stop_sign = 0
    best_acc = 0
    for epoch in range(args.epochs):
        for X_batch, y_batch in train_loader:

            X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev())
            y_batch = y_batch.type(torch.long)
            
            optimizer.zero_grad()

            noisy_X_batch = diffusion.q_sample(X_batch, t, noise=noise[0:X_batch.shape[0], :, :, :])
            y_pred = classifier(noisy_X_batch, diffusion._scale_timesteps(t))

            loss = criterion(y_pred, y_batch)
            
            y_pred_softmax = torch.log_softmax(y_pred, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            
            y_pred_tags = y_pred_tags.data.cpu().flatten()
            y_test = y_batch.data.cpu().flatten()
            unknown_index = np.where(y_test == args.aim)               # 去除真实标签中 未知类的像素，未知类像素的预测情况不参与计算
            y_test = np.delete(y_test, unknown_index)
            y_pred_tags = np.delete(y_pred_tags, unknown_index)
            
            correct_pred = (y_pred_tags == y_test).float()
            
            acc = correct_pred.sum() / len(correct_pred)
            acc = acc * 100

            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 10 == 0:
                print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                
            if epoch > 900:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    break_count = 0
                else:
                    break_count += 1

                if break_count > 50:
                    stop_sign = 1
                    print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                    break
        if epoch % 5 == 0:
            with torch.no_grad():
                i = 0
                all_acc = 0
                for X_batch_val, y_batch_val in val_loader:
                    i += 1
                    X_batch_val, y_batch_val = X_batch_val.to(dev()), y_batch_val.to(dev())
                    y_batch_val = y_batch_val.type(torch.long)

                    noisy_X_batch_val = diffusion.q_sample(X_batch_val, t, noise=noise[0:X_batch_val.shape[0], :, :, :])
                    y_pred_val = classifier(noisy_X_batch_val, diffusion._scale_timesteps(t))
                    
                    y_pred_val_softmax = torch.log_softmax(y_pred_val, dim=1)
                    _, y_pred_val_tags = torch.max(y_pred_val_softmax, dim=1)
            
                    y_pred_val_tags = y_pred_val_tags.data.cpu().flatten()
                    y_test_val = y_batch_val.data.cpu().flatten()
                    unknown_index = np.where(y_test_val == args.aim)               # 去除真实标签中 未知类的像素，未知类像素的预测情况不参与计算
                    y_test_val = np.delete(y_test_val, unknown_index)
                    y_pred_val_tags = np.delete(y_pred_val_tags, unknown_index)
            
                    correct_pred_val = (y_pred_val_tags == y_test_val).float()
            
                    acc = correct_pred_val.sum() / len(correct_pred_val)
                    all_acc += acc
                print('Epoch : ', str(epoch), 'val_acc:', all_acc/i, 'val_iter:', i)
            if all_acc/i > best_acc:
                best_acc = all_acc/i
                model_path = os.path.join(args.output_dir, 'model' + '.pth')
                print('save to:',model_path)
                torch.save({'model_state_dict': classifier.state_dict()},
                             model_path)

        if stop_sign == 1:
            break
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
    
