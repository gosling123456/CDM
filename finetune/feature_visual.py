import argparse
import gc
import json
import os
import time
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
import numpy as np
import torch
import torch.nn as nn
from improved_diffusion.dist_util import dev
from improved_diffusion.script_util import (add_dict_to_argparser,
                                            model_and_diffusion_defaults)
from matplotlib import pyplot as plt
from PIL import Image
from src.datasets import (FeatureDataset, ImageLabelDataset,
                          SARImageLabelDataset)
from src.feature_extractors_2 import collect_features, create_feature_extractor
from src.utils import multi_acc, setup_seed
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

def make_transform(model_type: str, resolution: int):
    """ Define input transforms for pretrained models """
    if model_type == 'ddpm':
        transform = transforms.Compose([
            # transforms.Resize((resolution,resolution)),
            transforms.ToTensor(),
            lambda x: 2 * x - 1
        ])
    elif model_type in ['mae', 'swav', 'swav_w2', 'deeplab']:
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return transform

def denorm(x):
        """将取值范围从 [-1, 1] 转换为 [0, 1]."""
        out = (x + 1) / 2
        # out = x * 0.2701 + 0.4180
        return out.clamp_(0, 1)

def feature_visual(args):
    feature_extractor = create_feature_extractor(**args)  # 生成特征提取器
    
    print(f"Preparing data...")
    dataset = SARImageLabelDataset(
        root = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/test_vis',
        list_path = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/test_vis/ImageSets/test_vis.txt',
        # root = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/256size_sar9/test_slide_86_new',
        # list_path = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/256size_sar9/test_slide_86_new/ImageSets/test_vis.txt',
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )       # 导入数据集

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None

    for img, label in tqdm(dataset):
        img = img[None].to(dev())
        #print(img.size())
        features, res_x = feature_extractor(img, noise=noise)
        features = collect_features(args, features)
        # noise_features, _ = feature_extractor(res_x)
        # noise_features = collect_features(args, noise_features)
        save_path = os.path.join(args['save_feature'], 'chanel_0.jpg')
        #print(features[0][0].size())
        # save_image(res_x[0].data.cpu(), os.path.join(args['save_feature'], 'res_image.jpg'))
        # print(res_x.size())
    '''    
    # for index, channel in enumerate(range(9)):
    # img = img[0].transpose((1,2,0))
    img = np.asarray(denorm(img[0]).mul(255).clamp(0,255).data.cpu(), dtype=np.uint8)
    img = img.transpose((1,2,0))
    # img = np.uint8(255 * img)
    #print(features[0].size())
    heatmap = np.asarray(features[5].data.cpu(), dtype=np.uint8)
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    #print(heatmap.shape)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(save_path, result)
    cv2.imwrite(os.path.join(args['save_feature'], 'img.jpg'), img)
    cv2.imwrite(os.path.join(args['save_feature'], 'heatmap.jpg'), heatmap)
    plt.imshow(features[5].data.cpu(),cmap='jet')  # 灰度图参数cmap="gray"
    save_path2 = os.path.join(args['save_feature'], 'chanel_0_2.jpg')
    plt.savefig(save_path2,dpi=300)
    '''
    plt.figure(figsize=(10, 10))
    for index, channel in enumerate(range(9)):
        ax = plt.subplot(3, 3, index+1,)
        plt.imshow(features[channel].data.cpu(),cmap='jet')  # 灰度图参数cmap="gray"
    plt.savefig(save_path,dpi=300)

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str, default = 'json_setting/feature_visual/ddpm.json') # json脚本文件路径
    parser.add_argument('--seed', type=int,  default=0)
    # 测试增加参数
    parser.add_argument('--index', type=str, default='sar1', help='the index of test dataset')
    parser.add_argument('--aim', type=int, default=4)
    parser.add_argument('--slidesize', type=int, default=0, help='sample bound')
    # parser.add_argument('--save_feature', type=str, default='feature/img9_202/step_50_blocks_8/')

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp,'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]
    
    if not os.path.exists(opts['save_feature']):
        os.makedirs(opts['save_feature'])
        print('creat result file ---->', opts['save_feature'])
        
    feature_visual(opts)
    
    