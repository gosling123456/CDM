import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
import gc
import numpy as np
import time
from PIL import Image

from torch.utils.data import DataLoader

import argparse
from src.utils import setup_seed, multi_acc
from src.SDN_classifier import  AE_model, tow_satege_classification, tow_stage_predict_labels
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform, AE_SARImageLabelDataset, SARImageLabelDataset
from src.feature_extractors import create_feature_extractor, collect_features
from utils.tools import  read_img_information, Record_result_evaluation
from utils.calculate import OverallAccuracy, Precision, Recall, F1Score, Frequency_Weighted_Intersection_over_Union
from utils.calculate import Kappa, Mean_Intersection_over_Union, cal_hist
from utils.label_color import label_color, sar_path, weight_np

from improved_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from improved_diffusion.dist_util import dev


def prepare_data(args,p1,p2):
    feature_extractor = create_feature_extractor(**args)  # 生成特征提取器
    
    print(f"Preparing the train set for {args['category']}...")
    if args['index'] == 'sar9':
        root2 = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/256size_sar9/test_slide_86_new'
        train_path2 = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/256size_sar9/test_slide_86_new/ImageSets/train9.txt'
    else:
        root2 = root
        train_path2 = train_path
    
    dataset = SARImageLabelDataset(
        root = root2,
        list_path = train_path2,
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )       # 导入数据集
    X = torch.zeros((len(dataset), *args['dim'][::-1]), dtype=torch.float) # 生成[len（dataset）,C,H,W]的零张量
    y = torch.zeros((len(dataset), *args['dim'][:-1]), dtype=torch.uint8)   # 生成[len（dataset）,H,W]的零张量

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    for row, (img, label) in enumerate(tqdm(dataset)):
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        X[row] = collect_features(args, features).cpu()
        y[row] = label
    
    d = X.shape[1]
    print(f'Total dimension {d}')
    print('save feature')
    for i in range(len(dataset)):
        a1 = p1 + '/' + str(i) + '.npy'
        a2 = p2 + '/' + str(i) + '.npy'
        np.save(a1,X[i])
        np.save(a2,y[i])
    print('save complete')
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str, default = 'json_setting/feature_dataset/ddpm.json') # json脚本文件路径
    parser.add_argument('--seed', type=int,  default=0)
    # 测试增加参数
    parser.add_argument('--index', type=str, default='sar15', help='the index of test dataset')
    parser.add_argument('--feature_root', type=str, default='data/feature_dataset/HH')

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp,'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]
    root, train_path, val_path, test_path = sar_path(opts['index'])
    # Prepare the experiment folder 
    
    feature_suffix = 'feature/' + opts['index']
    label_suffix = 'label/' + opts['index']
    feature_path = os.path.join(opts['feature_root'], feature_suffix)
    label_path = os.path.join(opts['feature_root'], label_suffix)

    os.makedirs(feature_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    
    
    
    prepare_data(opts, feature_path, label_path)
    
    
    
    
    