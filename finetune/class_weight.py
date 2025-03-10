import torch
import torch.nn as nn
from tqdm import tqdm
import os
import gc
import numpy as np
import time

from torch.utils.data import DataLoader

import argparse

from src.datasets import SARImageLabelDataset, make_transform



def prepare_data():
    dataset = SARImageLabelDataset(
        root = './data/exp_try/256size_sar9/test_slide_86_new',
        list_path = './data/exp_try/256size_sar9/test_slide_86_new/ImageSets/train9.txt',
        transform=make_transform(
            'ddpm',
            256
        )
    )       # 导入数据集
    y = torch.zeros((len(dataset), 256, 256), dtype=torch.uint8)   # 生成[len（dataset）,C,H,W]的零张量

    for row, (_, label) in enumerate(tqdm(dataset)):
        y[row] = label
    
    y = y.flatten()
    return y[y != 255]

image_label = prepare_data()
# print(torch.max(image_label))
class_nums = torch.zeros(5)
for i in range(5):
    class_nums[i] = torch.sum(image_label == i)
    
class_nums = class_nums[:-1]
class_pre = class_nums / len(image_label)
print(class_pre)
class_num_max = torch.max(class_nums)
class_maxweight = class_num_max / class_nums
print(torch.sum(class_nums))
print(class_maxweight)
