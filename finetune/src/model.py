import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

from torch.distributions import Categorical
from src.utils import colorize_mask, oht_to_scalar
from src.data_util import get_palette, get_class_names
from PIL import Image


class seg_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(seg_classifier, self).__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(dim, 512, kernel_size=1 ,stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=512),
                nn.Conv2d(512, 256, kernel_size=1 ,stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=256),
                nn.Conv2d(256, 128, kernel_size=1 ,stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=128),
                nn.Conv2d(128, numpy_class, kernel_size=1 ,stride=1, padding=0)
            )
    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
    def forward(self, x):
        return self.layers(x)


def predict_labels(model, features, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    
    all_seg = []
    

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        preds = model(features.cuda())
        img_seg = oht_to_scalar(preds)

    return img_seg

