import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import List
from torch.distributions import Categorical
from src.SDN import SDNet
from src.utils import colorize_mask, oht_to_scalar
# from src.data_util import get_palette, get_class_names
from PIL import Image

class AE_classifier(nn.Module):
    '''
    特征的表示部分第一个方法改进模型,首先通过AE模块降低特征的通道数,再进行分类任务
    '''
    def __init__(self, num_class, input_feature_dim, AE_dim = 128):
        super(AE_classifier, self).__init__()
        self.AE_dim = AE_dim
        self.AE_layer = nn.Sequential(
            # nn.Conv2d(input_feature_dim, AE_dim, kernel_size=1, stride=1),
            nn.Conv2d(input_feature_dim, AE_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(AE_dim)
            )
        #self.rec_AE_layer = nn.Conv2d(AE_dim, input_feature_dim, kernel_size=1, stride=1)
        self.rec_AE_layer = nn.Conv2d(AE_dim, input_feature_dim, kernel_size=3, stride=1, padding=1)
        self.classifier_layers = nn.Sequential(
            nn.Conv2d(AE_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, num_class, kernel_size=3, stride=1, padding=1),
            )
        #self.classifier_layers = nn.Sequential(
        #    nn.Conv2d(AE_dim, 32, kernel_size=3, stride=1,padding=1),
        #    nn.ReLU(),
        #    nn.BatchNorm2d(32),
        #    nn.Conv2d(32, num_class, kernel_size=3, stride=1,padding=1),
        #    )
    
    def forward(self, x):
        AE_feature = self.AE_layer(x)
        rec_x = self.rec_AE_layer(AE_feature)
        out = self.classifier_layers(AE_feature)
        return out, rec_x
    
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
        
class AE_model(nn.Module):
    def __init__(self, input_feature_dim, AE_dim = 128):
        super(AE_model, self).__init__()
        self.AE_dim = AE_dim
        self.sample = nn.Sequential(
            nn.Conv2d(input_feature_dim, input_feature_dim//2, kernel_size=1, stride=1),
            nn.ReLU(),
            # nn.BatchNorm2d(AE_dim)
            nn.Conv2d(input_feature_dim//2, input_feature_dim//4, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(input_feature_dim//4, AE_dim, kernel_size=1, stride=1),
            nn.ReLU()
            )
        self.rec = nn.Sequential(
            nn.Conv2d(AE_dim, input_feature_dim//4, kernel_size=1, stride=1),
            nn.ReLU(),
            # nn.BatchNorm2d(AE_dim)
            nn.Conv2d(input_feature_dim//4, input_feature_dim//2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(input_feature_dim//2, input_feature_dim, kernel_size=1, stride=1),
            )
    def forward(self, x):
        out = self.sample(x)
        rec_x = self.rec(out)
        return out, rec_x
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

class tow_satege_classification(nn.Module):
    def __init__(self, num_class, AE_dim = 128):
        super(tow_satege_classification, self).__init__()
        self.classifier_layers = nn.Sequential(
            nn.Conv2d(AE_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, num_class, kernel_size=3, stride=1, padding=1),
            )
    def forward(self, x):
        out = self.classifier_layers(x)
        return out
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
    

class FP_classifier(nn.Module):
    '''
    特征表示部分的第二个方法改进模型，具体来说，从扩散模型中提取的不同层级的特征激活，采用特征金字塔的方式进行特征融合。
    activations: List[torch.Tensor] 中的特征激活应该按照低分变率到高分辨率排列（高层特征激活到底层特征激活）
    '''
    def __init__(self, chenel_size, size_list, num_class, size):
        super(FP_classifier,self).__init__()
        self.C_list = chenel_size
        self.size_list = size_list
        assert len(chenel_size) == len(size_list)
        self.lateral_layers = nn.ModuleList([])
        
        for i in range(len(size_list)):
            inch = chenel_size[i]
            self.lateral_layers.append(nn.Conv2d(in_channels=inch, out_channels=256, kernel_size=1, stride=1, padding=0))
        self.smooth_layer = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(256,num_class,kernel_size=3, stride=1, padding=1)
        self.size = size

    def forward(self, x):
        input = []
        for j in range(len(self.C_list)):
            c_idx = self.C_list[0:j].sum()
            input.append(x[:, c_idx:c_idx+self.C_list[j], 0:self.size_list[j], 0:self.size_list[j]])
        for i in range(len(input)):
            if i == 0 :
                last_feature = self.lateral_layers[i](input[i])
            else:
                last_feature = self.upsampleAdd(last_feature, self.lateral_layers[i](input[i]))
        soomth_feature = self.smooth_layer(last_feature)
        soomth_feature = F.interpolate(soomth_feature, size=self.size, mode='nearest' )
        out = self.classifier(soomth_feature)
        return out
    
    def upsampleAdd(self, last_feature, cur_feature):
        # 上采用使用插值的方式，合并采用相加的方式而不是cat的方式。
        _, _, H, W = cur_feature.size()
        return F.interpolate(last_feature, size=(H, W), mode='nearest') + cur_feature
    
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

class AE_SDNet(AE_classifier):
    def __init__(self, SDN=SDNet, **kwargs):
        super().__init__(**kwargs)
        self.segConv = nn.Conv2d(self.AE_dim, self.AE_dim, 3,1,1)
        self.SDNet = SDN(self.AE_dim, self.AE_dim, lambda_=0.7)      #输出的通道和输入的相同

    def forward(self, x):
        # 特征提取
        AE_feature = self.AE_layer(x)
        rec_x = self.rec_AE_layer(AE_feature)
        # SDN网络
        
        V = self.segConv(AE_feature)
        F_s = self.SDNet(AE_feature, V)
        
        # 分类网络
        out = self.classifier_layers(F_s)
        
        # out = self.classifier_layers(AE_feature)
        return out, rec_x


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L68
class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier, self).__init__()
        if numpy_class < 30:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class)
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

    with torch.no_grad():
        preds, _ = model(features.cuda())
        img_seg = oht_to_scalar(preds)
    return img_seg

def tow_stage_predict_labels(model, features, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)

    with torch.no_grad():
        preds = model(features.cuda())
        img_seg = oht_to_scalar(preds)
    return img_seg
    
def FPN_predict_labels(model, features):
    
    with torch.no_grad():
        preds = model(features.cuda())
        img_seg = oht_to_scalar(preds)
    return img_seg

