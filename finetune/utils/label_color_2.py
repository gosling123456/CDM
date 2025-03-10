import numpy as np


def label_color(index):
    # 赋值索引值相应的颜色，P模式索引图(单通道)
    if index == 'sar1' or index == 'sar2' or index == 'sar3' or index == 'sar8' or index == 'sar9':
        colors = [
            0, 0, 255,      # label 0 的色彩      # blue ---water
            0, 255, 0,      # label 1 的色彩      # green --- trees
            255, 0, 0,      # label 2 的色彩      # red --- buildings
            255, 255, 0,    # label 3 的色彩      # yellow --- framland
            0, 0, 0,        # label 4 的色彩      # black ---Unknown
            # .....   可以随便加, 最多应该是uint8的上限
        ]
    elif index == 'sar4' :
        colors = [
            0, 0, 255,     # label 0 的色彩       # blue ---water
            0, 255, 0,     # label 1 的色彩       # green --- trees
            255, 0, 0,     # label 2 的色彩       # red --- buildings
            210,180,140,   # label 3 的色彩       # brown --- road/bridge
            255,255,255,   # label 4 的色彩       # white ---Unknown
        ]
    elif index == 'sar5':

        colors = [
            0, 0, 255,      # label 0 的色彩      # blue ---water
            0, 255, 0,      # label 1 的色彩      # green --- trees
            255, 0, 0,      # label 2 的色彩      # red --- low_buildings
            150, 6, 205,    # label 3 的色彩      # purple --- framland
            255, 255, 255,  # label 4 的色彩      # white ---Unknown
        ]

    elif index == 'sar6':
        colors = [
            0, 0, 255,      # label 0 的色彩      # blue ---water
            0, 255, 0,      # label 1 的色彩      # green --- trees
            255, 255, 0,    # label 2 的色彩      # yellow --- high_buildings
            255, 0, 0,      # label 3 的色彩      # red --- low_buildings
            210,180,140,    # label 4 的色彩      # brown --- road/bridge
            255,0,255,      # label 5 的色彩      # rosein ---Bareland/grassland
            255, 255, 255,  # label 6 的色彩      # white ---Unknown
        ]

    elif index == 'sar7':
        colors = [
            0, 0, 255,      # label 0 的色彩      # blue ---water
            0, 255, 0,      # label 1 的色彩      # green --- trees
            255, 255, 0,    # label 2 的色彩      # yellow --- high_buildings
            255, 0, 255,    # label 3 的色彩      # rosein --- low_buildings
            130, 80, 20,    # label 4 的色彩      # dark brown --- road
            210,180,140,    # label 5 的色彩      # brown --- bare_land/small_grassland
            0,255,255,      # label 6 的色彩      # lightcyan --- large_grassland
            255, 0, 0,      # label 7 的色彩      # red --- airport runway
            0, 0, 0,        # label 8 的色彩      # white ---Unknown
        ]
    elif index == 'sar10' or index == 'sar11':
        colors = [
            0, 0, 255,      # label 0 的色彩      # 蓝色 --水域
            0, 255, 0,      # label 1 的色彩      # 绿色 ---农田
            255, 0, 0,      # label 2 的色彩      # 红色 ---城市
            255, 0, 255,    # label 3 的色彩      # 紫色 ---水藻
            0, 0, 0,        # label 4 的色彩      # 黑色 ---其他
        ]
    elif index == 'sar12':
        colors = [
            0, 0, 255,      # label 0 的色彩      # 蓝色 --水域
            192, 192, 192,  # label 1 的色彩      # 灰色 ---沼泽
            255, 0, 255,    # label 2 的色彩      # 紫色 ---水藻
            0, 0, 0,        # label 3 的色彩      # 黑色 ---其他
        ]
    elif index == 'sar13' or index == 'sar14' or index == 'sar15':
        colors = [
            0, 0, 0,         # label 0 的色彩      # black ---无效类别
            255, 0, 0,       # label 1 的色彩      # red ---建筑
            0, 0, 255,       # label 2 的色彩      # blue ---水体
            255, 255, 0,     # label 3 的色彩      # yellow --- 耕地
            0, 255, 0,       # label 4 的色彩      # green --- 绿化
            210,180,140,     # label 5 的色彩      # brown --- 路
        ]
    return colors
    
    
def sar_path(index):
    if index == 'sar1':
        root = '/data/dz/data/SAR_small_img/256size/0_slide/SAR1'
        train_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR1/ImageSets/train1.txt'
        val_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR1/ImageSets/val1.txt'
        test_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR1/ImageSets/test1.txt'
    elif index == 'sar2':
        root = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/sar2/256size/test_slide_0'
        train_path = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/sar2/256size/test_slide_0/ImageSets/train2.txt'
        val_path = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/sar2/256size/test_slide_0/ImageSets/val2.txt'
        test_path = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/sar2/256size/test_slide_0/ImageSets/test2.txt'
    elif index == 'sar3':
        root = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/sar3/256size/test_slide_0'
        train_path = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/sar3/256size/test_slide_0/ImageSets/train3.txt'
        val_path = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/sar3/256size/test_slide_0/ImageSets/val3.txt'
        test_path = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/sar3/256size/test_slide_0/ImageSets/test3.txt'
    elif index == 'sar4':
        root = '/data/dz/data/SAR_small_img/256size/0_slide/SAR4'
        train_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR4/ImageSets/train4.txt'
        val_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR4/ImageSets/val4.txt'
        test_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR4/ImageSets/test4.txt'
    elif index == 'sar5':
        root = '/data/dz/data/SAR_small_img/256size/0_slide/SAR5'
        train_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR5/ImageSets/train5.txt'
        val_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR5/ImageSets/val5.txt'
        test_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR5/ImageSets/test5.txt'
    elif index == 'sar6':
        root = '/data/dz/data/SAR_small_img/256size/0_slide/SAR6'
        train_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR6/ImageSets/train6.txt'
        val_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR6/ImageSets/val6.txt'
        test_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR6/ImageSets/test6.txt'
    elif index == 'sar7':
        root = '/data/dz/data/SAR_small_img/256size/0_slide/SAR7'
        train_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR7/ImageSets/train7.txt'
        val_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR7/ImageSets/val7.txt'
        test_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR7/ImageSets/test7.txt'
    elif index == 'sar8':
        root = '/data/dz/data/SAR_small_img/256size/0_slide/SAR8'
        train_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR8/ImageSets/train8.txt'
        val_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR8/ImageSets/val8.txt'
        test_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR8/ImageSets/test8.txt'
    elif index == 'sar9':
        root = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/256size_sar9/test_slide_0_new'
        train_path = ''
        val_path = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/256size_sar9/test_slide_0_new/ImageSets/val9.txt'
        test_path = '/data/dz/diffusion_model/DDPN_seg/data/exp_try/256size_sar9/test_slide_0_new/ImageSets/test9.txt'
    elif index == 'sar10':
        root = '/data/dz/data/SAR_small_img/256size/0_slide/SAR10'
        train_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR10/ImageSets/train10.txt'
        val_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR10/ImageSets/val10.txt'
        test_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR10/ImageSets/test10.txt'
    elif index == 'sar11':
        root = '/data/dz/data/SAR_small_img/256size/0_slide/SAR11'
        train_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR11/ImageSets/train11.txt'
        val_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR11/ImageSets/val11.txt'
        test_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR11/ImageSets/test11.txt'
    elif index == 'sar12':
        root = '/data/dz/data/SAR_small_img/256size/0_slide/SAR12'
        train_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR12/ImageSets/train12.txt'
        val_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR12/ImageSets/val12.txt'
        test_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR12/ImageSets/test12.txt'
    elif index == 'sar13':
        root = '/data/dz/data/SAR_small_img/256size/0_slide/SAR13'
        train_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR13/ImageSets/train13.txt'
        val_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR13/ImageSets/val13.txt'
        test_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR13/ImageSets/test13.txt'
    elif index == 'sar14':
        root = '/data/dz/data/SAR_small_img/256size/0_slide/SAR14'
        train_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR14/ImageSets/train14.txt'
        val_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR14/ImageSets/val14.txt'
        test_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR14/ImageSets/test14.txt'
    elif index == 'sar15':
        root = '/data/dz/data/SAR_small_img/256size/0_slide/SAR15'
        train_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR15/ImageSets/train15.txt'
        val_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR15/ImageSets/val15.txt'
        test_path = '/data/dz/data/SAR_small_img/256size/0_slide/SAR15/ImageSets/test15.txt'
    return root,train_path,val_path,test_path

def weight_np(index):
    if index == 'sar1' or index == 'sar2' or index == 'sar3' or index == 'sar4' or index == 'sar5' or index == 'sar8' or index == 'sar9' or index == 'sar10' or index == 'sar11':
        out = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    elif index == 'sar12':
        out = np.array([1.0, 1.0, 1.0, 0.0])
    elif index == 'sar13' or index == 'sar14' or index == 'sar15':
        out = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    elif index == 'sar6':
        out = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    elif index == 'sar7':
        out = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    return out

