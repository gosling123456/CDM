from PIL import Image
import PIL
import numpy as np
from list_path import get_crfs_path
PIL.Image.MAX_IMAGE_PIXELS = 20000000000



def mask_img(label, index):
    # RGB to label
    maskimg = np.zeros((label.shape[0], label.shape[1]))

    if index == 'sar1' or index == 'sar2' or index == 'sar3' or index == 'sar8' or index == 'sar9':
        maskimg[(label == (0, 0, 255)).all(axis=-1)] = 0
        maskimg[(label == (0, 255, 0)).all(axis=-1)] = 1
        maskimg[(label == (255, 0, 0)).all(axis=-1)] = 2
        maskimg[(label == (255, 255, 0)).all(axis=-1)] = 3
        maskimg[(label == (0, 0, 0)).all(axis=-1)] = 4
            
    elif index == 'sar4' :
        maskimg[(label == (0, 0, 255)).all(axis=-1)] = 0
        maskimg[(label == (0, 255, 0)).all(axis=-1)] = 1
        maskimg[(label == (255, 0, 0)).all(axis=-1)] = 2
        maskimg[(label == (210,180,140)).all(axis=-1)] = 3
        maskimg[(label == (255,255,255)).all(axis=-1)] = 4

    elif index == 'sar5':
        maskimg[(label == (0, 0, 255)).all(axis=-1)] = 0
        maskimg[(label == (0, 255, 0)).all(axis=-1)] = 1
        maskimg[(label == (255, 0, 0)).all(axis=-1)] = 2
        maskimg[(label == (150, 6, 205)).all(axis=-1)] = 3
        maskimg[(label == (255, 255, 255)).all(axis=-1)] = 4

    elif index == 'sar6':
        maskimg[(label == (0, 0, 255)).all(axis=-1)] = 0
        maskimg[(label == (0, 255, 0)).all(axis=-1)] = 1
        maskimg[(label == (255, 255, 0)).all(axis=-1)] = 2
        maskimg[(label == (255, 0, 0)).all(axis=-1)] = 3
        maskimg[(label == (210,180,140)).all(axis=-1)] = 4
        maskimg[(label == (255,0,255)).all(axis=-1)] = 5
        maskimg[(label == (255, 255, 255)).all(axis=-1)] = 6

    elif index == 'sar7':
        maskimg[(label == (0, 0, 255)).all(axis=-1)] = 0
        maskimg[(label == (0, 255, 0)).all(axis=-1)] = 1
        maskimg[(label == (255, 255, 0)).all(axis=-1)] = 2
        maskimg[(label == (255, 0, 255)).all(axis=-1)] = 3
        maskimg[(label == (130, 80, 20)).all(axis=-1)] = 4
        maskimg[(label == (210,180,140)).all(axis=-1)] = 5
        maskimg[(label == (0,255,255)).all(axis=-1)] = 6
        maskimg[(label == (255, 0, 0)).all(axis=-1)] = 7
        maskimg[(label == (0, 0, 0)).all(axis=-1)] = 8

    elif index == 'sar10' or index == 'sar11':
        maskimg[(label == (0, 0, 255)).all(axis=-1)] = 0
        maskimg[(label == (0, 255, 0)).all(axis=-1)] = 1
        maskimg[(label == (255, 0, 0)).all(axis=-1)] = 2
        maskimg[(label == (255, 0, 255)).all(axis=-1)] = 3
        maskimg[(label == (0, 0, 0)).all(axis=-1)] = 4
        
    elif index == 'sar12':
        maskimg[(label == (0, 0, 255)).all(axis=-1)] = 0
        maskimg[(label == (192, 192, 192)).all(axis=-1)] = 1
        maskimg[(label == (255, 0, 255)).all(axis=-1)] = 2
        maskimg[(label == (0, 0, 0)).all(axis=-1)] = 3
        
    elif index == 'sar13' or index == 'sar14' or index == 'sar15':
        maskimg[(label == (0, 0, 0)).all(axis=-1)] = 0
        maskimg[(label == (255, 0, 0)).all(axis=-1)] = 1
        maskimg[(label == (0, 0, 255)).all(axis=-1)] = 2
        maskimg[(label == (255, 255, 0)).all(axis=-1)] = 3
        maskimg[(label == (0, 255, 0)).all(axis=-1)] = 4
        maskimg[(label == (210,180,140)).all(axis=-1)] = 5
        
    return maskimg

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

def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int64) + b[k], minlength=n ** 2).astype(np.int64).reshape(n, n)

def OverallAccuracy(confusionMatrix):
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA

def Kappa(confusion_matrix):
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)

def Mean_Intersection_over_Union(index, confusion_matrix):
    IoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) +
                np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
    if index == 'sar13' or index == 'sar14' or index == 'sar15':
        IoU = IoU[1:]
    else:
        IoU = IoU[:-1]

    MIoU = np.nanmean(IoU)
    return IoU, MIoU

def Precision(index, confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    if index == 'sar13' or index == 'sar14' or index == 'sar15':
        precision = precision[1:]
    else:
        precision = precision[:-1]
    return precision

def Recall(index, confusionMatrix):
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    if index == 'sar13' or index == 'sar14' or index == 'sar15':
        recall = recall[1:]
    else:
        recall = recall[:-1]
    return recall

def F1Score(index, confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    f1score = 2 * precision * recall / (precision + recall)
    if index == 'sar13' or index == 'sar14' or index == 'sar15':
        f1score = f1score[1:]
    else:
        f1score = f1score[:-1]
    return f1score

def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis=1) +
            np.sum(confusionMatrix, axis=0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def colorize_mask(index, mask):
    palette = label_color(index)
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask
    
def Record_result_evaluation(result_save_dir, hist, target_names, precision, recall, f1ccore, OA, kappa, MIoU, FWIoU):
    with open(result_save_dir + '/Record_test_parameters_and_pred_result.txt', 'a') as f:
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

def calculate_eval(result_save_dir, index, hist, target_names):
    precision = Precision(index, hist)
    recall = Recall(index, hist)
    f1ccore = F1Score(index, hist)
    OA = OverallAccuracy(hist)
    kappa = Kappa(hist)
    IoU, MIoU = Mean_Intersection_over_Union(index, hist)
    FWIoU = Frequency_Weighted_Intersection_over_Union(hist)
    Record_result_evaluation(result_save_dir, hist, target_names, precision, recall, f1ccore, OA, kappa, MIoU, FWIoU)


def recal(gt_path, label_path, save_path, index):
    gt = Image.open(gt_path).convert('P')
    np_gt = np.asarray(gt).astype('uint8')
    
    label = Image.open(label_path)
    np_label = np.asarray(label).astype('uint8')
    h, w = np_label.shape[0], np_label.shape[1]
    
    np_gt = np_gt[:h,:w]
    #print(np_gt.shape, np_label.shape)
    assert np_gt.shape[0] == np_label.shape[0] and np_gt.shape[1] == np_label.shape[1]
    
    mask_label = mask_img(np_label, index)
    assert np_gt.shape == mask_label.shape
    
    if index == 'sar1' or index == 'sar2' or index == 'sar3' or index == 'sar4' or index == 'sar5' or index == 'sar8' or index == 'sar9' or index == 'sar10' or index == 'sar11':
        num_class = 5
    elif index == 'sar12':
        num_class = 4
    elif index == 'sar13' or index == 'sar14' or index == 'sar15':
        num_class = 6
    elif index == 'sar6':
        num_class = 7
    elif index == 'sar7':
        num_class = 9
        
    if index in ['sar13', 'sar14', 'sar15']:
        aim = 0
    else:
        aim = num_class - 1
        
    hist = np.zeros((num_class, num_class))
    
    mask_label[np_gt == aim] = aim
    
    np_gt_fl = np_gt.flatten()
    mask_label_fl = mask_label.flatten()
    unknown_index = np.where(np_gt_fl == aim)               # 去除真实标签中 未知类的像素，未知类像素的预测情况不参与计算
    np_gt_fl = np.delete(np_gt_fl, unknown_index)
    mask_label_fl = np.delete(mask_label_fl, unknown_index)

    hist = cal_hist(np_gt_fl.astype(np.int64), mask_label_fl.astype(np.int64), num_class)
    
    
    result_all = colorize_mask(index, mask_label)
    result_all.save('%s/%s_color.png' % (save_path, 'result'))
    
    calculate_eval(save_path, index, hist, 'result')

if __name__ == '__main__':
    label_path = '/data/dz/diffusion_model/DDPN_seg/CRFs/SAR5/cssl_block4_10/result.png'
    save_path = '/data/dz/diffusion_model/DDPN_seg/CRFs/SAR5/cssl_block4_10'
    index = 'sar5'
    _, gt_path, _ = get_crfs_path(index)

    recal(gt_path, label_path, save_path, index)




