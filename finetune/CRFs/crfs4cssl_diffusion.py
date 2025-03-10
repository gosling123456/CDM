from postprocessing_CRFs import CRFs
from recalculate_SAR import recal
from list_path import get_crfs_path
from pathlib import Path

index = [10,11,12,13,14,15]

for i in index:
    sar_idx = 'sar' + str(i)
    print('开始进行CRFs处理和重计算:', sar_idx)
    sar_image, gt, predict = get_crfs_path(sar_idx)
    
    # CRFs processing
    original_image_path = sar_image
    predicted_image_path = predict
    
    CRF_image_path = '/data/dz/diffusion_model/DDPN_seg/CRFs/SAR' + str(i) + '/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8'
    if CRF_image_path:
        Path(CRF_image_path).mkdir(parents=True, exist_ok=True)
        
    CRFs(original_image_path, predicted_image_path, CRF_image_path)
    
    # recalculate 
    gt_path = gt
    label_path = CRF_image_path + '/result.png'
    save_path = CRF_image_path

    recal(gt_path, label_path, save_path, sar_idx)
    
    print('CRFs处理和重计算完成:', sar_idx)
    print('保存路径为:', CRF_image_path)