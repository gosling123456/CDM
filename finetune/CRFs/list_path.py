def get_crfs_path(index):
    if index == 'sar1':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/1-Traunstein/TraunsteinSAR8.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/1-Traunstein/p_mask_1.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR1/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8 /_1_Traunstein_color.png'
    elif index == 'sar2':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/2-CSK - Napoli/NapoliSAR8.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/2-CSK - Napoli/p_mask_2.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR2/train_cssl_t25_b4:10:2_lr0.0001_bs4_ep700_prebs8 /_2_Napoli_color.png'
    elif index == 'sar3':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/3-CSK - PoDelta/PoDeltaSAR8.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/3-CSK - PoDelta/p_mask_3.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR3/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8 /_3_PoDelta_color.png'
    elif index == 'sar4':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/4-Istanbul/IstanbulSAR8_thre_95.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/4-Istanbul/p_mask_4.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR4/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700/_4_Istanbul_color.png'
    elif index == 'sar5':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/5-Rosenheim/RosenheimSAR8_thre_95.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/5-Rosenheim/p_mask_5.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR5/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8 /_5_Rosenheim_color.png'
    elif index == 'sar6':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/6-Washington/WashingtonSAR8_thre_95.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/6-Washington/p_mask_6.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR6/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700/_6_Washington_color.png'
    elif index == 'sar7':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/7-HongKongAirport/HongKongAirportSAR8_thre_95.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/7-HongKongAirport/p_mask_7.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR7/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700/_7_HongKongAirport_color.png'
    elif index == 'sar8':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/8-RosenheimTwo/RosenheimTwoSAR8_thre_95.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/8-RosenheimTwo/p_mask_8.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR8/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8 /_8__color.png'
    elif index == 'sar9':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/9-JiuJiang/JiuJiangSAR8_thre_95.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/9-JiuJiang/p_mask_9_2.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR9/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8 /_9_JiuJiang_color.png'
    elif index == 'sar10':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/10-Mississippi(middle)/mssp2.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/10-Mississippi(middle)/mssp2_out.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR10/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8 /_10_mixxmid_color.png'
    elif index == 'sar11':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/11-Mississippi(north)/mssp1.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/11-Mississippi(north)/mssp1_out.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR11/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8 /_11_mixxnor_color.png'
    elif index == 'sar12':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/12-Mississippi(south)/mssp3.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/12-Mississippi(south)/mssp3_out.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR12/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8 /_12_mixxsou_color.png'
    elif index == 'sar13':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/13-GF3_shandong/shandong.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/13-GF3_shandong/SARGf3_p_mask_1.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR13/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8 /_gf3sar1_shandong_color.png'
    elif index == 'sar14':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/14-GF3_korea/korea.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/14-GF3_korea/SARGf3_p_mask_2.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR14/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8 /_gf3sar2_korea_color.png'
    elif index == 'sar15':
        sar_image_path = '/data/dz/data/SAR_DATA/InitialImage/15-GF3_xian/xian.png'
        gt_path = '/data/dz/data/SAR_DATA/InitialImage/15-GF3_xian/SARGf3_p_mask_3.png'
        predict_path = '/data/dz/diffusion_model/DDPN_seg/cls_model_result/SAR15/train_cssl_t25_b4:10:2_lr0.00001_bs4_ep700_prebs8 /_gf3sar3_xian_color.png'
    return sar_image_path, gt_path, predict_path