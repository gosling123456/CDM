def read_img_information(index):
    # 读取SAR图和GT原图
    if index == 'sar1':
        single_size = (715, 734,3)
        gt_shape = (6187, 4278, 3)  # (h,w,c)
        img_name = '_1_Traunstein'
        target_names = ['water', 'trees', 'buildings', 'framland', 'unknown']
        aim = 4
        image_num = 96

    elif index == 'sar2':
        single_size = (716, 723, 3)
        gt_shape = (16000, 18332, 3)
        img_name = '_2_Napoli'
        target_names = ['water', 'trees', 'buildings', 'framland', 'unknown']
        aim = 4
        image_num = 1085

    elif index == 'sar3':
        single_size = (722, 723, 3)
        gt_shape = (16716, 18308, 3)
        img_name = '_3_PoDelta'
        target_names = ['water', 'trees', 'buildings', 'framland', 'unknown']
        aim = 4
        image_num = 1120
    elif index == 'sar4':
        single_size = (724, 721, 3)
        gt_shape = (17296, 26606, 3)
        img_name = '_4_Istanbul'
        target_names = ['water', 'tree', 'low_buildings', 'road/bridges', 'unknown']
        aim = 4
        image_num = 1683

    elif index == 'sar5':
        single_size = (716, 712, 3)
        gt_shape = (7224, 7691, 3)
        img_name = '_5_Rosenheim'
        target_names = ['water', 'trees', 'low_buildings', 'framland', 'unknown']
        aim = 4
        image_num = 210

    elif index == 'sar6':
        single_size = (712, 719, 3)
        gt_shape = (13836, 25454, 3)
        img_name = '_6_Washington'
        target_names = ['water', 'trees', 'hign_buildings', 'low_buildings', 'road/bridge', 'bare land grassland','unknown']
        aim = 6
        image_num = 1323

    elif index == 'sar7':
        single_size = (715, 726, 3)
        gt_shape = (9281, 16309, 3)
        img_name = '_7_HongKongAirport'
        target_names = ['water', 'tree', 'high_buildings', 'low_buildings', 'road', 'bare_land/small_grassland', 'large_grassland', 'airport runway', 'unknow/parking lot']
        aim = 8
        image_num = 558
    elif index == 'sar8':
        single_size = (715, 726, 3)
        gt_shape = (3600, 3600, 3)
        img_name = '_8_'
        target_names = ['water', 'tree', 'low_buildings', 'road/bridges', 'unknown']
        aim = 4
        image_num = 0
    elif index == 'sar9':
        single_size = (733, 733, 3)
        gt_shape = (8000, 8000, 3)
        img_name = '_9_JiuJiang'
        target_names = ['water', 'forest', 'buildings', 'framland', 'unknown']
        aim = 4
        image_num = 225
    elif index == 'sar10':
        single_size = (733, 733, 3)
        gt_shape = (16125, 15166, 3)
        img_name = '_10_mixxmid'
        target_names = ['water', 'forest', 'buildings', 'framland', 'unknown']
        aim = 4
        image_num = 225
    elif index == 'sar11':
        single_size = (733, 733, 3)
        gt_shape = (21500, 15333, 3)
        img_name = '_11_mixxnor'
        target_names = ['water', 'forest', 'buildings', 'framland', 'unknown']
        aim = 4
        image_num = 225
    elif index == 'sar12':
        single_size = (733, 733, 3)
        gt_shape = (21666, 15166, 3)
        img_name = '_12_mixxsou'
        target_names = ['shuizao', 'zaoze', 'water', 'unknown']
        aim = 3
        image_num = 225
    elif index == 'sar13':
        single_size = (512, 512, 3)
        gt_shape = (9216, 10240, 3)
        img_name = '_gf3sar1_shandong'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree', 'load']
        aim = 0
        image_num = 928   # train: 360

    elif index == 'sar14':
        single_size = (512, 512, 3)
        gt_shape = (7680, 9728, 3)
        img_name = '_gf3sar2_korea'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree', 'load']
        aim = 0
        image_num = 744 # train: 285

    elif index == 'sar15':
        single_size = (512, 512, 3)
        gt_shape = (3840, 2304, 3)
        img_name = '_gf3sar3_xian'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree', 'load']
        aim = 0
        image_num = 84 # train: 28
    elif index == 'sar19':
        single_size = (712, 719, 3)
        gt_shape = (13836, 25454, 3)
        img_name = '_6_Washington'
        target_names = ['water', 'trees', 'buildings', 'road/bridge', 'unknown']
        aim = 4
        image_num = 1323

    return single_size, gt_shape, img_name, target_names, aim, image_num


def Record_result_evaluation(args, hist, target_names, precision, recall, f1ccore, OA, kappa, MIoU, FWIoU):
    with open(args['result_save_dir'] + '/Record_test_parameters_and_pred_result.txt', 'a') as f:
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