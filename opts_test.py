import argparse, os, json
import torch
import torchvision as tv
from utils import transforms

def get_opts():
    opt = argparse.Namespace()
     
    opt.task_name = ''
    opt.exp_name = 'Test_MobileNetV2_small_epoch60/'#test1231_4
    opt.fold = 1
    opt.data_root = ''
    
    opt.val_ori_data_list = '/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/all_test_ori.txt' 
    opt.val_crop_data_list = '/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/all_test_crop.txt'
    
#     opt.val_ori_data_list = '/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/ori_test_new.txt' 
#     opt.val_crop_data_list = '/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/crop_test_new.txt'
    
#     opt.val_ori_data_list = '/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/liveness_rawdata_test_for_zidonghua_live.txt'
#     opt.val_crop_data_list = '/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/liveness_rawdata_test_for_zidonghua_live_crop.txt'

#     opt.val_ori_data_list = './utils/liveness_rawdata_test_for_zidonghua_live_1.txt'
#     opt.val_crop_data_list = './utils/liveness_rawdata_test_for_zidonghua_live_1_crop.txt'

#     opt.val_ori_data_list = '/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/liveness_rawdata_test_0312.txt'
#     opt.val_crop_data_list ='/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/liveness_rawdata_test_0312_crop.txt'
    
#     opt.val_ori_data_list = '/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/liveness_rawdata_test_2020316.txt'
#     opt.val_crop_data_list ='/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/liveness_rawdata_test_2020316_crop.txt'
    
#     opt.val_ori_data_list ='/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/liveness_rawdata_test_videoReplay.txt'
#     opt.val_crop_data_list ='/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/liveness_rawdata_test_videoReplay_crop.txt'
    
    opt.out_root = './result/'
    opt.out_path = os.path.join(opt.out_root,opt.exp_name)
    opt.out_path_test = '/data/ba4ca28549c644ee81aa16ec37351ed3/weights/'
    #test0414_mobilenet_1024_Adam_128_bn_smooth_36e-4_0415_data_add_2_arguement test0530_mobilenetv2_small_dropout_0d1_1024_Adam_128_bn_smooth_30e-4_0529_train_new_data_add_3dmodel 
    #opt.out_path_test = '/mnt/sdb4/2d_project/sigle_result/test0325_resnet_18_128_Adam_64_bn_smooth_low_3e-4_new_data' test0412_mobilenet_1024_Adam_128_bn_smooth_48e-4_0410_data 
    #test0414_mobilenet_1024_Adam_128_bn_smooth_36e-4_0413_data_add_2_arguement
    opt.out_error_flag = False
    opt.out_error_list='./result/error_list/error.txt'
    opt.threshhold = 0.68
    

    opt.tpr_thd = 0.02
    ### Dataloader options ###
    opt.nthreads = 2
    opt.batch_size = 512 #280
    opt.ngpu = 2
    opt.img_size = 128
    opt.ori_img_size = (512,384)

    ### Learning ###
    opt.freeze_epoch = 0
    opt.optimizer_name = 'Adam'  #Adam
    opt.weight_decay = 0
    opt.lr = 4e-4
    opt.lr_decay_lvl = 0.5
    opt.lr_decay_period = 4   #50
    opt.lr_type = 'step_lr'  #cosine_repeat_lr
    opt.num_epochs=50
    opt.resume = 'model_best.pth'
    opt.debug = 0 
 
    ### Other ###  
    opt.manual_seed = 704
    opt.log_batch_interval=1
    opt.log_checkpoint = 1
    opt.net_type = 'mobilenetv2_small' #ResNet34CaffeSingle_fc Resnet18_single_online  densenet_121 Resnet18_single_online_bn mobilenetv2 mobilenetv2_small
    opt.pretrained = ''
    opt.classifier_type = 'linear'
    opt.loss_type= 'cce'
    opt.alpha_scheduler_type = None
    opt.nclasses = 3
    opt.fake_class_weight = 1
    opt.visdom_port = 8097
    
    opt.ori_test_transform = tv.transforms.Compose([
          #  transforms.CustomResize((125,125)),
            # transforms.CustomRotate(0),
            # transforms.CustomRandomHorizontalFlip(p=0),
            # transforms.CustomCrop((112,112), crop_index=0),
            #tv.transforms.Resize((int(opt.img_size),int(opt.img_size))),
            tv.transforms.Resize(opt.ori_img_size),
            tv.transforms.ToTensor(),
            #transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    opt.crop_test_transform = tv.transforms.Compose([
          #  transforms.CustomResize((125,125)),
            # transforms.CustomRotate(0),
            # transforms.CustomRandomHorizontalFlip(p=0),
            # transforms.CustomCrop((112,112), crop_index=0),
            #tv.transforms.Resize((int(opt.img_size),int(opt.img_size))),
            tv.transforms.ToTensor(),
            #transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    
    
    return opt


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--savepath', type=str, default = 'data/opts/', help = 'Path to save options')
    conf = parser.parse_args()
    opts = get_opts()
    save_dir = os.path.join(conf.savepath, opts.exp_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir,opts.exp_name + '_' + 'fold{0}'.format(opts.fold) + '_' + opts.task_name+'.opt')
    torch.save(opts, filename)
    print('Options file was saved to '+filename)
