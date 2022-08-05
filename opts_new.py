import argparse, os, json
import torch
import torchvision as tv
from utils import transforms
#import torchvision.transforms as transforms
def get_opts():
    opt = argparse.Namespace()
    
    opt.task_name = ''
    opt.exp_name = 'test0724_mobilenetv2_small_dropout_0d1_1024_Adam_128_bn_cce_30e-4_0724_train_new_data_add_video/'
    #opt.exp_name = 'test0704_mobilenet_64_Adam_128_bn_smooth_focal_cce_7d5e-4_0704_train_new_data_only_add_3dmodel_add_Mask_split_aug/'
    #opt.exp_name = 'test1/'
    opt.fold = 1
    opt.data_root = '/data/04ab8be3dd3742a7bac01b89faab5316/' 
    #opt.data_list = '2get_list/train_result/1230_train_test_for_learning.txt' 
    #opt.val_list = '2get_list/train_result/1230_train_test_for_learning.txt' 
    #opt.data_list = 'utils/0818_gloabal_train_new_data_add_low_phone_bmp_new.txt' 
    opt.train_ori_data_list = '/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/liveness_rawdata_train_new.txt'
    opt.train_crop_data_list = '/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/liveness_rawdata_train_crop_new.txt' 
    #train1204  0107_train.txt 0102_test_data_djy_less.txt 0323_train_data 0408_train_new_data_add_iphone 1230_train_test_for_learning  0415_train_new_data
    opt.val_ori_data_list = '/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/all_test_ori.txt' 
    opt.val_crop_data_list = '/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/all_test_crop.txt'#test1209_shorter  0101_test_data_djy.txt  0120_test_data_djy.txt  0120_test_data.txt 0406_end_to_end_test_data  0424_end_to_end_test_data_with_error_data_04_e2e_data
 
    opt.out_root = './result/log_files/'
    opt.out_path = os.path.join(opt.out_root,opt.exp_name)
    opt.out_path_test = '/mnt/sda1/nb_project/sigle_result/test12_31_test/'
     
    ### Dataloader options ###
    opt.nthreads = '8' #32 16
    opt.batch_size = 1024 # 256 1024
    opt.ngpu = 4
    opt.img_size = 128  #change to 224 128 160 144
    opt.ori_img_size = (512,384)
    opt.patchsize = 3
    ### Learning ###
    opt.freeze_epoch = 0
    opt.optimizer_name = 'Adam' #Adam  SGD
    opt.weight_decay = 1e-4
    opt.lr = 2e-4 #2e-4  1e-4 2e-3 2e-3 1e-4 6e-4
    opt.lr_decay_lvl = 0.5  #0.5 0.1
    opt.lr_decay_period = 4 # 4 2
    opt.lr_type = 'step_lr'#'cosine_repeat_lr'# step_lr ReduceLROnPlateau
    opt.lr_reduce_mode = 'None' #max for acc min for the min loss
    opt.num_epochs=100  #50
    opt.resume = None# 'model_40.pth'
    opt.debug = 0 
    opt.start_record_epoch = 1
    opt.num_itr= 500 #500 250 1000
    
    opt.log_dir = '/data/5a1c8816341c4a3bbdebf3e20dcf7219/replay_data/model_2_0'
    ### Other ###  
    opt.manual_seed = 704
    opt.log_batch_interval= 10  ##display batch's process 300 125
    opt.log_checkpoint = 1
    opt.net_type = 'mobilenetv2_small' # ResNet34CaffeSingle_fc Resnet101_single_online_bnResNet34CaffeMulti_fc  ResNet18Siam  Resnet18_single_online se_resnet_18 densenet_121 Resnet18_single_online_bn Resnet18_single_online_bn_sig mobilenetv2 ghost_net
    opt.pretrained = ''
    opt.classifier_type = 'linear'
    opt.smooth_rate = 0.0125
    opt.loss_type= 'smooth_cce'  #smooth_cce cce  smooth_focal_cce circle focal_loss am_softmax
    opt.alpha_scheduler_type = None
    opt.nclasses = 3
    opt.fake_class_weight = 1
    opt.visdom_port = 8097
    
    
    opt.ori_train_transform = tv.transforms.Compose([
                #transforms.RandomCrop(opt.img_size),   
                tv.transforms.RandomApply([
                   #transforms.ColorJitter(0.25,0.25,0.25,0.125)],p=0.2),
                	tv.transforms.RandomRotation(45)],p=0.3),
                #tv.transforms.RandomRotation(45,resample=2,p = 0.3),
                tv.transforms.RandomApply([
                   #transforms.ColorJitter(0.25,0.25,0.25,0.125)],p=0.2),
                tv.transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1), shear=(6, 9), fillcolor=0)],p=0.1),                
                tv.transforms.RandomHorizontalFlip(p=0.3),
                tv.transforms.RandomApply([
                    transforms.CustomCutout(1, 25, 75)],p=0.1),
                transforms.CustomGaussianBlur(max_kernel_radius=3, p=0.2),
                #transforms.CustomRandomResizedCrop(128, scale=(0.5, 1.0)),
                tv.transforms.RandomApply([
                   tv.transforms.Resize((int(opt.ori_img_size[0]*1.2),int(opt.ori_img_size[1]*1.2))),
                   tv.transforms.CenterCrop(opt.ori_img_size)],p=0.3),#center_crop must be right 128
                 tv.transforms.RandomApply([
                    tv.transforms.Resize((int(opt.ori_img_size[0]*1.2),int(opt.ori_img_size[1]*1.2))),
                   tv.transforms.RandomCrop(opt.ori_img_size,4,True,0,'constant')],p=0.3),#center_crop must be right 128
                tv.transforms.RandomApply([
                    transforms.CustomGaussianNoise(var = 0.0025, p = 0.3),
                    transforms.CustomPoissonNoise(p = 0.3)],p = 0.05),
                transforms.Motion_blur(degree = 15, angle = 15, p = 0.05),
                tv.transforms.RandomApply([
                    tv.transforms.ColorJitter(0.125,0.125,0.125,0)],p=0.2),
                tv.transforms.Resize(opt.ori_img_size),
                tv.transforms.ToTensor(),
                #transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        
    opt.crop_train_transform = tv.transforms.Compose([
                tv.transforms.RandomApply([
                	tv.transforms.RandomRotation(45)],p=0.3),
                tv.transforms.RandomApply([
                  tv.transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1), shear=(6, 9), fillcolor=0)],p=0.1),                
                tv.transforms.RandomHorizontalFlip(p=0.3),
                tv.transforms.RandomApply([
                  transforms.CustomCutout(1, 25, 75)],p=0.1),
                transforms.CustomGaussianBlur(max_kernel_radius=3, p=0.2),
                tv.transforms.RandomApply([
                  tv.transforms.Resize((int(opt.img_size*1.2),int(opt.img_size*1.2))),
                  tv.transforms.CenterCrop(opt.img_size)],p=0.3),#center_crop must be right 128
                tv.transforms.RandomApply([
                  tv.transforms.Resize((int(opt.img_size*1.2),int(opt.img_size*1.2))),
                  tv.transforms.RandomCrop(opt.img_size,4,True,0,'constant')],p=0.3),#center_crop must be right 128
                tv.transforms.RandomApply([
                    transforms.CustomGaussianNoise(var = 0.0025, p = 0.3),
                    transforms.CustomPoissonNoise(p = 0.3)],p = 0.05),   
                transforms.Motion_blur(degree = 15, angle = 15, p = 0.05),
                tv.transforms.RandomApply([
                    tv.transforms.ColorJitter(0.625,0.625,0.625,0.25)],p=0.2),
                tv.transforms.RandomGrayscale(p=0.20),
                #tv.transforms.Resize((int(opt.img_size),int(opt.img_size))),
                tv.transforms.ToTensor(),
                #transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        
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