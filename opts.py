import argparse, os, json
import torch
import torchvision as tv
from utils import transforms
#import torchvision.transforms as transforms


def get_opts():
    opt = argparse.Namespace()
    
    opt.task_name = ''
    #opt.exp_name = 'test0610_mobilenetv2_small_dropout_0d1_1024_Adam_128_bn_smooth_30e-4_0604_train_new_data_add_3dmodel/'
   # opt.exp_name = 'test0902_fintune_real_0901_mobilenetv2_small_dropout_0d2_128_Adam_128_bn_smooth_cce_5e-5_20200902_train_list_add_lyb_live'
    opt.exp_name = 'test_1209_01'
    opt.fold = 1
    opt.data_root = '' 
    #opt.data_list = '2get_list/train_result/1230_train_test_for_learning.txt' 
    #opt.val_list = '2get_list/train_result/20200724_0406_end_to_end_test_data.txt' 
    #opt.data_list = '2get_list/train_result/20200902_train_list_add_lyb_live.txt' 
    opt.data_list = '2get_list/train_result/0818_gloabal_train_new_data_add_low_phone.txt'
    
    #train1204  0107_train.txt 0102_test_data_djy_less.txt 0323_train_data 0408_train_new_data_add_iphone 1230_train_test_for_learning  0415_train_new_data
    opt.val_list = '2get_list/test_result/20200604_0406_end_to_end_test_data.txt' #test1209_shorter  0101_test_data_djy.txt  0120_test_data_djy.txt  0120_test_data.txt 0406_end_to_end_test_data 1230_test_data_for_learning.txt
 
    opt.out_root = '/mnt/sdb4/2d_project/sigle_result/'
    opt.out_path = os.path.join(opt.out_root,opt.exp_name)
    opt.out_path_test = '/mnt/sda1/nb_project/sigle_result/test12_31_test/'
    
    ### Dataloader options ###
    opt.nthreads = 16 #32 16
    opt.batch_size = 1024 # 256 1024
    opt.ngpu = 1 
    opt.img_size = 128  #change to 224 128 160 144

    ### Learning ###
    opt.freeze_epoch = 0
    opt.optimizer_name = 'Adam' #Adam  SGD
    opt.weight_decay = 0 #1e-4
    opt.lr = 4e-4 #2e-4  1e-4 2e-3 2e-3 1e-4 6e-4 
    opt.lr_decay_lvl = 0.5  #0.5 0.1
    opt.lr_decay_period = 4 # 4 2
    opt.lr_type = 'step_lr'#'cosine_repeat_lr'# step_lr ReduceLROnPlateau
    opt.lr_reduce_mode = 'None' #max for acc min for the min loss
    opt.num_epochs=300  #50
    opt.resume = None # 'model_40.pth'
    opt.debug = 0 
    opt.start_record_epoch = 1
    opt.num_itr= 250  #500 250 1000
      
    ### Other ###   
    opt.manual_seed = 704
    opt.log_batch_interval= 125  ##display batch's process 300 125
    opt.log_checkpoint = 1
    opt.net_type = 'mobilenetv2_small' # ResNet34CaffeSingle_fc ResNet34CaffeMulti_fc  ResNet18Siam  Resnet18_single_online se_resnet_18 densenet_121 Resnet18_single_online_bn Resnet18_single_online_bn_sig mobilenetv2 ghost_net
    opt.pretrained = True
    opt.pretrained_path = "/mnt/sdb4/2d_project/fintune_model/0805/model_8_67000"
    opt.isSaveTrainHard = False
    opt.isSaveTrainHardThresh = 0.3
    opt.SaveTrainHardPath = "/mnt/sdb5/20200723_train_hard_img"
    opt.classifier_type = 'linear'
    opt.smooth_rate = 0.02
    opt.loss_type= 'smooth_cce'  #smooth_cce cce  smooth_focal_cce circle focal_loss am_softmax
    opt.alpha_scheduler_type = None
    opt.nclasses = 2
    opt.fake_class_weight = 1
    opt.visdom_port = 8097
    #opt.isFintune = 'true'
    
    
    opt.train_transform = tv.transforms.Compose([
                #transforms.RandomCrop(opt.img_size),   
                tv.transforms.RandomApply([
                   #transforms.ColorJitter(0.25,0.25,0.25,0.125)],p=0.2),
                	#tv.transforms.RandomRotation(45)],p=0.3),
                 tv.transforms.RandomRotation(15)],p=0.2),
                #tv.transforms.RandomRotation(45,resample=2,p = 0.3),
                tv.transforms.RandomApply([
                   #transforms.ColorJitter(0.25,0.25,0.25,0.125)],p=0.2),
                tv.transforms.RandomAffine(degrees=15, translate=(0, 0.2), scale=(0.9, 1), shear=(6, 9), fillcolor=0)],p=0.1),                
                tv.transforms.RandomHorizontalFlip(p=0.2),
                tv.transforms.RandomApply([
                    transforms.CustomCutout(1, 25, 75)],p=0.1),
                transforms.CustomGaussianBlur(max_kernel_radius=3, p=0.2),
                #transforms.CustomRandomResizedCrop(128, scale=(0.5, 1.0)),
                #tv.transforms.RandomApply([
                #   tv.transforms.Resize((int(opt.img_size*1.2),int(opt.img_size*1.2))),
                 #  tv.transforms.CenterCrop(opt.img_size)],p=0.2),#center_crop must be right 128
                #tv.transforms.RandomApply([
                #    tv.transforms.Resize((int(opt.img_size*1.2),int(opt.img_size*1.2))),
                #   tv.transforms.RandomCrop(opt.img_size,4,True,0,'constant')],p=0.2),#center_crop must be right 128
                tv.transforms.RandomApply([
                    tv.transforms.ColorJitter(0.25,0.25,0.25,0.5)],p=0.3),
                #tv.transforms.ColorJitter(0.25,0.25,0.25,0.5)],p=0.4),
                tv.transforms.RandomGrayscale(p=0.02),
                
                tv.transforms.RandomApply([
                    transforms.CustomGaussianNoise(var = 0.0025, p = 0.3),
                    transforms.CustomPoissonNoise(p = 0.3)],p = 0.05),
                    
                transforms.Motion_blur(degree = 15, angle = 15, p = 0.05),
                
                #tv.transforms.RandomApply([
                #tv.transforms.ColorJitter(0.5,0.0,0.0,0.3),
                #transforms.CustomGaussianNoise(var = 0.0025, p = 0.5),
                #transforms.CustomPoissonNoise(p = 0.5)],p=0.10),
                
                #center_crop must be right 128
                #transforms.LabelSmoothing(eps=0.1, p=0.3),
                #transforms.CenterCrop(112),
                
                #transforms.GaussianBlur(max_kernel_radius=3, p=0.2),
                #transforms.CustomRandomResizedCrop(128, scale=(0.5, 1.0)),
                #transforms.RandomApply([
                #transforms.CenterCrop(200)],p=0.2),
                #transforms.RandomHorizontalFlip(), 
                #transforms.CenterCrop(112),
                #transforms.RandomGrayscale(p=0.3),
              #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                tv.transforms.ToTensor(),
                #transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    opt.test_transform = tv.transforms.Compose([
          #  transforms.CustomResize((125,125)),
            # transforms.CustomRotate(0),
            # transforms.CustomRandomHorizontalFlip(p=0),
            # transforms.CustomCrop((112,112), crop_index=0),
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
