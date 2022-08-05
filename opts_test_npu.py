import argparse, os, json
import torch
import torchvision as tv
from utils import transforms

def get_opts():
    opt = argparse.Namespace()
     
    opt.task_name = ''
    opt.exp_name = 'test0107_resnet_34_Adam/'#test1231_4
    opt.fold = 1
    opt.data_root = ''
    opt.data_list = 'list/train1125.txt'
    opt.val_list = '2get_list/test_result/20200723_live_fanyifu_npu_data_crop.txt' #test1203 train1203_0first 0120_test_data_djy.txt 0210_test_data 0224_test_data .txt Test3_2020316_crop
     
    opt.out_root = '/mnt/sdb4/2d_project/sigle_result/'
    opt.out_path = os.path.join(opt.out_root,opt.exp_name)
    opt.out_path_test = '/mnt/sdb4/2d_project/sigle_result//test0530_mobilenetv2_small_dropout_0d1_1024_Adam_128_bn_smooth_30e-4_0529_train_new_data_add_3dmodel/'
    #test0414_mobilenet_1024_Adam_128_bn_smooth_36e-4_0415_data_add_2_arguement test0530_mobilenetv2_small_dropout_0d1_1024_Adam_128_bn_smooth_30e-4_0529_train_new_data_add_3dmodel
    #opt.out_path_test = '/mnt/sdb4/2d_project/sigle_result/test0325_resnet_18_128_Adam_64_bn_smooth_low_3e-4_new_data' test0412_mobilenet_1024_Adam_128_bn_smooth_48e-4_0410_data 
    #test0414_mobilenet_1024_Adam_128_bn_smooth_36e-4_0413_data_add_2_arguement
    opt.out_error_flag = True
    opt.out_error_list='./error_list/20200721_chinese2w_error.tx'
    opt.threshhold = 0.5
    opt.tpr_thd = 0.02
    ### Dataloader options ###
    opt.nthreads = 8
    opt.batch_size = 512 #280
    opt.ngpu = 1
    opt.img_size = 128

    ### Learning ###
    opt.freeze_epoch = 0
    opt.optimizer_name = 'Adam'  #Adam
    opt.weight_decay = 0
    opt.lr = 4e-4
    opt.lr_decay_lvl = 0.5
    opt.lr_decay_period = 4   #50
    opt.lr_type = 'step_lr'  #cosine_repeat_lr
    opt.num_epochs=50
    opt.resume = 'model_18_82000'
    opt.debug = 0 

 
    ### Other ###  
    opt.manual_seed = 704
    opt.log_batch_interval=100
    opt.log_checkpoint = 1
    opt.net_type = 'mobilenetv2_small' #ResNet34CaffeSingle_fc Resnet18_single_online  densenet_121 Resnet18_single_online_bn mobilenetv2 mobilenetv2_small
    opt.pretrained = ''
    opt.classifier_type = 'linear'
    opt.loss_type= 'cce'
    opt.alpha_scheduler_type = None
    opt.nclasses = 2
    opt.fake_class_weight = 1
    opt.visdom_port = 8097
    
   # opt.git_commit_sha = '3ab79d6c8ec9b280f5fbdd7a8a363a6191fd65ce' 
    opt.train_transform = tv.transforms.Compose([
            #transforms.MergeItems(True, p=0.1),
            #transforms.LabelSmoothing(eps=0.1, p=0.2),
            # transforms.CustomRandomRotation(30, resample=2),
         #   transforms.CustomResize((125,125)),
            # tv.transforms.RandomApply([
            #     transforms.CustomCutout(1, 25, 75)],p=0.1),
            transforms.CustomGaussianBlur(max_kernel_radius=3, p=0.2),
            transforms.CustomRandomResizedCrop(112, scale=(0.5, 1.0)),
            transforms.CustomRandomHorizontalFlip(),
             tv.transforms.RandomApply([
                 transforms.CustomColorJitter(0.25,0.25,0.25,0.125)],p=0.2),
            transforms.CustomRandomGrayscale(p=0.1),
            transforms.CustomToTensor(),
        #    transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

    opt.test_transform = tv.transforms.Compose([
          #  transforms.CustomResize((125,125)),
            #transforms.CustomRotate(0),
            #transforms.CustomRandomHorizontalFlip(p=0),
            #transforms.CustomCrop((112,112), crop_index=0),
            
            #tv.transforms.RandomApply([
           	  #  tv.transforms.RandomRotation(45)],p=1),
       
            #tv.transforms.RandomApply([
                #tv.transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1), shear=(6, 9), fillcolor=0)],p=1),                
            #tv.transforms.RandomHorizontalFlip(p=1),
            #tv.transforms.RandomApply([
            #transforms.CustomCutout(1, 25, 75)],p=1),
            #transforms.CustomGaussianBlur(max_kernel_radius=3, p=1),
                #transforms.CustomRandomResizedCrop(128, scale=(0.5, 1.0)),
            #tv.transforms.RandomApply([
                #tv.transforms.Resize((int(opt.img_size*1.2),int(opt.img_size*1.2))),
                #tv.transforms.CenterCrop(opt.img_size)],p=1),#center_crop must be right 128
           # tv.transforms.RandomApply([
              #  tv.transforms.Resize((int(opt.img_size*1.2),int(opt.img_size*1.2))),
              #  tv.transforms.RandomCrop(opt.img_size,4,True,0,'constant')],p=0.3),#center_crop must be right 128
            #tv.transforms.RandomApply([
                #tv.transforms.ColorJitter(0.125,0.125,0.125,0.0)],p=1),
                #tv.transforms.ColorJitter(0.25,0.25,0.25,0.5)],p=1),
            #tv.transforms.RandomGrayscale(p=0.2),
                
            #tv.transforms.RandomApply([
                #transforms.CustomGaussianNoise(var = 0.0025, p = 1),
               # transforms.CustomPoissonNoise(p = 0.3)],p = 1),
     
            #transforms.Motion_blur(degree = 15, angle = 15, p = 1),
            #tv.transforms.Resize((int(opt.img_size),int(opt.img_size))),
            transforms.CustomResize(128),
            tv.transforms.ToTensor(),
        #    transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
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
