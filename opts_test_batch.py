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
    opt.val_list = ' ' #test1203 train1203_0first 0120_test_data_djy.txt 0210_test_data 0224_test_data Test0312.txt Test3_2020316_crop all_test_file.txt 20200802_all_live_test_file
    opt.vals_list = '2get_list/test_result/20200802_all_live_test_file.txt'
    
    
    opt.out_root = '/mnt/sdb4/2d_project/sigle_result/'
    opt.out_path = os.path.join(opt.out_root,opt.exp_name)
    opt.out_path_test = '/mnt/sdb4/2d_project/model_sel/1210_sel/'
    #opt.out_path_test = '/mnt/sdb4/2d_project/sigle_result/test0902_fintune_0901_mobilenetv2_small_dropout_0d2_128_Adam_128_bn_smooth_cce_1e-3_20200902_train_list_del_lyb/checkpoints'
    opt.out_error_list='./error_list//1210_sel/'
    opt.model_result_path = './model_result//1210_sel.txt'
    opt.all_model_result_path = './model_result//all_model_1210_sel.txt'
    opt.threshhold = 0.50
    opt.tpr_thd = 0.02
 
 
 
    ### Dataloader options ###
    opt.nthreads = 8
    opt.batch_size = 1024 #280
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
    opt.resume = 'model_14.pth'
    opt.debug = 0

 
    ### Other ###  
    opt.manual_seed = 704
    opt.log_batch_interval= 100
    opt.log_checkpoint = 1
    opt.net_type = 'mobilenetv2_small' #ResNet34CaffeSingle_fc Resnet18_single_online  densenet_121 Resnet18_single_online_bn mobilenetv2
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
            #tv.transforms.CustomToTensor(),
            #tv.transforms.Resize((int(opt.img_size),int(opt.img_size))),
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
