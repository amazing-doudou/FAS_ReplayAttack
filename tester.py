import torch
import torch.nn as nn
import numpy as np
import time, os
import math
#import torchvision.transforms as transforms
#from visdom import Visdom
import cv2
import models, datasets, utils

class Model:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if opt.ngpu else "cpu")
        
        self.model = models.get_model(opt.net_type,opt.pretrained)
        


        self.model = self.model.to(self.device)


        if opt.ngpu>1:
            self.model = nn.DataParallel(self.model)
            
        self.loss = models.init_loss(opt.loss_type)
        self.loss = self.loss.to(self.device)

        self.optimizer = utils.get_optimizer(self.model, self.opt)
        self.lr_scheduler = utils.get_lr_scheduler(self.opt, self.optimizer)
        self.alpha_scheduler = utils.get_margin_alpha_scheduler(self.opt)

        #self.train_loader = datasets.generate_loader(opt,'train') 
        self.test_loader = datasets.generate_loader_test(opt,'val',True) 
        #self.test_loader = datasets.generate_loader(opt,'val',True)       
        
        self.epoch = 0
        self.best_epoch = False
        self.training = False
        self.state = {}
        

        self.train_loss = utils.AverageMeter()
        self.test_loss  = utils.AverageMeter()
        self.batch_time = utils.AverageMeter()
        self.test_metrics = utils.ROCMeter()
        self.best_test_loss = utils.AverageMeter()                    
        self.best_test_loss.update(np.array([np.inf]))

        '''self.visdom_log_file = os.path.join(self.opt.out_path, 'log_files', 'visdom.log')
        self.vis = Visdom(port = opt.visdom_port,
                          log_to_filename=self.visdom_log_file,
                          env=opt.exp_name + '_' + str(opt.fold))

        self.vis_loss_opts = {'xlabel': 'epoch', 
                              'ylabel': 'loss', 
                              'title':'losses', 
                              'legend': ['train_loss', 'val_loss']}

        self.vis_tpr_opts = {'xlabel': 'epoch', 
                              'ylabel': 'tpr', 
                              'title':'val_tpr', 
                              'legend': ['tpr@fpr10-2', 'tpr@fpr10-3', 'tpr@fpr10-4']}

        self.vis_epochloss_opts = {'xlabel': 'epoch', 
                              'ylabel': 'loss', 
                              'title':'epoch_losses', 
                              'legend': ['train_loss', 'val_loss']}'''

    def test(self):
        
        # Init Log file
        for root,dirs,model_names in os.walk('./model/'):
            for model_name in model_names:
                model_path = root+model_name
                print(self.opt.resume)
                if self.opt.resume:
                    self.log_msg('resuming...\n')
                    # Continue training from checkpoint
                    self.load_checkpoint(model_path)
                else:
                     self.log_msg()

                self.test_epoch()

    

        
    def test_epoch(self):
        """
        Calculates loss and metrics for test set
        """
        self.training = False
        torch.set_grad_enabled(self.training)
        self.model.eval()
        
        self.batch_time.reset()
        self.test_loss.reset()
        self.test_metrics.reset()
        time_stamp = time.time()

        f = open(self.opt.out_error_list,'w',buffering = 1)
        print('file is ready')
        count = 0
        for batch_idx, data in enumerate(self.test_loader):
            print('batch_idx',batch_idx)
            (ori_data,target,path),(crop_data,_,crop_path) = data[0],data[1]
            ori_dataset = datasets.patchify(self.opt,ori_data)
            rgb_data = torch.cat([ori_dataset,crop_data],dim = 1)
            rgb_data = rgb_data.to(self.device)
            target = target.to(self.device)
            output = self.model(rgb_data)
            self.test_metrics.update(target.cpu().numpy(), output.cpu().numpy())
            self.check_errorfile(target.cpu().numpy().tolist(), output.cpu().numpy().tolist(),path,crop_path,f)   
            self.log_batch(batch_idx)
            
            if self.opt.loss_type == 'bce':
                target = target.float()
                loss_tensor = self.loss(output.squeeze(), target)
            else:
                loss_tensor = self.loss(output, target)
            self.test_loss.update(loss_tensor.item())

            if self.opt.loss_type == 'cce' or self.opt.loss_type == 'focal_loss':
                output = torch.nn.functional.softmax(output, dim=1)

            elif self.opt.loss_type == 'bce':
                output = torch.sigmoid(output)

            self.test_metrics.update(target.cpu().numpy(), output.cpu().numpy())

            if count == 0:
                target_array = target.cpu().numpy().tolist()
                output_array = output.cpu().numpy().tolist()

                count = count + 1
            else:
                target_array.extend(target.cpu().numpy().tolist())
                output_array.extend(output.cpu().numpy().tolist())
                
            
        
        self.test_metrics.update(target_array, output_array)
        tpr_aver,fpr_aver=self.test_metrics.get_live_tpr_fpr()
        print('test epoch '+str(self.epoch)+' result is:'+'tpr_average:',tpr_aver,'\tfpr_average:',fpr_aver)
        result_msg='test epoch '+str(self.epoch)+' result is:'+'tpr_average:',tpr_aver,'\tfpr_average:',fpr_aver
        self.log_msg(result_msg)
                       
    def calculate_metrics(self, output, target):   
        """
        Calculates test metrix for given batch and its input
        """
        t = target
        o = output
            
        if self.opt.loss_type == 'bce':
            accuracy = (t.byte()==(o>0.5)).float().mean(0).cpu().numpy()  
            batch_result.append(binary_accuracy)
        
        elif self.opt.loss_type == 'cce':
            top1_accuracy = (torch.argmax(o, 1)==t).float().mean().item()

            batch_result.append(top1_accuracy)
        else:
            raise Exception('This loss function is not implemented yet')
                
        return batch_result    
    
    def log_batch(self, batch_idx):
        if batch_idx % self.opt.log_batch_interval == 0:
            cur_len = len(self.train_loader) if self.training else len(self.test_loader)
            cur_loss = self.train_loss if self.training else self.test_loss
            #print('cur_len:',cur_len)
            output_string = 'Train ' if self.training else 'Test '
            output_string +='Epoch {}[{:.2f}%]: [{:.2f}({:.3f}) s]\t'.format(self.epoch,
                                                                          100.* batch_idx/cur_len, self.batch_time.val,self.batch_time.avg)
            
            loss_i_string = 'Loss: {:.5f}({:.5f})\t'.format(cur_loss.val, cur_loss.avg)
            output_string += loss_i_string
                    
            if not self.training:
                output_string+='\n'

                metrics_i_string = 'Accuracy: {:.5f}\t'.format(self.test_metrics.get_accuracy())
                output_string += metrics_i_string
                #metrics_tpr_fpt_string='Accuracy: {:.5f,.5f}\t'.format(self.test_metrics.get_accuracy_tpr_fpr())
                tpr,fpr=self.test_metrics.get_live_tpr_fpr()
                metrics_tpr_fpt_string='\ntpr:'+ str(tpr) +'\tfpr'+ str(fpr)
                output_string += metrics_tpr_fpt_string
                tpr=0
                fpr=0
            print(output_string)
    
    
    def log_msg(self, msg='1'):
        mode = 'a' if msg else 'w'
        data_name = self.opt.val_ori_data_list.split('/')[-1][:-4]
        f = open(self.opt.out_path+'/log_files/'+data_name, mode)
        f.write(msg)
        f.close()
             
    def log_epoch(self):
        """ Epoch results log string"""
        out_train = 'Train: '
        out_test = 'Test:  '
        loss_i_string = 'Loss: {:.5f}\t'.format(self.train_loss.avg)
        out_train += loss_i_string
        loss_i_string = 'Loss: {:.5f}\t'.format(self.test_loss.avg)
        out_test += loss_i_string
            
        out_test+='\nTest:  '
        metrics_i_string = 'TPR@FPR=10-2: {:.4f}\t'.format(self.test_metrics.get_tpr(0.01))
        metrics_i_string += 'TPR@FPR=10-3: {:.4f}\t'.format(self.test_metrics.get_tpr(0.001))
        metrics_i_string += 'TPR@FPR=10-4: {:.4f}\t'.format(self.test_metrics.get_tpr(0.0001))
        out_test += metrics_i_string
            
        is_best = 'Best ' if self.best_epoch else ''
        out_res = is_best+'Epoch {} results:\n'.format(self.epoch)+out_train+'\n'+out_test+'\n'
        
        print(out_res)
        self.log_msg(out_res)

    def vislog_epoch(self):
        x_value = self.epoch
        self.vis.line([self.train_loss.avg], [x_value], 
                        name='train_loss', 
                        win='epoch_losses', 
                        update='append')
        self.vis.line([self.test_loss.avg], [x_value], 
                        name='val_loss', 
                        win='epoch_losses', 
                        update='append')
        self.vis.update_window_opts(win='epoch_losses', opts=self.vis_epochloss_opts)


        self.vis.line([self.test_metrics.get_tpr(0.01)], [x_value], 
                        name='tpr@fpr10-2', 
                        win='val_tpr', 
                        update='append')
        self.vis.line([self.test_metrics.get_tpr(0.001)], [x_value], 
                        name='tpr@fpr10-3', 
                        win='val_tpr', 
                        update='append')
        self.vis.line([self.test_metrics.get_tpr(0.0001)], [x_value], 
                        name='tpr@fpr10-4', 
                        win='val_tpr', 
                        update='append')
        self.vis.update_window_opts(win='val_tpr', opts=self.vis_tpr_opts)
                    
    def create_state(self):
        self.state = {       # Params to be saved in checkpoint
                'epoch' : self.epoch,
                'model_state_dict' : self.model.state_dict(),
                'best_test_loss' : self.best_test_loss,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
            }
    
    def save_state(self):
        if self.opt.log_checkpoint == 0:
                self.save_checkpoint('checkpoint.pth')
        else:
            if (self.epoch % self.opt.log_checkpoint == 0):
                self.save_checkpoint('model_{}.pth'.format(self.epoch)) 
                  
    def save_checkpoint(self, filename):     # Save model to task_name/checkpoints/filename.pth
        fin_path = os.path.join(self.opt.out_path,'checkpoints', filename)
        torch.save(self.state, fin_path)
        if self.best_epoch:
            best_fin_path = os.path.join(self.opt.out_path, 'checkpoints', 'model_best.pth')
            torch.save(self.state, best_fin_path)
           

    def load_checkpoint(self,model_path):                            # Load current checkpoint if exists
        fin_path = os.path.join(model_path)
        print('pth_path:',fin_path,' state:',os.path.isfile(fin_path))
        if os.path.isfile(fin_path):
            print("=> loading checkpoint '{}'".format(fin_path))
            checkpoint = torch.load(fin_path, map_location=lambda storage, loc: storage)
            self.epoch = checkpoint['epoch'] + 1
            self.best_test_loss = checkpoint['best_test_loss']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            #self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

        '''if os.path.isfile(self.visdom_log_file):
                self.vis.replay_log(log_filename=self.visdom_log_file)'''
            
    def check_errorfile(self, target, output, path, crop_path, f):
        output = np.argmax(output,axis=1)
        for index,label in enumerate(target):
            if (label == 0 and output[index] in [1,2]):
                img_path = path[index]
                crop_img_path = crop_path[index]
                f.write(img_path+'\n')
                img = cv2.imread(img_path)
                crop_img = cv2.imread(crop_img_path)
                img_name = img_path.split('/')[-1]
                data_name = self.opt.val_ori_data_list.split('/')[-1][:-4]
                print(img_path)
                print(crop_img_path)
                result_path = './result/'+data_name+'/pre_photo_and_replay/'
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                cv2.imwrite(result_path+'/'+img_name+'.jpg',img)
                cv2.imwrite(result_path+'/'+img_name+'_crop.jpg',crop_img)
            elif (label in [1,2] and output[index] == 0):
                img_path = path[index]
                crop_img_path = crop_path[index]
                img = cv2.imread(img_path)
                crop_img = cv2.imread(crop_img_path)
                img_name = img_path.split('/')[-1]
                data_name = self.opt.val_ori_data_list.split('/')[-1][:-4]
                result_path = './result/'+data_name+'/pre_live/'
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                cv2.imwrite(result_path+'/'+img_name+'.jpg',img)
                cv2.imwrite(result_path+'/'+img_name+'_crop.jpg',crop_img)