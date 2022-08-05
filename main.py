import argparse,json,random,os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision as tv

from trainer import Model
from opts import get_opts

################ setup ############

opt = get_opts()
out_path = opt.out_root + opt.exp_name
################ setup ############

def main():
    
    # Create working directories
    print('**********************************lxf traning******************************* ')
    try:
        if (os.path.exists(out_path) is True):
            print('Directory {} has exists.'.format(out_path))
        os.makedirs(out_path)
        print(out_path)
        os.makedirs(os.path.join(out_path,'checkpoints'))
        os.makedirs(os.path.join(out_path,'log_files'))
        os.makedirs(os.path.join(out_path,'convert_to_caffe'))
        print( 'Directory {} was successfully created.'.format(out_path))
                   
    except OSError:
        print( 'Directory {} already exists.'.format(out_path))
        pass
    
    
    # Training
    M = Model(opt)
    M.train()
    '''
    TODO: M.test()
    '''
    
if __name__ == '__main__':
    main()


