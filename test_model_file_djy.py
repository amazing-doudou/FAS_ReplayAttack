import argparse,json,random,os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision as tv

from tester_file import Model
from opts_test_djy import get_opts

################ setup ############

opt = get_opts()
out_path = opt.out_root + opt.exp_name
################ setup ############

def main():
    
    # Create working directories
    try:
        os.makedirs(out_path)
        os.makedirs(os.path.join(out_path,'checkpoints'))
        os.makedirs(os.path.join(out_path,'log_files'))
        print( 'Directory {} was successfully created.'.format(out_path))
                   
    except OSError:
        print( 'Directory {} already exists.'.format(out_path))
        pass
    
    
    # Training
    M = Model(opt)
    M.test()

    
if __name__ == '__main__':
    main()


