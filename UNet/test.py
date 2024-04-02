import os
import sys
import time
import glob
import math
import torch
import shutil
import random
import logging
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from itertools import cycle
import matplotlib.pyplot as plt
from collections import OrderedDict


import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from nets.unet_drop_noSig import UNet
from utils.metrics import dice as dice_all
from utils.metrics import batch_dice,compute_dice
from utils.losses import FocalLoss,DiceLoss,logits_mse_loss,BinaryDiceLoss
from utils.util import set_logging,Logger,read_list,plot_base,plot_dice,AverageMeter,save_img_inter,plot_dice_loss
from predict_brats import test_images,compute_3d_dice
from dataloader.dataset import BaseDataSets,PatientBatchSampler,RandomGenerator

gpu_list = [7]
set_args = False
val_3d_interval = 8 
class_num = 3 

data_path_name = {'img':'train/img','seg':'train/seg'}
data_path_name_val = {'img':'val/img','seg':'val/seg'}
data_path_name_test = {'img':'test/img','seg':'test/seg'}

def set_argparse():
    parser = argparse.ArgumentParser()
    # data path
    parser.add_argument('--base_dir', type=str,default='/path/to/your/data',help='base dir name')
    parser.add_argument('--train_list', type=str,default='list_tvt/slice_all_train.txt',help='a list of train data')
    parser.add_argument('--val_list', type=str,default='list_tvt/slice_all_val.txt',help='a list of val data')
    parser.add_argument('--data_path', type=str,default='/path/to/your/data',help='a list of val data')
    # gpu 
    parser.add_argument('--gpu_id', type=int,default=1, help='base dir name')
    parser.add_argument('--seed', type=int,default=1111,help='seed')
    # base param
    parser.add_argument('--max_epoch', type=int,default=73,help='maximum epoch')
    parser.add_argument('--batch_size', type=int,default=45,help='batch size per gpu')
    parser.add_argument('--base_lr', type=float,default=0.0001,help='segmentation network learning rate')
    parser.add_argument('--weight_decay', type=float,default=1e-4,help='weight decay(L2 Regularization)')   
    parser.add_argument('--optim_name', type=str,default='adam',help='optimizer name')
    parser.add_argument('--loss_name', type=str,default='bce', help='loss name')   
    parser.add_argument('--lr_scheduler', type=str,default='warmupCosine',help='lr scheduler') 
    # change param
    parser.add_argument('--images_ratio', type=float,default=1,help='contras research need')
    parser.add_argument('--percent', type=float,default=1,help='percent research need')
    parser.add_argument('--labeled_ratio', type=float,default=1,help='labeled_ratio')
    parser.add_argument('--modal', nargs='+',default=['flair'],help='contras research need eg. python --modal t1ce da')     
    args = parser.parse_args()
    return args

def main(init_info):
    """选择GPU ID"""
    gpu_num = int(init_info.gpu_id)
    gpu_list = [gpu_num]
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device : {device}\n'
                 f'\tGPU ID is [{os.environ["CUDA_VISIBLE_DEVICES"]}],using {torch.cuda.device_count()} device\n'
                 f'\tdevice name:{torch.cuda.get_device_name(0)}')

    start_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    global img_h,img_w
    data_path = init_info.data_path
    img_h,img_w = 256,256

    # load data path
    global train_volume,train_list
    train_volume_path = '/list_tvt/volume_train.txt'
    train_volume = read_list(data_path + train_volume_path)
    train_list = 'list_tvt/slice_all_train.txt' #train 2D
 
    global volume_val_list
    volume_val_list = read_list(data_path+'/list_tvt/volume_val.txt') # val 3D
    global val_list_full
    val_list_full = 'list_tvt/slice_all_val.txt'

    test_list_path = data_path+'/list_tvt/test_imgs.txt'
    global volume_test_list
    volume_test_list = read_list(data_path+'/list_tvt/volume_test.txt')
    global test_list_full
    test_list_full = 'list_tvt/slice_all_test.txt'
    time_tic = time.time()

    logging.info('============= Start  Test ==============')
    model_path_last = os.path.join(args.base_dir,'model_UNet_last.pth')
    net = torch.load(model_path_last, map_location=device)
    net.to(device=device)
    net.eval()
    logging.info("Model loaded !")
    #定义数据集路径
    img_path = os.path.join(data_path,data_path_name_test['img'])
    true_path = os.path.join(data_path,data_path_name_test['seg'])
    
    logging.info("==========Evaluate 3d dice and 2d dice======")
    full_val_list_path = '{}/{}'.format(data_path,test_list_full)
    in_files_full_list = read_list(full_val_list_path) #png
    if class_num==1:
        mean_dice_all_3d = compute_3d_dice(net,device,
                                img_path,true_path,in_files_full_list,volume_test_list,out_dir=args.base_dir)
    elif class_num==3:
        mean_dice_3d_WT,mean_dice_3d_TC,mean_dice_3d_ET = compute_3d_dice(args.modal,net,device,img_path,
                                true_path,in_files_full_list,volume_test_list,out_dir=args.base_dir)
        logging.info("Mean 3d dice WT on all patients:{:.4f} ".format(mean_dice_3d_WT))
        logging.info("Mean 3d dice TC on all patients:{:.4f} ".format(mean_dice_3d_TC))
        logging.info("Mean 3d dice ET on all patients:{:.4f} ".format(mean_dice_3d_ET))
        mean_dice_all_3d = (mean_dice_3d_WT + mean_dice_3d_TC + mean_dice_3d_ET) / 3
    
    logging.info("Mean 3d dice on all patients: {:.4f} ".format(mean_dice_all_3d))
    time_toc = time.time()
    time_s = time_toc - time_tic
    time_end = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time_toc))
    logging.info("Train finished time: {}".format(time_end))
    logging.info("Time consuming:{:.2f} min in train and test".format(time_s / 60))

def split_labeled_volume(train_volume=[],labeled_num=0,is_predefined=False):
    if labeled_num == len(train_volume):
        return train_volume,None
    if is_predefined != True:
        train_lab_volumes = train_volume[:labeled_num]
        train_unlab_volumes = train_volume[labeled_num:]
        return train_lab_volumes,train_unlab_volumes
        
def set_random_seed(seed_num):
    if seed_num != '':
        # logging.info(f'set random seed: {seed_num}')
        cudnn.benchmark = False      #False：不进行最优卷积搜索，以控制CUDNN种子
        cudnn.deterministic = True   #True ：调用相同的CuDNN的卷积操作，以控制CUDNN种子
        random.seed(seed_num)     #为python设置随机种子
        np.random.seed(seed_num)  #为numpy设置随机种子
        torch.manual_seed(seed_num) # 为CPU设置随机种子
        torch.cuda.manual_seed(seed_num) # 为当前GPU设置随机种子 torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

def backup_code(base_dir):
    ###备份当前train代码文件及dataset代码文件
    code_path = os.path.join(base_dir, 'code') 
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    train_name = os.path.basename(__file__)
    dataset_name = 'dataset.py'
    net_name = 'unet_drop_noSig.py'
    shutil.copy('nets/' + net_name, code_path + '/' + net_name)
    shutil.copy('dataloader/' + dataset_name, code_path + '/' + dataset_name)
    shutil.copy(train_name, code_path + '/' + train_name)

if __name__ == '__main__':

    if set_args==True:
        import nni
        from nni.utils import merge_parameter
        print('WARNING!!! Using argparse for parameters to obtain ')
        logging = logging.getLogger('NNI')
        try:
            # get parameters form tuner
            tuner_params = nni.get_next_parameter()
            experiment_id = nni.get_experiment_id()
            trial_id = nni.get_trial_id()
            logging.debug(tuner_params)
            params = vars(merge_parameter(set_argparse(), tuner_params))
            base_dir_init = params['base_dir'] + '-' + experiment_id
            if not os.path.exists(base_dir_init):
                os.makedirs(base_dir_init)
            print(params)
            init_info = [params,experiment_id,trial_id]
            main(init_info)
        except Exception as exception:
            logging.exception(exception)
            raise
    else:
        args= set_argparse()
        # assert 'res-' in args.base_dir, \
        #         f'base_dir should include string:\'res-\',but base_dir is \'{args.base_dir}\'.'
        # base_dir = args.base_dir.replace('res-',f'res-{data_name}-',1)
        if not os.path.exists(args.base_dir):
            os.makedirs(args.base_dir)
        backup_code(args.base_dir)
        log_path = os.path.join(args.base_dir, 'test.log') 
        sys.stdout = Logger(log_path=log_path)
        set_logging(log_path=log_path)
        init_info = args
        try:
            main(init_info)
        except Exception as exception:
            logging.exception(exception)
            raise
