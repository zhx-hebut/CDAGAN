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
from predict import test_images,compute_3d_dice
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
    parser.add_argument('--base_dir', type=str,default='RES-NNI-train-brats',help='base dir name')
    parser.add_argument('--train_list', type=str,default='list_tvt/slice_all_train.txt',help='a list of train data')
    parser.add_argument('--val_list', type=str,default='list_tvt/slice_all_val.txt',help='a list of val data')
    parser.add_argument('--data_path', type=str,default='/path/to/your/data',help='a list of val data')
    # gpu 
    parser.add_argument('--gpu_id', type=int,default=5, help='base dir name')
    parser.add_argument('--seed', type=int,default=1111,help='seed')
    # base param
    parser.add_argument('--max_epoch', type=int,default=73,help='maximum epoch')
    parser.add_argument('--batch_size', type=int,default=42,help='batch size per gpu')
    parser.add_argument('--base_lr', type=float,default=0.0001,help='segmentation network learning rate')
    parser.add_argument('--weight_decay', type=float,default=1e-4,help='weight decay(L2 Regularization)')   
    parser.add_argument('--optim_name', type=str,default='adam',help='optimizer name')
    parser.add_argument('--loss_name', type=str,default='bce', help='loss name')   
    parser.add_argument('--lr_scheduler', type=str,default='warmupCosine',help='lr scheduler') 
    # change param
    parser.add_argument('--images_ratio', type=float,default=1,help='contras research need')
    parser.add_argument('--percent', type=float,default=1,help='percent research need')
    parser.add_argument('--labeled_ratio', type=float,default=1,help='labeled_ratio')
    parser.add_argument('--modal', nargs='+',default=['t1ce'],help='contras research need eg. python --modal t1ce da')     
    args = parser.parse_args()
    return args

def train_net(start_time,base_dir,data_path,device,
            lr_scheduler='warmupMultistep',
            max_epoch=80,
            batch_size=60,
            images_ratio=1,
            base_lr=0.001,
            weight_decay=0.000001,
            optim_name='adam',
            loss_name='bce',
            labeled_ratio=1,
            modal = ['t1ce','da'],
            percent = 0.2):
    local_vars_dict = {}
    for var in train_net.__code__.co_varnames: # 遍历train_net函数中所有的变量
        if var == 'local_vars_dict':
            break
        local_vars_dict[var] = locals()[var]

    global save_threshold
    warm_up_epochs = int(max_epoch * 0.1)

    """定义网络"""
    image_channels = len(modal)
    net = UNet(image_channels,class_num,32,bilinear=False)
    net.to(device=device)  #RandomGenerator()
    net_name = str(net)[0:str(net).find('(')]
    """load data"""
    labeled_num = round(len(train_volume)*labeled_ratio) # 
    train_lab_volumes,_ = split_labeled_volume(train_volume,labeled_num,is_predefined=False)
    vol_percent = random.sample(train_lab_volumes,int(percent*len(train_lab_volumes))) # one 
    train_db = BaseDataSets(modal,data_path,data_path_name,"train",train_lab_volumes,train_list,images_ratio,vol_percent)
    val_db = BaseDataSets(modal,data_path,data_path_name_val,"val",volume_val_list,val_list_full,images_ratio,vol_percent)
    val_db_3d = BaseDataSets(modal,data_path,data_path_name_val,"val_3d",volume_val_list,val_list_full,images_ratio,vol_percent)
    logging.info("Train Data: Total data: {}.".format(len(train_db)))

    def worker_init_fn(worker_id):#为每一个worker设置固定的seed
        set_random_seed(seed_num + worker_id)

    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory= True, drop_last=False,worker_init_fn=worker_init_fn)
    val_loader_2d = DataLoader(val_db, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False,worker_init_fn=worker_init_fn)
    slices_list = val_db_3d.__sampleList__()
    batch_samplerPatient = PatientBatchSampler(slices_list,volume_val_list)
    val_loader_3d = DataLoader(val_db_3d,batch_sampler=batch_samplerPatient, num_workers=0, pin_memory=True,worker_init_fn=worker_init_fn)

    # optimizier
    if optim_name=='adam':
        optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=weight_decay)
    elif optim_name=='sgd':
        optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9,weight_decay=weight_decay)
    elif optim_name=='adamW':
        optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=weight_decay)    
    # lr scheduler
    if lr_scheduler=='warmupMultistep':
        lr1,lr2,lr3 = int(max_epoch*0.25) , int(max_epoch*0.4) , int(max_epoch*0.6)
        lr_milestones = [lr1,lr2,lr3]
        warm_up_with_multistep_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
                                                else 0.1**len([m for m in lr_milestones if m <= epoch])
        scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = warm_up_with_multistep_lr)
    elif lr_scheduler=='warmupCosine':
        warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
                                else 0.5 * ( math.cos((epoch - warm_up_epochs) /(max_epoch - warm_up_epochs) * math.pi) + 1)
        scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = warm_up_with_cosine_lr)
    elif lr_scheduler=='autoReduce':
        scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=6, verbose=True, cooldown=2,min_lr=0)
    # loss
    if loss_name=='bce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_name=='focal':
        criterion = FocalLoss(alpha=0.75, gamma=2,reduce=False) 
    dice_loss = DiceLoss(n_classes=3)
    dice_loss1 = BinaryDiceLoss()
    param_str = "Starting training:"
    for var in list(local_vars_dict.keys()):
        if var != 'device':
            var_value = local_vars_dict[var]
            param_str += "\n\t" + var + ":" + " "*(15-len(var)) + str(var_value)
    logging.info(param_str+f'''\n\tNet Name:\t\t{net_name}\n\tInput Channel:\t{image_channels}\n\tClasses Num:\t{class_num}\n\tImages Shape:\t{img_h}*{img_w}''')


    """Star Train"""
    lr_curve = list()
    train_log = AverageMeter()
    val_log = AverageMeter()
    if labeled_ratio == 1:
        nums_batch = len(train_loader)
    logging.info("iter img num per epoch: {}".format(nums_batch * batch_size))
    for epoch in range(max_epoch):
        net.train()
        with tqdm(total=nums_batch * batch_size, desc=f'Epoch {epoch + 1}/{max_epoch}', unit='img', leave=is_leave) as pbar:
            for i,batch in enumerate(cycle(train_loader)):
                if i == nums_batch: break
                imgs = batch['image']
                true_masks = batch['mask']
            
                assert imgs.shape[1] == image_channels, \
                    f'Network has been defined with {image_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                outputs = net(imgs)

                # loss
                masks_pred = torch.sigmoid(outputs) # 0-1
                loss_base = criterion(outputs, true_masks) # 
                loss_dice = dice_loss(masks_pred, true_masks) # (masks_pred, true_masks,[1,0,0]) weight class
                loss_dice1 = dice_loss1(masks_pred, true_masks)
                loss = loss_base + loss_dice*8 + loss_dice1
                train_log.add_value({"loss": loss.item()}, n=1)  
                if torch.isnan(loss).any() is True:
                    print(nums_batch)
                if torch.isinf(loss).any() is True:
                    print(nums_batch)
                # print(loss)
                optimizer.zero_grad()
                loss.backward()  
                optimizer.step() 
                
                pbar.set_postfix(**{'loss_base(b)': loss_base.item(),'loss_dice(b)': loss_dice.item(),'loss_dice1(b)': loss_dice1.item()})
                # 2D dice     
                pred = (masks_pred > 0.5).float()    
                dice_sum,num = batch_dice(pred.cpu().data, true_masks.cpu())
                dice = dice_sum / num
                train_log.add_value({"dice": dice}, n=1)
                if set_args == False:
                    pbar.update(batch_size) #imgs.shape[0]
                
            train_log.updata_avg()
            mean_loss = train_log.res_dict["loss"][epoch]
            mean_dice = train_log.res_dict["dice"][epoch]

       # validate the model
        net.eval()
        n_val_2d = len(val_loader_2d)  
        n_val_3d = len(val_loader_3d)  
        if epoch % val_3d_interval==0:
            compute_3d = True
            n_val = n_val_3d
            val_loader = val_loader_3d
        else:
            compute_3d = False
            n_val = n_val_2d
            val_loader = val_loader_2d

        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for j,batch_val in enumerate(val_loader):
                imgs = batch_val['image']
                true_masks = batch_val['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                if compute_3d==True:
                    batch_num = imgs.shape[0]
                    batch_fore = int(batch_num / 2)
                    imgs_fore = imgs[:batch_fore]
                    # imgs_mid = imgs[batch_fore:batch_fore*2]
                    imgs_after = imgs[batch_fore:]

                    with torch.no_grad():
                        outputs_fore = net(imgs_fore)
                        # outputs_mid = net(imgs_mid)
                        outputs_after = net(imgs_after)
                    outputs = torch.cat([outputs_fore, outputs_after], dim=0)
                    mask_pred = torch.sigmoid(outputs)
                    # 计算dice
                    pred = mask_pred.ge(0.5).float()
                    pred_np = pred.cpu().data.numpy().astype("uint8")
                    true_np = true_masks.cpu().numpy().astype("uint8")
                    metric_list = []
                    for i in range(0, class_num):
                        metric_list.append(dice_all(
                            pred_np[:,i,:,:], true_np[:,i,:,:]))
                    dice_val_3d = np.mean(metric_list)
                    val_log.add_value({"dice_3d": dice_val_3d}, n=1)
                    dice_val_sum,nidus_num,nidus_start = compute_dice(pred.cpu().data, true_masks.cpu(),deNoNidus=True)
                    nidus_end = nidus_start+nidus_num-1
                    outputs = outputs[nidus_start:nidus_end+1]
                    mask_pred = mask_pred[nidus_start:nidus_end+1]
                    true_masks = true_masks[nidus_start:nidus_end+1]
                else:
                    with torch.no_grad():
                        outputs = net(imgs)
                    # 计算dice
                    mask_pred = torch.sigmoid(outputs)
                    pred = mask_pred.ge(0.5).float()
                    dice_val_sum,nidus_num = batch_dice(pred.cpu().data, true_masks.cpu())
                val_log.add_value({"dice": dice_val_sum}, n=nidus_num)
                # 计算loss
                loss_val_base = criterion(outputs, true_masks)
                loss_val_dice = dice_loss(mask_pred, true_masks)
                loss_val = loss_val_base + loss_val_dice
                val_log.add_value({"loss": loss_val.item()}, n=1)
                if set_args == False:
                    pbar.update()
            val_log.updata_avg()
            valid_loss_mean = val_log.res_dict["loss"][epoch]
            valid_dice_mean = val_log.res_dict["dice"][epoch]

            if epoch % val_3d_interval==0:
                valid_dice_3d_mean = val_log.res_dict["dice_3d"][-1]
                logging.info(
                    'Epoch:[{:0>3}/{:0>3}], Train Loss: {:.4f} , Val Loss: {:.4f} ,Train Dice: 2d {:.4f},  Val Dice: 2d  {:.4f} 3D {:.4f}'.format(
                            epoch,max_epoch,          mean_loss,  valid_loss_mean,            mean_dice ,       valid_dice_mean , valid_dice_3d_mean))
                all_valid_3d = valid_dice_3d_mean
                metrics = {
                    "default":valid_dice_3d_mean,
                    "dice_2D":valid_dice_mean,
                }
                if set_args == True:
                    nni.report_intermediate_result(metrics)
            else:
                logging.info(
                    'Epoch:[{:0>3}/{:0>3}], Train Loss: {:.4f} , Val Loss: {:.4f} ,Train Dice: 2d {:.4f},  Val Dice: 2d  {:.4f}'.format(
                            epoch,max_epoch,          mean_loss,  valid_loss_mean,            mean_dice ,       valid_dice_mean))
                
        ##更新学习率
        if lr_scheduler=='autoReduce':
            scheduler_lr.step(valid_loss_mean)
        else:
            scheduler_lr.step()
        lr_epoch = optimizer.param_groups[0]['lr']
        lr_curve.append(lr_epoch)

    valid_dice_3d_mean = val_log.res_dict["dice_3d"][-1]
    metrics_final = {
        "default":valid_dice_3d_mean,
        "dice_2D":valid_dice_mean,
    }
    if set_args == True:
        nni.report_final_result(metrics_final)
    return train_log.res_dict,val_log.res_dict,lr_curve,net,all_valid_3d

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
    img_h,img_w = 240,240

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

    logging.info('============ Start  train ==============')
    global is_leave
    global seed_num
    if set_args==True:
        args = init_info[0]
        seed_num = args['seed']
        logging.info(f'set random seed: {seed_num}')
        set_random_seed(seed_num)
        is_leave = False
        base_dir = args['base_dir']
        train_log,val_log,lr_curve,net,all_valid_3d = train_net(start_time,base_dir,data_path,device,
            args['data_name'],args['lr_scheduler'],args['max_epoch'],args['batch_size'],args['images_ratio'],args['base_lr'],
            args['weight_decay'],args['optim_name'],args['loss_name'],args['labeled_ratio'])
    else:
        args = init_info
        seed_num = 1111
        logging.info(f'set random seed: {seed_num}')
        set_random_seed(seed_num) 
        is_leave = True       
        base_dir = args.base_dir             
        train_log,val_log,lr_curve,net,all_valid_3d = train_net(start_time,base_dir,data_path,device,
            args.lr_scheduler,args.max_epoch,args.batch_size,args.images_ratio,args.base_lr,
            args.weight_decay,args.optim_name,args.loss_name,args.labeled_ratio,args.modal,args.percent)

    if set_args==True and all_valid_3d <= save_threshold:
        logging.info('Train Finish!') 
    else:
        if set_args==True:
            experiment_id = init_info[1]
            base_dir = args['base_dir']
            base_dir = base_dir + '-' + experiment_id + '/res-'
            base_dir = base_dir + 'LR_{}-WD_{}-seed_{}-{:.4f}'.format(args['base_lr'],args['weight_decay'],args['seed'],all_valid_3d)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            backup_code(base_dir)
        else:
            base_dir = init_info.base_dir

        net_name = str(net)[0:str(net).find('(')]
        mode_path_name = base_dir + '/' + f'model_{net_name}_last.pth'
        torch.save(net,mode_path_name)
        logging.info('Model saved !')    

        """画    图"""
        plot_dice_loss(train_log,val_log,val_3d_interval,lr_curve,base_dir)
    
        logging.info('============= Start  Test ==============')
        model_path_last = os.path.join(base_dir,'model_{}_last.pth'.format(net_name))
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
                                    img_path,true_path,in_files_full_list,volume_test_list,out_dir=base_dir)
        elif class_num==3:
            mean_dice_3d_WT,mean_dice_3d_TC,mean_dice_3d_ET = compute_3d_dice(args.modal,net,device,img_path,
                                    true_path,in_files_full_list,volume_test_list,out_dir=base_dir)
            logging.info("Mean 3d dice WT on all patients:{:.4f} ".format(mean_dice_3d_WT))
            logging.info("Mean 3d dice TC on all patients:{:.4f} ".format(mean_dice_3d_TC))
            logging.info("Mean 3d dice ET on all patients:{:.4f} ".format(mean_dice_3d_ET))
            mean_dice_all_3d = (mean_dice_3d_WT + mean_dice_3d_TC + mean_dice_3d_ET) / 3
        
        logging.info("Mean 3d dice on all patients: {:.4f} ".format(mean_dice_all_3d))
        new_lr_path = os.path.join(base_dir,'lr-3d_dice-{:.4f}.jpg'.format(mean_dice_all_3d))
        lr_path = glob.glob(base_dir + '/lr*.jpg')
        os.rename(lr_path[0],new_lr_path)

        time_toc = time.time()
        time_s = time_toc - time_tic
        time_end = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time_toc))
        logging.info("Train finished time: {}".format(time_end))
        logging.info("Time consuming:{:.2f} min in train and test".format(time_s / 60))
        if set_args==True:
            experiment_id = init_info[1]
            trial_id = init_info[2]
            trial_log_path = os.path.join('RES_NNI_LOG',experiment_id,'trials',trial_id,'trial.log')
            shutil.copy(trial_log_path, base_dir + '/' + 'trial.log')

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
        log_path = os.path.join(args.base_dir, 'training.log') 
        sys.stdout = Logger(log_path=log_path)
        set_logging(log_path=log_path)
        init_info = args
        try:
            main(init_info)
        except Exception as exception:
            logging.exception(exception)
            raise
