import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataset import med
from model import VGG19_bn,Resnet18,Alexnet
from datapre import balance
from distutils.version import LooseVersion
import os
import shutil
import numpy as np
import argparse
import datetime
import logging
import random

import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score, precision_recall_fscore_support

# import nni
# from nni.utils import merge_parameter 

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--gpu_id', type=int, default= 1 ,
                    help='location of the log path')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--val_batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epoch_ratio', type=int, default=2, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--num_epochs', type=int, default=80, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--min_epochs', type=int, default=40, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--best_epoch', type=int, default=0, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--max_acc', type=float, default=0., metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--DA', type=int, default=1,
                    help='Tumor of the validation dataset path')
parser.add_argument('--ov_do', type=str, default='no',
                    help='downsample|| oversample|| no')
parser.add_argument('--percent', type=float, default= 0.2 ,
                    help='0-1 percent')
# image path
parser.add_argument('--img1dir', type=str, default='/path/to/your/dataset/trainB/flair',
                    help='No tumor of the training dataset path')
parser.add_argument('--img0dir', type=str, default='/path/to/your/dataset/trainA/flair',
                    help='Tumor of the training dataset path')
parser.add_argument('--DApaths', type=str, default='/path/to/your/results/DA',
                    help='Syn of the training dataset path')
parser.add_argument('--val1dir', type=str, default='/path/to/your/dataset/valB/flair',
                    help='No tumor of the validation dataset path')
parser.add_argument('--val0dir', type=str, default='/path/to/your/dataset/valA/flair',
                    help='Tumor of the validation dataset path')                    
parser.add_argument('--test1dir', type=str, default='/path/to/your/dataset/testB/flair',
                    help='Tumor of the validation dataset path')
parser.add_argument('--test0dir', type=str, default='/path/to/your/dataset/testA/flair',
                    help='Tumor of the validation dataset path')
# others
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--use_mixed_precision', action='store_true', default=True,
                    help='use mixed precision for training')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def makelists(dir):
    imglist = []
    for root,_,files in os.walk(dir):
        for file in files:
            if file.endswith('.png'):
                imglist.append(os.path.join(root,file))
    return imglist

def makedapaths(dir,da):
    da1dir = os.path.join(dir,'trainA')
    da0dir = os.path.join(dir,'trainB')
    return da1dir,da0dir

def set_random_seed(seed_num):
    cudnn.benchmark = False      #False：不进行最优卷积搜索，以控制CUDNN种子
    cudnn.deterministic = True   #True ：调用相同的CuDNN的卷积操作，以控制CUDNN种子
    random.seed(seed_num)     #为python设置随机种子
    np.random.seed(seed_num)  #为numpy设置随机种子
    torch.manual_seed(seed_num) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_num) # 为当前GPU设置随机种子 torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

def train_mixed_precision(args, epoch, scaler, model, optimizer, train_loader, train_dataset,criterion):
    model.train()
    print("Current Learning rate:%f" % (optimizer.param_groups[0]['lr']))
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward and optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()        
        #scheduler.step()

        if (i+1) % args.log_interval == 0:
            with open(os.path.join(args.mode_path,'Log_train.txt'), 'a') as log:
                log.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, i * len(images), len(train_dataset),
                        100. * i / len(train_loader), loss.item()))
            print ("Epoch: {}, Step [{}/{}], Loss: {:.4f}"
                .format(epoch,  i, len(train_dataset)//args.batch_size, loss.item()))

    # torch.save(model, os.path.join(mode_path,'{}-{}.ckpt'.format(modelname,epoch)))

def train_epoch(args, epoch, model, optimizer, train_loader, train_dataset,criterion):
    model.train()
    print("Current Learning rate: %f" % (optimizer.param_groups[0]['lr']))
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()        
        #scheduler.step()

        if (i+1) % args.log_interval == 0:
            with open(os.path.join(args.mode_path,'Log_train.txt'), 'a') as log:
                log.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, i * len(images), len(train_dataset),
                        100. * i / len(train_loader), loss.item()))
            print ("Epoch: {}, Step [{}/{}], Loss: {:.4f}"
                .format(epoch,  i, len(train_dataset), loss.item()))

def val(args, model, epoch, val_loader, val_dataset, criterion):   
    # Validation
    # 修改了指标数：从2个指标变为5个指标

    predicts = []
    truelabels = []
    loss = 0.
    with torch.no_grad():
        for i,(images, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()
            model.eval()
            outputs = model(images)
            _,predict = torch.max(outputs.data,1)
            loss += criterion(outputs, labels)
            predicts += predict.tolist()
            truelabels += labels.tolist()
        loss /= len(val_dataset)
        acc = accuracy_score(truelabels,predicts)
        auc = roc_auc_score(truelabels,predicts)
        # 以下为增加的指标
        recall = recall_score(truelabels,predicts)
        f1score = f1_score(truelabels,predicts)
        pre = precision_score(truelabels,predicts)
        # pre_recall = precision_recall_fscore_support(truelabels,predicts)
        with open(os.path.join(args.mode_path,'Logval.txt'),'a') as log:
            log.write('epoch:{}, lr:{}, val acc:{}, val auc:{}, val recall:{}, val f1score:{},\
                      val pre:{}\n'.format(epoch,args.lr,acc,auc,recall,f1score,pre))
        print('acc,auc,recall,f1score,pre:\n',acc,auc,recall,f1score,pre)
        # all_metric = (acc+auc+recall+f1score+pre)/5
        all_metric = (acc+auc)/2
        if all_metric > args.max_acc:
            args.best_epoch = epoch
            args.max_acc = all_metric
            args.min_epochs = min(args.num_epochs,max(args.min_epochs,int(epoch * args.epoch_ratio)))
            with open(os.path.join(args.mode_path,'Logval.txt'),'a') as log:
                log.write('Find the best epoch:{}\n val acc:{}, val auc:{}, val recall:{}, \
                          val f1score:{}, val pre:{}\n'.format(epoch,acc,auc,recall,f1score,pre))
            print('Find the best epoch:{}\n the acc is {}\n the auc is {}\n the recall is {}\n \
                the f1score is {}\n the pre is {}\n'.format(epoch,acc,auc,recall,f1score,pre))
            # with open(os.path.join(args.mode_path,'Logval.txt'),'a') as log:
            #     log.write('Find the best epoch:{}, val acc:{}, val auc:{}\n'.format(epoch,acc,auc))
            # print('Find the best epoch:{}, the acc & auc is {},{}'.format(epoch,acc,auc))
            
            # if epoch > 5:
            #     torch.save(model, os.path.join(args.mode_path,'{}_{}.ckpt'.format(epoch,args.lr)))
            torch.save(model, os.path.join(args.mode_path,'{}_{}.ckpt'.format(epoch,args.lr)))

        # metrics = {
        #     "default": acc,
        #     # "AUC": auc,
        #     # "val_loss":loss,
        # }
        # nni.report_intermediate_result(metrics)   

        return args.best_epoch,acc,auc,recall,f1score,pre 
    
def test(args,best_epoch,test_loader):
   
    #####################################################################
    model_path = os.path.join(args.mode_path,'{}_{}.ckpt'.format(best_epoch,args.lr))  #训练完成后最好的模型的位置
    test_path = args.mode_path                                       
    with torch.no_grad():
        model = torch.load(model_path,map_location=lambda storage, loc: storage.cuda()).eval()
        predicts = []
        truelabels = []
        for i, (images, labels) in enumerate(test_loader):
            # print(i)
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _,predict = torch.max(outputs.data,1)
            predicts += predict.tolist()
            truelabels += labels.tolist()
        acc = accuracy_score(truelabels,predicts)
        auc = roc_auc_score(truelabels,predicts)
        # 以下为增加的指标
        recall = recall_score(truelabels,predicts)
        f1score = f1_score(truelabels,predicts)
        pre = precision_score(truelabels,predicts)
        # pre_recall = precision_recall_fscore_support(truelabels,predicts)
        # print(acc,auc)
        with open(os.path.join(test_path,'Logtest.txt'),'a') as log:
            log.write('Test acc:{}\n Test auc:{}\n Test recall:{}\n \
                Test f1score:{}\n Test pre:{}\n'.format(acc,auc,recall,f1score,pre))
       
# Input configuration
###########################################################################
def main(args):
    # save_threshold = 0.8
    seed = args.seed
    set_random_seed(seed)
    logging.info(f'set random seed: {seed}')
    if (args.use_mixed_precision and LooseVersion(torch.__version__)
            < LooseVersion('1.6.0')):
        raise ValueError("""Mixed precision is using torch.cuda.amp.autocast(),
                            which requires torch >= 1.6.0""")
       
    balance_mode = args.ov_do
    #balance_func = getattr(balance,'balance_'+balance_mode)
    img1dir = args.img1dir
    img0dir = args.img0dir
    img1list = makelists(img1dir)
    img0list = makelists(img0dir)

    DA = args.DA
    with_da = 'none' if DA == 0 else DA
    DApaths = args.DApaths
    
    if DA:
        da1dir,da0dir = makedapaths(DApaths,DA)
        da1list = makelists(da1dir)
        da0list = makelists(da0dir)
        da1list_per = da1list[:int((args.percent)*(len(da1list)))]
        da0list_per = da0list[:int((args.percent)*(len(da0list)))]       
        img1list += da1list_per
        img0list += da0list_per
        syn_name = DApaths.split('/')[-1]
    else:
        syn_name = 'or'
        balance_func = balance(img1list,img0list)
        img1list,img0list = getattr(balance_func,'balance_'+balance_mode)()

    val1dir = args.val1dir
    val0dir = args.val0dir
    val1list = makelists(val1dir)
    val0list = makelists(val0dir)

    test1dir = args.test1dir  
    test0dir = args.test0dir 
    test1list = makelists(test1dir)
    test0list = makelists(test0dir)

    # balance_func = balance(val1list,val0list)
    # val1list,val0list = getattr(balance_func,'balance_'+balance_mode)()

    ###########################################################################

    ###########################################################################
    log_path = '{}_{}_{}_{}'.format(syn_name,args.percent,args.lr,args.ov_do)
    mode_path = os.path.join('./classification',log_path)
    args.mode_path = mode_path
    mkdir(mode_path)


    # Device configuration
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(6)
    ###########################################################################
    torch.cuda.set_device(args.gpu_id)
    model = Resnet18(1,2).cuda()
    ###########################################################################


    # Hyper-parameters
    ###########################################################################

    ###########################################################################
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))


    # Image preprocessing modules
    def worker_init_fn(worker_id):
        set_random_seed(seed + worker_id)
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))])
    transform_val = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))])

    train_dataset = med(img1list,img0list,transforms = transform)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = args.batch_size,
                                                shuffle = True,
                                                num_workers=16, 
                                                pin_memory=True, 
                                                drop_last=True, 
                                                worker_init_fn=worker_init_fn)
    val_dataset = med(val1list,val0list,transforms = transform_val)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = args.val_batch_size,
                                            shuffle = False,
                                            num_workers = 16,
                                            pin_memory = True)

    test_dataset = med(test1list,test0list,transforms = transform_val)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size =128,
                                            shuffle = False)
    if args.use_mixed_precision:
        # Initialize scaler in global scale
        scaler = torch.cuda.amp.GradScaler()    

    # Train the model
    for epoch in range(args.num_epochs):
        if epoch < args.min_epochs:
        # Training
            if args.use_mixed_precision:
                train_mixed_precision(args, epoch, scaler, model, optimizer, train_loader, train_dataset,criterion)
            else:
                train_epoch(args, epoch, model, optimizer, train_loader, train_dataset,criterion)                

            best_epoch,acc,auc,recall,f1score,pre = val(args, model, epoch, val_loader, val_dataset,criterion) 
        else:
            print('Training Done, the best epoch is {}\n the best acc is {}\n the best auc is {}\n \
                  the best recall is {}\n the best f1score is {}\n the best pre is {}\n'.format(best_epoch,acc,auc,recall,f1score,pre))
            with open(os.path.join(args.mode_path,'Log_{}.txt'.format(args.lr)),'a') as log:
                log.write('Training Done, best epoch:{}\n best acc:{}\n best auc:{}\n \
                    best recall:{}\n best f1score:{}\n best pre:{}\n'.format(best_epoch,acc,auc,recall,f1score,pre))
            break 
        
    test(args, best_epoch,test_loader)

if __name__ == '__main__':

    args = parser.parse_args()
    try:
        main(args)
    except Exception as exception:
        logging.exception(exception)
        raise
