import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset import med
# from model import VGG19_bn,Resnet18
# from datapre import balance,balance_train
import os
from sklearn.metrics import accuracy_score, roc_auc_score

#from torchvision.utils import save_image

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


test1dir = '/path/to/your/testset/testA'
test0dir = '/path/to/your/testset/testB'
test1list = makelists(test1dir)
test0list = makelists(test0dir)

# Device configuration
gpu_id = 7
torch.cuda.set_device(gpu_id)

#####################################################################
model_path = 'classification/oversample_1/save.ckpt'   #训练完成后最好的模型的位置
test_path = 'classification/oversample_1'
transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))])

custom_dataset = med(test1list,test0list,transforms = transform)
test_loader = torch.utils.data.DataLoader(dataset = custom_dataset,
                                          batch_size =128,
                                          shuffle = False)
                                    
with torch.no_grad():
    model = torch.load(model_path,map_location=lambda storage, loc: storage.cuda(7)).eval()
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
    print(acc,auc)
    with open(os.path.join(test_path,'Logtest.txt'),'a') as log:
        log.write('Test acc:{}, test auc:{}\n'.format(acc,auc))
