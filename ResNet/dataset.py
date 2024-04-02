import numpy as np
import torch
import torch.utils.data
import os
import random
import torchvision
from PIL import Image
from torchvision import transforms

class med(torch.utils.data.Dataset):
    def __init__(self,img1_list,img0_list,transforms = None):
        label1 = [1]*len(img1_list)
        label0 = [0]*len(img0_list)
        img_list = img1_list + img0_list
        label = label1 + label0
        self.ziplist = list(zip(img_list,label))
        self.transform = transforms

    def __getitem__(self, index):
        img_path,label = self.ziplist[index]
        imgi = Image.open(img_path).convert('L')
        imgi = self.transform(imgi)
        return imgi,label

    def __len__(self):
        return len(self.ziplist)

class med_infer(torch.utils.data.Dataset):
    def __init__(self,img_list,transforms = None):
        self.img_list = list(img_list)
        self.transform = transforms

    def __getitem__(self,index):
        img_path = self.img_list[index]
        _,img_name = os.path.split(img_path)
        imgi = Image.open(img_path)
        imgi = self.transform(imgi)
        return imgi,img_name

    def __len__(self):
        return len(self.img_list)
