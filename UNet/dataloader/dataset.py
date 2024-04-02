import os
import cv2
import torch
import random
import logging
import numpy as np
from PIL import Image
from glob import glob
from utils.util import read_list
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from skimage import exposure

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

class BaseDataSets(Dataset):
    def __init__(self, modal,data_dir=None, data_path_name={}, mode='train', list_vol='',list_slice='',images_ratio=1,vol_percent=''):
        self._data_dir = data_dir 
        self.img_dir = os.path.join(data_dir,data_path_name['img']) 
        self.gt_dir = os.path.join(data_dir,data_path_name['seg'])
        self.vol_percent_list = vol_percent  # percent args
        self.modal = modal
        self.mode = mode
        self.img_wh = [240,240]
        self.num_cls = 3
        self.sample_list = []
        self.list_vol = list_vol
        slice_list = read_list(os.path.join(self._data_dir,list_slice)) 
        if mode=='train':
            for i , id in enumerate(list_vol):
                # a = filter(lambda x: x.startswith(id), slice_list) # 选取slice_list中 符合startswith(id)函数的值
                img_names = list(filter(lambda x: x.startswith(id), slice_list))
                self.sample_list.extend(img_names)
        else:
            self.sample_list = slice_list

        logging.info(f'Creating total {self.mode} dataset with {len(self.sample_list)} examples')         

        # downsample     
        if images_ratio < 1.0 and self.mode == "train":
            images_num = int(len(list_vol) * images_ratio)
            self.sample_list = slice_list[:images_num]
        # upsample
        elif images_ratio > 1.0 and self.mode == "train":
            images_num = int(len(list_vol) * (images_ratio-1))
            self.sample_list = self.sample_list + slice_list[:images_num]

    def __len__(self):
        return len(self.sample_list)
    def __sampleList__(self):
        return self.sample_list

    def __getitem__(self, idx):
        if self.mode=='val_3d':
            case = idx 
        else:
            case = self.sample_list[idx] 
        # 4 modal 
        imgm_path = []
        img_s = []
        # judge percent
        if self.mode == 'train': 
            if len(self.vol_percent_list) != len(self.list_vol):
                case_p = case.split('_')[0]+'_'+case.split('_')[1]
                if case_p in self.vol_percent_list:
                    for m_name in self.modal:
                        imgm_path.append(os.path.join(self.img_dir,m_name,'{}.png'.format(case)))
                    
                    for img in imgm_path:
                        img_init = Image.open(img).convert('L')
                        img_init_np = np.array(img_init)
                        if np.sum(img_init_np) != 0:
                            img_init_np = (img_init_np - np.min(img_init_np)) / (np.max(img_init_np) - np.min(img_init_np))
                        img_s.append(np.expand_dims(img_init_np, axis=0))
                else:
                    img_path=os.path.join(self.img_dir,self.modal[0],'{}.png'.format(case))
                    img_init = Image.open(img_path).convert('L')
                    img_init_np = np.array(img_init)
                    if np.sum(img_init_np) != 0:
                        img_init_np = (img_init_np - np.min(img_init_np)) / np.max(img_init_np) - np.min(img_init_np)
                    img_s.append(np.expand_dims(img_init_np, axis=0))
                    img_s.append(np.expand_dims(np.zeros((img_init_np.shape[0],img_init_np.shape[1]),dtype= np.float64), axis=0))
            else:
                for m_name in self.modal:
                    imgm_path.append(os.path.join(self.img_dir,m_name,'{}.png'.format(case)))
                
                for img in imgm_path:
                    img_init = Image.open(img).convert('L')
                    img_init_np = np.array(img_init)
                    if np.sum(img_init_np) != 0:
                        img_init_np = (img_init_np - np.min(img_init_np)) / np.max(img_init_np) - np.min(img_init_np)
                    img_s.append(np.expand_dims(img_init_np, axis=0))
        else:
            for m_name in self.modal:
                imgm_path.append(os.path.join(self.img_dir,m_name,'{}.png'.format(case)))
            
            for img in imgm_path:
                img_init = Image.open(img).convert('L')
                img_init_np = np.array(img_init)
                if np.sum(img_init_np) != 0:
                    img_init_np = (img_init_np - np.min(img_init_np)) / np.max(img_init_np) - np.min(img_init_np)
                img_s.append(np.expand_dims(img_init_np, axis=0))

        img_modal_np = np.concatenate(img_s,0)

        # read mask 
        mask_path = os.path.join(self.gt_dir,'{}.png'.format(case))
        mask_init = Image.open(mask_path).convert('L')
        mask_np_1c = np.array(mask_init)
        # print(np.unique(mask_np_1c ))
        mask_np_1c[np.where(mask_np_1c==255)] = 3
        mask_np_1c[np.where(mask_np_1c==128)] = 2
        mask_np_1c[np.where(mask_np_1c==64)] = 1
        mask_np_3c = np.stack([mask_np_1c == c for c in range(1,self.num_cls+1)], 0).astype(mask_np_1c.dtype)

        sample = {'image': img_modal_np.copy(), 'mask': mask_np_3c.copy(),'idx':case}
        return sample
class PatientBatchSampler(Sampler):
    def __init__(self, slices_list,patientID_list):
        self.slices_list = slices_list
        self.patientID_list = patientID_list
        assert len(self.slices_list) >= len(self.patientID_list) > 0

    def __iter__(self):
        return (
            list(filter(lambda x: x.startswith(id), self.slices_list))
            for i,id
            in enumerate(self.patientID_list)
        )

    def __len__(self):
        return len(self.patientID_list)
        
def random_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, cval):
    angle = np.random.randint(-45, 45)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label


def random_noise(image, label, mu=0, sigma=0.1):
    noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1]),
                    -2 * sigma, 2 * sigma)
    noise = noise + mu
    image = image + noise
    return image, label


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t**(n-i)) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x, label, prob=0.5):
    if random.random() >= prob:
        return x, label
    points = [[0, 0], [random.random(), random.random()], [
        random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x, label


def random_rescale_intensity(image, label):
    image = exposure.rescale_intensity(image)
    return image, label


def random_equalize_hist(image, label):
    image = exposure.equalize_hist(image)
    return image, label


class RandomGenerator(object):
    def __init__(self):
        self.output_size = ()

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label, cval=0)
        # if random.random() > 0.5:
        #     image, label = random_noise(image, label)
        # if random.random() > 0.33:
        #     image, label = nonlinear_transformation(image, label)
        if random.random() > 0.66:
            image, label = random_equalize_hist(image, label)
        elif random.random() < 0.66 and random.random() > 0.33:
            image, label = random_rescale_intensity(image, label)

            
        # x, y = image.shape
        # image = zoom(
        #     image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # label = zoom(
        #     label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # image = torch.from_numpy(
        #     image.astype(np.float32)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.int16))
        # sample = {'image': image, 'label': label}
        return image, label
