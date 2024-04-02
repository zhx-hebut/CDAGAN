import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import os 
import skimage.io as io
import cv2

def get_box_from_mask(mask):
    """
    生成框的坐标
    """
    # mask = mask.data.cpu().numpy()   
    # mask =mask*255
    ind = np.where(mask>0.5)    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    if len(ind[0])==0:
        return [0,0,0,0]
    box = [ind[1].min()-1, ind[0].min()-1, ind[1].max()+1, ind[0].max()+1]  #[x1,y1,x2,y2]
    # return ind[1].min(),ind[0].min(), ind[1].max(), ind[0].max()
    return box

def save_img_inter(out_path,img_tensor,data_type,patientSliceID,exp):
    # from utils.util import draw_bbox, save_img_inter
    # save_img_inter('./',mask_tensor,'mask','001','exp')
    if data_type=='img':
        img_np = img_tensor.cpu().data.numpy().squeeze()
    elif data_type=='mask':
        # img_np = img_tensor.cpu().data.numpy().astype("uint8").squeeze()
        img_np = img_tensor.cpu().data.numpy().squeeze()
        img_np = img_np * 255
    sum_a = np.sum(img_np)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    io.imsave('{}/{}_{}_{:.1f}.png'.format(out_path,patientSliceID,exp,sum_a),img_np)

def draw_bbox(out_path, mask, bbox, exp):
    # from utils.util import draw_bbox, save_img_inter
    # draw_bbox('./',mask,bbox,'epx')
    # mask = mask.cpu().data.numpy().squeeze()
    mask = mask * 255
    start_point, end_point = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])) # (x0, y0), (x1, y1)
    color = (0, 0, 255)  # Red color in BGR；红色：rgb(255,0,0)
    thickness = 1  # Line thickness of 1 px
    mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为3通道图，使得color能够显示红色。
    mask_bboxs1 = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
    cv2.imwrite(os.path.join(out_path, '{}_box.png'.format(exp)), mask_bboxs1)

def draw_feature(x, out_path, exp):
    img = x.cpu().data.numpy().squeeze()
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    img = img.astype(np.uint8)  # 转成unit8
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
    img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
    cv2.imwrite(os.path.join(out_path, '{}_hot.png'.format(exp)), img)
    # plt.imshow(img)
    # plt.savefig('{}/{}_{}_{:.1f}.png'.format(out_path,patientSliceID,exp,sum_a))


def plot_dot(m_indx,m_value,num=0,):
    plt.plot(m_indx, m_value, 'ks')
    show_max = '[' + str(m_indx) + ',' + str("{:.4f}".format(m_value)) + ']'
    plt.annotate(show_max, xytext=(m_indx, m_value+num), xy=(m_indx, m_value))

def plot_dice(train_c,valid_c,base_dir,mode,valid_c_3d=[],interval=0):
    train_x = range(len(train_c))
    train_y = train_c
    plt.plot(train_x, train_y)
    
    valid_x = range(len(valid_c))
    valid_y = valid_c
    plt.plot(valid_x, valid_y)

    m_indx=np.argmax(valid_c)
    #以下处理Dice 3d val曲线
    m_indx_3d=np.argmax(valid_c_3d)
    valid_x_3d_init = list(range(len(valid_c_3d)))
    valid_x_3d = [x*interval for x in valid_x_3d_init]
    valid_y_3d = valid_c_3d
    plt.plot(valid_x_3d, valid_c_3d)
    if m_indx_3d < int(len(valid_c_3d)*0.8):
        plot_dot(m_indx_3d*interval,valid_c_3d[m_indx_3d])
    last_indx_3d = valid_x_3d_init[-1]
    v_last_2d = valid_c[-1]
    v_last_3d = valid_c_3d[-1]
    #以下代码防止val和val_3d最后一个值相近导致相互遮挡
    abs_vLast = abs(v_last_3d-v_last_2d)
    if abs_vLast < 0.04:
        num = 0.04-abs_vLast if v_last_3d > v_last_2d else -(0.06-abs_vLast)
        plot_dot(last_indx_3d*interval,valid_y_3d[last_indx_3d],num)
    else:
        plot_dot(last_indx_3d*interval,valid_y_3d[last_indx_3d])
    plt.legend(['train', 'val','val_3d'],loc='upper left')

    if m_indx < int(len(valid_c)*0.8):
        plot_dot(m_indx,valid_c[m_indx])
    # 标出最后一个值
    last_indx = valid_x[-1]
    plot_dot(last_indx,valid_c[last_indx])
    plt.ylabel(mode + ' value')
    plt.xlabel('epoch')
    plt.title("Model " + mode)
    plt.savefig('{}/{}-{:.4f}.jpg'.format(base_dir,mode,valid_c_3d[-1]))
    plt.close()

def plot_base(train_c,valid_c,base_dir,mode):
    train_x = range(len(train_c))
    train_y = train_c
    plt.plot(train_x, train_y)
    if len(valid_c)>0:
        valid_x = range(len(valid_c))
        valid_y = valid_c
        plt.plot(valid_x, valid_y)
        # m_indx=np.argmin(valid_c)
        # if m_indx < int(len(valid_c)*0.8):
        #     plot_dot(m_indx,valid_c[m_indx])
        # 标出最后一个值
        last_indx = valid_x[-1]
        plot_dot(last_indx,valid_c[last_indx])
        plt.legend(['train', 'val'],loc='upper left')
    else:
        plt.legend(['train'],loc='upper left')
    last_indx = train_x[-1]
    plot_dot(last_indx,train_c[last_indx])
    
    plt.ylabel(mode + ' value')
    plt.xlabel('epoch')
    plt.title("Model " + mode)
    plt.savefig('{}/{}-{:.4f}.jpg'.format(base_dir,mode,train_c[last_indx]))
    plt.close()
def plot_dice_loss(train_dict,val_dict,val_3d_interval,lr_curve,base_dir):
    # plot dice curve
    print(val_dict['dice_3d'])
    plot_dice(train_dict['dice'],val_dict['dice'],base_dir,'Dice',val_dict['dice_3d'],val_3d_interval)
    # plot loss curve
    for key in train_dict:
        if 'loss' in key:
            if key in val_dict:
                plot_base(train_dict[key],val_dict[key],base_dir,mode=key)
            else:
                plot_base(train_dict[key],[],base_dir,mode=key)

    # plot lr curve
    lr_x = range(len(lr_curve))
    lr_y = lr_curve
    plt.plot(lr_x, lr_y)
    plt.legend(['learning_rate'],loc='upper right')
    plt.ylabel('lr value')
    plt.xlabel('epoch')
    plt.title("Learning Rate" )
    plt.savefig('{}/lr.jpg'.format(base_dir))
    plt.close()   
def set_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = '%(levelname)s: %(message)s'
    # DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT)
    chlr = logging.StreamHandler() # 输出到控制台的handler
    chlr.setFormatter(formatter)
    # chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
    fhlr = logging.FileHandler(log_path) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
class Logger(object):
    def __init__(self, log_path="Default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def read_list(list_path):
    list_data = []
    with open(list_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            list_data.append(line)  
    return list_data

class AverageMeter_old(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.curve = list()
    def init_zero(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
    def update(self, value, n=1):
        self.sum += value
        self.count += n
    def updata_avg(self):
        self.avg = self.sum / self.count
        self.curve.append(self.avg)
        return self.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.count_dict = {}
        self.value_dict = {}
        self.res_dict = {}

    def add_value(self,tag_dict,n=1):
        for tag,value in tag_dict.items():
            if tag not in self.value_dict.keys():
                self.value_dict[tag] = 0
                self.count_dict[tag] = 0
            self.value_dict[tag] += value
            self.count_dict[tag] += n

    def updata_avg(self):
        for tag in self.value_dict:
            if tag not in self.res_dict.keys():
                self.res_dict[tag] = []
            avg = self.value_dict[tag] / self.count_dict[tag]
            self.res_dict[tag].append(avg)
        self.count_dict = {}
        self.value_dict = {}

from torch import Tensor     
import torch
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast
from scipy.ndimage import distance_transform_edt as eucl_distance
# Assert utils

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape  

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res
    
def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res

