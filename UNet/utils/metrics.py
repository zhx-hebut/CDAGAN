import numpy as np
import torch.nn as nn
import torch
similarity_cos = nn.CosineSimilarity(dim=0)

def dice(y_pred, y_true):
    """
    single data only
    :param y_pred: 4-d tensor, value = [0,1]
    :param y_true: 4-d tensor, value = [0,1]
    :return:
    """
    eps = 1e-6
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    a_unin_b = np.sum(y_pred[y_true == 1])

    a_plus_b = np.sum(y_pred) + np.sum(y_true) + eps
    return (a_unin_b * 2.0 + eps) / a_plus_b

def dice_3d(y_pred, y_true):
    b,c,w,h = y_true.shape
    dice_group = []
    for i in range(c):
        dice_group.append(dice(y_pred[:,i,...],y_true[:,i,...]))
    return np.mean(dice_group)

def dice_3(y_pred, y_true):
    """
    single data only
    :param y_pred: 4-d tensor, value = [0,1]
    :param y_true: 4-d tensor, value = [0,1]
    :return:
    """
    eps = 0.0001
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    a_unin_b = np.sum(y_pred[y_true == 1])
    a_plus_b = np.sum(y_pred) + np.sum(y_true) + eps
    # dice
    dice_value = (a_unin_b * 2.0 + eps) / a_plus_b
    #PPV
    ppv_value = (a_unin_b + eps) / (np.sum(y_pred) + eps)
    #sensitivity
    sen_val = (a_unin_b + eps) / (np.sum(y_true) + eps)

    return dice_value, ppv_value, sen_val
    
def iou(y_pred , y_true):
    eps = 1e-6
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    a_unin_b = np.sum(y_pred[y_true == 1]) + eps
    a_plus_b = np.sum(y_pred) + np.sum(y_true) + eps
    return a_unin_b / (a_plus_b - a_unin_b)
def batch_dice(input,target):
    """Dice coeff for batches"""
    s = 0
    num = 0
    for i, v in enumerate(zip(input, target)):
        y_pred = v[0].numpy().astype("uint8")
        y_true = v[1].numpy().astype("uint8")
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        d = dice(y_pred, y_true)
        s = s + d
        num+=1
    return s,num

def batch_cos_sim(input,target):
    """Dice coeff for batches"""
    s = 0
    num = 0
    for i, v in enumerate(zip(input, target)):
        y_pred = v[0].numpy().astype("uint8")
        y_true = v[1].numpy().astype("uint8")
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        d = dice(y_pred, y_true)
        sim_L_LM = similarity_cos(label_Dis_head.squeeze(0), labelM_Dis_head.squeeze(0))
        s = s + d
        num+=1
    return s,num

def compute_dice(input,target,deNoNidus=False):
    """Dice coeff for batches"""
    s = 0
    num = 0
    nidus_start = 0
    a = torch.max(target)
    for i, v in enumerate(zip(input, target)):
        y_pred = v[0].numpy().astype("uint8")
        y_true = v[1].numpy().astype("uint8")
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        if deNoNidus==True and np.max(y_true)==0:
            continue

        d = dice(y_pred, y_true)
        s = s + d
        num+=1
        if num == 1:
            nidus_start = i

    return s,num,nidus_start