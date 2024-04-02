import torch
import torch.nn as nn
import torch.nn.functional as F
from cv2 import cv2
from utils.util import one_hot, simplex
from torch import Tensor,  einsum
import numpy as np

def dice_loss(score, target):
    smooth = 1e-6
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    dice_value = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return 1- dice_value


def box_project_bceWloss(output, mask_gt):
    bce_loss = nn.BCEWithLogitsLoss(reduce=False)
    mask_pred = torch.sigmoid(output).detach()

    y_sum = mask_pred.sum(dim=2, keepdim=True)
    x_len = (mask_pred>0.0).float().max(dim=3, keepdim=True)[0].sum(dim=2, keepdim=True)
    # x_len = mask_gt.max(dim=3, keepdim=True)[0].sum(dim=2, keepdim=True)
    y_weight = y_sum / x_len
    mask_losses_y = bce_loss(
        output.max(dim=2, keepdim=True)[0],
        y_weight * mask_gt.max(dim=2, keepdim=True)[0]
    )
    x_sum = mask_pred.sum(dim=3, keepdim=True)
    y_len = (mask_pred>0.0).float().max(dim=2, keepdim=True)[0].sum(dim=3, keepdim=True)
    # x_len = mask_gt.max(dim=3, keepdim=True)[0].sum(dim=2, keepdim=True)
    x_weight = x_sum / y_len
    mask_losses_x = bce_loss(
        output.max(dim=3, keepdim=True)[0],
        x_weight * mask_gt.max(dim=3, keepdim=True)[0]
    )

    return mask_losses_x.mean() + mask_losses_y.mean()

def box_project_bceloss(output, mask_gt):
    bce_loss = nn.BCEWithLogitsLoss()
    mask_losses_y = bce_loss(
        output.max(dim=2, keepdim=True)[0],
        mask_gt.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = bce_loss(
        output.max(dim=3, keepdim=True)[0],
        mask_gt.max(dim=3, keepdim=True)[0]
    )

    return (mask_losses_x + mask_losses_y).mean()

def box_project_loss(mask_pred, mask_gt):
    mask_losses_y = dice_loss(
        mask_pred.max(dim=2, keepdim=True)[0],
        mask_gt.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_loss(
        mask_pred.max(dim=3, keepdim=True)[0],
        mask_gt.max(dim=3, keepdim=True)[0]
    )

    return (mask_losses_x + mask_losses_y).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            smooth = 1e-6
            return torch.log(torch.sum(F_loss) + smooth)

class BinaryDiceLoss_xent(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss_xent, self).__init__()
    def _dice_loss(self, score, target):
        smooth = 1e-6
        dim_len = len(score.size())
        if dim_len == 5:
            dim=(2,3,4)
        elif dim_len == 4:
            dim=(2,3)
        elif dim_len == 3:
            dim=(2)
        intersect = torch.sum(score * target,dim=dim)
        y_sum = torch.sum(target * target,dim=dim)
        z_sum = torch.sum(score * score,dim=dim)
        dice_loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return dice_loss

    def forward(self, inputs, target):
        assert inputs.size()[2:] == target.size()[2:], 'predict & target shape do not match'
        dice_loss = self._dice_loss(inputs, target)
        # loss = 1 - dice_loss
        return dice_loss

class BinaryDiceLoss_xent2(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss_xent2, self).__init__()
    def _dice_loss(self, score, target):
        smooth = 1e-6
        dim_len = len(score.size())
        if dim_len == 5:
            dim=(2,3,4)
        elif dim_len == 4:
            dim=(2,3)
        elif dim_len == 3:
            dim=(2)
        intersect = torch.sum(score * target,dim=dim)
        y_sum = torch.sum(target * target,dim=dim)
        z_sum = torch.sum(score * score,dim=dim)
        dice_loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return dice_loss

    def forward(self, inputs, target):
        assert inputs.size()[2:] == target.size()[2:], 'predict & target shape do not match'
        _,_,b,_ = target.size()
        dice_loss = 0
        for i in range(b):   
            dice_loss += self._dice_loss(inputs[:,:,i,:], target[:,:,i,:])
        # loss = 1 - dice_loss
        return dice_loss/b

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
    def _dice_loss(self, score, target):
        smooth = 1e-6
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice_loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return dice_loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        target = target.float()
        dice_loss = self._dice_loss(inputs, target)
        loss = 1 - dice_loss
        return loss
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        # if softmax:
        #     inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
    def _dice_loss(self, score, target):
        smooth = 1e-6
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice_loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return dice_loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        # target = target.float()
        dice_loss = self._dice_loss(inputs, target)
        loss = 1 - dice_loss
        return loss           
def logits_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes logits on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert input_logits.size() == target_logits.size()
    mse_loss = (input_logits-target_logits)**2
    # return mse_loss
    return torch.mean(mse_loss)

class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc= kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss


BoundaryLoss = SurfaceLoss

def generalized_iou_loss(gt_bboxes, pr_bboxes, reduction='sum'):
    """
    gt_bboxes: tensor (-1, 4) xyxy
    pr_bboxes: tensor (-1, 4) xyxy
    loss proposed in the paper of giou
    """
    gt_area = (gt_bboxes[ 2]-gt_bboxes[ 0])*(gt_bboxes[ 3]-gt_bboxes[ 1])
    pr_area = (pr_bboxes[ 2]-pr_bboxes[ 0])*(pr_bboxes[ 3]-pr_bboxes[ 1])

    # iou
    lt = torch.max(gt_bboxes[ :2], pr_bboxes[ :2])
    rb = torch.min(gt_bboxes[ 2:], pr_bboxes[ 2:])
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[ 0] * wh[ 1]
    union = gt_area + pr_area - inter
    iou = inter / union
    loss = 1.-iou
    # # enclosure
    # lt = torch.min(gt_bboxes[ :2], pr_bboxes[ :2])
    # rb = torch.max(gt_bboxes[ 2:], pr_bboxes[ 2:])
    # wh = (rb - lt + TO_REMOVE).clamp(min=0)
    # enclosure = wh[ 0] * wh[ 1]

    # giou = iou - (enclosure-union)/enclosure
    # loss = 1. - giou
    
    # if reduction == 'mean':
    #     loss = loss.mean()
    # elif reduction == 'sum':
    #     loss = loss.sum()
    # elif reduction == 'none':
    #     pass
    return loss


def IOU(x1,y1,X1,Y1, x2,y2,X2,Y2):

    xx = max(x1,x2)

    XX = min(X1,X2)

    yy = max(y1,y2)

    YY = min(Y1,Y2)

    m = max(0., XX-xx)

    n = max(0., YY-yy)

    Jiao = m*n

    Bing = (X1-x1)*(Y1-y1)+(X2-x2)*(Y2-y2)-Jiao

    return 1-Jiao/Bing


def get_IoU(pred_bbox, gt_bbox):
    """
    return iou score between pred / gt bboxes
    :param pred_bbox: predict bbox coordinate
    :param gt_bbox: ground truth bbox coordinate
    :return: iou score
    """

    # bbox should be valid, actually we should add more judgements, just ignore here...
    # assert ((abs(pred_bbox[2] - pred_bbox[0]) > 0) and
    #         (abs(pred_bbox[3] - pred_bbox[1]) > 0))
    # assert ((abs(gt_bbox[2] - gt_bbox[0]) > 0) and
    #         (abs(gt_bbox[3] - gt_bbox[1]) > 0))

    # -----0---- get coordinates of inters
    ixmin = max(pred_bbox[0], gt_bbox[0]).data.numpy()
    iymin = max(pred_bbox[1], gt_bbox[1]).data.numpy()
    ixmax = min(pred_bbox[2], gt_bbox[2]).data.numpy()
    iymax = min(pred_bbox[3], gt_bbox[3]).data.numpy()
    # ixmin
    # iymin
    # ixmax
    # iymax
    
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    # -----1----- intersection
    inters = iw * ih

    # -----2----- union, uni = S1 + S2 - inters
    uni = ((pred_bbox[2] - pred_bbox[0] + 1.) * (pred_bbox[3] - pred_bbox[1] + 1.) +
           (gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.) -
           inters)

    # -----3----- iou
    overlaps = inters / uni
    # overlaps = torch.Tensor(overlaps)
    return 1-overlaps

def bbox_iou_np(box1, box2, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter = (np.min(b1_x2, b2_x2) - np.max(b1_x1, b2_x1)).clamp(0) * \
            (np.min(b1_y2, b2_y2) - np.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    return iou

def bbox_iou(box1, box2, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    return iou
def batch_bbox_iou(input,target):
    """bbox_iou coeff for batches"""
    iou_list = []
    for i, v in enumerate(zip(input, target)):
        iou = bbox_iou(v[0],v[1])
        iou_list.append(iou)
    return torch.stack(iou_list,dim=0)
def box_iou(box1, box2,eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # box1=np.array(box1)
    # box2=np.array(box2)
    
    # box1=torch.stack(box1,dim=0)
    # box2=torch.stack(box2,dim=0)


    lbox = torch.zeros(1).cuda()
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    
    return  inter / ((area1[:, None] + area2 - inter)+ eps) # iou = inter / (area1 + area2 - inter)
    # lbox = lbox+ (1.0 - iou).mean()  # iou loss
    # return lbox


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss