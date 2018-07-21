import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def compute_auc(pred,truth):
    AUROCs = []
    num_classes = pred.shape[1]
    for i in range(num_classes):
        AUROCs.append(roc_auc_score(truth[:, i], pred[:, i]))
    return AUROCs

def focal_loss_with_logits(logits,target,gamma = 2,alpha = 0.25,average = True,eps=1e-10):
    pred = F.sigmoid(logits)
    pred = torch.clamp(pred, eps, 1 - eps)
    loss = -  alpha * torch.pow((1 - pred),gamma) * target * torch.log(pred) - (1 - alpha) * torch.pow(pred,gamma) * (1 - target) * torch.log(1 - pred)
    loss = loss.sum(1)
    if average:
        loss = loss.mean()
    else:
        loss = loss.sum()
    return loss

def focal_loss(pred,target,gamma = 2,alpha = 0.25,average = True,eps=1e-10):
    pred = torch.clamp(pred, eps, 1 - eps)
    loss = -  alpha * torch.pow((1 - pred),gamma) * target * torch.log(pred) - (1 - alpha) * torch.pow(pred,gamma) * (1 - target) * torch.log(1 - pred)
    loss = loss.sum(1)
    if average:
        loss = loss.mean()
    else:
        loss = loss.sum()
    return loss