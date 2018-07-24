import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np


## micro auc
def compute_auc(pred,truth):
    AUROCs = []
    num_classes = pred.shape[1]
    for i in range(num_classes):
        AUROCs.append(roc_auc_score(truth[:, i], pred[:, i]))
    return AUROCs


## macro recall
def compute_topk_recall(pred,truth,topk):
    topk_labels = np.argsort(pred,axis=1)[:,:-1*(topk + 1):-1]

    sample_num = len(truth)
    recall = .0
    for i,labels in enumerate(topk_labels):
        truth_labels = np.nonzero(truth[i])[0]
        recall += len(np.intersect1d(labels,truth_labels)) * 1.0 / min(len(truth_labels),topk)
    
    recall = recall / sample_num

    return recall

def average_softmax_logits(logits,truth,eps=1e-10,average=True):
    pred = F.softmax(logits)
    pred = torch.clamp(pred, eps, 1 - eps)

    norm = torch.norm(q, p=1, dim=1).detach()
    truth = truth.div(norm.expand_as(truth))

    loss = -1 * truth * torch.log(pred)
    loss = loss.sum(1)

    if average:
        loss = loss.mean()
    else:
        loss = loss.sum()
    return loss


def focal_loss_with_logits(logits,target,positive_ratio,gamma = 2,average = True,eps=1e-10):
    
    pred = F.sigmoid(logits)
    pred = torch.clamp(pred, eps, 1 - eps)
    # weight = torch.exp(target + (1 - target * 2) * positive_ratio)
    weight = torch.exp(-1 * positive_ratio)
    loss = - weight * (torch.pow((1 - pred),gamma) * target * torch.log(pred) + torch.pow(pred,gamma) * (1 - target) * torch.log(1 - pred))
    loss = loss.sum(1)
    if average:
        loss = loss.mean()
    else:
        loss = loss.sum()
    return loss

def focal_loss(pred,target,positive_ratio,gamma = 2,average = True,eps=1e-10):
    pred = torch.clamp(pred, eps, 1 - eps)
    # weight = torch.exp(target + (1 - target * 2) * positive_ratio)
    weight = torch.exp(-1 * positive_ratio)
    loss = - weight * (torch.pow((1 - pred),gamma) * target * torch.log(pred) + torch.pow(pred,gamma) * (1 - target) * torch.log(1 - pred))
    loss = loss.sum(1)
    if average:
        loss = loss.mean()
    else:
        loss = loss.sum()
    return loss

def focal_loss_without_balance(pred,target,alpha = 0.15,gamma = 2,average = True,eps=1e-10):
    pred = torch.clamp(pred, eps, 1 - eps)
    # weight = torch.exp(target + (1 - target * 2) * positive_ratio)
    loss = - (alpha * torch.pow((1 - pred),gamma) * target * torch.log(pred) + (1 - alpha) * torch.pow(pred,gamma) * (1 - target) * torch.log(1 - pred))
    loss = loss.sum(1)
    if average:
        loss = loss.mean()
    else:
        loss = loss.sum()
    return loss