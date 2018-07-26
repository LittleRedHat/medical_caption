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

    if torch.isnan(loss):
        print(pred,weight)
        
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


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
