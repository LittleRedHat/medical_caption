import torch
import torch
import torch.nn as nn
import torchvision.models as M
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import time
import os 
import sys
sys.path.append('..')

class ResnetExtractor(nn.Module):
    def __init__(self,resnet):
        super(ResnetExtractor,self).__init__()
        self.resnet = resnet
        self.resnet_layers = [len(self.resnet.layer1),len(self.resnet.layer2),len(self.resnet.layer3),len(self.resnet.layer4)]

    def forward(self,x,layers=[]):
        feats = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)


        for i,block in enumerate(self.resnet.layer1.children()):
            x = block(x)
            if i in layers:
                feats.append(x)

        for i,block in enumerate(self.resnet.layer2.children()):
            x = block(x)
            if (i + sum(self.resnet_layers[:1])) in layers:
                feats.append(x)

        for i,block in enumerate(self.resnet.layer3.children()):
            x = block(x)
            if (i + sum(self.resnet_layers[:2])) in layers:
                feats.append(x)

        for i,block in enumerate(self.resnet.layer4.children()):
            x = block(x)
            if (i + sum(self.resnet_layers[:2])) in layers:
                feats.append(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        return x,feats


class VGGExtractor(nn.Module):
    def __init__(self,vgg):
        super(VGGExtractor,self).__init__()
        self.vgg = vgg
    
    def forward(self,x,layers):
        feats = []
        for i,block in enumerate(self.vgg.features.children()):
            x = block(x)
            if i in layers:
                feats.append(x)
        x = x.view(x.size(0), -1)
        return x,feats

class MLC(nn.Module):
    def __init__(self,num_classes,backbone='resnet50',layers=[]):
        super(MLC,self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        if backbone == 'resnet50':
            resnet50 = M.resnet50(pretrained=True)
            self.model = ResnetExtractor(resnet50)
            num_features = resnet50.fc.in_features
            self.classifier = nn.Linear(num_features,num_classes)
        elif backbone == 'vgg19':
            vgg19 = M.vgg19(pretrained=True)
            self.model = VGGExtractor(vgg19)
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        else:
            assert backbone not in ['vgg19','resnet50']
        self.layers = layers
        
    def forward(self,x,softmax=False):
        final_layout,feats = self.model(x,self.layers)
        logits = self.classifier(final_layout)
        if not softmax:
            preds = F.sigmoid(logits)
        else:
            preds = F.softmax(logits)
        return preds,feats


class SRN(nn.Module):
    def __init__(self,c,hidden_channle):
        super(SRN,self).__init__()
        self.channel = c
        self.hidden_channle = hidden_channle
        self.attn_map = nn.Sequential(
            nn.Conv2d(self.channel,self.hidden_channle,(1,1)),
            nn.Conv2d(self.hidden_channle,self.hidden_channle,(3,3),padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channle,self.channel,(1,1))
        )
        self.confidence = nn.Sequential(
            nn.Conv2d(self.channel,self.channel,(1,1),stride=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        ## batch * channle * w * h
        attention_map = self.attn_map(x)
        confidence_weight = self.confidence(x)
        batch_size,channle,w,h = attention_map.size()
        attention_map = attention_map.reshape(batch_size,channle,-1)
        attention_map = F.softmax(attention_map)
        attention_map = attention_map.reshape(batch_size,channle,w,h)
        output = attention_map * confidence_weight

        return output
        

class AttnMLC(MLC,nn.Module):
    def __init__(self,num_classes,backbone='resnet50',layers=[]):
        super(AttnMLC,self).__init__(num_classes,backbone,layers)
        # print(backbone)
        ## for vgg19
        ## use 
        if backbone == 'vgg19':
            ## conv5_4 conv4_4
            self.layers = [34,20]
            srn1 = SRN(512,256)
            srn2 = SRN(256,128)
            self.branch = nn.ModuleList([nn.Sequential(srn1,nn.AvgPool2d(14,14)),nn.Sequential(nn.Sequential(srn2,nn.AvgPool2d(28,28)))])
            
        elif backbone == 'resnet50':
            
            ## 13th block and 16th block
            print('backbone resnet50')


            self.layers = [16,13]
            srn1 = SRN(2048,256)
            srn2 = SRN(1024,256)
            self.branch = nn.ModuleList([nn.Sequential(srn1,nn.AvgPool2d(14,14)),nn.Sequential(nn.Sequential(srn2,nn.AvgPool2d(28,28)))])

    

    def forward(self,x):
        final_layout,feats = self.model(x,self.layers)
        logits = self.classifier(final_layout)
        
        global_pred = F.sigmoid(logits)
        total_pred = global_pred
        
        for i,feat in enumerate(feats):
            logits = self.branch[i](feat)
            logits = logits.view(logits.size(0),-1)
            branch_pred = F.sigmoid(logits)
            total_pred = total_pred + branch_pred

        return total_pred / (1 + len(self.layers)),feats
