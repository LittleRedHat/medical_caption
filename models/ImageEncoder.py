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
        return x,output


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
            resnet50 = M.resnet50(pretrain=True)
            self.model = ResnetExtractor(M.resnet50())
            num_features = resnet50.fc.in_features
            self.classifier = nn.Linear(num_features,num_classes)
        elif backbone == 'vgg19':
            vgg19 = M.vgg19(pretrain=True)
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
        
    def forward(self,x):
        final_layout,feats = self.model(x,self.layers)
        logits = self.classifier(final_layout)
        return logits,feats

class AttnMLC(MLC):
    
    def __init__(self,num_classes,backbone='resnet50',layers=[],config=[]):
        super(AttnMLC,self).__init__(num_classes,backbone,layers)
        ## for vgg19
        ## use 
        self.layer_w = 



    def forward(self,x):
        
