import torch
import torch.nn as nn
import torchvision.models as M
import torch.nn.functional as F

import sys
sys.path.append('..')
from dataset.IU_Chest_XRay import ChestIUXRayDataset


class SententDecoder(nn.Module):
    pass

class WordDecoder(nn.Module):
    pass

class HierarchyLSTM(nn.Module):
    pass


class MLC(nn.module):
    def __init__(self,config):
        super(MLC,self).__init__()
        self.model = M.vgg19(pretrained=True)

    def forward(self,x):
        features = self.model.features(x)

        






    

