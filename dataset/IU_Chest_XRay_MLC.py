import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import pandas as pd
import numpy as np
from PIL import Image
import os

import sys
sys.path.append('..')
from utils import normalize_string


class IUChestXRayMLCDataset(Dataset):
    def __init__(self,file_findings,file_tags,image_root,transformer):
        super(IUChestXRayMLCDataset,self).__init__()
        self.file_findings = file_findings
        self.file_tags = file_tags
        self.image_root = image_root

        self.transformer = transformer
        
        self.tag2idx = {}
        with open(self.file_findings,'r') as f:
            self.findings = json.load(f)

        self.tags = list(pd.read_csv(self.file_tags)['tag'].values)

        for index,tag in enumerate(self.tags):
            self.tag2idx[tag] = index
    
    def description(self):
        pass
        

    def __getitem__(self,index):
        record = self.findings[index]
        tags = record['tags']
        image_path = os.path.join(self.image_root,record['id']+'.png')
        image = self.image_preprocess(image_path,self.transformer)
        # print(image)
        target_tags = self.tag_preprocess(tags)
        return torch.tensor(image,dtype=torch.float), \
               torch.tensor(target_tags,dtype=torch.float)

    def image_preprocess(self,image_path,transformer):
        image = Image.open(image_path)
        image = image.convert('RGB')
        # image = np.array(image)
        image = transformer(image)
        return image
    
    def tag_preprocess(self,tags):
        tags_vector = np.zeros(len(self.tag2idx))
        for tag in tags:
            tag = tag.lower().strip()
            if tag in self.tag2idx:
                index = self.tag2idx[tag]
                tags_vector[index] = 1
        return tags_vector
                

    
    def get_tags_size(self):
        return len(self.tags)
    
    def __len__(self):
        return len(self.findings)
        

