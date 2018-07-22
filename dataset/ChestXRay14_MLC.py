import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

tags = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 
        'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
        'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
positive_ratio = [
    0.10190141204057705, 
    0.02485089463220676, 
    0.1180226334301881, 
    0.17732069123719224, 
    0.05082326553499516, 
    0.0557552123158485, 
    0.01246367946169139, 
    0.04721669980119284, 
    0.041583830351225974, 
    0.021537442014579192, 
    0.022926543304276903, 
    0.014757608196972014, 
    0.029043686598358567, 
    0.001835142988224499
]

class ChestXRay14Dataset(Dataset):
    def __init__(self,image_dir,image_list_file,transformer = None):
        super(ChestXRay14Dataset,self).__init__()


        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_names.append(image_name)
                labels.append(label)
        self.image_dir = image_dir
        self.image_names = image_names
        self.labels = labels
        self.transformer = transformer
    

    def get_positive_ratio(self):
        return torch.tensor(positive_ratio,dtype=torch.float)

    def get_tag_size(self):
        return len(self.labels[0])
    
    def __getitem__(self,index):
        image_name = self.image_names[index]
        image = self.image_preprocess(os.path.join(self.image_dir,image_name),self.transformer)
        label = self.labels[index]
        return image, \
               torch.tensor(label,dtype=torch.float)


    def image_preprocess(self,image_path,transformer):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = transformer(image)
        return image

    def __len__(self):
        return len(self.image_names)
    