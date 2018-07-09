import torch
from torch.utils.data import DataLoader,Dataset
import unicodedata
import json
import sys
sys.path.append('..')


class ChestIUXRayDataset(Dataset):
    def __init__(self,file_data,file_findings,file_words,file_tags):
        super(ChestIUXRayDataset,self).__init__()

        self.file_data = file_data
        self.file_findings = file_findings
        self.file_words = file_words
        self.file_tags = file_tags

        self.word2idx = {'SOS':0,'EOS':1,'UNK':2}
        self.n_words = 3

        self.idx2word = {}
        self.tag2idx = {}

        with open(file_words,'r') as f:
            line = f.readline()
            
        with open(file_tags,'r') as f:
            line = f.readline()

        with open(file_findings,'r') as f:
            self.findings = json.load(f)
        


    def __getitem__(self,index):
        
        record = self.findings[]
        pass

    def __len__(self):
        return len(self.pairs)

    


        
