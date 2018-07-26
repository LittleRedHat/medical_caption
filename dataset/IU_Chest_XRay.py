import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import json
import pandas as pd
import numpy as np
from PIL import Image
import os
import csv


import sys
sys.path.append('..')
from utils import normalize_string

SOS_INDEX = 0
EOS_INDEX = 1
UNK_INDEX = 2
MAX_SENT = 10
MAX_WORDS = 15


class ChestIUXRayDataset(Dataset):
    def __init__(self,file_findings,file_words,file_tags,image_root,transformer=None):
        super(ChestIUXRayDataset,self).__init__()

        self.file_findings = file_findings
        self.file_words = file_words
        self.file_tags = file_tags
        self.image_root = image_root
        # self.word2idx = {'SOS':SOS_INDEX,'EOS':EOS_INDEX,'UNK':UNK_INDEX}
        self.word2idx = {}
        self.n_words = 0
        self.transformer = transformer

        self.idx2word = {}
        self.tag2idx = {}

        with open(self.file_findings,'r') as f:
            self.findings = json.load(f)

        self.words = list(pd.read_csv(self.file_words)['word'].values)
        self.tags = list(pd.read_csv(self.file_tags)['tag'].values)
        for word in self.words:
            if word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.n_words = self.n_words + 1
        self.word2idx['UNK'] = self.n_words
        self.word2idx['EOS'] = self.n_words + 1
        self.word2idx['SOS'] = self.n_words + 2

        self.idx2word = {index:word for word,index in self.word2idx.items()}

        for index,tag in enumerate(self.tags):
            self.tag2idx[tag] = index

    def get_config(self):
        return {
            'SOS_INDEX':SOS_INDEX,
            'EOS_INDEX':EOS_INDEX,
            'UNK_INDEX':UNK_INDEX,
            'MAX_SENT':MAX_SENT,
            'MAX_WORDS':MAX_WORDS
        }

    def get_word_embed(self):
        return []

    def get_dict(self):
        return self.idx2word
    
    def get_tags_size(self):
        return len(self.tags)

    def get_words_size(self):
        return len(self.word2idx)

    def image_preprocess(self,image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transformer(image)
        return image

    def caption_preprocess(self,report):
        caption = normalize_string(report)
        caption = [sent.strip() for sent in caption.split(' .') if len(sent.strip()) > 0]

        sent_embds = []

        sent_length = []
        sent_num = min(len(caption),MAX_SENT)


        ## 截断过长的caption以及过长的sent
        for i_sent in range(MAX_SENT):
            
            if i_sent >= len(caption):
                sent_embds.append([self.word2idx['EOS']] * (MAX_WORDS + 1))
                sent_length.append(0)
                continue
            sent = caption[i_sent]
            words = sent.split()

            length = min(len(words),MAX_WORDS)
            sent_length.append(length)

            sent_embd = []
            
            for i_word in range(MAX_WORDS + 1):
                if i_word >= len(words):
                    sent_embd.append(self.word2idx['EOS'])
                else:
                    word = words[i_word]
                    sent_embd.append(self.word2idx.get(word,self.word2idx['UNK']))

            sent_embds.append(sent_embd)

        return sent_embds,sent_num,sent_length

    def tag_preprocess(self,tags):
        tags_vector = np.zeros(len(self.tag2idx))
        for tag in tags:
            tag = tag.lower().strip()
            if tag in self.tag2idx:
                index = self.tag2idx[tag]
                tags_vector[index] = 1
        return tags_vector
                
    def __getitem__(self,index):
        record = self.findings[index]
        tags = record['tags']
        report = record['report']
        image_path = os.path.join(self.image_root,record['id']+'.png')

        image = self.image_preprocess(image_path)
        sent_idxs,sent_num,sent_length = self.caption_preprocess(report)
       
        stop = []
        stop = [0 if (i+1) < sent_num else 1 for i in range(MAX_SENT)]

        tags = self.tag_preprocess(tags)
        # tags_vector = np.zeros(self.get_tags_size())

        # for i in tags:
        #     tags_vector[i] = 1


        return torch.tensor(image,dtype=torch.float), \
               torch.tensor(sent_idxs,dtype=torch.long), \
               torch.tensor(sent_length,dtype=torch.long), \
               torch.tensor(sent_num,dtype=torch.long), \
               torch.tensor(stop,dtype=torch.float), \
               torch.tensor(tags,dtype=torch.float)

            #    torch.tensor(tags_vector,dtype=torch.float)
               


    def __len__(self):
        return len(self.findings)

    


        
