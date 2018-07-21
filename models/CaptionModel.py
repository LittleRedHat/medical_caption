import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
    
    def forward(self,feats):
        pass


class ChannleAttention(nn.Module):
    
    def __init__(self):
        super(ChannleAttention,self).__init__()
    
    def forward(self,feats):
        pass


class AggregationAttn(nn.Module):
    pass





class SentDecoder(nn.Module):
    def __init__(self,config):
        super(SentDecoder,self).__init__()
        self.hidden_size = config.hidden_size
        self.topic_size = config.topic_size
        self.num_layers = config.num_layers
        self.bidirection = config.bidirection
        self.batch_size = config.batch_size
        self.lstm = nn.LSTM(self.ctx_size,self.hidden_size,num_layers=self.num_layers,batch_first=True,bidirection=self.bidirection)

        ## topic
        self.topic_h_W = torch.nn.Linear(self.hidden_size, self.topic_size)
        self.topic_ctx_W = torch.nn.Linear(self.ctx_size, self.topic_size)

        # stop distribution output
        self.stop_h_W = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.stop_prev_h_W = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.stop_W = torch.nn.Linear(self.hidden_size, 2)

        self.hidden = self.init_hidden()
        self.init_weights()


    def init_weights(self):
        self.topic_ctx_W.weight.data.uniform_(-0.1,0.1)
        self.topic_h_W.weight.data.uniform_(-0.1,0.1)

        self.stop_h_W.weight.data.uniform_(-0.1,0.1)
        self.stop_prev_h_W.weight.data.uniform_(-0.1,0.1)
        self.stop_W.weight.data.uniform_(-0.1,0.1)



    def init_hidden(self,batch_size=None):
        if batch_size is None:
            batch_size  = self.batch_size

        h = torch.zeros(self.num_layers * self.bidirection, batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(self.num_layers * self.bidirection, batch_size, self.hidden_size, device=self.device)
        nn.init.orthogonal_(h)
        nn.init.orthogonal_(c)
        return h,c


    ## 这里ctx是对image feats的attention,可以是spatial attention，semantic attention以及channle attention
    def forward(self,ctx,hidden):
        prev_hidden = hidden
        output,hidden = self.lstm(ctx,hidden)
        output = output[0]
        ## predict topic vector
        topic = self.topic_h_W(output) + self.topic_ctx_W(ctx)
        topic = F.tanh(topic)

        ## predict stop distribution
        stop = self.stop_h_W(output) + self.stop_prev_h_W(prev_hidden)
        stop = F.tanh(stop)
        stop = self.stop_W(stop)
        return topic,stop,hidden

class WordDecoder(nn.Module):
    def __init__(self,config):
        super(WordDecoder,self).__init__()
        self.dict_size = config.dict_size
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size
        self.num_layers = config.num_layers
        self.bidirection = config.bidirection
        self.dropout = config.dropout

        self.pt = self.pt
        if not self.pt:
            ## +2 for SOS and UNK token
            self.embedding = nn.Embedding(self.words_size + 2, self.hidden_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.init_embed),freeze=False)
        
        ## +2 for EOS and UNK token
        self.lstm = nn.LSTM(self.embd_size,self.hidden_size,batch_first=True,num_layers=num_layers,bidirection=self.bidirection)
        
        self.out = nn.Linear(self.hidden_size,self.words_size + 2)

    def forward(self,ctx,):
        pass


