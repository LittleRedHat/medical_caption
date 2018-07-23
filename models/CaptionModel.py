import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

class Dict2Class():
    def __init__(self,dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])

class AggragationAttn(nn.Module):
    def __init__(self):
        super(AggragationAttn,self).__init__()

    def forward(self,feats,hidden):
        
        

class SentDecoder(nn.Module):
    def __init__(self,config):
        super(SentDecoder,self).__init__()
        self.hidden_dim = config.hidden_dim
        self.topic_dim = config.topic_dim
        self.num_layers = config.num_layers
        self.bidirection = config.bidirection
        self.batch_size = config.batch_size
        self.lstm = nn.LSTM(self.ctx_dim,self.hidden_dim,num_layers=self.num_layers,batch_first=True,bidirection=self.bidirection)

        ## topic
        self.topic_h_W = torch.nn.Linear(self.hidden_dim, self.topic_dim)
        self.topic_ctx_W = torch.nn.Linear(self.ctx_dim, self.topic_dim)

        # stop distribution output
        self.stop_h_W = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.stop_prev_h_W = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.stop_W = torch.nn.Linear(self.hidden_dim, 2)

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

        h = torch.zeros(self.num_layers * self.bidirection, batch_size, self.hidden_dim, device=self.device)
        c = torch.zeros(self.num_layers * self.bidirection, batch_size, self.hidden_dim, device=self.device)
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
        self.hidden_dim= config.hidden_dim
        self.embd_dim = config.embd_dim
        self.num_layers = config.num_layers
        self.bidirection = config.bidirection
        self.dropout = config.dropout

        self.pt = self.pt
        if not self.pt:
            ## +2 for SOS and UNK token
            self.embedding = nn.Embedding(self.words_size + 2, self.hidden_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.init_embed),freeze=False)
        
        ## build LSTM
        self.i2h = nn.Linear(self.embd_dim,self.hidden_dim)
        self.h2h = nn.Linear(self.hidden_dim,self.hidden_dim)

        self.dropout = nn.Dropout(self.dropout)


        ## +2 for EOS and UNK token
        self.out = nn.Linear(self.hidden_size,self.words_size + 2)

    def forward(self,input,topic,ctx,hidden):
        word_embd = self.embedding(x)


        output,hidden = self.lstm(x,hidden)
        output = self.out(output)
        return output,hidden

class AggregationModel(nn.Module):
    def __init__(self,config):
        super(AggregationAttn,self).__init__(self)
        sent_decoder_config = {
            'hidden_size':config.sent_hidden_size,
            'topic_size':config.topic_size,
            'num_layers':config.sent_num_layers,
            'bidirection':config.sent_bidirection,
            'batch_size':config.sent_batch_size,
        } 
        sent_decoder_config = Dict2Class(sent_decoder_config)

        self.sent_decoder = SentDecoder(sent_decoder_config)

        word_decoder_config = {
            'batch_size':config.word_batch_size,
            'dict_size':config.dict_size,
            'num_layers':config.word_num_layers,
            'embd_size':config.embd_size,
            'hidden_size':config.word_hidden_size,
            'dropout':config.word_drop,
            'bidirection':config.word_bidirection
        }
        self.word_decoder = WordDecoder(word_decoder_config)
        

        self.image_encoder = config.image_encoder
    
    def compute_sent_ctx(self,)

    def forward(self,x):
        image,caption = x
        logit,feats = self.image_encoder(image)

        


        

        
        
        



