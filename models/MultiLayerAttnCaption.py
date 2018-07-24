import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

class Dict2Class():
    def __init__(self,dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


## first channle attention then spatial attention
class AggregationAttn(nn.Module):
    def __init__(self,hidden_size,feat_size):
        super(AggragationAttn,self).__init__()

        self.hidden_size = hidden_size
        self.feat_size = feat_size ## channle * w * h
        ## spatial attention
        self.s_h2v = nn.Linear(self.hidden_size,self.hidden_size)
        self.s_im2v = nn.Linear(self.feat_size[0],self.hidden_size)
        self.s_ctx_w = nn.Linear(self.hidden_size,self.feat_size[1] * self.feat_size[2])

        ## channle attention
        self.c_h2v = nn.Linear(self.hidden_size,self.hidden_size)
        self.c_w = nn.Parameter(torch.tensor((self.hidden_size,),dtype=torch.float,require_grad=True))
        self.c_bias = nn.Parameter(torch.tensor((self.hidden_size,),dtype=torch.float,require_grad=True))
        self.c_ctx_w = nn.Parameter(torch.tensor((self.hidden_size,),dtype=torch.float,require_grad=True))
        self.c_ctx_bias = nn.Parameter(torch.tensor((self.feat_size[0],),dtype=torch.float,require_grad=True))

    def channle_attention(self,feat_conv,hidden):
        batch_size,channle,w,h = feat_conv.size()
        feat_conv_reshape = feat_conv.reshape(batch_size,channle,-1)
        ## batch_size * channle
        feat_conv_mean = feat_conv_reshape.mean(dim=2,keepdim=False)
        proj_feat = torch.matmul(feat_conv.unsqueeze(2),self.c_w.unsqueeze(0)) + self.c_bias
        proj_h = self.c_h2v(hidden)
        proj_ctx = proj_feat + proj_h
        proj_ctx = F.tanh(proj_ctx)
        proj_ctx = torch.mm(proj_ctx,self.c_ctx_w) + self.c_ctx_bias
        ## batch * c
        c_ctx = F.softmax(proj_ctx,dim=1)
        return c_ctx
    
    def spatial_attention(self,feat_conv,hidden):
        batch_size,channel,w,h = feat_conv.size()
        ## reshape 
        feat_conv_reshape = feat_conv.reshape(batch_size,channel,-1)
        feat_conv_reshape = feat_conv_reshape.transpose(1,2) ## batch * (w*h) * channle
        
        proj_h = self.s_h2v(hidden)
        proj_feat = self.s_im2v(feat_conv_reshape)
        proj_ctx = proj_ctx + proj_h
        proj_ctx = F.tanh(proj_ctx)
        proj_ctx = self.s_ctx_w(proj_ctx))

        ## batch * (w*h)
        s_ctx = F.softmax(proj_ctx,dim=1)

        return s_ctx

    def forward(self,feat_conv,hidden):
        ''' 
            fea_conv batch * c * w * h
            c_ctx batch * c
            s_ctx batch * (w*h)
        '''
        batch_size,c,w,h = feat_conv.size()
        
        ## batch * c
        c_ctx = self.channle_attention(feat_conv,hidden) 

        ## batch * c * (w*h)
        weighted_feat_conv = feat_conv.reshape(batch_size,c,-1) * c_ctx.unsqueeze(2) * c
        ## batch * c * w * h
        weighted_feat_conv = weighted_feat_conv.reshape(batch_size,channle,w,h)

        ## batch * (w * h)
        s_ctx = self.spatial_attention(weighted_feat_conv,hidden)
        '''
        final features
        '''
        weighted_feat_conv = weighted_feat_conv.reshape(batch,c,(w*h)) * s_ctx.unsqueeze(1) * w * h
        weighted_feat_conv = weighted_feat_conv.reshape(batch,c,w,h)

        ## batch * c * w * h
        return weighted_feat_conv

class SentDecoder(nn.Module):
    def __init__(self,config):
        super(SentDecoder,self).__init__()
        self.hidden_dim = config.hidden_dim
        self.topic_dim = config.topic_dim
        self.num_layers = config.num_layers
        self.bidirection = config.bidirection
        self.batch_size = config.batch_size
        self.lstm = nn.LSTM(self.ctx_dim,self.hidden_dim,num_layers=self.num_layers,batch_first=True,bidirectional=self.bidirection)

        ## topic
        self.topic_h_W = torch.nn.Linear(self.hidden_dim, self.topic_dim)
        self.topic_ctx_W = torch.nn.Linear(self.ctx_dim, self.topic_dim)

        # stop distribution output
        self.stop_h_W = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.stop_prev_h_W = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.stop_W = torch.nn.Linear(self.hidden_dim, 2)

        self.state = self.init_state()
        self.init_weights()

    def init_weights(self):
        self.topic_ctx_W.weight.data.uniform_(-0.1,0.1)
        self.topic_h_W.weight.data.uniform_(-0.1,0.1)

        self.stop_h_W.weight.data.uniform_(-0.1,0.1)
        self.stop_prev_h_W.weight.data.uniform_(-0.1,0.1)
        self.stop_W.weight.data.uniform_(-0.1,0.1)

    def init_state(self,batch_size=None):
        if batch_size is None:
            batch_size  = self.batch_size

        h = torch.zeros(self.num_layers * self.bidirection, batch_size, self.hidden_dim, device=self.device)
        c = torch.zeros(self.num_layers * self.bidirection, batch_size, self.hidden_dim, device=self.device)
        nn.init.orthogonal_(h)
        nn.init.orthogonal_(c)
        return h,c

    ## 这里ctx是对image feats的attention,可以是spatial attention，semantic attention以及channle attention
    ## ctx batch_size * ctx_len
    def forward(self,ctx,state):
        
        ctx = ctx.unsqueeze(1)
        
        prev_state = state

        output,state = self.lstm(ctx,state)

        ## batch * hidden_dim
        output = output.squeeze(1)

        ## predict topic vector
        topic = self.topic_h_W(output) + self.topic_ctx_W(ctx)
        topic = F.tanh(topic)

        ## predict stop distribution
        stop = self.stop_h_W(output) + self.stop_prev_h_W(prev_state[0])

        stop = F.tanh(stop)
        stop = self.stop_W(stop)
        return topic,stop,state

class WordDecoder(nn.Module):
    def __init__(self,config):
        super(WordDecoder,self).__init__()
        self.dict_size = config.dict_size
        self.hidden_dim= config.hidden_dim
        # self.embd_dim = config.embd_dim
        self.num_layers = config.num_layers

        self.dropout = config.dropout
        self.att_feat_size = config.att_feat_size

        ## build LSTM
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.hidden_dim)
        self.i2h = nn.Linear(self.embd_dim,self.hidden_dim)
        self.h2h = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout)

        ## +2 for EOS and UNK token
        self.out = nn.Linear(self.hidden_size,self.words_size + 2)

        self.init_weights()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    def init_weights(self):
        self.a2c.weight.data.uniform_(-0.1,0.1)
        self.i2h.weight.data.uniform_(-0.1,0.1)
        self.h2h.weight.data.uniform_(-0.1,0.1)
        self.out.weight.data.uniform_(-0.1,0.1)

    def init_state(self,batch_size):
        
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        nn.init.orthogonal_(h)
        nn.init.orthogonal_(c)
        return h,c


    def forward(self,xt,att_res,state):
        
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunlk = all_input_sums.narrow(1,0,3 * self.hidden_dim)
        sigmoid_chunlk = F.sigmoid(sigmoid_chunlk)

        in_gate = sigmoid_chunk.narrow(1, 0, self.hidden_dim)
        forget_gate = sigmoid_chunk.narrow(1, self.hidden_dim, self.hidden_dim)
        out_gate = sigmoid_chunk.narrow(1, self.hidden_dim * 2, self.hidden_dim)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + self.a2c(att_res)
        in_transform = torch.max(in_transform.narrow(1, 0, self.rnn_size),in_transform.narrow(1, self.rnn_size, self.rnn_size))

        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)
        output = self.dropout(next_h)

        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class AggregationCaptionModel(nn.Module):
    def __init__(self,config):
        super(AggregationCaptionModel,self).__init__(self)
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
            'hidden_size':config.word_hidden_size,
            'dropout':config.word_drop,
            'bidirection':config.word_bidirection,
            'att_feat_size':config.att_feat_size,
        }
        self.word_decoder = WordDecoder(word_decoder_config)
        # self.image_encoder = config.image_encoder
        self.max_sent_len = config.max_sent_len

        self.embd_size = config.embd_size
        self.pt = config.pt

        if not self.pt:
            ## +2 for SOS and UNK token
            self.embedding = nn.Embedding(self.words_size + 2, self.embd_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.init_embed),freeze=False)
        
        self.sent_feat_size = config.sent_feat_size

        self.sent_attention = AggregationAttn(config.sent_hidden_size,self.sent_feat_size)
        self.sent_avgpool = nn.AvgPool2d(kernel_size=(14,14))

        self.word_feat_size = config.word_feat_size
        self.word_attention = AggregationAttn(config.word_hidden_size,self.word_feat_size)
        self.word_avgpool = nn.AvgPool2d(kernel_size=(14,14))


    def feat_normlize(self,conv_feats):
        norm = conv_feats.norm(p=2,dim=1,keepdim=True)
        return conv_feats / norm


    def generate_sentences(self,sent_topics,feat,seq,seq_length):
        
        for i in range(seq_length):
            

        
        
        
        
        
        
        
        
        
        
    
    ## feat batch * c * h * w
    def _forward(self,feat,seq,att_masks=None):
        batch_size = feats.size(0)

        sent_decoder_state = self.sent_decoder.init_state
        sent_topics = []
        sent_stop = []


        ## generate topics
        for sent_i in range(self.sent_max_len):
            ## batch * c * h * w
            sent_conv_feats = self.sent_attention.forward(feat,sent_decoder_hidden)
            sent_conv_feats = self.sent_avgpool(sent_conv_feats)
            ## batch * c
            sent_conv_feats = sent_conv_feats.reshape(sent_conv_feats.size(0),sent_conv_feats(1))
            ## normalize
            sent_ctx = self.feat_normlize(sent_conv_feats) # batch * c


            ## batch * topic_size, batch * 2, batch * (h,c)
            topic,stop,sent_decoder_state = self.sent_decoder.forward(sent_ctx,sent_decoder_state)

            sent_topics.append(topic)
            sent_stop.append(stop)

        ## generate sentences
        for batch_i in range(batch_size):
            self._generate_sentences(sent_topics[:,batch_i,:],)
        


        
        

            
            
            
            
            
            
            


    



        


        

        
        
        



