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
        super(AggregationAttn,self).__init__()
        self.hidden_size = hidden_size
        self.feat_size = feat_size ## channle * w * h

        ## spatial attention
        self.s_h2v = nn.Linear(self.hidden_size,self.hidden_size)
        self.s_im2v = nn.Linear(self.feat_size[0],self.hidden_size)
        self.s_ctx_w = nn.Linear(self.hidden_size,self.feat_size[1] * self.feat_size[2])

        ## channle attention
        self.c_h2v = nn.Linear(self.hidden_size,self.hidden_size)
        self.c_w = nn.Linear(1,self.hidden_size)
        self.c_ctx = nn.Linear(self.hidden_size,1)
        # self.c_w = nn.Parameter(torch.tensor((self.hidden_size,),dtype=torch.float,requires_grad=True))
        # self.c_bias = nn.Parameter(torch.tensor((self.hidden_size,),dtype=torch.float,requires_grad=True))
        # self.c_ctx_w = nn.Parameter(torch.tensor((self.hidden_size,),dtype=torch.float,requires_grad=True))
        # self.c_ctx_bias = nn.Parameter(torch.tensor((self.feat_size[0],),dtype=torch.float,requires_grad=True))


    def channle_attention(self,feat_conv,hidden):
        batch_size,channle,w,h = feat_conv.size()
        feat_conv_reshape = feat_conv.reshape(batch_size,channle,-1)



        # batch_size,channle,w,h = feat_conv.size()
        # feat_conv_reshape = feat_conv.reshape(batch_size,channle,-1)
        # ## batch_size * channle
        # feat_conv_mean = feat_conv_reshape.mean(dim=2,keepdim=False)

        # proj_feat = torch.matmul(feat_conv_mean.unsqueeze(2),self.c_w.unsqueeze(0)) + self.c_bias

        # proj_h = self.c_h2v(hidden)
        # proj_ctx = proj_feat + proj_h.unsqueeze(1)

        # proj_ctx = F.tanh(proj_ctx)
        # print(proj_ctx.size(),self.c_ctx_w.size(0),self.c_ctx_bias.size(0))

        # proj_ctx = torch.mm(proj_ctx,self.c_ctx_w) + self.c_ctx_bias
        # ## batch * c
        # c_ctx = F.softmax(proj_ctx,dim=1)
        # return c_ctx
    
    def spatial_attention(self,feat_conv,hidden):
        batch_size,channel,w,h = feat_conv.size()
        ## reshape 
        feat_conv_reshape = feat_conv.reshape(batch_size,channel,-1)
        feat_conv_reshape = feat_conv_reshape.transpose(1,2) ## batch * (w*h) * channle
        
        proj_h = self.s_h2v(hidden)
        proj_feat = self.s_im2v(feat_conv_reshape)
        proj_ctx = proj_ctx + proj_h
        proj_ctx = F.tanh(proj_ctx)
        proj_ctx = self.s_ctx_w(proj_ctx)

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
        self.ctx_dim = config.ctx_dim
        self.feat_size = config.feat_size

        self.lstm = nn.LSTM(self.ctx_dim,self.hidden_dim,num_layers=self.num_layers,batch_first=True,bidirectional=self.bidirection)
        self.device = config.device
        ## topic
        self.topic_h_W = torch.nn.Linear(self.hidden_dim, self.topic_dim)
        self.topic_ctx_W = torch.nn.Linear(self.ctx_dim, self.topic_dim)

        # stop distribution output
        self.stop_h_W = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.stop_prev_h_W = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.stop_W = torch.nn.Linear(self.hidden_dim, 2)


        self.attention = AggregationAttn(self.hidden_dim,config.feat_size)
        self.avgpool = nn.AvgPool2d(kernel_size=(7,7))

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
     

    def feat_normlize(self,conv_feats):
        norm = conv_feats.norm(p=2,dim=1,keepdim=True)
        return conv_feats / norm

    def prepare_sent_ctx(self,feat_conv,hidden):
        ## batch * c * h * w
        sent_conv_feats = self.attention(feat_conv,hidden)
        sent_conv_feats = self.avgpool(sent_conv_feats)
        ## batch * c
        sent_conv_feats = sent_conv_feats.reshape(sent_conv_feats.size(0),sent_conv_feats(1))
        ## normalize
        sent_ctx = self.feat_normlize(sent_conv_feats) # batch * c
        return sent_ctx

    def forward(self,feat_conv,state):
        
        ## ctx是对image feats的attention,可以是spatial attention，semantic attention以及channle attention
        ## ctx batch_size * ctx_len
        ctx = self.prepare_sent_ctx(feat_conv,state[0][-1])
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
        self.embd_dim = config.embd_dim
        self.num_layers = config.num_layers
        self.topic_dim = config.topic_dim
        self.dropout = config.dropout
        self.att_feat_size = config.att_feat_size
        self.device = config.device
        self.feat_size = config.feat_size
        self.pt = config.pt

        ## build LSTM
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.hidden_dim)
        self.i2h = nn.Linear(self.embd_dim,self.hidden_dim)
        self.h2h = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout)

        if not self.pt:
            ## +2 for SOS and UNK token
            self.embedding = nn.Embedding(self.dict_size + 2, self.embd_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.init_embed),freeze=False)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.ss_prob = 0.0
        self.max_words = config.max_words
        
        self.logit_layers = getattr(config,'logit_layers',1)

        if self.logit_layers == 1:
            self.logit = nn.Linear(self.hidden_dim, self.dict_size + 2)
        else:
            self.logit = [[nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.hidden_size, self.dict_size + 2)]))

        word_attn_hidden_size = self.topic_dim + self.hidden_dim

        self.attention = AggregationAttn(word_attn_hidden_size,self.feat_size)
        self.init_weights()

    
    def set_ss_prob(self,prob):
        self.ss_prob = prob

    def init_weights(self):
        self.a2c.weight.data.uniform_(-0.1,0.1)
        self.i2h.weight.data.uniform_(-0.1,0.1)
        self.h2h.weight.data.uniform_(-0.1,0.1)
        self.logit.weight.data.uniform_(-0.1,0.1)

    def init_state(self,batch_size):
        
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        nn.init.orthogonal_(h)
        nn.init.orthogonal_(c)
        return h,c
    
     
    def core(self,xt,feat_conv,state):
        ## state 上一时刻的h与c, h和c -> layers(=1) * batch_size * hidden_dim
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


    def feat_normlize(self,conv_feats):
        norm = conv_feats.norm(p=2,dim=1,keepdim=True)
        return conv_feats / norm

    ## we use [topic,lstm_hidden] as attention_hidden
    def prepare_word_ctx(self,feat_conv,topic,hidden):
        ## hidden batch * hidden_dim
        ## topic batch * topic_dim
        
        hidden = torch.cat((topic,hidden),dim=1)
        
        ## batch * c * h * w
        sent_conv_feats = self.attention(feat_conv,hidden)
        sent_conv_feats = self.avgpool(sent_conv_feats)
        ## batch * c
        sent_conv_feats = sent_conv_feats.reshape(sent_conv_feats.size(0),sent_conv_feats(1))
        ## normalize
        sent_ctx = self.feat_normlize(sent_conv_feats) # batch * c
        return sent_ctx

    ## feat_conv 1 * c * w * h

    def forward(self,feat_conv,sent_topic,seq,seq_len):
        
        batch_size = seq.size(0)
        outputs = att_res.new(batch_size, self.max_words, self.dict_size + 2)
        max_seq_len = torch.max(seq_len)
        
        state = self.init_state(batch_size)
        for i in range(max_seq_len):
            ## schedule sampling
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = att_res.new(batch_size).uniform_(0,1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:,i].clone()
                else:
                    sample_index = sample_mask.nonzero().view(-1)
                    it = seq[:,i].data.clone()
                    prob_prev = torch.exp(outputs[:,i-1].detach())
                    sampled_word_index = torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_index)
                    it.index_copy_(0, sample_index,sampled_word_index)
            else:
                it = seq[:,i].clone()

            att_res = self.prepare_word_ctx(feat_conv,sent_topic,state[0][-1])

            output,state = self.get_logprobs_state(it,att_res,state)
            outputs[:,i] = output
        return outputs

    def get_logprobs_state(self, it, att_res,state):
        xt = self.embedding(it)
        output, state = self.core(xt, att_res,state)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state
        

class AggregationCaptionModel(nn.Module):
    def __init__(self,config):
        super(AggregationCaptionModel,self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        sent_decoder_config = {
            'hidden_dim':config.sent_hidden_dim,
            'topic_dim':config.topic_dim,
            'num_layers':config.sent_num_layers,
            'bidirection':config.sent_bidirection,
            'batch_size':config.batch_size,
            'ctx_dim':config.att_feat_size,
            'att_feat_size':config.att_feat_size,
            'feat_size':config.feat_size,
            'device':self.device
        } 
        sent_decoder_config = Dict2Class(sent_decoder_config)
        self.sent_decoder = SentDecoder(sent_decoder_config)

        word_decoder_config = {
            'dict_size':config.dict_size,
            'num_layers':config.word_num_layers,
            'hidden_dim':config.word_hidden_dim,
            'dropout':config.word_drop,
            'bidirection':config.word_bidirection,
            'att_feat_size':config.att_feat_size,
            'feat_size':config.feat_size,
            'max_words':config.max_words,
            'init_embed':config.init_embed,
            'embd_dim':config.embd_dim,
            'topic_dim':config.topic_dim,
            'pt':config.pt,
            'device':self.device,
           

        }
        word_decoder_config = Dict2Class(word_decoder_config)

        self.word_decoder = WordDecoder(word_decoder_config)
        self.topic_dim = config.topic_dim
        self.max_words = config.max_words
        self.max_sent = config.max_sent
        self.dict_size = config.dict_size

    ## feats batch * c * h * w
    ## seq batch * max_sent * max_words
    ## seq_num batch
    ## seq_len batch * max_sent
    def forward(self,feats,seq,seq_num,seq_len):
        batch_size = feats.size(0)
        sent_topics = feats.new(self.max_sent,batch_size,self.topic_dim)
        sent_stop = feats.new(self.max_sent,batch_size,2)
        sent_decoder_state = self.sent_decoder.init_state()
        ## generate topics
        for sent_i in range(self.max_sent):
            ## batch * topic_dim, batch * 2, batch * (h,c)
            topic,stop,sent_decoder_state = self.sent_decoder.forward(feats,sent_decoder_state)
            sent_topics[sent_i] = topic
            sent_stop[sent_i] = stop

        ## batch_size * max_len * topic_dim
        sent_topics = sent_topics.transpose(0,1)
        ## batch_size * max_len * 2
        sent_stop = sent_stop.transpose(0,1)

        outputs = feats.new(batch_size,self.max_sent,self.max_words,self.dict_size + 2)

        ## generate sentences for all images
        for batch_i in range(batch_size):
            ## max_len * topic_dim
            sent_topics_i = sent_topics[batch_i]
            ## 1 * c * w * h
            feat_i = feats[batch_i].unsqueeze(0)
            sent_num_i = sent_num[batch_i]


            seq_i = seq[batch_i][:sent_num_i]
            seq_len_i = seq_len[i][:sent_num_i]
            ## 1 * max_sent * max_words * (dict_size + 2)
            output = self.word_decoder.forward(feat_conv,sent_topic,seq,seq_len)

            outputs[batch_i] = output
        ## batch * max_sent * max_words * (dict_size + 2)
        return outputs,sent_stop,sent_topics
            


        


        
        

            
            
            
            
            
            
            


    



        


        

        
        
        



