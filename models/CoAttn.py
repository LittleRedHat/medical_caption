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

SOS_INDEX = 0

class SentDecoder(nn.Module):
    def __init__(self,config):
        super(SentDecoder,self).__init__()

        self.hidden_size = config.sent_hidden_size
        self.topic_size = config.topic_size
        self.num_layers = config.sent_num_layers
        self.bidirection = config.sent_bidirection
        self.batch_size = config.sent_batch_size
        self.device = config.device

        self.semantic_attention = config.semantic_attention
        self.spatial_attention = config.spatial_attention

        self.ctx_size = self.sent_hidden_size

        self.attention_dim = 0

        ## visual attention
        if self.spatial_attention:
            ## feature size (h*w) * channel
            self.feature_size = config.feature_size
            self.attention_dim = self.attention_dim + self.feature_size[1]
            self.ctx_h_v = nn.Linear(self.hidden_size,self.hidden_size)
            self.ctx_im_w = nn.Linear(self.feature_size[1],self.hidden_size)
            self.ctx_w_v = nn.Linear(self.hidden_size,self.feature_size[0])

        ## semantic attention
        if self.semantic_attention:
            ## semantic size == num_tags * tag_embedding
            self.semantic_size = config.semantic_size

            if not config.semantic_pretrain:
                self.semantic_embedding = nn.Embedding(self.semantic_size[0],self.semantic_size[1])
            else:
                self.semantic_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.init_embed),freeze=False)
                
            self.attention_dim = self.attention_dim + self.semantic_size[1]
            self.ctx_h_s = nn.Linear(self.hidden,self.hidden_size)
            self.ctx_s_w = nn.Linear(self.semantic_size[1],self.hidden_size)
            self.ctx_w_s = nn.Linear(self.hidden_size,self.semantic_size[0])

        ## final attention
        self.ctx_w = nn.Linear(self.attention_dim,self.hidden_size)

        ## rnn init
        self.lstm = nn.LSTM(self.ctx_size,self.hidden_size,num_layers=self.num_layers,batch_first=True,bidirection=self.bidirection)

        ## topic
        self.topic_h_W = torch.nn.Linear(self.hidden_size, self.topic_size)
        self.topic_ctx_W = torch.nn.Linear(self.ctx_size, self.topic_size)

        # stop distribution output
        self.stop_h_W = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.stop_prev_h_W = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.stop_W = torch.nn.Linear(self.hidden_size, 2)

        self.hidden = self.init_hidden()


    def init_hidden(self):
        h = torch.zeros(self.num_layers * self.bidirection, self.batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(self.num_layers * self.bidirection, self.batch_size, self.hidden_size, device=self.device)
        nn.init.orthogonal_(h)
        nn.init.orthogonal_(c)
        return h,c

    def coattention(self,features,semantic,hidden):
        ## batch * visual_size
        visual_attn = self.get_visual_attention(features,hidden)

        ## batch * semantic_embedding_size
        semantic_attn = self.get_semantic_attention(semantic,hidden)
        semantic_ctx = torch.mm(semantic_attn,semantic)
        visual_ctx = torch.mm(visual_attn,features)

        ## batch * (visual_size + semantic_embedding_size)
        cat_ctx = torch.cat((visual_ctx,semantic_ctx),dim=1)

        ctx = self.ctx_w(cat_ctx)
        return ctx,visual_attn,semantic_attn

    def semantic_attention_only(self,semantic,hidden):
        semantic_attn = self.get_semantic_attention(semantic,hidden)
        semantic_ctx = torch.mm(semantic_attn,semantic)
        ctx = self.ctx_w(semantic_ctx)
        return ctx,semantic_attn
    
    def spatial_attention_only(self,features,hidden):
        visual_attn = self.get_visual_attention(features,hidden)
        visual_ctx = torch.mm(visual_attn,features)
        ctx = self.ctx_w(visual_ctx)
        return ctx,visual_attn

    def get_visual_attention(self,features,hidden):
        x = self.ctx_im_w(features) +  self.ctx_h_v(hidden)
        x = F.tanh(x)
        x = self.ctx_w_v(x)
        x = F.softmax(x)
        return x
    
    def get_semantic_attention(self,semantic,hidden):
        x = self.ctx_s_w(semantic) + self.ctx_h_s(hidden)
        x = F.tanh(x)
        x = self.ctx_w_s(x)
        x = F.softmax(x)
        # x = torch.mm(x,semantic)
        return x 
        
    def forward(self,features,semantic,hidden):
        attention_weights = []
        if self.semantic_attention and self.spatial_attention:
            semantic_embds = self.semantic_embedding(semantic)
            ctx,visual_w,semantic_w = self.coattention(features,semantic_embds,hidden)
            attention_weights.append(visual_w)
            attention_weights.append(semantic_w)

        elif self.semantic_attention:
            semantic_embds = self.semantic_embedding(semantic)
            ctx,semantic_w = self.semantic_attention_only(semantic_embds,hidden)
            attention_weights.append(semantic_w)

        elif self.spatial_attention:
            ctx,visual_w = self.spatial_attention_only(features,hidden)
            attention_weights.append(visual_w)

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
        # stop = F.softmax(stop)


        return topic,stop,hidden,attention_weights


class WordDecoder(nn.Module):
    def __init__(self,config):
        super(WordDecoder,self).__init__()
        self.words_size = config.dict_size
        self.hidden_size = config.word_hidden_size
        self.embd_size = config.word_embd_size
        self.num_layers = config.word_num_layers
        self.pt = self.pt

        if not self.pt:
            ## for SOS and UNK 
            self.embedding = nn.Embedding(self.words_size + 2, self.hidden_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.init_embed),freeze=False)

        self.lstm = nn.LSTM(self.embd_size,self.hidden_size,batch_first=True,num_layers=num_layers)
        
        self.out = nn.Linear(self.hidden_size,self.words_size)
    
    def forward(self,x,hidden):
        x = self.embedding(x)
        output,hidden = self.lstm(x,hidden)
        output = self.out(output)
        return output,hidden

class MLCEncoder(nn.Module):
    def __init__(self,num_classes):
        super(MLCEncoder,self).__init__()
        self.num_classes = num_classes
        # self.model = M.vgg19(pretrained=False)
        self.model = M.resnet50(pretrained=False)
        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(num_features, num_classes)
        
        

        # self.classifier = nn.Linear()
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     # nn.Dropout(),
        #     # nn.Linear(4096, 4096),
        #     # nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
    def focal_loss(self,pred,target,gamma = 2,alpha = 0.25,average = True):
        loss = -  alpha * torch.pow((1 - pred),gamma) * target * torch.log(pred) - (1 - alpha) * torch.pow(pred,gamma) * (1 - target) * torch.log(1 - pred)
        loss = loss.sum(1)
        if average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    
    def compute_auc(self,pred,truth):
        AUROCs = []
        truth_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
        for i in range(self.num_classes):
            AUROCs.append(roc_auc_score(truth_np[:, i], pred_np[:, i]))
        return AUROCs

    def predict(self,val_dataset,batch_size,device):
        dataloader = DataLoader(val_dataset,shuffle=False,batch_size=batch_size,num_workers=4)
        self.eval()
        total_loss = 0.0
        for sample in dataloader:
            batch_image,batch_tags = sample
            
            self.to(device)
            batch_image = batch_image.to(device)
            batch_tags = batch_tags.to(device)
            logits,_ = self.forward(batch_image)
            preds = F.sigmoid(logits)
            eps=1e-10
            preds = torch.clamp(preds, eps, 1 - eps)

            # prob = F.softmax(logits,dim=1).data.cpu().numpy()
            prob = preds.data.cpu().numpy()
            r = np.random.randint(0,len(prob))
            topk = np.argsort(prob[r])[::-1][:10]
            truth = batch_tags[r].nonzero().data.cpu().numpy()
            print('***************')
            print(topk)
            print(truth.squeeze())
            

            loss = self.focal_loss(preds,batch_tags,average=False)
            total_loss += loss.data.numpy() if device == 'cpu' else loss.data.cpu().numpy()
        return total_loss / len(val_dataset)
    
    def update(self,train_dataset,val_dataset,config):
        if config.start_from != -1:
            self.load_state_dict(torch.load(os.path.join(config.save_dir, 'model_params_{}.pkl'.format(config.start_from))))
        device = torch.device(config.device)

        valid_file = open(os.path.join(config.save_dir, 'valid_result.csv'), 'w')

        train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.nw)
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(params=parameters,lr=config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)
        
        if config.device == 'cuda':
            self.to(device)
        for epoch in range(1,config.epoches + 1):
            scheduler.step()
            self.train()

            sum_loss = .0
            time_start_spoch = time.time()

            for step,sample in enumerate(train_dataloader,1):
                optimizer.zero_grad()
                batch_image,batch_tags = sample
                if config.device == 'cuda':
                    # self.to(device)
                    batch_image = batch_image.to(device)
                    batch_tags = batch_tags.to(device)

                logits,_ = self.forward(batch_image)
                preds = F.sigmoid(logits)
                eps=1e-10
                preds = torch.clamp(preds, eps, 1 - eps)
                loss = self.focal_loss(preds,batch_tags)
                sum_loss += loss.data.numpy() if config.device == 'cpu' else loss.data.cpu().numpy()
                loss.backward()
                optimizer.step()

                if step % config.log == 0:
                    # prob = F.softmax(logits,dim=1).data.cpu().numpy()
                    # topk = np.argsort(prob[0])[::-1][:10]
                    # truth = batch_tags[0].nonzero().data.cpu().numpy()
                    # print(topk)
                    # print(truth.squeeze())

                    info = 'epoch {} batch step {}/{}: loss = {:.5f} {:.4f}mins'
                    print(info.format(epoch,step, len(train_dataloader),sum_loss / step,(time.time() - time_start_spoch) / 60.))

            if epoch % config.eval_frq == 0:
                loss = self.predict(val_dataset,config.batch_size,device)
                info = 'epoch {}, loss_test = {:.6f} time/epoch={:.1f}mins'
                valid_file.write(info.format(epoch, loss, (time.time() - time_start_spoch) / 60.) + '\n')
                valid_file.flush()
                print(info.format(epoch, loss, (time.time() - time_start_spoch) / 60.) + '\n')
            
            if epoch % config.save_frq == 0:
                torch.save(self.state_dict(),os.path.join(config.save_dir, 'model_params_{}.pkl'.format(epoch)))

        valid_file.close()

    def forward(self,x):
        # features = self.model.features(x)
        # logits = self.classifier(features.view(features.size(0),-1))
        logits = self.model(x)
        return logits,None

class SpatialAttn():
    def __init__(self,config):
        super(CoAttn,self).__init__()

        self.config = config
        self.mlc = MLCEncoder(config.num_classes)
        self.sent_decoder = SentDecoder(self.config)
        self.word_decoder = WordDecoder(self.config)

    def pretrain_mlc(self,train_dataset,val_dataset,config):
        device = torch.device(config.device)
        train_loader = DataLoader(dataset=train_dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.nw)

    def one_pass(self,image_input_variable,image_target_output,caption_input_variable,stop_target_variable,tags_input_variable,config):
        


        preds,feature_maps = self.mlc(image_input_variable.squeeze())

        l1_norm = torch.norm(image_target_output, p=1, dim=1).detach()
        image_target_output = image_target_output / l1_norm.unsqueeze(1)

        mlc_loss = nn.BCELoss(size_average=True)(preds,image_target_output)

        ## transfer feature_maps to batch * (h*w) * channel
        feature_maps = feature_maps.view(feature_maps.size()[0],feature_maps.size()[1],-1)
        feature_maps = feature_maps.transpose(1,2)

        ## generate max_sent_num topic
        sent_decoder_hidden = self.sent_decoder.hidden
        sent_semantic = None
        sent_topics = []
        stop_loss = 0.0

        for i in range(config.max_sent_num):
            sent_decoder_topic, sent_decoder_stop, sent_decoder_hidden, attn_weights = self.sent_decoder(feature_maps,sent_semantic,sent_decoder_hidden)
            sent_topics.append(sent_decoder_topic[0])
            stop_loss += nn.CrossEntropyLoss(sent_decoder_stop,stop_target_variable[:,i])
        
        
            


            
            
            



        

    
    def update(self,train_dataset,val_dataset,config):
        device = torch.device(config.device)
        train_loader = DataLoader(dataset=train_dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.nw)
        
        if config.device == 'cuda':
            self.mlc.to(device)
            self.sent_decoder.to(device)
            self.word_decoder.to(device)

        for epoch in range(config.epoches):
            
            time_start_epoch = time.time()
            sum_loss = 0.0

            for step,sample in enumerate(train_loader,0):
                

                
                batch_images,batch_caption,batch_stop,batch_tags,batch_tags_vector = sample

                if config.device == 'cuda':
                    batch_images.to(device)
                    batch_caption.to(device)
                    batch_tags.to(device)
                    batch_stop.to(device)
                    batch_tags_vector.to(device)

                
                
                    
                ## batch * num_classes batch * channel * h * w
                preds,features = self.mlc(batch_images)
                ## transfer features to spatial batch * (h*w) * channel
                features = features.view(batch_size,features.size()[1],-1)
                features = features.transpose(1,2)

                sent_decoder_hidden = self.sent_decoder.hidden
                sent_semantic = None


                

                ## max_sent_num * batch * topic_embd
                sent_topics = []
                sent_stops = []
                ## max_sent_num * batch * (1 or 2)
                sent_attn_weights = []

               

                for i in range(config.max_sent_num):
                    sent_decoder_topic, sent_decoder_stop, sent_decoder_hidden, attn_weights= self.sent_decoder(features,sent_semantic,sent_decoder_hidden)
                    sent_topics.append(sent_decoder_topic)
                    sent_stop.append(sent_decoder_stop)
                    sent_attn_weights.append(attn_weights)
                
                batch_size = batch_images.size()[0]

                _cap_loss = 0.0 
                _stop_loss = 0.0
                _mlc_loss = 0.0 
                _attn_loss = 0.0
                cross_entropy_criterion = nn.CrossEntropyLoss()
                bce_loss_criterion = nn.BCELoss()

                for i in range(batch_size):
                    
                    ## max_sent_num * topic_embd
                    topics = sent_decoder_topic[:,i,:]

                    pred_stops = sent_stops
                    ## sent_num * max_len
                    caption = batch_caption[i]
                    sent_num = caption.size()[0]
                    for sent_i in range(sent_num):
                        sent_target_variable = caption[sent_i]
                        _sent_len = (sent_target_variable == EOS_INDEX).nonzero().data[0][0] + 1
                        word_decoder_hidden = topics[sent_i]
                        word_decoder_input = torch.autograd.Variable(torch.LongTensor([[SOS_INDEX, ]]))
                        if config.device == 'cuda':
                            word_decoder_input.to(device)

                        for word_i in range(_sent_len):
                            word_decoder_output, word_decoder_hidden = self.word_decoder(word_decoder_input, word_decoder_hidden)
                            word_decoder_input = sent_target_variable[word_i]
                            _cap_loss += criterion(word_decoder_output[0], sent_target_variable[word_i])




                    

               
        


        
            



        


