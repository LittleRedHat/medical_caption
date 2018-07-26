import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import transforms
from torch.backends import cudnn
from tensorboardX import SummaryWriter

import numpy as np
import argparse
import time
import os
import random
import sys
import csv
sys.path.append('..')
from dataset.IU_Chest_XRay import ChestIUXRayDataset

from models.MultiLayerAttnCaption import AggregationCaptionModel
from models.ImageEncoder import MLC,VGGExtractor

class Dict2Class():
    def __init__(self,dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    cudnn.benchmark = True 
    torch.cuda.manual_seed_all(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

class Config(object):
    def __init__(self):
        self.sent_hidden_dim = 512
        self.word_hidden_dim = 512
        self.topic_dim = 512

        self.stop_loss_weight = 0.5
        self.cap_loss_weight = 0.5
       
        
        self.sent_num_layers = 1
        self.sent_bidirection = True
        
        self.word_num_layers = 1
        self.word_drop = 0.5
        self.word_bidirection = False

        self.att_feat_size = 512
        self.feat_size = (512,14,14)

        self.backbone = 'vgg19'
        self.tag_size = 14

        self.embd_dim = 256

def parse_args():
    parser = argparse.ArgumentParser("IU Chest XRay Report Generation")
    parser.add_argument('--epoches', type=int, help="epoch", default=50)
    parser.add_argument('--device', type=str, help="device", default='cuda')
    parser.add_argument('--log_frq', type=int, help="log iter", default=20)
    parser.add_argument('--save_frq', type=int, help="model save frequency", default=1)
    parser.add_argument('--eval_frq', type=int, help="model eval frequency", default=1)
    parser.add_argument('--nw',type=int,help="number of workers",default=4)
    parser.add_argument('--lr', type=float, help="learning_rate", default=1e-4)
    parser.add_argument('--batch_size', type=int, help="batch_size", default=4)
    parser.add_argument('--gamma', type=float, help="learning rate gamma", default=0.95)
    parser.add_argument('--decay', type=float, help="learning_rate", default=1e-5)
    parser.add_argument('--save_dir', type=str, help="model save dir", default='iu_chest')
    parser.add_argument('--seed', type=int, help="seed", default=-1)

    parser.add_argument('--train_finding_file',type=str,help='train datas')
    parser.add_argument('--val_finding_file',type=str,help='val datas')
    parser.add_argument('--words_file',type=str,help="")
    parser.add_argument('--tags_file',type=str,help='tags')

    parser.add_argument('--image_dir',type=str,help='medical image dir')
    parser.add_argument('--start_from',type=int,help='start from',default=-1)
    parser.add_argument('--device_ids',type=str,help='gpu device ids',default='0,1,2,3')

    parser.add_argument('--image_encoder_checkpoint',type=str,help='image feature extractor checkpoint')
    parser.add_argument('--backbone',type=str,help='backbone for image extractor')
    parser.add_argument('--tag_size',type=int,default=14)
    parser.add_argument('--weight_decay',type=float,default=0.0001)

    parser.add_argument('--pt', type=bool, help="seed", default=False)


    args = parser.parse_args()
    
    args.save_dir = os.path.join('../output/models/caption',args.save_dir)

    if args.seed == -1:
        args.seed = random.randint(-2^30,2^30)

    return args


def write_dict(s, file_path):
    w = csv.writer(open(file_path, "w"))
    for key, val in s.items():
        w.writerow([key, val])

def normalize(image):
    for channel in range(image.size(0)):
        mean = image[channel,:,:].mean()
        std = image[channel,:,:].std()
        image[channel,:,:] = (image[channel,:,:] - mean) / std
    return image

def display_args(args):
    print('*********************** configuration ***********************')
    for k,v in args.__dict__.items():
        print(k,':',v)
    print('*********************** configuration ***********************')

def merge_config(default,update):
    for k,v in default.__dict__.items():
        setattr(update,k,v)
        
    return update

## batch * max_sent * 2
def cal_stop_loss(self,pred_stops,target_stops):
    pred_stops = pred_stops.reshape(-1,pred_stops.size(2))
    target_stops = target_stops.reshape(-1)
    loss = nn.CrossEntropyLoss(size_average=False)(pred_stops,target_stops)
    return loss / batch_size


## batch * max_sent * max_words * (dict_size + 2)
def cal_language_loss(self,pred,target,sent_num,sent_length):
    batch_size = pred.size(0)
    loss = 0.0
    crition = nn.CrossEntropyLoss(size_average=False)
    for batch_i in range(batch_size):
        sent_num_i = sent_num[batch_i]
        valid_pred_i = pred[batch_i,:sent_num_i]
        valid_target_i = target[batch_i,:sent_num_i]
        
        ## sent_num_i 
        sent_length_i = sent_length[batch_i,:sent_num_i]

        for sent_i,single_sent_length in enumerate(sent_length_i):
            sent_pred = valid_pred_i[sent_i,:single_sent_length].unsequeeze(0)
            sent_target = valid_target_i[sent_i,:single_sent_length].unsequeeze(0)
            loss += crition(sent_pred,sent_target)
    return loss / batch_size

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def create_image_encoder(tag_size,backbone,checkpoint):
    mlc = MLC(tag_size,backbone)
    state_dict = torch.load(checkpoint)

    ## change multi-gpu to single-gpu
    state_dict = {k[7:]: v for k,v in state_dict.items()}

    mlc.load_state_dict(state_dict)
    extractor = mlc.model
    return extractor

image_size = (224,224)

def train(model,image_encoder,train_dataset,val_dataset,config):
    
    infos = {}
    histories = {}
    summary_writer = SummaryWriter(config.save_dir)
    
    if torch.cuda.is_available():
        model = model.cuda()
        image_encoder = image_encoder.cuda()
    
    valid_file = open(os.path.join(config.save_dir, 'valid_result.csv'), 'w')
    train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.nw)
    val_dataloader = DataLoader(val_dataset,shuffle=False,batch_size=config.batch_size,num_workers=config.nw)

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(parameters,lr=config.lr,momentum=0.9,nesterov=True,weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=2,threshold=0.01)

    for epoch in range(1,config.epoches + 1):
        # scheduler.step()
        model.train()
        time_start_spoch = time.time()
        sum_loss = 0.0
        for step,sample in enumerate(train_dataloader,1):
            optimizer.zero_grad()

            image,sent_idxs,sent_length,sent_num,stop,tags = sample

            if torch.cuda.is_available():
                image = image.cuda()
                sent_idxs = sent_idxs.cuda()
                sent_length = sent_length.cuda()
                sent_num = sent_num.cuda()

            _,image_feature = image_encoder(image,layers=[51])
            ## batch * c * w * h
            image_feature = image_feature[0]

            image_feature = torch.tensor(image_feature,requires_grad=False)

            outputs,sent_stop,sent_topics = model(image_feature,sent_idxs,sent_num,sent_length)
            stop_loss = cal_stop_loss(sent_stop,stop)

            caption_loss = cal_language_loss(outputs,sent_idxs,sent_num,sent_length)

            final_loss = config.stop_loss_weight * stop_loss + config.cap_loss_weight * caption_loss

            final_loss.backward()
            optimizer.step()

            _loss = final_loss.data.numpy() if config.device == 'cpu' else final_loss.data.cpu().numpy()
            sum_loss += _loss

            if step % config.log_frq == 0:
                info = 'epoch {} batch step {}/{}: loss = {:.5f} {:.4f}mins'
                print(info.format(epoch,step, len(train_dataloader),sum_loss / step,(time.time() - time_start_spoch) / 60.))
            




            


def predict():
    pass

def main():
    start_time = time.time()
    args = parse_args()
    default_config = Config()

    args = merge_config(default_config,args)
    

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    write_dict(vars(args), os.path.join(args.save_dir, 'arguments.csv'))

    torch.manual_seed(args.seed)
    cudnn.benchmark = True 
    torch.cuda.manual_seed_all(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    train_transformer = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
        transforms.Lambda(normalize)
    ])

    val_transformer = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
        transforms.Lambda(normalize)

    ])

    


    train_dataset = ChestIUXRayDataset(args.train_finding_file,args.words_file,args.tags_file,args.image_dir,transformer=train_transformer)

    val_dataset = ChestIUXRayDataset(args.val_finding_file,args.words_file,args.tags_file,args.image_dir,transformer=val_transformer)


    setattr(args,'dict_size',train_dataset.get_words_size())
    setattr(args,'max_words',train_dataset.get_config()['MAX_WORDS'])
    setattr(args,'max_sent',train_dataset.get_config()['MAX_SENT'])
    setattr(args,'init_embed',train_dataset.get_word_embed())

    display_args(args)

    image_encoder = create_image_encoder(args.tag_size,args.backbone,args.image_encoder_checkpoint)
    model = AggregationCaptionModel(args)
    train(model,image_encoder,train_dataset,val_dataset,args)




    

    
    


if __name__ == '__main__':
    main()
