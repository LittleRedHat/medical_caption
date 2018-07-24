import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import transforms
from torch.backends import cudnn
import numpy as np
import argparse
import time
import os
import csv
import random
import sys
sys.path.append('..')
import misc.utils as utils
from dataset.ChestXRay14_MLC import ChestXRay14Dataset
from models.ImageEncoder import AttnMLC as mlc_model


def args_parser():
    parser = argparse.ArgumentParser("IU Chest XRay MLC Pretrain")
    parser.add_argument('--epoches', type=int, help="epoch", default=100)
    parser.add_argument('--device', type=str, help="device", default='cuda')
    parser.add_argument('--log', type=int, help="log iter", default=10)
    parser.add_argument('--save_frq', type=int, help="model save frequency", default=1)
    parser.add_argument('--eval_frq', type=int, help="model eval frequency", default=1)
    parser.add_argument('--nw',type=int,help="number of workers",default=4)
    parser.add_argument('--lr', type=float, help="learning_rate", default=1e-4)
    parser.add_argument('--batch_size', type=int, help="batch_size", default=64)
    parser.add_argument('--gamma', type=float, help="learning rate gamma", default=0.95)
    parser.add_argument('--decay', type=float, help="learning_rate", default=1e-5)
    parser.add_argument('--save_dir', type=str, help="model save dir", default='iu_chest14_mlc')
    parser.add_argument('--seed', type=int, help="seed", default=-1)
    parser.add_argument('--train_file',type=str,help='train datas')
    parser.add_argument('--val_file',type=str,help='val datas')
    parser.add_argument('--image_dir',type=str,help='image root')
    parser.add_argument('--start_from',type=int,help='image root',default=-1)
    parser.add_argument('--device_ids',type=str,help='gpu device ids',default='0,1,2,3')
    parser.add_argument('--backbone',type=str,default='resnet50')

    args = parser.parse_args()
    args.save_dir = os.path.join('../output/models/mlc',args.save_dir)
    if args.seed == -1:
        args.seed = random.randint(-2^30,2^30)
    # args.device_ids = args.device_ids.split(',').map(lambda x:int(x))
    return args

def write_dict(s, file_path):
    w = csv.writer(open(file_path, "w"))
    for key, val in s.items():
        w.writerow([key, val])

image_size = (224,224)

def random_print_preds(prob,target,threhold=0.5):
    prob = prob.data.cpu().numpy()
    r = np.random.randint(0,len(prob))
    topk = np.argsort(prob[r])[::-1][:10]
    topk = topk[topk > threhold]

    truth = target[r].nonzero().data.cpu().numpy()
    print('***************')
    print(topk)
    print(truth.squeeze())


def normalize(image):
    for channel in range(image.size(0)):
        mean = image[channel,:,:].mean()
        std = image[channel,:,:].std()
        image[channel,:,:] = (image[channel,:,:] - mean) / std
    return image

def predict(model,val_dataloader,batch_size=32):
    model.eval()
    
    targets = []
    preds = []
    total_loss = 0.0

    forward_start_time = time.time()

    with torch.no_grad():
        for sample in val_dataloader:
            image,tags = sample
            if torch.cuda.is_available():
                image = image.cuda()
                tags = tags.cuda()

            # logits,_ = model.forward(image)
            # probs = F.sigmoid(logits)
            probs,_ = model.forward(image)

            # random_print_preds(probs,tags)

            _loss = utils.focal_loss(probs,tags,val_dataloader.dataset.get_positive_ratio().cuda())
            total_loss += _loss.cpu().item()
            if len(targets):
                preds = np.concatenate((preds,probs.data.cpu().numpy()),0)
                targets = np.concatenate((targets,tags.data.cpu().numpy()),0)
            else:
                targets = tags.data.cpu().numpy()
                preds = probs.data.cpu().numpy()

    forward_stop_time = time.time()
    print('forward val dataset use time {:.4f}'.format((forward_stop_time - forward_start_time) / 60.0))

    auc_start_time = time.time()
    auc = utils.compute_auc(preds,targets)
    print('cal val dataset auc use time {:.4f}'.format((time.time() - auc_start_time) / 60.0)) 
    loss = total_loss / len(val_dataloader)

    return loss,auc
    
    

def train(model,train_dataset,val_dataset,config):
    # device = torch.device(config.device)
    
    if config.start_from != -1:
        model.load_state_dict(torch.load(os.path.join(config.save_dir, 'model_params_{}.pkl'.format(config.start_from))))
    # if torch.cuda.is_available():
    #     model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
    elif torch.cuda.is_available():
        model = model.cuda()

    num_classes = train_dataset.get_tag_size()
    valid_file = open(os.path.join(config.save_dir, 'valid_result.csv'), 'w')
    train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.nw)
    val_dataloader = DataLoader(val_dataset,shuffle=False,batch_size=config.batch_size,num_workers=config.nw)

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # optimizer = torch.optim.Adam(params=parameters,lr=config.lr)
    optimizer = torch.optim.SGD(parameters,lr=config.lr,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)


    for epoch in range(1,config.epoches + 1):
        scheduler.step()
        model.train()
        time_start_spoch = time.time()
        sum_loss = 0.0
        for step,sample in enumerate(train_dataloader,1):
            optimizer.zero_grad()
            image,tags = sample
            if torch.cuda.is_available():
                image = image.cuda()
                tags = tags.cuda()

            preds,_ = model.forward(image)
            # preds = F.sigmoid(logits)
            loss = utils.focal_loss(preds,tags,train_dataset.get_positive_ratio().cuda())
            _loss = loss.data.numpy() if config.device == 'cpu' else loss.data.cpu().numpy()
            sum_loss += _loss
            loss.backward()
            optimizer.step()

            if step % config.log == 0:
                info = 'epoch {} batch step {}/{}: loss = {:.5f} {:.4f}mins'
                print(info.format(epoch,step, len(train_dataloader),sum_loss / step,(time.time() - time_start_spoch) / 60.))
        if epoch % config.eval_frq == 0:
            loss,auc = predict(model,val_dataloader,config.batch_size)

            avg_auc = np.mean(auc)

            info = 'epoch {}, loss_test = {:.6f} auc = {} time/epoch={:.1f}mins'
            valid_file.write(info.format(epoch,loss,avg_auc,(time.time() - time_start_spoch) / 60.) + '\n')
            valid_file.flush()
            print(info.format(epoch, loss, avg_auc,(time.time() - time_start_spoch) / 60.) + '\n')
            
        if epoch % config.save_frq == 0:
            torch.save(model.state_dict(),os.path.join(config.save_dir, 'model_params_{}.pkl'.format(epoch)))
    valid_file.close()

def main():
    start_time = time.time()
    args = args_parser()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    write_dict(vars(args), os.path.join(args.save_dir, 'arguments.csv'))

    torch.manual_seed(args.seed)
    cudnn.benchmark = True 
    torch.cuda.manual_seed_all(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    

    train_transformer = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.RandomCrop(size=image_size),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(normalize)

    ])
    train_dataset = ChestXRay14Dataset(args.image_dir,args.train_file,train_transformer)
    
    val_transformer = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(normalize)

    ])

    val_dataset = ChestXRay14Dataset(args.image_dir,args.val_file,val_transformer)
    print(vars(args))

    model = mlc_model(train_dataset.get_tag_size(),backbone=args.backbone)

    train(model,train_dataset,val_dataset,args)
        



if __name__ == '__main__':
    main()