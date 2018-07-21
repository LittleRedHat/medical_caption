import torch
import torch.nn as nn
from torchvision.transforms import transforms
import argparse
import time
import os
import csv
import sys
sys.path.append('..')
from dataset.IU_Chest_XRay_MLC import IUChestXRayMLCDataset
from models.CoAttn import MLCEncoder


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
    parser.add_argument('--save_dir', type=str, help="model save dir", default='iu_chest_mlc')
    parser.add_argument('--seed', type=int, help="seed", default=12345)
    parser.add_argument('--train_findings',type=str,help='train datas')
    parser.add_argument('--val_findings',type=str,help='train datas')
    parser.add_argument('--tags',type=str,help='tags file')
    parser.add_argument('--image_root',type=str,help='image root')
    parser.add_argument('--start_from',type=int,help='image root',default=-1)


    args = parser.parse_args()
    args.save_dir = os.path.join('../output/models/chest_mlc',args.save_dir)
    return args

def write_dict(s, file_path):
    w = csv.writer(open(file_path, "w"))
    for key, val in s.items():
        w.writerow([key, val])

image_size = (224,224)

def main():
    start_time = time.time()
    args = args_parser()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    write_dict(vars(args), os.path.join(args.save_dir, 'arguments.csv'))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_transformer = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.RandomCrop(size=image_size),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = IUChestXRayMLCDataset(args.train_findings,args.tags,args.image_root,train_transformer)
    
    val_transformer = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = IUChestXRayMLCDataset(args.val_findings,args.tags,args.image_root,val_transformer)
    model = MLCEncoder(train_dataset.get_tags_size())
    model.update(train_dataset,val_dataset,args)



if __name__ == '__main__':
    main()