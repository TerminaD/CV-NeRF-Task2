from models.nerf import NeRF
from models.render import render_ray
from utils.positional_encoding import PositionalEncoding
from utils.dataset import BlenderDataset

import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='data/lego'
                        help='Path to collection of images to fit NeRF on. Should follow COLMAP format.')
    parser.add_argument('-c', '--ckpt', type=str, 
                        help='Name of checkpoint to save to. Defaults to timestamp.')
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-l', type=int, default=15, 
                        help='Parameter L in positional encoding.')
    args = parser.parse_args()
    
    if not args.ckpt:
        now = datetime.now()
        args.ckpt = now.strftime("%m-%d-%H-%M-%S")
    
    writer = SummaryWriter()
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device)
    print(f"Device is {device}")
    
    trainset = BlenderDataset(root_dir=args.data, split='train')
    trainloader = DataLoader(trainset,
                             shuffle=True,
                             num_workers=4, 
                             batch_size=args.batch_size,
                             pin_memory=True)
    
    model = NeRF()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    for e in range(epoch):
        print(f"Epoch {e}")
        for data in dataloader:
            coords, gt_color = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            pred_color = model(coords)
            loss = criterion(gt_color, pred_color)
            loss.backward()
            optimizer.step()
        if e % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/{ckpt_name}-e{e}.pth")
        psnr_v = test(model, device, gt_img, ckpt_name, training_epoch=e)
        writer.add_scalar('training/psnr', psnr_v, e)
    
    
    
    
    
    
    writer.flush()
    
    
    
    
    

if __name__ == '__main__':
    train()
    