from models.nerf import NeRF
from models.render import render_rays, render_image
from utils.dataset import BlenderDataset
from utils.psnr import psnr_func

import argparse
import datetime

from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='data/lego',
                        help='Path to collection of images to fit NeRF on. Should follow COLMAP format.')
    parser.add_argument('-c', '--ckpt', type=str, 
                        help='Name of checkpoint to save to. Defaults to timestamp.')
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('--xyz_L', type=int, default=10, 
                        help='Parameter L in positional encoding for xyz.')
    parser.add_argument('--dir_L', type=int, default=4, 
                        help='Parameter L in positional encoding for direction.')
    parser.add_argument('-s', '--sample_num', type=int, default=75, 
                        help='How many points to sample on each ray.')
    parser.add_argument('-t', '--test_every', type=int, default=10, 
                        help='Performs testing after we\'ve trained for this many epochs.')
    parser.add_argument('--test_in_training', action='store_true',
                        help='Perform testing during training')
    args = parser.parse_args()
    
    if not args.ckpt:
        now = datetime.datetime.now()
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
                             num_workers=8, 
                             batch_size=args.batch_size,
                             pin_memory=True)
    testset = BlenderDataset(root_dir=args.data, split='test')
    model = NeRF(in_channels_xyz=6*args.xyz_L, in_channels_dir=6*args.dir_L)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    for e in range(args.epoch):
        print(f"epoch:{e}")
        cum_loss = 0.0
        
        for sample in tqdm(trainloader, desc="Training", leave=False):
            rays = sample['rays']
            gt_rgbs = sample['rgbs']
            
            optimizer.zero_grad()
            
            pred_rgbs = render_rays(rays, args.sample_num, model, device)
            
            loss = criterion(gt_rgbs, pred_rgbs)
            loss.backward()
            cum_loss += loss
            
            optimizer.step()
        
        cum_loss /= len(trainloader)
        writer.add_scalar('Loss/train', cum_loss, e)
        print(cum_loss)
        
        # Perform testing periodically
        if args.test_in_training and e % args.test_every == 0:
            with torch.no_grad:
                sample = testset[0]
                pred_img = render_image(rays=sample['rays'],
                                        batch_size=args.batch_size,
                                        img_shape=(800, 800),
                                        sample_num=args.sample_num,
                                        nerf=model,
                                        device=device)
                gt_img = rearrange(sample['rgbs'], '(h w) 3 -> h w 3',
                                h=800, w=800)
                
                loss = criterion(gt_img, pred_img)
                psnr = psnr_func(gt_img, pred_img)
                
                writer.add_scalar('Loss/test', loss, e)
                writer.add_scalar('PSNR/test', psnr, e)
                
                torch.save(model.state_dict(), f"checkpoints/{args.ckpt}/{e}.pth")
                plt.imsave(f'renders/{args.ckpt}/train/{e}.png', pred_img)
    
    torch.save(model.state_dict(), f"checkpoints/{args.ckpt}/final.pth")           
                
    writer.flush()
    

if __name__ == '__main__':
    train()
    