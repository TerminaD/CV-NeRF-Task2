from models.nerf import NeRF
from models.render import render_rays, render_image
from utils.dataset import BlenderDataset
from utils.psnr import psnr_func
from utils.loss import NeRFMSELoss

import os
import argparse
import datetime

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='data/lego_small',
                        help='Path to collection of images to fit NeRF on. Should follow COLMAP format.')
    parser.add_argument('-c', '--ckpt', type=str, default='debug8',
                        help='Name of checkpoint to save to. Defaults to timestamp.')
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=16384)
    parser.add_argument('--xyz_L', type=int, default=10, 
                        help='Parameter L in positional encoding for xyz.')
    parser.add_argument('--dir_L', type=int, default=4, 
                        help='Parameter L in positional encoding for direction.')
    parser.add_argument('--sample_num_coarse', type=int, default=64, 
                        help='How many points to sample on each ray for coarse model.')
    parser.add_argument('--sample_num_fine', type=int, default=128, 
                        help='How many points to sample on each ray for fine model.')
    parser.add_argument('-t', '--test_every', type=int, default=5, 
                        help='Performs testing after we\'ve trained for this many epochs.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('-l', '--length', type=int, default=200,
                        help='Length of images. Currently only support square images.')
    args = parser.parse_args()
        
    return args


def train() -> None:
    args = parse_args()
    
    if not args.ckpt:
        now = datetime.datetime.now()
        args.ckpt = now.strftime("%m-%d-%H-%M-%S")
    
    writer = SummaryWriter()
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device)
    print(f"Device is {device}")
    
    trainset = BlenderDataset(root_dir=args.data, 
                              split='train', 
                              img_wh=(args.length, args.length))
    trainloader = DataLoader(trainset,
                             shuffle=True,
                             num_workers=8, 
                             batch_size=args.batch_size,
                             pin_memory=True)
    testset = BlenderDataset(root_dir=args.data, 
                             split='test', 
                             img_wh=(args.length, args.length))
    
    model_coarse = NeRF(in_channels_xyz=6*args.xyz_L, in_channels_dir=6*args.dir_L)
    model_fine = NeRF(in_channels_xyz=6*args.xyz_L, in_channels_dir=6*args.dir_L)
    model_coarse.to(device)
    model_fine.to(device)
    
    all_params = list(model_coarse.parameters()) + list(model_fine.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1.0 / 100))     # lr *= 0.1 in first 100 epoch, static after
    
    nerf_criterion = NeRFMSELoss()
    mse_criterion = nn.MSELoss()
    
    os.makedirs(f'checkpoints/{args.ckpt}/coarse', exist_ok=True)
    os.makedirs(f'checkpoints/{args.ckpt}/fine', exist_ok=True)
    os.makedirs(f'renders/{args.ckpt}/train', exist_ok=True)
    
    for e in range(args.epoch):
        print(f"epoch:{e}")
        cum_loss = 0.0
        
        for sample in tqdm(trainloader, desc="Training", leave=False):
            rays = sample['rays'].to(device)
            gt_rgbs = sample['rgbs'].to(device)
            
            optimizer.zero_grad()
            
            pred_rgbs_coarse, pred_rgbs_fine = render_rays(rays,
                                                           args.sample_num_coarse,
                                                           args.sample_num_fine,
                                                           model_coarse,
                                                           model_fine,
                                                           device=device)
            
            loss = nerf_criterion(gt_rgbs, pred_rgbs_coarse, pred_rgbs_fine)
            loss.backward()
            cum_loss += loss
            
            optimizer.step()
        
        cum_loss /= len(trainloader)
        writer.add_scalar('Loss/train', cum_loss, e)
        print(cum_loss.item())
        
        if e < 100:
            scheduler.step()
        
        # Perform testing periodically
        if e % args.test_every == 0:
            with torch.no_grad():
                print("Testing...")
                sample = testset[0]
                pred_img = render_image(rays=sample['rays'],
                                        batch_size=args.batch_size,
                                        img_shape=(args.length, args.length),
                                        sample_num_coarse=args.sample_num_coarse,
                                        sample_num_fine=args.sample_num_fine,
                                        nerf_coarse=model_coarse,
                                        nerf_fine=model_fine,
                                        device=device)
                gt_img = sample['rgbs'].reshape(args.length, args.length, 3).to(device)
                
                loss = mse_criterion(gt_img, pred_img)
                psnr = psnr_func(gt_img, pred_img)
                
                writer.add_scalar('MSE/test', loss, e)
                writer.add_scalar('PSNR/test', psnr, e)
                
                torch.save(model_coarse.state_dict(), f"checkpoints/{args.ckpt}/coarse/{e}.pth")
                torch.save(model_fine.state_dict(), f"checkpoints/{args.ckpt}/fine/{e}.pth")
                plt.imsave(f'renders/{args.ckpt}/train/{e}.png', torch.clip(pred_img, 0, 1).cpu().numpy())
    
    torch.save(model_coarse.state_dict(), f"checkpoints/{args.ckpt}/coarse/final.pth")
    torch.save(model_fine.state_dict(), f"checkpoints/{args.ckpt}/fine/final.pth") 
                
    writer.flush()
    

if __name__ == '__main__':
    train()
    