from models.nerf import NeRF
from models.render import render_image
from utils.dataset import BlenderDataset
from utils.psnr import psnr_func

import os
import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

        
@torch.no_grad   
def test_all_depths() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='data/lego_small',
                        help='Path to collection of images to test NeRF on. Should follow COLMAP format.')
    parser.add_argument('-c', '--ckpt', type=str, required=True,
                        help='Name of checkpoint to load.')
    # parser.add_argument('-c', '--ckpt', type=str, default='cf',
    #                     help='Name of checkpoint to load.')
    parser.add_argument('-b', '--batch_size', type=int, default=4096)
    parser.add_argument('--xyz_L', type=int, default=10, 
                        help='Parameter L in positional encoding for xyz.')
    parser.add_argument('--dir_L', type=int, default=4, 
                        help='Parameter L in positional encoding for direction.')
    parser.add_argument('--sample_num_coarse', type=int, default=64, 
                        help='How many points to sample on each ray for coarse model.')
    parser.add_argument('--sample_num_fine', type=int, default=128, 
                        help='How many points to sample on each ray for fine model.')
    parser.add_argument('-l', '--length', type=int, default=200,
                        help='Length of images. Currently only support square images.')
    parser.add_argument('-t', '--threshold', type=float, default=0.8,
                        help='Cut-off value for T.')
    args = parser.parse_args()
        
    if torch.cuda.is_available():
        device = 'cuda:0'
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device)
    print(f"Device is {device}")
    
    testset = BlenderDataset(root_dir=args.data, 
                             split='test', 
                             img_wh=(args.length, args.length))
    
    model_coarse = NeRF(in_channels_xyz=6*args.xyz_L, in_channels_dir=6*args.dir_L)
    model_fine = NeRF(in_channels_xyz=6*args.xyz_L, in_channels_dir=6*args.dir_L)
    model_coarse.load_state_dict(torch.load(f'checkpoints/{args.ckpt}/coarse/final.pth', map_location=device))
    model_fine.load_state_dict(torch.load(f'checkpoints/{args.ckpt}/fine/final.pth', map_location=device))
    
    criterion = nn.MSELoss()
    
    c1s_losses = []
    c2s_losses = []
    a1s_losses = []
    a2s_losses = []
    
    c1s_psnrs = []
    c2s_psnrs = []
    a1s_psnrs = []
    a2s_psnrs = []
    
    os.makedirs(f'renders/{args.ckpt}/test_depth/c1s', exist_ok=True)
    os.makedirs(f'renders/{args.ckpt}/test_depth/c2s', exist_ok=True)
    os.makedirs(f'renders/{args.ckpt}/test_depth/a1s', exist_ok=True)
    os.makedirs(f'renders/{args.ckpt}/test_depth/a2s', exist_ok=True)
    
    for i in tqdm(range(len(testset))):
        sample = testset[i]
        rays = sample['rays'].to(device)
        gt_depth = torch.reshape(sample['depths'], (args.length, args.length)).to(device)
        
        c1s, c2s, a1s, a2s = render_image(rays=rays,
                                  batch_size=args.batch_size,
                                  img_shape=(args.length, args.length),
                                  sample_num_coarse=args.sample_num_coarse,
                                  sample_num_fine=args.sample_num_fine,
                                  nerf_coarse=model_coarse,
                                  nerf_fine=model_fine,
                                  threshold=args.threshold,
                                  depth_only=True,
                                  device=device)
        
        scale = -0.375
        bias = -6 * scale
        
        c1s = scale * c1s + bias
        c2s = scale * c2s + bias
        a1s = scale * a1s + bias
        a2s = scale * a2s + bias
        
        loss = criterion(gt_depth, c1s)
        psnr = psnr_func(gt_depth, c1s)
        c1s_losses.append(loss)
        c1s_psnrs.append(psnr)
        plt.imsave(f'renders/{args.ckpt}/test_depth/c1s/{i}.png', torch.clip(c1s, 0, 1).cpu().numpy())
        
        loss = criterion(gt_depth, c2s)
        psnr = psnr_func(gt_depth, c2s)
        c2s_losses.append(loss)
        c2s_psnrs.append(psnr)
        plt.imsave(f'renders/{args.ckpt}/test_depth/c2s/{i}.png', torch.clip(c2s, 0, 1).cpu().numpy())
        
        loss = criterion(gt_depth, a1s)
        psnr = psnr_func(gt_depth, a1s)
        a1s_losses.append(loss)
        a1s_psnrs.append(psnr)
        plt.imsave(f'renders/{args.ckpt}/test_depth/a1s/{i}.png', torch.clip(a1s, 0, 1).cpu().numpy())
        
        loss = criterion(gt_depth, a2s)
        psnr = psnr_func(gt_depth, a2s)
        a2s_losses.append(loss)
        a2s_psnrs.append(psnr)
        plt.imsave(f'renders/{args.ckpt}/test_depth/a2s/{i}.png', torch.clip(a2s, 0, 1).cpu().numpy())
        
    average_loss = sum(c1s_losses) / len(c1s_losses)
    average_psnr = sum(c1s_psnrs) / len (c1s_psnrs)
    with open(f'renders/{args.ckpt}/test/c1s/results.txt', 'w') as f:
        f.write(f'Average MSE Loss: {average_loss}\n')
        for i, loss in enumerate(c1s_losses):
            f.write(f'MSE Loss for Image {i}: {loss}\n')
        f.write('\n')
        f.write(f'Average PSNR: {average_psnr}\n')
        for i, psnr in enumerate(c1s_psnrs):
            f.write(f'PSNR for Image {i}: {psnr}\n')
            
    average_loss = sum(c2s_losses) / len(c2s_losses)
    average_psnr = sum(c2s_psnrs) / len (c2s_psnrs)
    with open(f'renders/{args.ckpt}/test/c2s/results.txt', 'w') as f:
        f.write(f'Average MSE Loss: {average_loss}\n')
        for i, loss in enumerate(c2s_losses):
            f.write(f'MSE Loss for Image {i}: {loss}\n')
        f.write('\n')
        f.write(f'Average PSNR: {average_psnr}\n')
        for i, psnr in enumerate(c2s_psnrs):
            f.write(f'PSNR for Image {i}: {psnr}\n')
            
    average_loss = sum(a1s_losses) / len(a1s_losses)
    average_psnr = sum(a1s_psnrs) / len (a1s_psnrs)
    with open(f'renders/{args.ckpt}/test/a1s/results.txt', 'w') as f:
        f.write(f'Average MSE Loss: {average_loss}\n')
        for i, loss in enumerate(a1s_losses):
            f.write(f'MSE Loss for Image {i}: {loss}\n')
        f.write('\n')
        f.write(f'Average PSNR: {average_psnr}\n')
        for i, psnr in enumerate(a1s_psnrs):
            f.write(f'PSNR for Image {i}: {psnr}\n')
            
    average_loss = sum(a2s_losses) / len(a2s_losses)
    average_psnr = sum(a2s_psnrs) / len (a2s_psnrs)
    with open(f'renders/{args.ckpt}/test/a2s/results.txt', 'w') as f:
        f.write(f'Average MSE Loss: {average_loss}\n')
        for i, loss in enumerate(a2s_losses):
            f.write(f'MSE Loss for Image {i}: {loss}\n')
        f.write('\n')
        f.write(f'Average PSNR: {average_psnr}\n')
        for i, psnr in enumerate(a2s_psnrs):
            f.write(f'PSNR for Image {i}: {psnr}\n')
        

if __name__ == '__main__':
    test_all_depths()
