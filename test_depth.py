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
    # parser.add_argument('-c', '--ckpt', type=str, required=True,
    #                     help='Name of checkpoint to load.')
    parser.add_argument('-c', '--ckpt', type=str, default='cf',
                        help='Name of checkpoint to load.')
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
    parser.add_argument('-t', '--threshold', type=float, default=0.875,
                        help='Cut-off value for T.')
    args = parser.parse_args()
        
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
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
    
    losses = []
    psnrs = []
    
    os.makedirs(f'renders/{args.ckpt}/test_depth', exist_ok=True)
    
    for i in tqdm(range(len(testset))):
        sample = testset[i]
        rays = sample['rays'].to(device)
        gt_depth = torch.reshape(sample['depths'], (args.length, args.length)).to(device)
        
        pred_depth = render_image(rays=rays,
                                  batch_size=args.batch_size,
                                  img_shape=(args.length, args.length),
                                  sample_num_coarse=args.sample_num_coarse,
                                  sample_num_fine=args.sample_num_fine,
                                  nerf_coarse=model_coarse,
                                  nerf_fine=model_fine,
                                  threshold=args.threshold,
                                  depth_only=True,
                                  device=device)
        # TODO: flip and normalize
        
        plt.imshow(gt_depth)
        plt.show()
        
        plt.imshow(pred_depth)
        plt.show()
        
        loss = criterion(gt_depth, pred_depth)
        psnr = psnr_func(gt_depth, pred_depth)
        losses.append(loss)
        psnrs.append(psnr)
        
        plt.imsave(f'renders/{args.ckpt}/test/{i}.png', torch.clip(pred_depth, 0, 1).cpu().numpy())
        
    average_loss = sum(losses) / len(losses)
    average_psnr = sum(psnrs) / len (psnrs)
    
    with open(f'renders/{args.ckpt}/test/results.txt', 'w') as f:
        f.write(f'Average MSE Loss: {average_loss}\n')
        for i, loss in enumerate(losses):
            f.write(f'MSE Loss for Image {i}: {loss}\n')
        f.write('\n')
        f.write(f'Average PSNR: {average_psnr}\n')
        for i, psnr in enumerate(psnrs):
            f.write(f'PSNR for Image {i}: {psnr}\n')
        

if __name__ == '__main__':
    test_all_depths()
