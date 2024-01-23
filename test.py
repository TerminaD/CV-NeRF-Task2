import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.render import render_rays, render_image
from models.nerf import NeRF
from einops import rearrange
import argparse
from utils.dataset import BlenderDataset




def PSNR(pred_img,gt_img):
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    psnr = 20 * np.log10(1/np.sqrt(mse)) 
    return psnr 

def test_pic(index=0,dataset=BlenderDataset("data/lego/test","test")):
    sample=dataset[index]

    pred_img=render_img(sample[0])
    gt_img=sample[1]

    concat_img = np.concatenate((pred_img, gt_img), axis=1)
    plt.imsave(f"result/pred_img{index}.png",pred_img)
    plt.imsave(f"result/con_img{index}.png",con_img)
    plt.imshow(concat_img)
    plt.show()

    psnr=PSNR(test_img,gt_img)
    print(psnr)
    return psnr

def test_pics():
    dataset=BlenderDataset("data/lego/test","test")
    for i,data in enum(dataset):
        test_pic(i,data)
          
@torch.no_grad   
def test_all() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='data/lego',
                        help='Path to collection of images to test NeRF on. Should follow COLMAP format.')
    parser.add_argument('-c', '--ckpt', type=str, required=True,
                        help='Name of checkpoint to load.')
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('--xyz_L', type=int, default=10, 
                        help='Parameter L in positional encoding for xyz.')
    parser.add_argument('--dir_L', type=int, default=4, 
                        help='Parameter L in positional encoding for direction.')
    parser.add_argument('-s', '--sample_num', type=int, default=75, 
                        help='How many points to sample on each ray.')
    args = parser.parse_args()
        
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device)
    print(f"Device is {device}")
    
    testset = BlenderDataset(root_dir=args.data, split='test')
    
    model = NeRF(in_channels_xyz=6*args.xyz_L, in_channels_dir=6*args.dir_L)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    
    
        

if __name__ == '__main__':
    test_all()
