from models.nerf import NeRF
from utils.positional_encoding import PositionalEncoding

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

def render_rays(rays: torch.Tensor,
                sample_num: int,
                nerf: NeRF,
                xyz_L: int,
                dir_L: int,
                device) -> torch.Tensor:
    """
    Render a number of rays.
    
    Inputs:
		rays: shape [num, 8], rays_o, rays_d, near bound & far bound concatenated
		sampling_num: number of points to sample for each ray
		nerf: a NeRF neural network
		xyz_L: the L in xyz's positional encoding
		dir_L: the L in direction's positional encoding
  
	Output:
		results: shape of [num, 4], RGB & sigma for each ray
    """
    
    assert rays.dim() == 2
    assert rays.shape[1] == 8
    
    # Sample uniformly randomly from uniform bins
    ray_num = rays.shape[0]
    rays_o = rays[:, :3]
    rays_d = rays[:, 3:6]
    near = rays[0][6]
    far = rays[0][7]
    
    bin_size = (far - near) / sample_num
    bin_edges = torch.linspace(near, far-bin_size, sample_num)
    bin_edges = bin_edges.repeat(ray_num, 1)
    rands = torch.rand(ray_num, sample_num)
    scaled_rands = rands*bin_size + bin_edges
    xyzs = rays_o + rays_d * rearrange(scaled_rands, 'n1 n2 -> n2 n1 1')
    xyzs = rearrange(xyzs, 'sample ray xyz -> ray sample xyz') # Shape: ray_num * sample_num * 3
    xyzs = rearrange(xyzs, 'ray sample xyz -> (ray sample) xyz') # Assume first axis is ray
    
    # Encode xyz and direction
    xyz_encoder = PositionalEncoding(xyz_L)
    xyz_encoded = xyz_encoder(xyzs)	# (ray_num * sample_num) * (6 * xyz_L)
    assert xyz_encoded.shape == torch.Size([ray_num * sample_num, 6 * xyz_L])
    
    dir_encoder = PositionalEncoding(dir_L)
    dir_encoded = dir_encoder(rays_d) # ray_num * (6 * dir_L)
    dir_encoded = torch.repeat_interleave(dir_encoded, torch.tensor([sample_num, 1])) # (ray_num * sample_num) * (6 * dir_L)
    assert dir_encoded.shape == torch.Size([ray_num * sample_num, 6 * dir_L])
    
    xyz_dir_encoded = torch.concat((xyz_encoded, dir_encoded), dim=1)
    
    result = nerf(xyz_dir_encoded)
    
    rgbs = result[:, :3]
    sigmas = result[]
    
    
    

def render_image():
    pass