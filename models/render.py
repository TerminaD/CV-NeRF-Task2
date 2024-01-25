from models.nerf import NeRF
from utils.positional_encoding import PositionalEncoding

from typing import Tuple

from einops import rearrange, repeat
import torch

def render_rays(rays: torch.Tensor,
                sample_num: int,
                nerf: NeRF,
                device) -> torch.Tensor:
    """
    Render a number of rays.
    
    Inputs:
        rays: shape [num, 8], rays_o, rays_d, near bound & far bound catenated
        sampling_num: number of points to sample for each ray
        nerf: a pre-trained NeRF neural network
        xyz_L: the L in xyz's positional encoding
        dir_L: the L in direction's positional encoding
  
    Output:
        results: shape of [num,], rendered color of each ray
    """
    
    rays.to(device)
    nerf.to(device)
    
    # Sample uniformly randomly from uniform bins
    ray_num = rays.shape[0]
    rays_o = rays[:, :3].to(device)
    rays_d = rays[:, 3:6].to(device)
    near = rays[0][6]
    far = rays[0][7]
    
    bin_size = (far - near) / sample_num
    bin_edges = torch.linspace(near, far-bin_size, sample_num).to(device)
    bin_edges = bin_edges.repeat(ray_num, 1)
    rands = torch.rand(ray_num, sample_num).to(device)
    depths = rands*bin_size + bin_edges
    xyzs = rays_o + rays_d * rearrange(depths, 'n1 n2 -> n2 n1 1').to(device)
    xyzs = rearrange(xyzs, 'sample ray xyz -> ray sample xyz') # Shape: ray_num * sample_num * 3
    xyzs = rearrange(xyzs, 'ray sample xyz -> (ray sample) xyz') # Assume first axis is ray
    
    # Encode xyz and direction
    xyz_L = int(nerf.in_channels_xyz / 6)
    dir_L = int(nerf.in_channels_dir / 6)
    
    xyz_encoder = PositionalEncoding(xyz_L)
    xyz_encoded = xyz_encoder(xyzs)	# (ray_num * sample_num) * (6 * xyz_L)
    
    dir_encoder = PositionalEncoding(dir_L)
    dir_encoded = dir_encoder(rays_d) # ray_num * (6 * dir_L)
    
    dir_encoded = torch.repeat_interleave(dir_encoded, sample_num, dim=0) # (ray_num * sample_num) * (6 * dir_L)
    
    xyz_dir_encoded = torch.cat((xyz_encoded, dir_encoded), dim=1)
    
    # Feed into NeRF
    result = nerf(xyz_dir_encoded)
    
    # Unpack results
    rgbs = result[:, :3]
    rgbs = rearrange(rgbs, '(ray sample) rgb -> ray sample rgb', 
                     ray=ray_num, sample=sample_num)
    sigmas = result[:, 3:4]
    sigmas = rearrange(sigmas, '(ray sample) 1 -> ray sample', 
                       ray=ray_num, sample=sample_num)
    clipped_sigmas = torch.relu(sigmas)     # Clip sigmas as it should be non-negative

    # Do neural rendering
    deltas = torch.diff(depths, dim=1)
    deltas = torch.cat((deltas, 1e7 * torch.ones((ray_num, 1)).to(device)), dim=1)
    
    exps = torch.exp(-clipped_sigmas*deltas)
    
    Ts = torch.cumprod(torch.cat((torch.ones(ray_num, 1).to(device), exps), dim=1), dim=1)[:, :-1]
    
    point_rgb = repeat(Ts * (1 - exps), 'ray sample -> ray sample 3') * rgbs
    pixel_rgb = torch.sum(point_rgb, dim=1)
    
    return pixel_rgb
    
@torch.no_grad
def render_image(rays: torch.Tensor,
                 batch_size: int,
                 img_shape: Tuple[int, int],
                 sample_num: int,
                 nerf: NeRF,
                 device) -> torch.Tensor:
    """
    Renders an image.
    This function should not be used for training purposes, as it does not
    calculate gradients.
    
    Inputs:
        rays: all rays of an image. Can directly use the `rays` key of a
              test-time dataloader.
        batch_size: how many rays to render in one go.
        img_shape: shape of the image.
        sample_num: how many points to sample on each ray.
        nerf: a pre-trained NeRF neural network. 
        device: device to run this function on.
        
    Output:
        The predicted RGB image. Shape: [img_shape[0], img_shape[1], 3]
    """
    
    rays.to(device)
    nerf.to(device)
    
    batches = torch.split(rays, batch_size)
    
    rgb_batches = []
    for ray_batch in batches:
        rgb_batch = render_rays(ray_batch, sample_num, nerf, device)
        rgb_batches.append(rgb_batch)
    last_rgb_batch = rgb_batches.pop()
    rgb_batches = torch.cat(rgb_batches, dim=0)
    rgb_batches = torch.cat((rgb_batches, last_rgb_batch), dim=0)
    
    rgb_batches = rgb_batches.reshape(img_shape[0], img_shape[1], 3)
    
    return rgb_batches
