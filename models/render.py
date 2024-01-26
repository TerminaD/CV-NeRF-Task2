from models.nerf import NeRF
from utils.positional_encoding import PositionalEncoding

from typing import Tuple

from einops import rearrange, repeat, reduce
import torch


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: (N_rays, N_importance) the sampled samples
        
    This function is ported from kwea123/nerf_pl 
    (https://github.com/kwea123/nerf_pl/tree/master)
    under the MIT license.
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0,
                         # in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(rays: torch.Tensor,
                sample_num_coarse: int,
                sample_num_fine: int,
                nerf_coarse: NeRF,
                nerf_fine: NeRF,
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
    
    assert nerf_coarse.in_channels_xyz == nerf_fine.in_channels_xyz
    assert nerf_coarse.in_channels_dir == nerf_fine.in_channels_dir
    
    rays.to(device)
    nerf_coarse.to(device)
    nerf_fine.to(device)
    
    # Get coarse depths
    ray_num = rays.shape[0]
    rays_o = rays[:, :3].to(device)
    rays_d = rays[:, 3:6].to(device)
    near = rays[0][6]
    far = rays[0][7]
    
    bin_size_coarse = (far - near) / sample_num_coarse
    bin_edges_coarse = torch.linspace(near, far-bin_size_coarse, sample_num_coarse).to(device)
    bin_edges_coarse = bin_edges_coarse.repeat(ray_num, 1)
    rands_coarse = torch.rand(ray_num, sample_num_coarse).to(device)
    
    depths_coarse = rands_coarse*bin_size_coarse + bin_edges_coarse
    
    # Sample coarsely along ray
    xyzs_coarse = rays_o + rays_d * rearrange(depths_coarse, 'n1 n2 -> n2 n1 1').to(device)
    xyzs_coarse = rearrange(xyzs_coarse, 'sample ray xyz -> ray sample xyz') # Shape: ray_num * sample_num_coarse * 3
    xyzs_coarse = rearrange(xyzs_coarse, 'ray sample xyz -> (ray sample) xyz') # Assume first axis is ray
    
    deltas_coarse = torch.diff(depths_coarse, dim=1)
    deltas_coarse = torch.cat((deltas_coarse, 1e7 * torch.ones((ray_num, 1)).to(device)), dim=1)
    
    # Encode xyz and direction
    xyz_L = int(nerf_coarse.in_channels_xyz / 6)
    dir_L = int(nerf_coarse.in_channels_dir / 6)
    
    xyz_encoder = PositionalEncoding(xyz_L)
    xyz_encoded_coarse = xyz_encoder(xyzs_coarse)	# (ray_num * sample_num_coarse) * (6 * xyz_L)
    
    dir_encoder = PositionalEncoding(dir_L)
    dir_encoded = dir_encoder(rays_d) # ray_num * (6 * dir_L)
    dir_encoded_coarse = torch.repeat_interleave(dir_encoded, sample_num_coarse, dim=0) # (ray_num * sample_num_coarse) * (6 * dir_L)
    
    xyz_dir_encoded_coarse = torch.cat((xyz_encoded_coarse, dir_encoded_coarse), dim=1)
    
    # Feed into NeRF
    result_coarse = nerf_coarse(xyz_dir_encoded_coarse)
    
    # Unpack results
    rgbs_coarse = result_coarse[:, :3]
    rgbs_coarse = rearrange(rgbs_coarse, '(ray sample) rgb -> ray sample rgb', 
                            ray=ray_num, sample=sample_num_coarse)
    sigmas_coarse = result_coarse[:, 3:4]
    sigmas_coarse = rearrange(sigmas_coarse, '(ray sample) 1 -> ray sample', 
                              ray=ray_num, sample=sample_num_coarse)

    # Do part of neural rendering to sample again
    exps_coarse = torch.exp(-sigmas_coarse*deltas_coarse)
    
    Ts_coarse = torch.cumprod(torch.cat((torch.ones(ray_num, 1).to(device), exps_coarse), dim=1), dim=1)[:, :-1]
    
    weights_coarse = Ts_coarse * (1 - exps_coarse)
    
    depths_mid_coarse = 0.5 * (depths_coarse[: ,:-1] + depths_coarse[: ,1:])
    
    depths_fine = sample_pdf(depths_mid_coarse, 
                             weights_coarse[:, 1:-1].detach(),
                             sample_num_fine,
                             det=False)
    
    xyzs_fine = rays_o + rays_d * rearrange(depths_fine, 'n1 n2 -> n2 n1 1').to(device)
    xyzs_fine = rearrange(xyzs_fine, 'sample ray xyz -> ray sample xyz') # Shape: ray_num * sample_num_fine * 3
    xyzs_fine = rearrange(xyzs_fine, 'ray sample xyz -> (ray sample) xyz') # Assume first axis is ray
    
    # Encode fine xyz & dir, prepare for NeRF again
    xyzs_encoded_fine = xyz_encoder(xyzs_fine)
    
    dir_encoded_fine = torch.repeat_interleave(dir_encoded, sample_num_fine, dim=0) # (ray_num * sample_num_fine) * (6 * dir_L)
    
    xyz_dir_encoded_fine = torch.cat((xyzs_encoded_fine, dir_encoded_fine), dim=1)
    
    result_fine = nerf_fine(xyz_dir_encoded_fine)
    
    # Unpack fine results
    rgbs_fine = result_fine[:, :3]
    rgbs_fine = rearrange(rgbs_fine, '(ray sample) rgb -> ray sample rgb', 
                          ray=ray_num, sample=sample_num_fine)
    sigmas_fine = result_fine[:, 3:4]
    sigmas_fine = rearrange(sigmas_fine, '(ray sample) 1 -> ray sample', 
                            ray=ray_num, sample=sample_num_fine)
    
    # Concat coarse and fine results and sort
    depths_all = torch.cat((depths_coarse, depths_fine), dim=1)
    depths_all, indices = torch.sort(depths_all, dim=1)
    
    rgbs_all = torch.gather(torch.cat((rgbs_coarse, rgbs_fine), dim=1), 1, 
                            repeat(indices, 'a b -> a b 3'))
    
    sigmas_all = torch.gather(torch.cat((sigmas_coarse, sigmas_fine), dim=1), 1, indices)
    
    # Re-do neural rendering
    deltas_all = torch.diff(depths_all, dim=1)
    deltas_all = torch.cat((deltas_all, 1e7 * torch.ones((ray_num, 1)).to(device)), dim=1)
    
    exps_all = torch.exp(-sigmas_all*deltas_all)
    
    Ts_all = torch.cumprod(torch.cat((torch.ones(ray_num, 1).to(device), exps_all), dim=1), dim=1)[:, :-1]
    
    point_rgb = repeat(Ts_all * (1 - exps_all), 'ray sample -> ray sample 3') * rgbs_all
    pixel_rgb = torch.sum(point_rgb, dim=1)
    
    return pixel_rgb
    
    
@torch.no_grad
def render_image(rays: torch.Tensor,
                 batch_size: int,
                 img_shape: Tuple[int, int],
                 sample_num_coarse: int,
                 sample_num_fine: int,
                 nerf_coarse: NeRF,
                 nerf_fine: NeRF,
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
    nerf_coarse.to(device)
    nerf_fine.to(device)
    
    batches = torch.split(rays, batch_size)
    
    rgb_batches = []
    for ray_batch in batches:
        rgb_batch = render_rays(ray_batch, 
                                sample_num_coarse, 
                                sample_num_fine,
                                nerf_coarse,
                                nerf_fine,
                                device)
        rgb_batches.append(rgb_batch)
    last_rgb_batch = rgb_batches.pop()
    rgb_batches = torch.cat(rgb_batches, dim=0)
    rgb_batches = torch.cat((rgb_batches, last_rgb_batch), dim=0)
    
    rgb_batches = rgb_batches.reshape(img_shape[0], img_shape[1], 3)
    
    return rgb_batches
