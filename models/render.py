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
                device):
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
    xyz_encoded_coarse = xyz_encoder(xyzs_coarse/3)	# (ray_num * sample_num_coarse) * (6 * xyz_L), divided by 3 here for positional encoding
    
    dir_encoder = PositionalEncoding(dir_L)
    dir_encoded_base = dir_encoder(rays_d)
    dir_encoded_coarse = torch.repeat_interleave(dir_encoded_base, sample_num_coarse, dim=0) # (ray_num * sample_num_coarse) * (6 * dir_L)
    
    xyz_dir_encoded_coarse = torch.cat((xyz_encoded_coarse, dir_encoded_coarse), dim=1)
    
    # Feed into NeRF
    results_coarse = nerf_coarse(xyz_dir_encoded_coarse)
    
    # Unpack results
    rgbs_coarse = results_coarse[:, :3]
    rgbs_coarse = rearrange(rgbs_coarse, '(ray sample) rgb -> ray sample rgb', 
                            ray=ray_num, sample=sample_num_coarse)
    sigmas_coarse = results_coarse[:, 3:4]
    sigmas_coarse = rearrange(sigmas_coarse, '(ray sample) 1 -> ray sample', 
                              ray=ray_num, sample=sample_num_coarse)

    # Sample finely & render coarse image
    exps_coarse = torch.exp(-sigmas_coarse*deltas_coarse)
    
    Ts_coarse = torch.cumprod(torch.cat((torch.ones(ray_num, 1).to(device), exps_coarse), dim=1), dim=1)[:, :-1]
    
    weights_coarse = Ts_coarse * (1 - exps_coarse)
    
    point_rgb_coarse = repeat(weights_coarse, 'ray sample -> ray sample 3') * rgbs_coarse
    pixel_rgb_coarse = torch.sum(point_rgb_coarse, dim=1)
    
    depths_mid_coarse = 0.5 * (depths_coarse[: ,:-1] + depths_coarse[: ,1:])
    
    depths_fine = sample_pdf(depths_mid_coarse, 
                             weights_coarse[:, 1:-1].detach(),
                             sample_num_fine,
                             det=False)
    
    depths_all, _ = torch.sort(torch.cat((depths_coarse, depths_fine), dim=1), dim=1)
    
    xyzs_all = rays_o + rays_d * rearrange(depths_all, 'n1 n2 -> n2 n1 1').to(device)
    xyzs_all = rearrange(xyzs_all, 'sample ray xyz -> ray sample xyz') # Shape: ray_num * sample_num_coarse * 3
    xyzs_all = rearrange(xyzs_all, 'ray sample xyz -> (ray sample) xyz') # Assume first axis is ray
    xyzs_encoded_all = xyz_encoder(xyzs_all/3)
    xyzs_encoded_all = xyz_encoder(xyzs_all/3)
    
    dir_encoded_all = torch.repeat_interleave(dir_encoded_base, sample_num_coarse+sample_num_fine, dim=0) # (ray_num * sample_num_coarse) * (6 * dir_L)
    
    xyz_dir_encoded_all = torch.cat((xyzs_encoded_all, dir_encoded_all), dim=1)
    
    results_all = nerf_fine(xyz_dir_encoded_all, sigma_only=False)
    
    # Unpack fine results
    rgbs_all = results_all[:, :3]
    rgbs_all = rearrange(rgbs_all, '(ray sample) rgb -> ray sample rgb', 
                         ray=ray_num, sample=sample_num_coarse+sample_num_fine)
    sigmas_all = results_all[:, 3:4]
    sigmas_all = rearrange(sigmas_all, '(ray sample) 1 -> ray sample', 
                           ray=ray_num, sample=sample_num_coarse+sample_num_fine)
    
    # Re-do neural rendering
    deltas_all = torch.diff(depths_all, dim=1)
    deltas_all = torch.cat((deltas_all, 1e7 * torch.ones((ray_num, 1)).to(device)), dim=1)
    
    exps_all = torch.exp(-sigmas_all*deltas_all)
    
    Ts_all = torch.cumprod(torch.cat((torch.ones(ray_num, 1).to(device), exps_all), dim=1), dim=1)[:, :-1]
    
    point_rgb_all = repeat(Ts_all * (1 - exps_all), 'ray sample -> ray sample 3') * rgbs_all
    pixel_rgb_all = torch.sum(point_rgb_all, dim=1)
    
    return pixel_rgb_coarse, pixel_rgb_all


@torch.no_grad
def render_depth(rays: torch.Tensor,
                 sample_num_coarse: int,
                 sample_num_fine: int,
                 nerf_coarse: NeRF,
                 nerf_fine: NeRF,
                 threshold,
                 device):
    
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
    
    xyz_encoder = PositionalEncoding(xyz_L)
    xyz_encoded_coarse = xyz_encoder(xyzs_coarse/3)	# (ray_num * sample_num_coarse) * (6 * xyz_L)
    
    # Feed into NeRF
    sigmas_coarse = nerf_coarse(xyz_encoded_coarse, sigma_only=True)
    sigmas_coarse = rearrange(sigmas_coarse, '(ray sample) 1 -> ray sample', 
                              ray=ray_num, sample=sample_num_coarse)

    # Sample finely & render coarse image
    exps_coarse = torch.exp(-sigmas_coarse*deltas_coarse)
    
    Ts_coarse = torch.cumprod(torch.cat((torch.ones(ray_num, 1).to(device), exps_coarse), dim=1), dim=1)[:, :-1]
    
    weights_coarse = Ts_coarse * (1 - exps_coarse)
    
    depths_c1 = torch.sum(weights_coarse * depths_coarse, dim=1)
    
    depth_indices_coarse = torch.sum(Ts_coarse > threshold, dim=1) - 1
    depths_c2 = torch.gather(depths_coarse, 1, depth_indices_coarse[:, None])
    
    depths_mid_coarse = 0.5 * (depths_coarse[: ,:-1] + depths_coarse[: ,1:])
    
    depths_fine = sample_pdf(depths_mid_coarse, 
                             weights_coarse[:, 1:-1].detach(),
                             sample_num_fine,
                             det=False)
    
    depths_all, _ = torch.sort(torch.cat((depths_coarse, depths_fine), dim=1), dim=1)
    
    xyzs_all = rays_o + rays_d * rearrange(depths_all, 'n1 n2 -> n2 n1 1').to(device)
    xyzs_all = rearrange(xyzs_all, 'sample ray xyz -> ray sample xyz') # Shape: ray_num * sample_num_coarse * 3
    xyzs_all = rearrange(xyzs_all, 'ray sample xyz -> (ray sample) xyz') # Assume first axis is ray
    xyzs_encoded_all = xyz_encoder(xyzs_all/3)
    
    sigmas_all = nerf_fine(xyzs_encoded_all, sigma_only=True)
    sigmas_all = rearrange(sigmas_all, '(ray sample) 1 -> ray sample', 
                           ray=ray_num, sample=sample_num_coarse+sample_num_fine)
    
    # Re-do neural rendering
    deltas_all = torch.diff(depths_all, dim=1)
    deltas_all = torch.cat((deltas_all, 1e7 * torch.ones((ray_num, 1)).to(device)), dim=1)
    
    exps_all = torch.exp(-sigmas_all*deltas_all)
    
    Ts_all = torch.cumprod(torch.cat((torch.ones(ray_num, 1).to(device), exps_all), dim=1), dim=1)[:, :-1]
    
    weights_all = Ts_all * (1 - exps_all)
    
    depths_a1 = torch.sum(weights_all * depths_all, dim=1)
    
    depth_indices_all = torch.sum(Ts_all > threshold, dim=1)-1
    depths_a2 = torch.gather(depths_all, 1, depth_indices_all[:, None])

    return depths_c1, depths_c2, depths_a1, depths_a2
    
    
@torch.no_grad
def render_image(rays: torch.Tensor,
                 batch_size: int,
                 img_shape: Tuple[int, int],
                 sample_num_coarse: int,
                 sample_num_fine: int,
                 nerf_coarse: NeRF,
                 nerf_fine: NeRF,
                 threshold=None,
                 depth_only=False,
                 device=None) -> torch.Tensor:
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
    
    if depth_only:
        c1s = []
        c2s = []
        a1s = []
        a2s = []
        
        for ray_batch in batches:
            c1, c2, a1, a2 = render_depth(ray_batch,
                                       sample_num_coarse, 
                                       sample_num_fine,
                                       nerf_coarse,
                                       nerf_fine,
                                       threshold,
                                       device)
            c1s.append(c1)
            c2s.append(c2)
            a1s.append(a1)
            a2s.append(a2)
            
        last_c1 = c1s.pop()
        c1s = torch.cat(c1s, dim=0)
        c1s = torch.cat((c1s, last_c1), dim=0)
        
        last_c2 = c2s.pop()
        c2s = torch.cat(c2s, dim=0)
        c2s = torch.cat((c2s, last_c2), dim=0)
        
        last_a1 = a1s.pop()
        a1s = torch.cat(a1s, dim=0)
        a1s = torch.cat((a1s, last_a1), dim=0)
        
        last_a2 = a2s.pop()
        a2s = torch.cat(a2s, dim=0)
        a2s = torch.cat((a2s, last_a2), dim=0)
        
        c1s = c1s.reshape(img_shape)
        c2s = c2s.reshape(img_shape)
        a1s = a1s.reshape(img_shape)
        a2s = a2s.reshape(img_shape)
        
        return c1s, c2s, a1s, a2s
    
    else:
        rgb_batches = []
        for ray_batch in batches:
            _, rgb_batch = render_rays(ray_batch, 
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
