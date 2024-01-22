from models.nerf import NeRF
from utils.positional_encoding import PositionalEncoding

import torch
import torch.nn as nn

def render_rays(rays: torch.Tensor, 
                nerf: NeRF) -> torch.Tensor:
    """
    Render a number of rays.
    
    Inputs:
		rays: shape [num, 8], rays_o, rays_d, near bound & far bound concatenated
  
	Output:
		results: shape of [num, 4], RGB & sigma for each ray
    """
    
    assert rays.dim() == 2
    assert rays.shape[1] == 8
    
    rays_o = rays[:, :3]
    rays_d = rays[:, 3:6]
    nears = rays[:, 6]
    fars = rays[:, 7]
    
    num = rays.shape[0]
    for i in range(num):

def render_image():
    pass