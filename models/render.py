from models.nerf import NeRF
from utils.positional_encoding import PositionalEncoding

import torch
import torch.nn as nn

def render_rays(rays_o: torch.Tensor, 
                rays_d: torch.Tensor, 
                nerf: NeRF) -> torch.Tensor:
    """
    Render a number of rays.
    
    Inputs:
		rays_o: shape of [num, 3], all ray origins
		rays_d: shape of [num, 3], all normalized ray directions
		nerf: a pre-trained NeRF neural network
  
	Output:
		results: shape of [num, 4], RGB & sigma for each ray
    """
    
    assert rays_o.dim() == 2
    assert rays_d.dim() == 2
    assert rays_o.shape()[0] == rays_d.shape()[0]
    assert rays_o.shape()[1] == 3
    assert rays_d.shape()[1] == 3
    
    