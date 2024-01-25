import torch


def init_p2c_directions(H: int = 200, 
                        W: int = 200, 
                        focal: float = 1100) -> torch.Tensor:
    """
    Calculates the directions of rays from the camera to each pixel.
    This only depends on the dimensions of the image and the camera focal length.
    
    Inputs:
        H: image height
        W: image width
        focal: camera focal length
        
    Outputs:
        directions: shape of (H, W, 3), the direction of the rays in camera coordinate
    """

    i, j = torch.meshgrid(torch.arange(H), torch.arange(W))
    directions = torch.stack([(i - W/2) / focal, (j - H/2) / focal, -torch.ones_like(i)], dim=-1)
    return directions

# print(init_p2c_directions())

def get_p2w_ray_directions(dir: torch.Tensor, c2w: torch.Tensor):
    """
    Get the origin and direction of rays in world coordinates.
    
    Inputs:
        dir: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    
    # Map ray directions to world coordinates
    rays_dir = dir @ c2w[:, :3].T
    
    # Normalizd direction vectors
    rays_nor = torch.norm(rays_dir, dim=-1, keepdim=True) + 1e-7
    rays_dir /= rays_nor
    
    # The origin is camera position in world coordinates
    rays_o = c2w[:, 3].expand(rays_dir.shape)
    
    rays_o = rays_o.view(-1, 3)
    rays_d = rays_dir.view(-1, 3)

    return rays_o, rays_d
