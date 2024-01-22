import torch

def get_ray_directions(H=800,W=800,focal=100):
    '''
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    '''

    i,j=torch.meshgrid(torch.arange(H),torch.arange(W))
    directions=torch.stack([(i-W/2)/focal,(j-H/2)/focal,-torch.ones_like(i)],dim=-1)
    return directions

def get_rays(dir,c2w):
    '''
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    '''
    rays_dir=dir@c2w[:,:3].T
    rays_nor=torch.norm(rays_dir,dim=-1,keepdim=1)
    rays_dir/=rays_nor
    rays_o=c2w[:,3].expand(rays_dir.shape)

    rays_o=rays_o.view(-1,3)
    rays_d=rays_dir.view(-1,3)

    return rays_o,rays_d
