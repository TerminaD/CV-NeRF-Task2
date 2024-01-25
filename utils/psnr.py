import torch

@torch.no_grad
def psnr_func(gt_img, pred_img):
    mse = torch.mean((gt_img - pred_img) ** 2) 
    
    # MSE is zero means no noise is present in the signal, therefore PSNR have no importance. 
    if(mse == 0):  
        return 100
    
    psnr = 20 * torch.log10(1/torch.sqrt(mse)) 
    return psnr 
