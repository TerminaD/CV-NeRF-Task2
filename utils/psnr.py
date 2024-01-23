import numpy as np

def PSNR(pred_img,gt_img):
    mse = np.mean((gt_img - pred_img) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    psnr = 20 * np.log10(1/np.sqrt(mse)) 
    return psnr 
