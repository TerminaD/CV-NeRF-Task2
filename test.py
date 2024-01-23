import torch
import matplotlib.pyplot as plt

def PSNR(pred_img,gt_img):
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    psnr = 20 * np.log10(1/np.sqrt(mse)) 
    return psnr 

def test_pic(index=0,dataset=BlenderDataset("data/lego/test","test")):
    sample=dataset[index]

    pred_img=render_img(sample[0])
    gt_img=sample[1]

    concat_img = np.concatenate((pred_img, gt_img), axis=1)
    plt.imsave(f"result/pred_img{index}.png",pred_img)
    plt.imsave(f"result/con_img{index}.png",con_img)
    plt.imshow(concat_img)
    plt.show()

    psnr=PSNR(test_img,gt_img)
    print(psnr)
    return psnr

def test_pics():
    dataset=BlenderDataset("data/lego/test","test")
    for i,data in enum(dataset):
        test_pic(i,data)

