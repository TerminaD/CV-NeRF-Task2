import argparse
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

    
class PositionalEncoding(nn.Module):
    def __init__(self, l):
        """
        Input:
            l: number
        """
        super().__init__()
        self.N_freqs = l
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = (2.0 ** torch.arange(l)) * torch.pi

    def forward(self, x):
        """
        Input:
            x: tensor
        Output: tensor(x.size() * 2l, )
        """
        out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class ImageDataset(Dataset):
    """
    A dataset class that returns the coordinates-RGB value tuple for all pixels
    in an image.
    
    Inputs:
        img: Numpy array
    Outputs:
        coords: torch tensor
        rgb: torch tensor
    """
    def __init__(self, img):
        self.image = img    # np array
        self.h, self.w, _ = self.image.shape

    def __len__(self):
        return self.h * self.w

    def __getitem__(self, idx):
        x = idx // self.w
        y = idx % self.w
        rgb = torch.tensor(self.image[x, y])
        if rgb.size(0) == 4: # to deal with PNG alpha channel:
            rgb = rgb[:-1]
        return torch.tensor((x/self.h, y/self.w)), rgb
    
    
class CoordinateMLP(nn.Module):
    """
    A neural network that represents a 2D image.
    Input:
        Regularized(0 to 1) pixel coordinates
    Output:
        Relularized RGB value
    """
    def __init__(self, l, in_dim, out_dim=3, no_posenc=False):
        super().__init__()
        
        self.no_posenc = no_posenc
        
        if not no_posenc:
            self.pos_embed = PositionalEncoding(l)
            in_dim *= (2 * l) 
        
        self.func = nn.Sequential(nn.Linear(in_dim, 256), 
                                  nn.ReLU(), 
                                  nn.Linear(256, 128),
                                  nn.ReLU(), 
                                  nn.Linear(128, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, out_dim),
                                  nn.Sigmoid())

    def forward(self, x):
        if not self.no_posenc:
            x = self.pos_embed(x)
        x = self.func(x)
        return x
    
def psnr(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    psnr = 20 * np.log10(1/np.sqrt(mse)) 
    return psnr 

def train(model, dataloader, epoch, device, ckpt_name, gt_img=None, writer=None):
    print("Training...")
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    
    for e in range(epoch):
        print(f"Epoch {e}")
        for data in dataloader:
            coords, gt_color = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            pred_color = model(coords)
            loss = criterion(gt_color, pred_color)
            loss.backward()
            optimizer.step()
        if e % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/{ckpt_name}-e{e}.pth")
        psnr_v = test(model, device, gt_img, ckpt_name, training_epoch=e)
        writer.add_scalar('training/psnr', psnr_v, e)
    
    torch.save(model.state_dict(), f"checkpoints/{ckpt_name}-final.pth")
    
    writer.flush()
    writer.close()


def test(model, device, gt_img, ckpt_name, training_epoch=-1):
    if training_epoch < 0:
        print("Testing...")
        model.load_state_dict(torch.load(f"checkpoints/{ckpt_name}-final.pth", map_location=device))

    if np.shape(gt_img)[2] == 4:    # Remove alpha channel
        gt_img = gt_img[:, :, :-1]
        
    h, w, _ = gt_img.shape
    
    pred_img = torch.zeros([h, w, 3], device=device)

    coords = torch.zeros([h, w, 2], device=device)
    coords[:, :, 0] = torch.arange(h, device=device).unsqueeze(1) / h
    coords[:, :, 1] = torch.arange(w, device=device).unsqueeze(0) / w

    with torch.no_grad():
        pred_color = model(coords.view(-1, 2))
        pred_img = pred_color.view(h, w, 3)

    pred_img = pred_img.cpu().numpy()
    
    if training_epoch >= 0:
        plt.imsave(f'results/{ckpt_name}-e{training_epoch}.png', pred_img)
        
    else:
        concat_img = np.concatenate((pred_img, gt_img), axis=1)

        np.save(f'results/{ckpt_name}.npy', pred_img)
        plt.imsave(f'results/{ckpt_name}-pred.png', pred_img)
        plt.imsave(f'results/{ckpt_name}-concat.png', concat_img)
        
        plt.imshow(concat_img)
        plt.show()

    return psnr(pred_img, gt_img)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Perform training?')
    parser.add_argument('--test', action='store_true', help='Perform testing?')
    parser.add_argument('-i', '--image', 
                        type=str, default="images/starry_night.png", 
                        help='Path of image to fit.')
    parser.add_argument('-e', '--epoch', 
                        type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('-l', 
                        type=int, default=15, help='Param L in positional encoding.')
    parser.add_argument('-c', '--ckpt', 
                        type=str, help='Name of checkpoint to render.')
    parser.add_argument('--no_posenc', action='store_false',
                        help='Turn off positional encoding?')
    args = parser.parse_args()
    if args.test and not args.ckpt and not args.train:
        parser.error("Cannot test on timestamp-named checkpoint if not training")
        
    writer = SummaryWriter(log_dir='runs/psnr')
    
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%H-%M-%S")
    
    model = CoordinateMLP(l=args.l, in_dim=2, no_posenc=args.no_posenc)
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device)
    print(f"Device is {device}")
    model.to(device)  
    
    gt_img = np.array(Image.open(args.image), dtype=np.float32) / 256
    dataset = ImageDataset(gt_img)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    if args.train:
        if args.ckpt:
            train(model, dataloader, args.epoch, device, args.ckpt, gt_img, writer)
        else:
            train(model, dataloader, args.epoch, device, timestamp, gt_img, writer)
    
    if args.test:
        if args.ckpt:
            test(model, device, gt_img, args.ckpt)
        else:
            test(model, device, gt_img, timestamp)
    

if __name__ == "__main__":
    train()
    