import torch


class NeRFMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, gt, pred_coarse, pred_fine):
        return torch.mean((gt-pred_coarse) ** 2 + (gt-pred_fine) ** 2)
    