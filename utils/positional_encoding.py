import torch


class PositionalEncoding(torch.nn.Module):
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
