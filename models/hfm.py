import torch
import torch.nn as nn
from .debug import PrintLayer

# High Filter Module
class HFM(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        
        self.k = k

        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size = self.k, stride = self.k),
            nn.Upsample(scale_factor = self.k, mode = 'nearest'),
        )
        self.pool = nn.AvgPool2d(kernel_size = self.k, stride = self.k)

    def forward(self, tL):
        x = self.pool(tL)
        upsample = nn.Upsample(size=(tL.shape[2], tL.shape[3]), mode='nearest')
        x = upsample(x)
        #assert tL.shape[2] % self.k == 0, 'h, w must divisible by k'
        #assert tL.shape[3] % self.k == 0, 'h, w must divisible by k'
        #return tL - self.net(tL)
        return tL - x


if __name__ == '__main__':
    m = HFM(2)
    x = torch.tensor([float(i+1) for i in range(16)]).reshape((1, 1, 4, 4))
    y = m(x)
