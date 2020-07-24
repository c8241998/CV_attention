import torch.nn as nn
import torch

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b,c,h,w = x.size()
        y = self.avgpool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)

if __name__ == '__main__':
    x = torch.randn(16,128,100,100)
    net = SELayer(channel = 128)
    y = net(x)
    print(y.shape)