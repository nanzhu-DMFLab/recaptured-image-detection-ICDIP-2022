import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, nonlinear='relu', pooling='max'):
        super().__init__()
        conc_block = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)]

        conc_block.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'elu':
            conc_block.append(nn.ELU(True))
        elif nonlinear == 'relu':
            conc_block.append(nn.ReLU(True))
        elif nonlinear == 'leaky_relu':
            conc_block.append(nn.LeakyReLU(0.2, True))
        elif nonlinear is None:
            pass

        if pooling is not None:
            conc_block.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.model = nn.Sequential(*conc_block)

    def forward(self, x):
        out = self.model(x)
        return out



class MrcNet(nn.Module):
    def __init__(self, ndf=5, nonlinear='relu'):
        super().__init__()

        # the branch of ScNet
        self.branch0 = nn.Sequential(nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=False), 
                                     Block(1, 8, kernel_size=3, stride=1, padding=1, nonlinear='relu', pooling=None),
                                     #Block(8, 8, kernel_size=3, stride=1, padding=1, nonlinear='relu', pooling=None),
                                     Block(8, 8, kernel_size=3, stride=1, padding=1, nonlinear='relu', pooling=None))
        self.branch1 = nn.Sequential(nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=False), 
                                     Block(1, 8, kernel_size=5, stride=1, padding=2, nonlinear='relu', pooling=None),
                                     #Block(8, 8, kernel_size=5, stride=1, padding=2, nonlinear='relu', pooling=None),
                                     Block(8, 8, kernel_size=5, stride=1, padding=2, nonlinear='relu', pooling=None))      
        self.branch2 = nn.Sequential(nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=False), 
                                     Block(1, 8, kernel_size=7, stride=1, padding=3, nonlinear='relu', pooling=None),
                                     #Block(8, 8, kernel_size=7, stride=1, padding=3, nonlinear='relu', pooling=None),
                                     Block(8, 8, kernel_size=7, stride=1, padding=3, nonlinear='relu', pooling=None))                       
 
        # the basic of ScNet
        self.basic = nn.Sequential(
            Block(in_channels=24, out_channels=2**ndf, nonlinear=nonlinear),
            Block(in_channels=2**(ndf + 0), out_channels=2**(ndf + 1), nonlinear='relu'),
            Block(in_channels=2**(ndf + 1), out_channels=2**(ndf + 2), nonlinear='relu'),
            Block(in_channels=2**(ndf + 2), out_channels=2**(ndf + 3), nonlinear='relu'),
            Block(in_channels=2**(ndf + 3), out_channels=2**(ndf + 3), nonlinear='relu', pooling=None)
        )

        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        y1 = self.branch0(x)
        y2 = self.branch1(x)
        y3 = self.branch2(x)
        x = torch.cat((y1, y2, y3), 1)   
        x = self.basic(x)

        x = F.adaptive_avg_pool2d(x, 1)   
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

'''
t = torch.randn([4,3,96,96])
net = MrcNet()
out = net(t)
print(out)
print(net)
'''
