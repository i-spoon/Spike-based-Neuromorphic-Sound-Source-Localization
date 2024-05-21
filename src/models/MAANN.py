import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
# import args_config
from torchvision import datasets, transforms
import gc
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from spike_related import LIFSpike
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
import torchinfo
from thop import profile
from thop import clever_format
from torchstat import stat
class FWJA(nn.Module):
    def __init__(self, kernel_size_t: int = 2, in_channels: int = 51, out_channels: int = 30):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=kernel_size_t,padding='same', bias=False)
        self.lif = LIFSpike()

    def forward(self,x_seq:torch.Tensor):
        # [T,N,6,H,W] --> [N,T,6,H,W]

        x = self.conv(x_seq)
        x = self.lif(x).squeeze(dim=1)
        y_seq = x_seq * x[:,None,:,:]
        return y_seq


class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(ConvBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.lif1 = LIFSpike()
        self.pool1 = nn.MaxPool2d((2,2))


    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.pool1(x)
        return x

class PoolBlock(nn.Module):
    def __init__(self,in_channel,out_channel,at_w,at_h,kernel_size=3,padding=1):
        super(PoolBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.lif = LIFSpike()
        self.att = FWJA(4,in_channel,out_channel)
        self.pool1 = nn.MaxPool2d((2,2))
        self.lif1 = LIFSpike()


    def forward(self,x):
        x = self.att(x)
        x = self.lif(x)

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.lif1(x)
        x = self.pool1(x)

        return x
    
class MAA(nn.Module):
    def __init__(self):
        super(MAA,self).__init__()

        self.conv1 = ConvBlock(6,12)
        ## 脉冲的形式
        self.fconv = nn.Conv2d(12,12,kernel_size=3,padding=1)
        self.lif = LIFSpike()
        self.fblock = nn.Sequential(self.fconv)
        self.conv2 = PoolBlock(12,24,12,10)
        self.conv3 = PoolBlock(24,48,12,10)
        # self.conv4 = PoolBlock(48,96,6,5)
        self.dropout = nn.Dropout(0.15)
        self.lin = nn.Linear(48*6*5,360,bias=False)
        self.lif = LIFSpike()
    def forward(self,x):
        
        x = x.permute(1,0,2,3,4)
        u_out = []
        qtim_1 = torch.zeros((128,12,25,20)).cuda()
        for t in range(x.shape[0]):
            q_x = self.conv1(x[t])
            qtim_1 = self.lif(self.fblock(qtim_1) + 4/3 * q_x)
            x_1 = self.conv2(qtim_1)
            x_1 = self.conv3(x_1)
            
            x_1 = x_1.view(x_1.size(0),-1)
            x_1 = self.lin(x_1)
            u_out += [x_1]
        return torch.stack(u_out,dim=0)

if __name__=='__main__':
    
    ratio = 0.1058311
    torch.cuda.set_device(2)  
    model = MAA().cuda()
    x = torch.randn((1,4,6,51,40)).cuda()
    flops, params = profile(model, inputs=(x,))
    print(flops)
    # 将结果转换为更易于阅读的格式
    flops, params = clever_format([flops, params], '%.3f')

    print(f"运算量：{flops}, 参数量：{params}")
