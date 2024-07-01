import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from deconv import FastDeconv
# from dcn_v2 import DCN
import math
import inspect

PI = math.pi
from torchvision import models
from transweather_model import Tenc, Tdec,convprojection
# from gpu_mem_track import MemTracker  # 引用显存跟踪代码
device = torch.device('cuda:0')


#######################################################  Attention Module
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Sigmoid3 = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            # nn.Sigmoid()
        )
        self.fc1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 3, 1, 1, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y=self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        y1 = self.fc1(x)
        y2 = y * y1
        y3 = self.Sigmoid3(y2)
        # y=self.fc(y)
        y4 = x * y3.expand_as(x)
        return y4


class SAM_cat(nn.Module):
    def __init__(self, in_channel):
        super(SAM_cat, self).__init__()
        self.conv_1 = nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0)
        self.conv_sigm = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_1(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, conv_out], 1)
        y = self.conv_sigm(y)
        return x * y


class SSB(nn.Module):
    def __init__(self, in_channel):
        super(SSB, self).__init__()
        self.CAU = SELayer(in_channel)
        self.SAU = SAM_cat(in_channel)
        self.Sigmoid = nn.Sigmoid()
        self.conv_out = nn.Sequential(
            nn.Conv2d(2 * in_channel, in_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        CAU_out = self.CAU(x)
        SAU_out = self.SAU(x)
        y = torch.cat([CAU_out, SAU_out], 1)
        # y=CAU_out*SAU_out
        # y=self.Sigmoid(y)
        # y=x*y
        # y = torch.cat([CAU_out, SAU_out], 1)
        y = self.conv_out(y)
        return y


########################################################
class DFconvResBlock(nn.Module):
    def __init__(self, in_channel, DF):
        super(DFconvResBlock, self).__init__()

        self.DFconv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=DF, dilation=DF),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.DFconv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=DF, dilation=DF),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(2 * in_channel, in_channel, kernel_size=1, stride=1, padding=0),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.AM = SSB(in_channel)

    def forward(self, x):
        y = self.DFconv1(x)
        y = self.conv_cat(torch.cat([y, self.DFconv2(y)], 1))
        y = self.AM(y)
        return y + x


class AGB_mean(nn.Module):
    def __init__(self, in_channel):
        super(AGB_mean, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(3, 3, bias=False),
            nn.Sigmoid()
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2, x3):
        y1 = torch.mean(x1, dim=1, keepdim=True)
        y2 = torch.mean(x2, dim=1, keepdim=True)
        y3 = torch.mean(x3, dim=1, keepdim=True)
        y = torch.cat([y1, y2, y3], 1)

        b, _, _, _ = y.size()
        y = self.avg_pool(y).view(b, 3)
        y = self.fc(y).view(b, 3, 1, 1)

        y1 = x1 * y[:, 0:1, :, :].expand_as(x1)
        y2 = x2 * y[:, 1:2, :, :].expand_as(x2)
        y3 = x3 * y[:, 2:3, :, :].expand_as(x3)

        y = self.conv_out(y1 + y2 + y3)

        return y


class MRFEB1(nn.Module):
    def __init__(self, in_channel, DF1=1, DF2=3, DF3=5):
        super(MRFEB1, self).__init__()

        self.MRFRB_1 = DFconvResBlock(in_channel, DF1)
        self.MRFRB_2 = DFconvResBlock(in_channel, DF2)
        self.MRFRB_3 = DFconvResBlock(in_channel, DF3)
        self.AGB = AGB_mean(in_channel)
        self.faix = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 8, 1, padding=0, bias=True),
            nn.BatchNorm2d(in_channel // 8),
            nn.ReLU(inplace=True),
        )
        self.dim_kernel = in_channel

    def forward(self, x):
        X_ = self.faix(x)
        N, D, _, _ = X_.shape
        w1 = torch.randn(self.dim_kernel, D).to('cuda')
        W = w1
        b = torch.FloatTensor(self.dim_kernel).uniform_(0, 2 * PI).to('cuda')
        B = b.repeat(N, 1)
        X_feature1 = torch.sqrt(torch.Tensor([2 / self.dim_kernel])).to('cuda')
        X_feature2 = torch.cos(torch.matmul(W, X_.T) + B.T).to('cuda')
        X_feature = (X_feature1 * X_feature2).T
        y1 = self.MRFRB_1(X_feature)
        y2 = self.MRFRB_2(x)
        y3 = self.MRFRB_3(x)
        y1 = self.AGB(y1, y2, y3)

        return y1 + x


class MRFEB2(nn.Module):
    def __init__(self, in_channel, DF1=1, DF2=3, DF3=5):
        super(MRFEB2, self).__init__()

        self.MRFRB_1 = DFconvResBlock(in_channel, DF1)
        self.MRFRB_2 = DFconvResBlock(in_channel, DF2)
        self.MRFRB_3 = DFconvResBlock(in_channel, DF3)
        self.AGB = AGB_mean(in_channel)
        self.faix = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 8, 1, padding=0, bias=True),
            nn.BatchNorm2d(in_channel // 8),
            nn.ReLU(inplace=True),

        )
        self.dim_kernel = in_channel

    def forward(self, x):
        y1 = self.MRFRB_1(x)
        y2 = self.MRFRB_2(x)
        y3 = self.MRFRB_3(x)
        y1 = self.AGB(y1, y2, y3)

        return y1 + x


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DehazeBlock1(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(DehazeBlock1, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        # self.act1 = nn.GELU()
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        # self.calayer = CALayer(dim)
        # self.palayer = PALayer(dim)
        self.MRFEB1 = MRFEB1(512)

        self.MRFEB2 = MRFEB2(512)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.MRFEB2(res)
        res += x
        return res


class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        # self.act1 = nn.GELU()
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.MRFEB1 = MRFEB1(512)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.MRFEB1(res)
        res += x
        return res
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class Dehaze(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect'):
        super(Dehaze, self).__init__()
        self.Tencoder= Tenc()
        self.Tdecoder=Tdec()
        self.convtail = convprojection()
        self.active = nn.Tanh()
        self.clean = ConvLayer(8, 3, kernel_size=3, stride=1)
        self.block = DehazeBlock(default_conv, ngf * 8, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv_output = ConvLayer(64, 3, kernel_size=3, stride=1)

    def forward(self, input ,**kwargs):
        a=input
        add=self.Tencoder(a)
        z = torch.flatten(self.avgpool(add[3]), 1)
        xd0 = self.block(add[3])
        # xd1 = self.block(xd0)
        # xd2 = self.block1(xd1)
        ad=[add[0],add[1],add[2],xd0]
        x_u=self.Tdecoder(ad)
        x = self.convtail(add, x_u)
        clean = self.active(self.clean(x))
        return clean, z
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from torchstat import stat

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    net = Dehaze(3, 3).to('cuda')
    summary(net, input_size=(3, 256, 256), batch_size=1)
    # stat(net, input_size=(3, 224, 224))
