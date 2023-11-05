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
        """
        # 只用一层空洞卷积
        self.MRFRB_1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=DF1, dilation=DF1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(in_channel)
        )
        self.MRFRB_2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=DF2, dilation=DF2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(in_channel)
        )
        self.MRFRB_3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=DF3, dilation=DF3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(in_channel)
        )
"""
        self.AGB = AGB_mean(in_channel)
        self.faix = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 8, 1, padding=0, bias=True),
            nn.BatchNorm2d(in_channel // 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            # nn.Conv2d(channel // 8, channel // 8, kernel_size=1, stride=1, padding=0),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(in_channel)

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
        """
        # 只用一层空洞卷积
        self.MRFRB_1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=DF1, dilation=DF1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(in_channel)
        )
        self.MRFRB_2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=DF2, dilation=DF2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(in_channel)
        )
        self.MRFRB_3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=DF3, dilation=DF3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(in_channel)
        )
"""
        self.AGB = AGB_mean(in_channel)
        self.faix = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 8, 1, padding=0, bias=True),
            nn.BatchNorm2d(in_channel // 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            # nn.Conv2d(channel // 8, channel // 8, kernel_size=1, stride=1, padding=0),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(in_channel)

        )
        self.dim_kernel = in_channel

    def forward(self, x):
        # X_ = self.faix(x)
        # N, D, _, _ = X_ .shape
        # w1 = torch.randn(self.dim_kernel, D).to('cuda')
        # W = w1
        # b = torch.FloatTensor(self.dim_kernel).uniform_(0, 2 * PI).to('cuda')
        # B = b.repeat(N, 1)
        # X_feature1 = torch.sqrt(torch.Tensor([2 / self.dim_kernel])).to('cuda')
        # X_feature2 = torch.cos(torch.matmul(W, X_ .T) + B.T).to('cuda')
        # X_feature = (X_feature1 * X_feature2).T
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
        # res = self.calayer(res)
        # res = self.palayer(res)
        res += x
        return res


class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        # self.act1 = nn.GELU()
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        # self.calayer = CALayer(dim)
        # self.palayer = PALayer(dim)
        self.MRFEB1 = MRFEB1(512)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.MRFEB1(res)
        # res = self.calayer(res)
        # res = self.palayer(res)
        res += x
        return res


# class DCNBlock(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(DCNBlock, self).__init__()
#         self.dcn = DCN(in_channel, out_channel, kernel_size=(3,3), stride=1, padding=1).cuda()
#     def forward(self, x):
#         return self.dcn(x)

# class Mix(nn.Module):
#     def __init__(self, m=-0.80):
#         super(Mix, self).__init__()
#         w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
#         w = torch.nn.Parameter(w, requires_grad=True)
#         self.w = w
#         self.mix_block = nn.Sigmoid()
#
#     def forward(self, fea1, fea2):
#         mix_factor = self.mix_block(self.w)
#         out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
#         return out


# class LAM_Module(nn.Module):
#     """ Layer attention module"""
#
#     def __init__(self, in_dim):
#         super(LAM_Module, self).__init__()
#         self.chanel_in = in_dim
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X N X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X N X N
#         """
#         m_batchsize, N, C, height, width = x.size()
#         proj_query = x.view(m_batchsize, N, -1)
#         proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
#         energy = torch.bmm(proj_query, proj_key)
#         energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
#         attention = self.softmax(energy_new)
#         proj_value = x.view(m_batchsize, N, -1)
#
#         out = torch.bmm(attention, proj_value)
#         out = out.view(m_batchsize, N, C, height, width)
#
#         out = self.gamma * out + x
#         out = out.view(m_batchsize, -1, height, width)
#         return out


# class UpsampleConvLayer(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super(UpsampleConvLayer, self).__init__()
#         self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
#
#     def forward(self, x):
#         out = self.conv2d(x)
#         return out


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

        # ###### downsample
        # self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
        #                            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
        #                            nn.ReLU(True)
        #                            )
        # self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
        #                            nn.ReLU(True))
        # self.down3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
        #                            nn.ReLU(True))
        # self.down4 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=5, stride=2, padding=2),
        #                            nn.ReLU(True))
        # ###### FFA blocks
        self.block = DehazeBlock(default_conv, ngf * 8, 3)
        # self.block1 = DehazeBlock1(default_conv, ngf * 8, 3)
        # ###### upsample
        # self.up0 = nn.Sequential(
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(True))
        # self.up1 = nn.Sequential(
        #     nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(True))
        # self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=5, stride=2, padding=2, output_padding=1),
        #                          nn.ReLU(True))
        # self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
        #                          nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
        #                          nn.Tanh())
        #
        # # self.dcn_block = DCNBlock(256, 256)
        #
        # self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)
        #
        # self.mix1 = Mix(m=-1)
        # self.mix2 = Mix(m=-0.6)
        # self.mix3 = Mix(m=-0.4)
        # # self.MRFEB = MRFEB(512, 1, 2, 5)
        #
        # self.la = LAM_Module(64)
        # self.conv_output = ConvLayer(64, 3, kernel_size=3, stride=1)
        #
        # self.convd8x_LAM = nn.Sequential(
        #     UpsampleConvLayer(512, 256, kernel_size=3, stride=2),
        #     UpsampleConvLayer(256, 128, kernel_size=3, stride=2),
        #     UpsampleConvLayer(128, 64, kernel_size=3, stride=2))
        # self.convd4x_LAM = nn.Sequential(
        #     UpsampleConvLayer(256, 128, kernel_size=3, stride=2),
        #     UpsampleConvLayer(128, 64, kernel_size=3, stride=2))
        # self.convd2x_LAM = nn.Sequential(
        #     UpsampleConvLayer(128, 64, kernel_size=3, stride=2))
        # self.last_LAM_conv = nn.Conv2d(64 * 3, 64, 3, 1, 1)
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
        # self.convtail
        # x2 = self.block1(x1)
        # x3 = self.block1(x2)




        # x_deconv = self.deconv(input)  # preprocess
        #
        # x_down1 = self.down1(x_deconv)  # [bs, 64, 256, 256]
        # x_down2 = self.down2(x_down1)  # [bs, 128, 128, 128]
        # x_down3 = self.down3(x_down2)  # [bs, 256, 64, 64]
        # x_down4 = self.down4(x_down3)  # [bs, 256, 64, 64]
        # z = torch.flatten(self.avgpool(x_down4), 1)
        #
        # x1 = self.block(x_down4)
        # x2 = self.block1(x1)
        # x3 = self.block1(x2)
        # # x4 = self.block1(x3)
        # # x5 = self.block(x4)
        # # x6 = self.block(x5)
        # # x_dcn1 = self.MRFEB(x6)
        # # x_dcn2 = self.MRFEB(x_dcn1)
        # # x_dcn2 = self.MRFEB(x_dcn2)
        # x_out_mix = self.mix1(x_down4, x3)
        # res8x_LAM = self.convd8x_LAM(x_out_mix)
        # x_up1 = self.up0(x_out_mix)  # [bs, 128, 128, 128]
        # x_up1_mix = self.mix2(x_down3, x_up1)
        # res4x_LAM = self.convd4x_LAM(x_up1_mix)
        # x_up2 = self.up1(x_up1_mix)  # [bs, 64, 256, 256]
        # x_up3_mix = self.mix3(x_down2, x_up2)
        # res2x_LAM = self.convd2x_LAM(x_up3_mix)
        # x_up4 = self.up2(x_up3_mix)  # [bs, 64, 256, 256]
        # res8x_LAM = F.upsample(res8x_LAM, x_up4.size()[2:], mode='bilinear')
        # res4x_LAM = F.upsample(res4x_LAM, x_up4.size()[2:], mode='bilinear')
        # res2x_LAM = F.upsample(res2x_LAM, x_up4.size()[2:], mode='bilinear')
        # res8x_LAM = res8x_LAM.unsqueeze(1)
        # res8x_LAM = torch.cat([res4x_LAM.unsqueeze(1), res8x_LAM], 1)
        # res8x_LAM = torch.cat([res2x_LAM.unsqueeze(1), res8x_LAM], 1)
        #
        # res8x_LAM = self.la(res8x_LAM)
        # res8x_LAM = self.last_LAM_conv(res8x_LAM)
        # # res8x_LAM =self.conv_output(res8x_LAM )
        # out = self.up3(x_up4 + res8x_LAM)
        # [bs,  3, 256, 256]
        # return clean, z


from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchstat import stat

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    net = Dehaze(3, 3).to('cuda')
    summary(net, input_size=(3, 256, 256), batch_size=1)
    # stat(net, input_size=(3, 224, 224))
