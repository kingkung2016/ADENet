
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(BaseConv2d, self).__init__()
        self.basicconv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.basicconv(x)


class SA(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SA, self).__init__()

        self.conv_mask = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        mask = self.conv_mask(torch.cat([avgout, maxout], dim=1))

        #out = self.sigmoid(mask)

        return mask


class CA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1))

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        weights = avgout + maxout
        return x * weights


class Cross_Modality_Fusion(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=16):
        super(Cross_Modality_Fusion, self).__init__()

        self.conv_RGB = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel)
        )
        self.conv_Depth = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel)
        )

        self.CA_fusion = CA(2 * out_channel, reduction)

        self.SA_rgb = SA()
        self.SA_depth = SA()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * out_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, RGB, Depth):
        RGB = self.conv_RGB(RGB)
        Depth = self.conv_Depth(Depth)

        enhanced_rgb = self.SA_depth(Depth) * RGB + RGB
        enhanced_depth = self.SA_rgb(RGB) * Depth + Depth

        fusion = torch.cat((enhanced_rgb, enhanced_depth), dim=1)
        out = self.fusion_conv(self.CA_fusion(fusion))

        return out

# CIR-Net: Cross-Modality Interaction and Refinement for RGB-D Salient Object Detection (IGF)
# CATNet: A Cascaded and Aggregated Transformer Network For RGB-D Salient Object Detection (MSAM)
# The implementation of the Cross-level Aggregation unit.
class CLA(nn.Module):
    def __init__(self, fea_high_channels, fea_low_channels, out_channels, upsample=True, rate=4):
        super(CLA, self).__init__()

        inter_channels = out_channels // rate
        self.upsample = upsample

        self.conv_high = nn.Sequential(
            BaseConv2d(fea_high_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_low = nn.Sequential(
            BaseConv2d(fea_low_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

        self.local_att = nn.Sequential(
            nn.Conv2d(out_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.conv3 = nn.Sequential(
            BaseConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            BaseConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, fea_high, fea_low):
        if self.upsample:
            fea_high = F.interpolate(fea_high, scale_factor=2, mode='bilinear', align_corners=True)

        fea_high_up = self.conv_high(fea_high)
        fea_low = self.conv_low(fea_low)

        fea_fuse = fea_high_up + fea_low
        fuse_l = self.local_att(fea_fuse)
        fuse_g = self.global_att(fea_fuse)
        fuse_lg = fuse_g + fuse_l

        p_block = torch.sigmoid(fuse_lg)
        one_block = torch.ones_like(p_block)

        fea_out = 2 * fea_high_up * (one_block - p_block) + 2 * fea_low * p_block
        fea_out = self.conv3(fea_out)

        return fea_out


#Cross-modal hierarchical interaction network for RGB-D salient object detection
#the multi-scale convolution (MC) module
class MC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MC, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BaseConv2d(in_channel, out_channel, 1, padding=0),
        )
        self.branch1 = nn.Sequential(
            BaseConv2d(in_channel, out_channel, 1, padding=0),
            BaseConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BaseConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BaseConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BaseConv2d(in_channel, out_channel, 1, padding=0),
            BaseConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BaseConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BaseConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BaseConv2d(in_channel, out_channel, 1, padding=0),
            BaseConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BaseConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BaseConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BaseConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BaseConv2d(in_channel, out_channel, 3, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


# Progressively Multiscale Aggregation Decoder
class PMA_decoder(nn.Module):
    def __init__(self, in_channels=[64, 128, 256, 512]):
        super(PMA_decoder, self).__init__()
        self.image_size = 256
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.cf43 = CLA(in_channels[3], in_channels[2], in_channels[2], upsample=False)
        self.cf32 = CLA(in_channels[2], in_channels[1], in_channels[1])
        self.cf21 = CLA(in_channels[1], in_channels[0], in_channels[0])

        self.conv_dilation = ASPP(in_channels[3], in_channels[3], rates=[1,2,3,4])

        self.Group_Conv_3 = MC(in_channels[3], in_channels[2])
        self.Group_Conv_2 = MC(in_channels[2], in_channels[1])
        self.Group_Conv_1 = MC(in_channels[1], in_channels[0])


        self.Sal_Head_1 = Sal_Head(in_channels[0])
        self.Sal_Head_2 = Sal_Head(in_channels[1])
        self.Sal_Head_3 = Sal_Head(in_channels[2])
        self.Sal_Head_4 = Sal_Head(in_channels[3])


    def forward(self, fuse1, fuse2, fuse3, fuse4):

        feature_4 = self.conv_dilation(self.upsamplex2(fuse4))
        feature_43 = self.cf43(feature_4, fuse3)
        feature_32 = self.cf32(feature_43, fuse2)
        feature_21 = self.cf21(feature_32, fuse1)

        fusion_4 = feature_4
        fusion_3 = torch.cat((fuse3, feature_43), dim=1)
        fusion_2 = torch.cat((fuse2, feature_32), dim=1)
        fusion_1 = torch.cat((fuse1, feature_21), dim=1)


        fusion_43 = fusion_4 + fusion_3
        f_out3 = self.Group_Conv_3(fusion_43)
        fusion_32 = self.upsamplex2(f_out3) + fusion_2
        f_out2 = self.Group_Conv_2(fusion_32)
        fusion_21 = self.upsamplex2(f_out2) + fusion_1
        f_out1 = self.Group_Conv_1(fusion_21)


        F4_out = F.interpolate(self.Sal_Head_4(fusion_4), self.image_size, mode='bilinear', align_corners=True)
        F3_out = F.interpolate(self.Sal_Head_3(f_out3), self.image_size, mode='bilinear', align_corners=True)
        F2_out = F.interpolate(self.Sal_Head_2(f_out2), self.image_size, mode='bilinear', align_corners=True)
        F1_out = F.interpolate(self.Sal_Head_1(f_out1), self.image_size, mode='bilinear', align_corners=True)

        return F1_out, F2_out, F3_out, F4_out


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
			stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, inplanes, planes, rates):
        super(ASPP, self).__init__()

        self.aspp1 = ASPP_module(inplanes, planes, rate=rates[0])
        self.aspp2 = ASPP_module(inplanes, planes, rate=rates[1])
        self.aspp3 = ASPP_module(inplanes, planes, rate=rates[2])
        self.aspp4 = ASPP_module(inplanes, planes, rate=rates[3])

        self.conv_f = nn.Sequential(
            nn.Conv2d(4*planes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
		)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x_fusion = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.conv_x(x) + self.conv_f(x_fusion)

        return out


class Sal_Head(nn.Module):
    def __init__(self, channel):
        super(Sal_Head, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channel, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, 1, 0)
        )

    def forward(self, x):
        y = self.conv(x)
        return y


if __name__== '__main__':
    rgb = torch.rand(1,512,32,32)
    depth = torch.rand(1,512,32,32)

    module = Cross_Modality_Fusion(512, 128)
    out = module(depth, rgb)
    print(out.shape)


    fusion_features = [torch.rand(1, 64, 64, 64).cuda(), torch.rand(1, 128, 32, 32).cuda(), torch.rand(1, 256, 16, 16).cuda(), torch.rand(1, 512, 8, 8).cuda()]

    in_channels = [64, 128, 256, 512]
    test_module = PMA_decoder(in_channels).cuda()
    out4321, out432, out43, feature_4321 = test_module(fusion_features[0], fusion_features[1], fusion_features[2], fusion_features[3])
    print(out4321.shape, out432.shape, out43.shape, feature_4321.shape)




