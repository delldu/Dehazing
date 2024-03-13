"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2020-2024 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 09月 09日 星期三 23:56:45 CST
# ***
# ************************************************************************************/
#
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import pdb


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


## Channel Attention (CA) Layer
class RC_CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False):
        super().__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(nn.ReLU())
        modules_body.append(RC_CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super().__init__()
        modules_body = []
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False)
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res + x

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, conv=default_conv):
        super().__init__()
        n_resgroups = 5
        n_resblocks = 10
        n_feats = 32
        kernel_size = 3
        reduction = 8

        modules_head = [conv(3, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, n_resblocks=n_resblocks)
            for _ in range(n_resgroups) # 5
        ]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [conv(n_feats, 3, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x_feat = self.head(x)
        res = self.body(x_feat)
        return res + x_feat


###################################################################################################################
class NormalBottle2Neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4):
        super().__init__()
        assert scale == 4, "valid scale is allways 4"
        assert baseWidth == 26, "valid baseWidth is allways 26"

        # planes -- 64 or 128 or 256
        width = int(math.floor(planes * (baseWidth / 64.0))) # 26 or 52 or 104

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale - 1 # === 3

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU()

        if downsample is not None:
            self.downsample = downsample
        else:  # Support torch.jit.script
            self.downsample = nn.Identity()

        self.scale = scale
        self.width = width

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, dim=1)
        # spx is tuple: len = 4
        #     tensor [item] size: [1, 26, 304, 404], min: 0.0, max: 1.414761, mean: 0.11166
        #     tensor [item] size: [1, 26, 304, 404], min: 0.0, max: 1.901305, mean: 0.082564
        #     tensor [item] size: [1, 26, 304, 404], min: 0.0, max: 1.251288, mean: 0.041927
        #     tensor [item] size: [1, 26, 304, 404], min: 0.0, max: 2.315903, mean: 0.042094

        # (Pdb) self.bns
        # ModuleList(
        #   (0-2): 3 x BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )
        # (Pdb) self.convs
        # ModuleList(
        #   (0-2): 3 x Conv2d(26, 26, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # )

        # i -- 0
        sp = spx[0]
        sp = self.convs[0](sp)
        sp = self.relu(self.bns[0](sp))
        out = sp

        # i -- 1
        sp = sp + spx[1] # normal
        sp = self.convs[1](sp)
        sp = self.relu(self.bns[1](sp))
        out = torch.cat((out, sp), dim=1)

        # i -- 2
        sp = sp + spx[2] # normal
        sp = self.convs[2](sp)
        sp = self.relu(self.bns[2](sp))
        out = torch.cat((out, sp), dim=1)

        out = torch.cat((out, spx[self.nums]), dim=1)

        out = self.conv3(out)
        out = self.bn3(out)
        residual = self.downsample(x)
        out = out + residual

        return self.relu(out)


class StageBottle2Neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4):
        super().__init__()
        assert scale == 4, "valid scale is allways 4"
        assert baseWidth == 26, "valid baseWidth is allways 26"

        # planes -- 64 or 128 or 256
        width = int(math.floor(planes * (baseWidth / 64.0))) # 26 or 52 or 104

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.nums = scale - 1
        self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) # stage

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU()

        if downsample is not None:
            self.downsample = downsample
        else:  # Support torch.jit.script
            self.downsample = nn.Identity()

        self.scale = scale
        self.width = width

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, dim=1)

        # i -- 0
        sp = spx[0]
        sp = self.convs[0](sp)
        sp = self.relu(self.bns[0](sp))
        out = sp

        # i -- 1
        sp = spx[1] # "stage"
        sp = self.convs[1](sp)
        sp = self.relu(self.bns[1](sp))
        out = torch.cat((out, sp), dim=1)

        # i -- 2
        sp = spx[2] # "stage"
        sp = self.convs[2](sp)
        sp = self.relu(self.bns[2](sp))
        out = torch.cat((out, sp), dim=1)

        out = torch.cat((out, self.pool(spx[self.nums])), dim=1)

        out = self.conv3(out)
        out = self.bn3(out)
        residual = self.downsample(x)

        return self.relu(out + residual)


class Res2Net(nn.Module):
    def __init__(self, block, layers=[3, 4, 23, 3], baseWidth=26, scale=4):
        self.inplanes = 64
        super().__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            StageBottle2Neck(
                self.inplanes,
                planes,
                stride,
                downsample=downsample,
                baseWidth=self.baseWidth,
                scale=self.scale,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                NormalBottle2Neck(
                    self.inplanes, 
                    planes, 
                    baseWidth=self.baseWidth, 
                    scale=self.scale,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x) # size() -- [1, 64, 304, 404]
        x_layer1 = self.layer1(x) # size() -- [1, 256, 304, 404]
        x_layer2 = self.layer2(x_layer1) # size() -- [1, 512, 152, 202]
        x = self.layer3(x_layer2)  # x16

        return x, x_layer1, x_layer2


class PALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super().__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU()
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x

        return res


class Enhancer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

        self.refine1 = nn.Conv2d(in_channels, 20, kernel_size=3, stride=1, padding=1)
        self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)

        self.refine3 = nn.Conv2d(20 + 4, out_channels, kernel_size=3, stride=1, padding=1)

        self.batch1 = nn.InstanceNorm2d(100, affine=True) # Useless, only match with weight dict

    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))

        shape_out = dehaze.size()
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 32)
        x102 = F.avg_pool2d(dehaze, 16)
        x103 = F.avg_pool2d(dehaze, 8)
        x104 = F.avg_pool2d(dehaze, 4)

        x1010 = F.interpolate(self.relu(self.conv1010(x101)), size=shape_out, mode="nearest")
        x1020 = F.interpolate(self.relu(self.conv1020(x102)), size=shape_out, mode="nearest")
        x1030 = F.interpolate(self.relu(self.conv1030(x103)), size=shape_out, mode="nearest")
        x1040 = F.interpolate(self.relu(self.conv1040(x104)), size=shape_out, mode="nearest")

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), dim=1)
        dehaze = self.tanh(self.refine3(dehaze))

        return dehaze

class Dehaze(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Res2Net(NormalBottle2Neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        self.mid_conv = DehazeBlock(default_conv, 1024, 3)

        self.up_block1 = nn.PixelShuffle(2)
        self.attention1 = DehazeBlock(default_conv, 256, 3)
        self.attention2 = DehazeBlock(default_conv, 192, 3)
        self.enhancer = Enhancer(28, 28)

    def forward(self, input):
        x, x_layer1, x_layer2 = self.encoder(input)

        x_mid = self.mid_conv(x)

        x = self.up_block1(x_mid)
        x = self.attention1(x)

        x = torch.cat((x, x_layer2), dim=1)
        x = self.up_block1(x)
        x = self.attention2(x)

        x = torch.cat((x, x_layer1), dim=1)
        x = self.up_block1(x)
        x = self.up_block1(x)

        dout2 = self.enhancer(x) #size() -- [2, 28, 256, 256]

        return dout2


class DehazeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define max GPU/CPU memory -- 8G(2048x2048), 3G(1024x1024)
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 16
        # GPU 3G, 400ms

        self.feature_extract = Dehaze()
        self.pre_trained_rcan = RCAN()
        self.tail1 = nn.Sequential(
            nn.ReflectionPad2d(3), 
            nn.Conv2d(60, 3, kernel_size=7, padding=0),
            nn.Tanh(),
        )
        self.load_weights()

    def load_weights(self, model_path="models/image_dehaze.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint))

    def forward(self, input):
        B, C, H, W = input.size()

        pad_h = self.MAX_TIMES - (H % self.MAX_TIMES)
        pad_w = self.MAX_TIMES - (W % self.MAX_TIMES)
        input = F.pad(input, (0, pad_w, 0, pad_h), 'reflect')

        feature = self.feature_extract(input)

        rcan_out = self.pre_trained_rcan(input)
        x = torch.cat([feature, rcan_out], dim=1)
        feat_hazy = self.tail1(x)
        feat_hazy = feat_hazy[:, :, 0:H, 0:W]
        
        return feat_hazy.clamp(0.0, 1.0)

