"""
Copyright (c) Jason Lee

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, channel, dropout_prob):
        super().__init__()
        factor = 4
        bottlenect_channel = channel // factor
        self.layers = nn.Sequential(
            nn.Conv2d(channel, bottlenect_channel, kernel_size=1, padding=0, stride=1),
            nn.InstanceNorm2d(bottlenect_channel),
            nn.PReLU(bottlenect_channel),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(bottlenect_channel, bottlenect_channel, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(bottlenect_channel),
            nn.PReLU(bottlenect_channel),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(bottlenect_channel, channel, kernel_size=1, padding=0, stride=1),
            nn.InstanceNorm2d(channel),
            nn.PReLU(channel),
            nn.Dropout2d(dropout_prob)
        )
        self.relu = nn.PReLU(channel)

    def forward(self, input):
        out = self.layers(input)
        out = torch.add(out, input)
        return self.relu(out)


class DoubleConv(nn.Module):
    def __init__(self, channel, dropout_prob):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channel),
            nn.PReLU(channel),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channel),
            nn.PReLU(channel),
            nn.Dropout2d(dropout_prob)
        )

    def forward(self, input):
        return self.layer(input)


class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_prob, resblock_num):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=2, padding=0, stride=2),
            nn.InstanceNorm2d(out_channel),
            nn.PReLU(out_channel)
        )
        # self.conv = DoubleConv(out_channel, dropout_prob)
        self.relu = nn.PReLU(out_channel)

        layer = []
        for i in range(resblock_num):
            layer.append(ResBlock(out_channel, dropout_prob))
        self.conv = nn.Sequential(*layer)

    def forward(self, input):
        down = self.down(input)
        out = self.conv(down)
        return self.relu(torch.add(out, down))


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_prob, resblock_num):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel // 2, kernel_size=2, padding=0, stride=2),
            nn.InstanceNorm2d(out_channel // 2),
            nn.PReLU(out_channel // 2)
        )
        # self.conv = DoubleConv(out_channel, dropout_prob)
        self.relu = nn.PReLU(out_channel)

        layer = []
        for i in range(resblock_num):
            layer.append(ResBlock(out_channel, dropout_prob))
        self.conv = nn.Sequential(*layer)

    def forward(self, input, short):
        up = self.up(input)
        input = torch.cat((up, short), dim=1)
        out = self.conv(input)
        return self.relu(torch.add(out, input))


class ResUNet(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, resblock_num):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_chans, chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(chans),
            nn.PReLU(chans),
            DoubleConv(chans, drop_prob)
        )
        self.down_layers = nn.ModuleList()
        chn = chans
        for i in range(num_pool_layers):
            self.down_layers += [DownBlock(chn, 2 * chn, drop_prob, resblock_num)]
            chn *= 2

        self.up_layers = nn.ModuleList([UpBlock(chn, chn, drop_prob, resblock_num)])
        for i in range(num_pool_layers - 1):
            self.up_layers += [UpBlock(chn, chn // 2, drop_prob, resblock_num)]
            chn //= 2

        self.conv_output = nn.Sequential(
            nn.Conv2d(chn, chn // 2, kernel_size=1),
            # nn.InstanceNorm2d(chn // 2),
            # nn.PReLU(chn // 2),
            nn.Conv2d(chn // 2, out_chans, kernel_size=1),
            # nn.InstanceNorm2d(out_chans),
            # nn.PReLU(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=1)
        )

    def forward(self, input):
        stack = []
        input = self.conv_input(input)

        for layer in self.down_layers:
            stack.append(input)
            input = layer(input)

        for layer in self.up_layers:
            input = layer(input, stack.pop())

        return self.conv_output(input)
