import torch.nn as nn
import torch

import models.modules.blocks as B


class vdsr(nn.Module):
    def __init__(self, in_channels, out_channels, num_branch):
        super(vdsr, self).__init__()
        self.conv = nn.ModuleList()

        self.conv_in = B.ConvBlock(in_channels, 64, kernel_size=3, norm_type=None, act_type='relu',valid_padding=True, bias=False)

        for i in range(18):
            self.conv.append(B.ConvBlock(64, 64, kernel_size=3, norm_type=None, act_type='relu',valid_padding=True, bias=False))

        self.conv_out = B.ConvBlock(64, out_channels, kernel_size=3, norm_type=None, act_type=None, valid_padding=True, bias=False)

    def forward(self, input):
        x = self.conv_in(input)

        for c in self.conv:
            x = c(x)

        output = torch.add(self.conv_out(x), input)

        return output

# TODO: vdsr_k

class vdsr_k(nn.Module):
    def __init__(self, in_channels, out_channels, num_branch):
        super(vdsr_k, self).__init__()
        split_layer = 4
        self.num_branch = num_branch

        self.conv = nn.Sequential()
        self.conv.add_module(B.ConvBlock(in_channels, 64, kernel_size=3, norm_type=None, act_type='relu',valid_padding=True, bias=False))

        for i in range(18-split_layer):
            self.conv.add_module(B.ConvBlock(64, 64, kernel_size=3, norm_type=None, act_type='relu',valid_padding=True, bias=False))

        self.conv_branch = nn.ModuleList()

        for _ in range(self.num_branch):
            sub_branch = nn.Sequential()
            for _i in range(split_layer):
                sub_branch.add_module(B.ConvBlock(64, 64, kernel_size=3, norm_type=None, act_type='relu',valid_padding=True, bias=False))

            sub_branch.add_module(B.ConvBlock(64, out_channels, kernel_size=3, norm_type=None, act_type=None, valid_padding=True, bias=False))
            self.conv_branch.append(sub_branch)

    def forward(self, x, coeff):
        x = self.conv(x)

        hr_list = []
        for idx in range(self.num_branch):
            hr_list.append(self.conv_branch[idx](x).mul(torch.reshape(coeff[idx], (-1, 1, 1, 1))))

        hr = hr_list[0]

        for idx in range(self.num_branch - 1):
            hr = torch.add(hr, hr_list[idx+1])

        return hr

