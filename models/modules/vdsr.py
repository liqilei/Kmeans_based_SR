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

class vdsr_k(nn.Module):
    def __init__(self, in_channels, out_channels, num_branch):
        super(vdsr_k, self).__init__()
        split_layer = 4
        self.num_branch = num_branch

        # self.conv = nn.Sequential()
        conv_in = B.ConvBlock(in_channels, 64, kernel_size=3, norm_type=None, act_type='relu',valid_padding=True, bias=False)
        conv_blocks = [B.ConvBlock(64, 64, kernel_size=3, norm_type=None, act_type='relu',valid_padding=True, bias=False) for _ in range(18 - split_layer)]
        self.conv = B.sequential(conv_in, *conv_blocks)

        self.conv_branch = nn.ModuleList()

        for _ in range(self.num_branch):
            sub_branch = [B.ConvBlock(64, 64, kernel_size=3, norm_type=None, act_type='relu',valid_padding=True, bias=False) for _ in range(split_layer)]
            conv_out = B.ConvBlock(64, out_channels, kernel_size=3, norm_type=None, act_type=None, valid_padding=True, bias=False)
            self.conv_branch.append(B.sequential(*sub_branch, conv_out))

    def forward(self, input, coeff):
        x = self.conv(input)

        res_list = []
        for idx in range(self.num_branch):
            res_list.append(self.conv_branch[idx](x).mul(torch.reshape(coeff[idx], (-1, 1, 1, 1))))

        res = res_list[0]

        for idx in range(self.num_branch - 1):
            res = torch.add(res, res_list[idx+1])

        hr = torch.add(res, input)
        return hr

