import torch.nn as nn
import torch

import models.modules.blocks as B

class srcnn_k(nn.Module):
    def __init__(self, in_channels, out_channels, num_branch):
        super(srcnn_k, self).__init__()
        self.num_branch = num_branch

        self.patch_extraction = B.ConvBlock(in_channels, 64, kernel_size=9, norm_type=None, act_type='relu',valid_padding=False, padding=0)
        self.mapping = B.ConvBlock(64, 32, kernel_size=1, norm_type=None, act_type='relu',valid_padding=False, padding=0)

        self.reconstruct = nn.ModuleList()

        for _ in range(num_branch):
            self.reconstruct.append(B.ConvBlock(32, out_channels, kernel_size=5, norm_type=None, act_type=None,valid_padding=False, padding=0))

    def forward(self, x, coeff):
        x = self.patch_extraction(x)
        x = self.mapping(x)

        hr_list = []
        for idx in range(self.num_branch):
            hr_list.append(self.reconstruct[idx](x).mul(torch.reshape(coeff[idx], (-1, 1, 1, 1))))

        hr = hr_list[0]

        for idx in range(self.num_branch - 1):
            hr = torch.add(hr, hr_list[idx+1])

        return hr
