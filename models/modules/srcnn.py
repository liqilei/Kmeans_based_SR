import torch.nn as nn

import models.modules.blocks as B

class srcnn(nn.Module):
    def __init__(self, in_channels, out_channels, num_features1=64, num_features2=32):
        super(srcnn, self).__init__()
        patch_extraction = B.ConvBlock(in_channels, num_features1, kernel_size=9, norm_type=None, act_type='relu',valid_padding=False,padding=0)
        mapping = B.ConvBlock(num_features1, num_features2, kernel_size=1, norm_type=None, act_type='relu',valid_padding=False,padding=0)
        recon = B.ConvBlock(num_features2, out_channels, kernel_size=1, norm_type=None, act_type=None,valid_padding=False,padding=0)

        self.network = B.sequential(patch_extraction, mapping, recon)

    def forward(self, x):
        return self.network(x)
