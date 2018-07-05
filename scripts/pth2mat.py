import torch
import argparse
import sys
import scipy.io
import os
import numpy as np
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Convert model from PyTorch to MATLAB.')
parser.add_argument('-pytorch_model', type=str, help='It should be a path to its .pth file')

args_ = parser.parse_args(sys.argv[1:])

def tensor_to_array(tensor):
    """Convert a PyTorch Tensor to a numpy array.

    (changes the order of dimensions to [width, height, channels, instance])
    """
    dims = len(tensor.size())
    raw = tensor.cpu().numpy()
    if dims == 4:
        return raw.transpose((3,2,1,0))
    elif dims == 3:
        return np.expand_dims(raw.transpose(), axis=3)
    else:
        return raw.transpose()

def params_to_dict(params, dict, key, layer_name):
    if layer_name not in list(dict.keys()):
        dict[layer_name] = OrderedDict()
    if 'weight' in key:
        dict[layer_name]['weight'] = tensor_to_array(params[key])
    else:
        dict[layer_name]['bias'] = tensor_to_array(params[key])

    return dict

checkpoint = torch.load(args_.pytorch_model)
params = checkpoint['state_dict'].state_dict()

mat_savepath = os.path.join(os.path.dirname(args_.pytorch_model), 'best_epoch.mat')

tmp = OrderedDict()
for key in params:
    key_list = key.split('.')
    new_name = '_'.join(key_list[1:-2])
    tmp = params_to_dict(params, tmp, key, new_name)

params = tmp

print('Saving network to {}'.format(mat_savepath))

scipy.io.savemat(mat_savepath, params, oned_as='column')