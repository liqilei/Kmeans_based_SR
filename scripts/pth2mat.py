import torch
import argparse
import sys
import scipy.io
import os
import numpy as np
from collections import OrderedDict
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

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

def main():
    pth_path = args_.pytorch_model
    checkpoint = torch.load(pth_path)
    params = checkpoint['state_dict'].state_dict()

    mat_path = os.path.join(os.path.dirname(pth_path), os.path.basename(pth_path).split('.')[0] + '.mat')

    tmp = OrderedDict()
    for key in params:
        key_list = key.split('.')
        new_name = '_'.join(key_list[1:-2])
        tmp = params_to_dict(params, tmp, key, new_name)

    params = tmp

    print('Saving network to {}'.format(mat_path))

    scipy.io.savemat(mat_path, params, oned_as='column')

if __name__ == '__main__':
    main()
