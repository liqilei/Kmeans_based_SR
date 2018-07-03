import os
import torch.utils.data as data
import h5py
import torch
import numpy as np

from data import common

class LRHRH5Dataset(data.Dataset):
    '''
    Read LR and HR image pair from H5 file.
    '''

    def name(self):
        return 'LRHRH5Dataset'

    def __init__(self, opt):
        super(LRHRH5Dataset, self).__init__()
        self.opt = opt
        self.file_data, self.file_coeff = None, None
        self.h5data_LR = None
        self.h5data_HR = None
        self.h5data_coeff = None

        # tie with h5 paths
        self.file_data = h5py.File(opt['dataroot_H5'], 'r', swmr=True)
        self.file_coeff = h5py.File(opt['coeffroot_H5'], 'r', swmr=True)

        # read dataset from h5 files
        self.h5data_HR = self.file_data['label']
        self.h5data_LR = self.file_data['data']
        self.h5data_coeff = [self.file_coeff[k] for k in list(self.file_coeff.keys())]

        assert self.h5data_HR.shape[0] > 0, 'Error: HR file are empty.'

        if (self.h5data_HR.shape[0] > 0) and (self.h5data_LR.shape[0] > 0):
            assert self.h5data_HR.shape[0] == self.h5data_LR.shape[0], \
                'HR and LR datasets have different number of images - {}, {}.'.format(\
                self.h5data_HR.shape[0], self.h5data_LR.shape[0])

    def __getitem__(self, index):
        patch_LR = self.h5data_LR[index]
        patch_HR = self.h5data_HR[index]
        coeff = [cof[index, :][0] for cof in self.h5data_coeff]
        if self.opt['phase'] == 'train':
            patch_LR, patch_HR = common.h5_augment([patch_LR, patch_HR], self.opt['use_flip'], self.opt['use_rot'])

        tensor_LR = torch.from_numpy(np.ascontiguousarray(patch_LR)).float()
        tensor_HR = torch.from_numpy(np.ascontiguousarray(patch_HR)).float()

        return {'LR': tensor_LR, 'HR': tensor_HR, 'coeff': coeff}

    def __len__(self):
        return self.h5data_HR.shape[0]

    """
    def __del__(self):
        self.file_data.close()
        self.file_coeff.close()
    """

