#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist

K_means = sio.loadmat('/home/qilei/data/H5Data/h5withcoeff/C2/kmeans')
# print(mat_contents)
data = sio.loadmat('/home/qilei/data/H5Data/h5withcoeff/C2/data')
dist1 = cdist(data['data_cluster'], K_means["C"][0,:].reshape(1,1681))
# print(dist1)
dist2 = cdist(data['data_cluster'], K_means["C"][1,:].reshape(1,1681))
dist = np.concatenate((dist1, dist2), axis=1)
print(dist.shape)