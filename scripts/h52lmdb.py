import os
import h5py
import lmdb
import numpy as np
import pickle

# read h5 file
file_path = '/home/qilei/data/H5Data/h5withcoeff/291/train.h5'
hf = h5py.File(file_path)
print('keys in hf are:', )
hf.visit(print)
data = hf['data'].value
label = hf['label'].value
#data = np.transpose(data, (0, 2, 3, 1))
#label = np.transpose(label, (0, 2, 3, 1))
print('shape of data is', data.shape)
print('shape of label is', label.shape)
print('Finish reading image data from h5 file\n')

# create lr lmdb file

lmdb_save_path = '/home/qilei/data/lmdb/291withcoeff/LR_291.lmdb'  # must end with .lmdb
map_size = data.nbytes * 10
env = lmdb.open(lmdb_save_path, map_size=map_size)
print('Write image data to LR_lmdb file')
with env.begin(write=True) as txn:
    for cnt in range(0,data.shape[0]):
        if cnt % 10000 == 0:
            print('processing {}'.format(cnt))
        key = str(cnt).encode('ascii')
        LR = data[cnt,:,:,:]
        C, H, W = LR.shape
        meta_key = (str(cnt) + '.meta').encode('ascii')
        meta = '{:d}, {:d}, {:d}'.format(C, H, W)
        txn.put(key=key,value=LR)
        txn.put(meta_key, meta.encode('ascii'))
print('Finish writing to LR_lmdb')

keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    print('creating LR_lmdb keys cache: {}'.format(keys_cache_file))
    keys = [key.decode('ascii') for key, _ in txn.cursor()]
    pickle.dump(keys, open(keys_cache_file, "wb"))
print('Finish creating LR_lmdb keys.')

# create HR lmdb file

lmdb_save_path = '/home/qilei/data/lmdb/291withcoeff/HR_291.lmdb'  # must end with .lmdb
map_size = data.nbytes * 10
env = lmdb.open(lmdb_save_path, map_size=map_size)
print('Write image data to HR_lmdb file')
with env.begin(write=True) as txn:
    for cnt in range(0,data.shape[0]):
        if cnt % 10000 == 0:
            print('processing {}'.format(cnt))
        key = str(cnt).encode('ascii')
        HR = label[cnt,:,:,:]
        C, H, W = HR.shape
        meta_key = (str(cnt) + '.meta').encode('ascii')
        meta = '{:d}, {:d}, {:d}'.format(C, H, W)
        txn.put(key=key,value=HR)
        txn.put(meta_key, meta.encode('ascii'))
print('Finish writing to HR_lmdb')

keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    print('creating HR_lmdb keys cache: {}'.format(keys_cache_file))
    keys = [key.decode('ascii') for key, _ in txn.cursor()]
    pickle.dump(keys, open(keys_cache_file, "wb"))
print('Finish creating HR_lmdb keys.')

# create coeff lmdb file
coeff_path = '/home/qilei/data/H5Data/h5withcoeff/291/train_coeff.h5'
hf_coeff = h5py.File(coeff_path)
print('keys in hf are:', )
hf_coeff.visit(print)
coeff1 = hf_coeff['coeff1'].value
coeff2 = hf_coeff['coeff2'].value
coeff = np.concatenate((coeff1, coeff2),axis=1)
print('shape of coeff is', coeff.shape)
print('test {}, {}'.format(coeff[0,0]+coeff[0,1],coeff[1,0]+coeff[1,1]))
print('Finish reading coeff data from h5 file\n')


coeff_save_path = '/home/qilei/data/lmdb/291withcoeff/coeff_set5.lmdb'  # must end with .lmdb
map_size = coeff.nbytes * 100

env = lmdb.open(coeff_save_path, map_size=map_size)
print('Write coeff data to coeff_lmdb file')
with env.begin(write=True) as txn:
    for cnt in range(0,len(coeff)):
        if cnt % 10000 == 0:
            print('processing {}'.format(cnt))
        key = str(cnt).encode('ascii')
        coeff_data = coeff[cnt,:]
        row, col = 1, coeff_data.shape[0]
        meta_key = (str(cnt) + '.meta').encode('ascii')
        meta = '{:d}, {:d}'.format(row, col)
        txn.put(key=key,value=coeff_data)
        txn.put(meta_key, meta.encode('ascii'))
print('Finish writing to coeff_lmdb')


keys_cache_file = os.path.join(coeff_save_path, '_keys_cache.p')
env = lmdb.open(coeff_save_path, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    print('creating coeff_lmdb keys cache: {}'.format(keys_cache_file))
    keys = [key.decode('ascii') for key, _ in txn.cursor()]
    pickle.dump(keys, open(keys_cache_file, "wb"))
print('Finish creating coeff_lmdb keys.')
