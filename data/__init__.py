import torch.utils.data


def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt['phase']
    if phase == 'train':
        batch_size = dataset_opt['batch_size']
        #shuffle = True
        num_workers = dataset_opt['n_workers']
    else:
        batch_size = dataset_opt['batch_size']
        #shuffle = False
        num_workers = 1
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode'].upper()
    if mode == 'LR':
        from data.LR_dataset import LRDataset as D
    elif mode == 'LRHR':
        from data.LRHR_dataset import LRHRDataset as D
    elif mode == 'LRHR_LMDB':
        from data.LRHR_dataset import LRHRlmdbDatasetwithcoeff as D
    elif mode == 'LRHRSEG':
        from data.LRHR_seg_dataset import LRHRSegDataset as D
    elif mode == 'LRHR_H5':
        from data.LRHR_H5dataset import LRHRH5Dataset as D
    elif mode == 'LRHR_H5_M':
        from data.LRHR_H5dataset4memory import LRHRH5Dataset as D
    else:
        raise NotImplementedError("Dataset [%s] is not recognized." % mode)
    dataset = D(dataset_opt)
    print('Dataset [%s - %s] is created.' % (dataset.name(), dataset_opt['name']))
    return dataset
