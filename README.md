This repository is developed by [@penguin1214](https://github.com/penguin1214) and [@Paper99](https://github.com/Paper99).

Code structure is inspired by [pytorch-cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [BasicSR](https://github.com/xinntao/BasicSR).

## Notice
- `num_workers` in **options/train/train.json** should fix at 1. If you want to use multiple `num_workers`. You should firstly install **hdf5 package with parallel settings**. You can find solutions in [here](http://docs.h5py.org/en/latest/build.html#building-against-parallel-hdf5)

## TODO
- 跑通VDSR
- Kmeans
- test code in matlab

## Requirements
- Python3
- Pytorch
- MatConvNet(It is included in `scripts\Test`, you should complie it firstly)

## Code architecture:
-- data (对数据进行操作)  

-- datasets (存放数据集)  
-- DIV2K  
-- VOC2012

-- models (算法模块)  
    models.modules --> modules within networks
    models.modules.blocks --> basic blocks    
    models.modules.xxx_arch --> 特定网络的 building blocks 和 网络的完整结构
    models --> solver classes
    models.base_model --> base class solver
    models.networks --> utilitie

-- options (输入参数)  
-- train (训练参数)  
-- test (测试参数)  

-- experiments (存储结果)  
-- train (训练结果)  
-- test （测试结果）

-- scripts (运行脚本)  

-- utils (其它)  

-- tmp_deprecated (暂时弃用的文件)  

-- train.py（训练代码）  

