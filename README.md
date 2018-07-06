This repository is developed by [@penguin1214](https://github.com/penguin1214) and [@Paper99](https://github.com/Paper99).

Code structure is inspired by [pytorch-cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [BasicSR](https://github.com/xinntao/BasicSR).

## Notice
- `num_workers` in **options/train/train.json** should fix at 1. If you want to use multiple `num_workers`. You should firstly install **hdf5 package with parallel settings**. You can find solutions from [here](http://docs.h5py.org/en/latest/build.html#building-against-parallel-hdf5)

## Running now
- SRCNN_K: x3: C1 C2

## TODO
- SRCNN_K: x3: C3 C4 C5 C6; x2: Best; x4: Best.
- VDSR_K: x3: C1 C2 C3 C4 C5 C6; x2: Best; x4: Best

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
    models.modules --> modules within networks </br>
    models.modules.blocks --> basic blocks </br>   
    models.modules.xxx_arch --> 特定网络的 building blocks 和 网络的完整结构 </br>
    models --> solver classes </br>
    models.base_model --> base class solver </br>
    models.networks --> utilitie </br>

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

