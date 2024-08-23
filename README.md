# Skin Segmentation
Image segmentation framework using PyTorch.

## Environment
- Windows 11 x64
- python 3.11.3
- cuda 12.1
- pytorch 2.2.0

The repo shows examples of FCN and UNet trained with [ECU dataset](https://ieeexplore.ieee.org/document/1359760). If you want to implement your work using this framework please refer to the blocks below to create your dataset and network.
## Network
### FCN
![FCN](https://github.com/HanXuMartin/SkinSegmentation/blob/main/docs/FCN.png)
### U-Net
The architecture of U-Net is quite similiar as it in the paper. What's different is that I do not apply center crop operation.

## Get Started
1. Create a conda environment and install the package mentioned above. Some commands are writen in the requirements.txt
2. cd to the directory where you put the code.

### Train
```
python train.py --config configs\unet_cfg.py
```
### Test
```
python test.py --config configs\unet_cfg.py --ckp workdir\Unet-10\Unet-epoch_25.pth --data E:\ECU\images\001\im00009.jpg
```
Change your own path to the config file, checkpoint path and image path.<br>
checkpoint link：https://pan.baidu.com/s/1B5bJCJEMTQw6x9o7RR2T4Q?pwd=lgew <br>
code：lgew

## Create your own project
### Custom your dataset
If you want to train the model on your own dataset, please look at [PyTorch document](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files) or the example of [ECU dataset](https://github.com/HanXuMartin/SkinSegmentation/blob/main/segmentation/datasets/ecudataset.py). 
### Custom your data augmentation
### Custom your network

Reminder: Do not forget to register your components (dataset, transforms, network and etc.) in each __init__.py file after you write down your code.
## TODO
1. Color space augmentation is not applied in this repo.
2. The training stage need to be finetune.


