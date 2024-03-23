# Dehazing_R

This is the official PyTorch implementation of our dehazing method for NTIRE 2024 Dense and NonHomogeneous Dehazing Challenge
# Environment:

CUDA Version: 11.0

Python 3.8

# Dependencies:

torch==1.10.0
torchvision==0.9.0
NVIDIA GPU and CUDA
pytorch_lightning=2.0.0
timm=0.6.12
mmcv-full==1.5.0
mmdet==2.28.1
opencv-python 
termcolor 
yacs 
yyaml 
scipy
einops
DCNv4


# Our saved Model
Download [our saved model for NTIRE 2024 Dense and NonHomogeneous Dehazing Challenge](https://drive.google.com/file/d/1HfaVBTkBSU8sScFW30cv9XiOuXAmaByW/view?usp=sharing) and unzip it into the folder ./weights to reproduce our test result.

# How to reproduce our test result
Download above saved models

Run test.py and find results in the folder ./results. Please note new generated results may have subtle varations with our submitted test results due to the device difference, please feel free to choose our submitted results or your generated images for user study.






