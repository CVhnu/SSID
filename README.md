# SSID (Code is available)
This is the official PyTorch code for the paper.  
【You Only Need Clear Images: Self-supervised Single Image Dehazing】

# Dependencies and Installation
Ubuntu >= 18.04  
CUDA >= 11.0  
Python 3.6  
Pytorch 1.2.0  

\# git clone this repository
git clone https://github.com/CVhnu/SSID.git  
cd SSID

\# create new anaconda env
conda create -n SSID python=3.8  
conda activate SSID  

# Get Started
1. The pretrained checkpoints in the Code/pretrained/SSID.pt.
2. Preparing data for training

# Quick test
Run demos to process the images in dir ./examples/input/ by following commands:  
python test.py  --dataroot  ./examples/input/ --output ./examples/output

# Train SSID
Step 1: collect clear images in the ./examples/input  
Step 2: Train our SSID

# The architecture of the image dehazing network.
<img src=https://github.com/CVhnu/SSID/blob/main/images/image_dehazing_network_paras.png >

# The generated pseudo labels (synthetic hazy images) by SSID.
 <img src=https://github.com/CVhnu/SSID/blob/main/images/pseudo%20labels.png >
 
<!-- * Comparison with the SOTA unsupervised image dehazing methods.
 <img src=https://github.com/CVhnu/SSID/blob/main/images/dehazed%20results.png > -->
 
# Citation
If you find our repo useful for your research, please cite us:

# License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only. Any commercial use should get formal permission first.

# Acknowledgement

This repository is maintained by [Jiyou Chen](https://scholar.google.com.hk/citations?user=BjgoH4cAAAAJ&hl=zh-CN&oi=ao).
