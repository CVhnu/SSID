# SSID (Code is available)
This is the official PyTorch code for the paper.  【You Only Need Clear Images: Self-supervised Single Image Dehazing】

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



* The architecture of the image dehazing network.
<img src=https://github.com/CVhnu/SSID/blob/main/images/image_dehazing_network_paras.png >

* The generated pseudo labels (synthetic hazy images) by SSID.
 <img src=https://github.com/CVhnu/SSID/blob/main/images/pseudo%20labels.png >

* Comparison with the SOTA unsupervised image dehazing methods.
 <img src=https://github.com/CVhnu/SSID/blob/main/images/dehazed%20results.png >
