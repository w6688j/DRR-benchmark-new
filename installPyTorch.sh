#!/bin/bash

# Download Anaconda3-5.3.0-Linux-x86_64.sh
if [ ! -f "/root/Anaconda3-5.3.0-Linux-x86_64.sh" ];then
    echo "===================== Downloading the Anaconda3-5.3.0-Linux-x86_64.sh ====================="
    wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh
fi

# Run Anaconda3-5.3.0-Linux-x86_64.sh
echo "===================== Installing the Anaconda3-5.3.0 ====================="
sh Anaconda3-5.3.0-Linux-x86_64.sh
if [ $? -ne 0 ];then
    rm -rf /home/app/anaconda3
    exit $?
fi

source ~/.bashrc

# Show Conda Version
conda -V
if [ $? -ne 0 ];then
    exit $?
fi

# Set up Tsinghua mirror image
# echo "===================== Set up Tsinghua mirror image ====================="
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

conda config --set show_channel_urls yes

# Create Anaconda Environments
echo "===================== Create Anaconda Environments ====================="
conda create --name PyTorch python=3.7

# Activate the PyTorch Environment
source activate PyTorch

# Show Python Version
python -V

# Installing PyTorch0.4.1
echo "===================== Installing PyTorch0.4.1 ====================="
conda install pytorch -c pytorch
if [ $? -ne 0 ];then
    exit $?
fi
pip install torchvision
if [ $? -ne 0 ];then
    exit $?
fi
