#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
# source /opt/conda/etc/profile.d/conda.sh
conda create --name py310 python=3.10.9 -y
conda activate py310
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt torch
pip install azure.storage.blob
pip install --ignore-installed --upgrade tensorflow==2.5.0
pip install pytorch_lightning
pip install pytorch-forecasting==0.10.3
# conda install ipykernel -y
ipython kernel install --user --name=py310
