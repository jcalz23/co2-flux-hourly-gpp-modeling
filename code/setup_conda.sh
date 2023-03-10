#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh

conda create --name py310 python=3.10.9 -y
conda activate py310
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt torch
conda install ipykernel -y