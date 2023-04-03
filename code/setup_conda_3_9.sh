#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh

conda create --name py39 python=3.9 -y
conda activate py39
pip install --upgrade pip
pip install --no-cache-dir -r requirements_3_9.txt torch
conda install ipykernel -y