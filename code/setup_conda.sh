#!/usr/bin/env bash
set -eux

export REPOSITORY_URL="https://github.com/jcalz23/co2-flux-hourly-gpp-modeling.git"
export REPOSITORY_NAME="co2-flux-hourly-gpp-modeling"

if [ -d "$REPOSITORY_NAME" ]; then
  echo "Directory 'git_directory' already exists"
else
  git clone $REPOSITORY_URL
fi

cd $REPOSITORY_NAME/code
conda create --name py310 python=3.10.9 -y
conda activate py310
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt torch