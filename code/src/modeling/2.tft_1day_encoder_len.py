MY_HOME_ABS_PATH = "/root/co2-flux-hourly-gpp-modeling/"

import os
os.chdir(MY_HOME_ABS_PATH)

import sys
import warnings
warnings.filterwarnings("ignore")
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting import BaseModel, MAE
from pytorch_forecasting.metrics.point import RMSE
from pytorch_forecasting.data.encoders import NaNLabelEncoder

from sklearn.metrics import r2_score
from timeit import default_timer
from datetime import datetime
import gc
import pickle

# Load locale custome modules
os.chdir(MY_HOME_ABS_PATH)
sys.path.append('./.cred')
sys.path.append('./code/src/tools')
sys.path.append(os.path.abspath("./code/src/tools"))
  
from CloudIO.AzStorageClient import AzStorageClient
from data_pipeline_lib import *
from model_pipeline_lib import *

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pl.seed_everything(42)

# Download full data
root_dir  = MY_HOME_ABS_PATH
tmp_dir   = root_dir + os.sep + '.tmp'
model_dir = root_dir + os.sep + 'data' + os.sep + 'models'

container = "all-sites-data"
blob_name = "full_2010_2015_v_mvp_raw.parquet"
local_file = tmp_dir + os.sep + blob_name

data_df = get_raw_datasets(container, blob_name)

# Define experiment
exp_name = "1YrTrain_1DEncode"

# Experiment constants
VAL_INDEX  = 3
TEST_INDEX = 4
SUBSET_LEN = 24*365 # 1 year
ENCODER_LEN = 24    # 1 day
print(f"Training timestemp length = {SUBSET_LEN}.")

# Create model result directory
experiment_ts = datetime.now().strftime("%y%m%d_%H%M")
exp_fname = f"tft_model_{exp_name}_{experiment_ts}"
exp_model_dir = model_dir + os.sep + exp_fname
if not (os.path.exists(exp_model_dir)):
    os.makedirs(exp_model_dir)
print(f"Experiment logs saved to {exp_model_dir}.")

# setup datasets data
train_df, val_df, _ = get_splited_datasets(data_df, VAL_INDEX, TEST_INDEX)
train_df, val_df, _ = subset_data(train_df, val_df, None, SUBSET_LEN)
training, validation, _ = setup_tsdataset(train_df, val_df, None, ENCODER_LEN)

# create dataloaders for model
# ref: https://pytorch-lightning.readthedocs.io/en/stable/guides/speed.html#dataloaders
batch_size = 128  # set this between 32 to 128
cpu_count = os.cpu_count()
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=cpu_count, pin_memory=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=cpu_count, pin_memory=False)

# Create TFT model from dataset
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.0001414321246300602,
    hidden_size=136,  # most important hyperparameter apart from learning rate
    attention_head_size=1, # Set to up to 4 for large datasets
    dropout=0.1239528685401419, # Between 0.1 and 0.3 are good values
    hidden_continuous_size=40,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    logging_metrics=nn.ModuleList([MAE(), RMSE()]), #SMAPE(), #MAPE() #<---- added metrics to report in TensorBoard
    reduce_on_plateau_patience=3, # reduce learning rate if no improvement in validation loss after x epochs
    optimizer="adam"
)
print(f"  Number of parameters in network: {tft.size()/1e3:.1f}k")

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=4, mode="min",
                                    check_finite=True, verbose=False,)
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger(exp_model_dir)  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=20,
    enable_model_summary=True,
    gradient_clip_val=0.1418798159870465,
    fast_dev_run=False,  # comment in to check that network or dataset has no serious bugs
    accelerator='gpu',
    devices=1,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

start = default_timer()
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
train_time = default_timer() - start
print(f"Training time: {train_time}")

# load the best model according to the validation loss
best_model_path = trainer.checkpoint_callback.best_model_path
print(" Best model path: " + best_model_path)
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

local_model_path = exp_model_dir + os.sep + f"model.pth"
torch.save(best_tft.state_dict(), local_model_path)
print(f"Saved model to {local_model_path}")