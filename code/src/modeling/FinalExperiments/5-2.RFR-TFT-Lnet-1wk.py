MY_HOME_ABS_PATH = "/root/co2-flux-hourly-gpp-modeling/"

import os
os.chdir(MY_HOME_ABS_PATH)

import sys
import warnings

#os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
warnings.filterwarnings("ignore")
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
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
blob_name = "hybrid-2010-2015-raw-v2.parquet"
local_file = tmp_dir + os.sep + blob_name

data_df = get_raw_datasets(container, blob_name)
data_df.rename(columns={"rfr_pred_gpp": "estimated_gpp"}, inplace=True)

# Define experiment
exp_name = "5YrTrain_1WkEncode_RFRv2_slim_Lnet"

# Experiment constants
VAL_INDEX  = 3
TEST_INDEX = 4
SUBSET_LEN = 24*365*5 # 5 year
ENCODER_LEN = 24*7
print(f"Training timestemp length = {SUBSET_LEN}.")

def setup_tsdataset_treeft_slim(train_df, val_df, test_df, min_encoder_len):
    # create training and validation TS dataset 
    training = TimeSeriesDataSet(
      train_df, 
      time_idx="timestep_idx_global",
      target="GPP_NT_VUT_REF",
      group_ids=["site_id"],
      allow_missing_timesteps=False, # <---- turned on bc some rows are removed.
      min_encoder_length=min_encoder_len,
      max_encoder_length=min_encoder_len,
      min_prediction_length=1,
      max_prediction_length=1,
      static_categoricals=["MODIS_IGBP","koppen_main"],
      static_reals=[],
      time_varying_known_categoricals=["month", "hour"],
      time_varying_known_reals=['TA_ERA', 'SW_IN_ERA', 'LW_IN_ERA', 'VPD_ERA', 'P_ERA', 'PA_ERA',
                                'NDVI', 'b2', 'b4', 'b6', 'b7', 
                                'BESS-PARdiff', 'CSIF-SIFdaily', 'ESACCI-sm', 'Percent_Snow', 
                                'Lai', 'LST_Day','LST_Night'],
      time_varying_unknown_categoricals=["gap_flag_month", "gap_flag_hour"], 
      time_varying_unknown_reals=['estimated_gpp'],
      target_normalizer=None,
      categorical_encoders={'MODIS_IGBP': NaNLabelEncoder(add_nan=True),
                            'koppen_main': NaNLabelEncoder(add_nan=True),
                            'koppen_sub': NaNLabelEncoder(add_nan=True),
                            },
      add_relative_time_idx=True,
      add_target_scales=False, # <------- turned off
      add_encoder_length=False, # <------- turned off
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
    
    if test_df is not None:
        testing = TimeSeriesDataSet.from_dataset(training, test_df, predict=False, stop_randomization=True)
    else:
        testing = None

    return (training, validation, testing)

# Create model result directory
experiment_ts = datetime.now().strftime("%y%m%d_%H%M")
exp_fname = f"treeft_{exp_name}_{experiment_ts}"
exp_model_dir = model_dir + os.sep + exp_fname
if not (os.path.exists(exp_model_dir)):
    os.makedirs(exp_model_dir)
print(f"Experiment logs saved to {exp_model_dir}.")

# setup datasets data
train_df, val_df, _ = get_splited_datasets(data_df, VAL_INDEX, TEST_INDEX)
train_df, val_df, _ = subset_data(train_df, val_df, None, SUBSET_LEN)
training, validation, _ = setup_tsdataset_treeft_slim(train_df, val_df, None, ENCODER_LEN)

# create dataloaders for model
# ref: https://pytorch-lightning.readthedocs.io/en/stable/guides/speed.html#dataloaders
batch_size = 64
cpu_count = os.cpu_count()
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=cpu_count, pin_memory=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=cpu_count, pin_memory=False)

# Create TFT model from dataset
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-5,
    hidden_size=128,  # most important hyperparameter apart from learning rate
    attention_head_size=4, # Set to up to 4 for large datasets
    dropout=0.2, # Between 0.1 and 0.3 are good values
    hidden_continuous_size=128,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    logging_metrics=nn.ModuleList([MAE(), RMSE()]), #SMAPE(), #MAPE() #<---- added metrics to report in TensorBoard
    reduce_on_plateau_patience=2, # reduce learning rate if no improvement in validation loss after x epochs
    optimizer="adam"
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, mode="min",
                                    check_finite=True, verbose=False,)
checkpoint_callback = ModelCheckpoint(dirpath=exp_model_dir,  save_top_k=3, monitor="val_loss",  mode="min") # save model objects for top k epoch val loss
lr_logger = LearningRateMonitor(logging_interval='epoch')  # log the learning rate
logger = TensorBoardLogger(exp_model_dir)  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=8,
    enable_model_summary=True,
    fast_dev_run=False,  # comment in to check that network or dataset has no serious bugs
    accelerator='gpu',
    devices="auto",
    callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
    logger=logger,
    strategy="ddp",
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
print("Best model path: " + best_model_path)
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

local_model_path = exp_model_dir + os.sep + f"model.pth"
torch.save(best_tft.state_dict(), local_model_path)
print(f"Saved model to {local_model_path}")