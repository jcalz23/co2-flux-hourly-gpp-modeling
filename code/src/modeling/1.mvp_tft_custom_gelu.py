MY_HOME_ABS_PATH = "/home/ec2-user/SageMaker/co2-flux-hourly-gpp-modeling"

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
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters, MetricsCallback
from pytorch_forecasting import BaseModel, MAE
from pytorch_forecasting.metrics.point import RMSE
from pytorch_forecasting.data.encoders import NaNLabelEncoder

# Add Custom modules
from temporal_fusion_transformer_gelu import TemporalFusionTransformer as TemporalFusionTransformer_GELU

import optuna
from optuna.integration import PyTorchLightningPruningCallback, TensorBoardCallback

from sklearn.metrics import r2_score
from timeit import default_timer
from datetime import datetime
import gc
import pickle

# Load locale custome modules
os.chdir(MY_HOME_ABS_PATH)
sys.path.append('./cred')
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
tmp_dir   = root_dir + os.sep + 'tmp'
model_dir = root_dir + os.sep + 'data' + os.sep + 'models'

container = "all-sites-data"
blob_name = "full_2010_2015_v_mvp_raw.parquet"
local_file = tmp_dir + os.sep + blob_name

data_df = get_raw_datasets(container, blob_name)

# Define experiment
exp_name = "1yrtrain_tuning"

# Experiment constants
VAL_INDEX  = 3
TEST_INDEX = 4
SUBSET_LEN = 24*365 # 1 year
ENCODER_LEN = 24*7
print(f"Training timestemp length = {SUBSET_LEN}.")

# Create model result directory
experiment_ts = datetime.now().strftime("%y%m%d_%H%M")
exp_fname = f"tft_model_{exp_name}_{experiment_ts}"
exp_model_dir = model_dir + os.sep + exp_fname
exp_model_dir = "/root/co2-flux-hourly-gpp-modeling/data/models/tft_custom_GELU_1yrtrain_230406_0900"
if not (os.path.exists(exp_model_dir)):
    os.makedirs(exp_model_dir)
print(f"Experiment logs saved to {exp_model_dir}.")

# save study results - also we can resume tuning at a later point in time
loaded_study = None
with open(exp_model_dir + os.sep + "study.pkl", "rb") as fin:
    loaded_study = pickle.load(fin)
if loaded_study is not None:
    print(f"Previous study has {len(loaded_study.trials) } trails.")

# setup datasets
train_df, val_df, _ = get_splited_datasets(data_df, VAL_INDEX, TEST_INDEX)
train_df, val_df, _ = subset_data(train_df, val_df, None, SUBSET_LEN)
training, validation, _ = setup_tsdataset(train_df, val_df, None, ENCODER_LEN)
############

# create dataloaders for model
# ref: https://pytorch-lightning.readthedocs.io/en/stable/guides/speed.html#dataloaders
batch_size = 64  # set this between 32 to 128
cpu_count = os.cpu_count()
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=cpu_count, pin_memory=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=cpu_count, pin_memory=True)

# Setup trainer callbacks
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, mode="min",
                                    check_finite=True, verbose=False,)
#########


# # create study
# study = custom_optimize_hyperparameters(
#     train_dataloader,
#     val_dataloader,
#     model_path=exp_model_dir,
#     n_trials=20,  # Defaults to 100.
#     max_epochs=100, # Defaults to 20.
#     gradient_clip_val_range=(0.01, 100.),  # Defaults to (0.01, 100.0)
#     hidden_size_range=(128, 320),           # Defaults to (16, 265)
#     hidden_continuous_size_range=(8, 64),  # Defaults to (8, 64).
#     attention_head_size_range=(4, 4),      # Defaults to (1, 4).
#     learning_rate_range=(1e-4, 1.0),       # Defaults to (1e-5, 1.0)
#     dropout_range=(0.1, 0.3),              # Defaults to (0.1, 0.3).
#     trainer_callbacks=[early_stop_callback],
#     reduce_on_plateau_patience=3,
#     use_learning_rate_finder=False,  # use Pytorch built-in solution to find ideal learning rate
#     loss=QuantileLoss(),
#     logging_metrics=nn.ModuleList([MAE(), RMSE()]), #SMAPE(), #MAPE() #<---- added metrics to report in TensorBoard
#     optimizer="adam",
#     log_dir = exp_model_dir,
#     study = loaded_study,
#     verbose = 2
# )

 # learning rate= 0.001, batchsize:128,  hiddent_layer:64, dropout:0.2, hiddent_countinues:32

# Create TFT model from dataset
tft = TemporalFusionTransformer_GELU.from_dataset(
    training,
    learning_rate=1e-5,
    hidden_size=32,  # most important hyperparameter apart from learning rate
    attention_head_size=4, # Set to up to 4 for large datasets
    dropout=0.2,           # Between 0.1 and 0.3 are good values
    hidden_continuous_size=32,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    logging_metrics=nn.ModuleList([MAE(), RMSE()]), #SMAPE(), #MAPE() #<---- added metrics to report in TensorBoard
    reduce_on_plateau_patience=4, # reduce learning rate if no improvement in validation loss after x epochs
    optimizer="adam"
)

print(f"  Number of parameters in network: {tft.size()/1e3:.1f}k")

lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger(exp_model_dir)  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=15,
    enable_model_summary=True,
    gradient_clip_val=0.1,
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
best_tft = TemporalFusionTransformer_GELU.load_from_checkpoint(best_model_path)

local_model_path = exp_model_dir + os.sep + f"model.pth"
torch.save(best_tft.state_dict(), local_model_path)
print(f"Saved model to {local_model_path}")
  
# save study results - also we can resume tuning at a later point in time
with open(exp_model_dir + os.sep + "study.pkl", "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print(study.best_trial.params)