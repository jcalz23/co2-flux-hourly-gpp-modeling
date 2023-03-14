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
import torch.nn as nn
import hydroeval as he

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
from eval_functions import *

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pl.seed_everything(42)

root_dir =  MY_HOME_ABS_PATH
tmp_dir =  root_dir + os.sep + '.tmp'
raw_data_dir = tmp_dir
data_dir = root_dir + os.sep + 'data'
model_dir = data_dir + os.sep + 'models'
cred_dir = root_dir + os.sep + '.cred'
az_cred_file = cred_dir + os.sep + 'azblobcred.json'

container = "all-sites-data"
blob_name = "full_2010_2015_v_mvp_raw.parquet"
local_file = tmp_dir + os.sep + blob_name

# Download full data
data_df = None

if not (os.path.exists(local_file)):
    azStorageClient = AzStorageClient(az_cred_file)
    file_stream = azStorageClient.downloadBlob2Stream(container, blob_name)
    data_df = pd.read_parquet(file_stream, engine='pyarrow')
    data_df.to_parquet(local_file)
else:
    data_df = pd.read_parquet(local_file)
print(f"Data size: {data_df.shape}")

# Convert Dtypes
cat_cols = ["year", "month", "day", "hour", "MODIS_IGBP", "koppen_main", "koppen_sub", 
            "gap_flag_month", "gap_flag_hour"]
for col in cat_cols:
    data_df[col] = data_df[col].astype(str).astype("category")

data_df.dropna(inplace=True)
print(f"Data size: {data_df.shape}")
print(f"Data Columns: {data_df.columns}")
print(f"NA count: {data_df.isna().sum().sum()}")

SITE_SPLITS =[
  ['AR-SLu', 'AU-ASM', 'AU-Cpr', 'AU-Cum', 'AU-RDF', 'CA-TP3', 'CA-TPD', 'CN-Sw2',
    'DE-SfN', 'NL-Hor', 'US-Me6', 'US-Syv', 'US-WCr', 'US-AR2', 'US-Tw4', 'US-UMB', 
    'US-Vcp', 'CH-Cha', 'CZ-BK1', 'CZ-KrP', 'DE-Obe', 'ES-LJu', 'FI-Let', 'FR-Lam', 
    'IT-Lav', 'SE-Lnn'], 
  ['CZ-BK2', 'DE-Spw', 'FR-Pue', 'IT-CA3', 'IT-Noe', 'IT-Ro2', 'US-IB2', 'US-Myb',
    'US-SRM', 'CA-Ca3', 'US-CRT', 'US-Fmf', 'US-KFS', 'US-Prr', 'US-UMd', 'US-Wjs',
    'BE-Bra', 'BE-Lon', 'CH-Lae', 'CZ-RAJ', 'DE-HoH', 'DE-Kli', 'DE-RuR', 'IL-Yat', 
    'IT-Tor', 'SE-Htm'], 
  ['AR-Vir', 'AT-Neu', 'AU-DaS', 'AU-TTE', 'AU-Wom', 'CA-TP1', 'IT-CA1', 'IT-SRo',
    'US-WPT', 'US-Wkg', 'CA-Ca2', 'CA-Cbo', 'CA-TP4', 'US-ARM', 'US-Ro1', 'US-Rws',
    'US-SRG', 'US-Vcm', 'BE-Dor', 'BE-Vie', 'CZ-Stn', 'DE-Geb', 'ES-LM2', 'FR-Fon', 
    'SE-Ros', 'DE-Hte'],
  ['AU-DaP', 'AU-Emr', 'AU-Gin', 'AU-How', 'AU-Rig', 'US-GLE', 'US-NR1', 'US-Twt',
    'CA-Ca1', 'CA-Gro', 'US-AR1', 'US-Bar', 'US-Mpj', 'US-Ses', 'CH-Fru', 'CH-Oe2',
    'DE-Hai', 'DK-Sor', 'FI-Hyy', 'FR-Aur', 'FR-Hes', 'GF-Guy', 'IT-SR2', 'SE-Deg',
    'SE-Nor', 'NL-Loo'],
  ['AU-Stp', 'AU-Whr', 'CA-Oas', 'DE-Lnf', 'ES-Amo', 'FI-Sod', 'IT-CA2', 'US-Ton',
    'US-Var', 'US-Whs', 'US-Ho1', 'US-Oho', 'US-Seg', 'CH-Dav', 'CZ-Lnz', 'CZ-wet',
    'DE-Gri', 'DE-Tha', 'ES-LM1', 'FR-Bil', 'FR-FBn', 'IT-BCi', 'IT-MBo', 'IT-Ren',
    'RU-Fyo']
]

def get_splited_datasets(df, val_index, test_index): 
    train_sites, val_sites, test_sites = [], [], []
    for i, subset in enumerate(SITE_SPLITS):
        if i == val_index:
            val_sites = SITE_SPLITS[i]
        elif i == test_index:
            test_sites = SITE_SPLITS[i]
        else:
            train_sites += SITE_SPLITS[i]

    train_df = data_df.loc[data_df['site_id'].isin(train_sites), ].copy()
    val_df   = data_df.loc[data_df['site_id'].isin(val_sites), ].copy()

    if len(train_df['site_id'].unique()) != len(train_sites):
        print(f"Expected Train({len(train_sites)}), Actual Train({len(train_df['site_id'].unique())})")
        sites_missing = [s for s in train_sites if s not in train_df['site_id'].unique()]
        print(f'  missing: {sites_missing}')

    if len(val_df['site_id'].unique()) != len(val_sites):
        print(f"Expected Train({len(val_sites)}), Actual Train({len(val_df['site_id'].unique())})")
        sites_missing = [s for s in val_sites if s not in val_df['site_id'].unique()]
        print(f'  missing: {sites_missing}')

    if test_index is not None:
        test_df = data_df.loc[data_df['site_id'].isin(test_sites), ].copy()
        if len(test_df['site_id'].unique()) != len(test_sites):
            print(f"Expected Test({len(test_sites)}): {test_sites}")
            print(f"Actual Test({len(test_df['site_id'].unique())}): {test_df['site_id'].unique()}")
    else:
        test_df = None

    return (train_df, val_df, test_df)

def subset_data(train_df, val_df, test_df, subset_len):
    print(f'Subest length: {subset_len} timesteps for each sites')
    # Subset the time series within sites to save more time
    train_df = train_df.loc[train_df['timestep_idx_local'] < subset_len, ].copy()
    print(f"Subset num train timesteps: {len(train_df)}")
    val_df = val_df.loc[val_df['timestep_idx_local'] < subset_len, ].copy()
    print(f"Subset num val timesteps: {len(val_df)}")
    if test_df is not None:
        test_df = test_df.loc[test_df['timestep_idx_local'] < subset_len, ].copy()
        print(f"Subset num test timesteps: {len(test_df)}")

    return (train_df, val_df, test_df)

max_prediction_length = 1

def setup_train_val_tsdataset(train_df, val_df, min_encoder_len):
    # create training and validation TS dataset 
    training = TimeSeriesDataSet(
      train_df, # <------ no longer subsetting, option 1 split can use entire train site sequence
      time_idx="timestep_idx_global",
      target="GPP_NT_VUT_REF",
      group_ids=["site_id"],
      allow_missing_timesteps=True, # <---- turned on bc some rows are removed.
      min_encoder_length=min_encoder_len,
      max_encoder_length=min_encoder_len,
      min_prediction_length=max_prediction_length,
      max_prediction_length=max_prediction_length,
      static_categoricals=["MODIS_IGBP","koppen_main","koppen_sub", "gap_flag_month", "gap_flag_hour"],
      static_reals=[], #elevation lat long
      time_varying_known_categoricals=["year", "month", "day", "hour"],
      time_varying_known_reals=["timestep_idx_global", 
                                'TA_ERA', 'SW_IN_ERA', 'LW_IN_ERA', 'VPD_ERA', 'P_ERA', 'PA_ERA',
                                'EVI', 'NDVI', 'NIRv', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 
                                'BESS-PAR', 'BESS-PARdiff', 'BESS-RSDN', 'CSIF-SIFdaily', 'PET', 'Ts', 
                                'ESACCI-sm', 'NDWI', 'Percent_Snow', 'Fpar', 'Lai', 'LST_Day','LST_Night'],
      time_varying_unknown_categoricals=[], 
      time_varying_unknown_reals=["GPP_NT_VUT_REF"],
      target_normalizer=None, # <---- not sure if we need this given we scale in data pipeline.... but might want to change to scale at Group level?
      categorical_encoders={'MODIS_IGBP': NaNLabelEncoder(add_nan=True),
                            'koppen_main': NaNLabelEncoder(add_nan=True),
                            'koppen_sub': NaNLabelEncoder(add_nan=True),
                            'year': NaNLabelEncoder(add_nan=True), # temp for subset
                            'month': NaNLabelEncoder(add_nan=True), # temp for subset
                            'day': NaNLabelEncoder(add_nan=True), # temp for subset
                            },
      add_relative_time_idx=True,
      add_target_scales=False, # <------- turned off
      add_encoder_length=False, # <------- turned off
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)

    return (training, validation)

def get_eval_metrics(actuals, predictions):
    
    mae = (actuals - predictions).abs().mean()
    
    criterion = nn.MSELoss()
    rmse = torch.sqrt(criterion(actuals, predictions))

    nse = he.nse(actuals.reshape(-1).numpy(), predictions.reshape(-1).numpy())

    r2 = r2_score(actuals.reshape(-1).numpy(), predictions.reshape(-1).numpy())

    return { 'mae': mae.item(), 'rmse': rmse.item(), 'nse': nse, 'r2':r2}

# Define experiment
exp_name = "1yrtrain_baseline"

max_encoder_len =  24*7

VAL_INDEX = 3
TEST_INDEX = 4
SUBSET_LEN = 24*365 # 1 year
 
print(f"training timestemp length= {SUBSET_LEN}")

# Create model result directory
experiment_ts = datetime.now().strftime("%y%m%d_%H%M")
exp_fname = f"tft_model_{exp_name}_{experiment_ts}"
exp_model_dir = model_dir + os.sep + exp_fname
if not (os.path.exists(exp_model_dir)):
    os.makedirs(exp_model_dir)
print(f"Experiment logs saved to {exp_model_dir}.")

# split data
train_df, val_df, test_df = get_splited_datasets(data_df, VAL_INDEX, TEST_INDEX)
train_df, val_df, test_df = subset_data(train_df, val_df, test_df, SUBSET_LEN)
(training, validation) = setup_train_val_tsdataset(train_df, val_df, max_encoder_len)

# create dataloaders for model
# ref: https://pytorch-lightning.readthedocs.io/en/stable/guides/speed.html#dataloaders
batch_size = 64  # set this between 32 to 128
cpu_count = int(os.cpu_count()/2)
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=cpu_count, pin_memory=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=cpu_count, pin_memory=False)

# Create TFT model from dataset
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=128,  # most important hyperparameter apart from learning rate
    attention_head_size=4, # Set to up to 4 for large datasets
    dropout=0.2,           # Between 0.1 and 0.3 are good values
    hidden_continuous_size=64,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    logging_metrics=nn.ModuleList([MAE(), RMSE()]), #SMAPE(), #MAPE() #<---- added metrics to report in TensorBoard
    reduce_on_plateau_patience=4, # reduce learning rate if no improvement in validation loss after x epochs
    optimizer="adam"
)
print(f"  Number of parameters in network: {tft.size()/1e3:.1f}k")

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=4, mode="min",
                                    check_finite=True, verbose=False,)
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
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

local_model_path = exp_model_dir + os.sep + f"model.pth"
torch.save(best_tft.state_dict(), local_model_path)
print(f"Saved model to {local_model_path}")

# Print Model Eval on Validation Set
start = default_timer()
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
eval_time = default_timer() - start
print(f"Val eval time: {eval_time}")

eval_metric = get_eval_metrics(actuals, predictions)
print(eval_metric)

# Print Model Eval on Masked Validation Set
start = default_timer()
masked_rmse, masked_mae, masked_nse, masked_r2 = masked_eval_metrics(val_dataloader, tft)
eval_time = default_timer() - start
print(f"Masked Val eval time: {eval_time}")
print(f"masked_rmse: {masked_rmse}")
print(f"masked_mae: {masked_mae}")
print(f"masked_nse: {masked_nse}")
print(f"masked_r2: {masked_r2}")
