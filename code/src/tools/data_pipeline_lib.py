import os
import sys
import pandas as pd
from io import BytesIO
from IPython.display import display
from CloudIO.AzStorageClient import AzStorageClient

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.sql.functions import col

from sklearn.preprocessing import LabelEncoder, StandardScaler

def get_min_max(df):
  return (df.min(), df.max())

def get_min_max_datetime(df):
  return (pd.to_datetime(df).min(), pd.to_datetime(df).max())

def is_leap_year(year):
  return year%4 == 0 ;

def data_cleanup(data_dir, site_id_file_df, target, target_qc, features):
  data_df = None
  # qc_flag_dtype = CategoricalDtype([0, 1, 2, 3], ordered=True)
  qc_flags_features = [s for s in features if "_QC" in s]

  # Iterate through each site:
  for i, r in site_id_file_df.iterrows():        
    if not r.filename or type(r.filename) != type(""):
      print(f'\nERROR: {r.site_id} is mssing hourly data.')
      continue

    # Get only `features` from file
    local_filename = data_dir + os.sep + r.filename
    site_df = pd.read_csv(local_filename, usecols = [target, target_qc] + features)
    site_df['datetime'] = pd.to_datetime(site_df['datetime'])
    site_df['date'] = pd.to_datetime(site_df['date'])
    site_df['minute'] = site_df['datetime'].dt.minute
    if len(qc_flags_features) != 0:
      site_df[qc_flags_features] = site_df[qc_flags_features].astype('int')
    site_df['site_id'] = r.site_id

    # Remove zero or negative SW
    site_df.drop(site_df[site_df['SW_IN_ERA'] <= 0].index, inplace = True)

    # Drop rows with NAs for Target Variable
    site_df.dropna(subset=[target], axis=0, inplace=True)

    # Drop rows with bad NEE_VUT_REF_QC (aka bad GPP records)
    site_df.drop(site_df[site_df[target_qc] == 3].index, inplace = True)
    site_df.drop([target_qc], axis=1, inplace=True)

    # Drop rows with any NA
    site_df.dropna(axis=0, inplace=True)

    print(f"{r.site_id}: {site_df.shape}")
    if type(data_df) == type(None):
      data_df = site_df
    else:
      data_df = pd.concat([data_df, site_df])
          
  return data_df

def merge_site_metadata(data_df, site_metadata_df):
  data_df = data_df.merge(site_metadata_df, how='left', left_on='site_id', right_on='site_id')
  return data_df

def check_and_drop_na(data_df):
  if data_df.isna().sum().sum() > 0:
    print("Data has NA.")
    display(pd.DataFrame(data_df.isna().sum()).T)
    data_df.dropna(axis=0, inplace=True)
  else:
    print("Datas has no NA.")

class PySparkMLDataTransformer:
  def __init__(self, spark_session, train_sites, test_sites, \
               data_file_path = None, data_df = None, ):
    
    self.spark_session = spark_session
    self.data_df = data_df 
    self.train_df = None 
    self.test_df = None 
    self.train_sites = train_sites
    self.test_sites = test_sites
    self.scaler = None

    if type(data_df) == type(None):
      if os.path.exists(data_file_path):
        self.data_df = self.spark_session.read.parquet(data_file_path)
        if '__index_level_0__' in self.data_df.columns:
          self.data_df = self.data_df.drop(*['__index_level_0__'])
      else:
        print(f"ERROR: {data_file_path} not found.")
    
    if 'date' in self.data_df.columns:
      self.data_df = self.data_df.drop(*['date'])
    print(f"Data loaded: {self.data_df.count()} rows x {len(self.data_df.columns)} columns.")
  
  def data_transform(self, categorical_cols, timestamp_col, target_col):
    self.categorical_cols = categorical_cols
    self.timestamp_col =  timestamp_col
    self.target_col =  target_col

    # One-Hot Encoding
    string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=[x + "_Index" for x in categorical_cols]) 
    self.data_df = string_indexer.fit(self.data_df).transform(self.data_df)
    one_hot_encoder  = OneHotEncoder(inputCols=string_indexer.getOutputCols(), outputCols=[x + "_OHE" for x in categorical_cols])
    self.data_df = one_hot_encoder.fit(self.data_df).transform(self.data_df)
    self.data_df = self.data_df.drop(*string_indexer.getOutputCols())

    print(f"Data size after encoding: {self.data_df.count()} rows x {len(self.data_df.columns)} columns.")
    self.data_df.show(5, False)

    # Get Features
    features = self.data_df.columns
    features.remove(target_col)
    features.remove(timestamp_col)
    features.remove('site_id')
    for f in categorical_cols:
      features.remove(f)
    print(f"Features({len(features)}): {features}")

    # Assemable Data
    assembler = VectorAssembler(inputCols=features, outputCol="vectorized_features")
    self.data_df = assembler.transform(self.data_df)
    print(f"Data size after assembling: {self.data_df.count()} rows x {len(self.data_df.columns)} columns.")
    self.data_df.show(5, False)

    # Split into train and test sets
    train_df = self.data_df.filter(col('site_id').isin(self.train_sites))
    test_df = self.data_df.filter(col('site_id').isin(self.test_sites))
    print(f"Train data size: {train_df.count()} rows x {len(train_df.columns)} columns.")
    print(f"Test data size: {test_df.count()} rows x {len(test_df.columns)} columns.")

    print("Train data peak:")
    train_df.show(5, False)
    print("Test data peak:")
    test_df.show(5, False)

    # Normalize data
    self.scaler = MinMaxScaler(inputCol='vectorized_features', outputCol='features').fit(train_df)
    train_df = self.scaler.transform(train_df)
    test_df = self.scaler.transform(test_df)

    train_df = train_df.drop(*['vectorized_features'])
    test_df = test_df.drop(*['vectorized_features'])
    print(f"Train data size: {train_df.count()} rows x {len(train_df.columns)} columns.")
    print(f"Test data size: {test_df.count()} rows x {len(test_df.columns)} columns.")

    self.train_df = train_df
    self.test_df = test_df
    return (train_df, test_df)
  
  def upload_train_test_to_azure(self, az_cred_file, container, train_blob_name, test_blob_name):
    # Initialize AzStorageClient 
    azStorageClient = AzStorageClient(az_cred_file)
    sessionkeys = azStorageClient.getSparkSessionKeys()
    self.spark_session.conf.set(sessionkeys[0],sessionkeys[1])

    # Upload train dataset
    train_blob_path = f"wasbs://{container}@{sessionkeys[2]}.blob.core.windows.net/{train_blob_name}"
    print(f"Uploading train dataset to {train_blob_path}...")
    self.train_df.write.format("parquet").mode("overwrite").save(train_blob_path)

    # Upload test dataset
    test_blob_path = f"wasbs://{container}@{sessionkeys[2]}.blob.core.windows.net/{test_blob_name}"
    print(f"Uploading test dataset to {test_blob_path}...")
    self.test_df.write.format("parquet").mode("overwrite").save(test_blob_path)

class TFTDataTransformer:
  def __init__(self, train_sites, test_sites, \
               data_file_path = None, data_df = None):
    
    self.data_df = data_df 
    self.train_df = None 
    self.test_df = None 
    self.train_sites = train_sites
    self.test_sites = test_sites
    self.scaler = None

    if type(data_df) == type(None):
      if os.path.exists(data_file_path):
        self.data_df = pd.read_parquet(data_file_path, engine='pyarrow')
      else:
        print(f"ERROR: {data_file_path} not found.")
    
    if 'date' in self.data_df.columns:
      self.data_df = self.data_df.drop(['date'], axis = 1)
    print(f"Data size: {self.data_df.shape}.")

  def get_test_train_raw(self):
    train_df = self.data_df[self.data_df['site_id'].isin(self.train_sites)]
    test_df  = self.data_df[self.data_df['site_id'].isin(self.test_sites)]
    print(f"Train data size: {train_df.shape}.")
    print(f"Test data size: {test_df.shape}.")

    self.train_df = train_df
    self.test_df = test_df
    return (train_df, test_df)

  def data_transform(self, categorical_cols, realNum_cols,\
                     backup_cols,\
                     timestamp_col, target_col):
    data_df = self.data_df
    # backup
    for f in backup_cols:
      data_df[f+'_name'] = data_df[f]
    print(f"Data size: {self.data_df.shape}.")

    # Label encode the categorical columns
    data_df[categorical_cols] = data_df[categorical_cols].apply(LabelEncoder().fit_transform)
    print(f"Data size after encoding: {data_df.shape}")
    display(data_df.head())

    # Get features
    features = data_df.columns.to_list()
    features.remove(target_col)
    features.remove(timestamp_col)

    for f in [x+"_name" for x in backup_cols]:
      features.remove(f)
    print(f"Features({len(features)}): {features}")
    
    # Split into train and test sets
    train_df = data_df[data_df['site_id_name'].isin(self.train_sites)]
    test_df  = data_df[data_df['site_id_name'].isin(self.test_sites)]
    print(f"Train data size: {train_df.shape}.")
    print(f"Test data size: {test_df.shape}.")

    # Normalize data
    for f in categorical_cols:
      features.remove(f)
    print(f"Normalizinf features ({len(features)}): {features}")

    scaler = StandardScaler().fit(train_df[features])
    train_df.loc[:,features] = scaler.transform(train_df[features])
    test_df.loc[:,features] = scaler.transform(test_df[features])
    print(f"Train data size: {train_df.shape}.")
    print(f"Test data size: {test_df.shape}.")

    self.train_df = train_df
    self.test_df = test_df
    return (train_df, test_df)

  def upload_train_test_to_azure(self, az_cred_file, container, train_blob_name, test_blob_name):
    # Initialize AzStorageClient 
    azStorageClient = AzStorageClient(az_cred_file)

    # Upload train dataset
    train_file = BytesIO()
    self.train_df.to_parquet(train_file, engine='pyarrow')
    train_file.seek(0)
    print(f"Uploading train dataset to {train_blob_name}...")
    azStorageClient.uploadBlob(container, train_blob_name, train_file, overwrite=True)

    # Upload test dataset
    test_file = BytesIO()
    self.test_df.to_parquet(test_file, engine='pyarrow')
    test_file.seek(0)
    print(f"Uploading test dataset to {test_blob_name}...")
    azStorageClient.uploadBlob(container, test_blob_name, test_file, overwrite=True)


