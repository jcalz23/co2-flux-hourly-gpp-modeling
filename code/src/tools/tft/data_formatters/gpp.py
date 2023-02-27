# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom formatting functions for GPP dataset.

Defines dataset specific column definitions and data transformations.
"""

import data_formatters.base
import libs_v2.utils as utils
import pandas as pd
import sklearn.preprocessing

DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class GppFormatter(data_formatters.base.GenericDataFormatter):
  """Defines and formats data for the GPP dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
      ('site_id', DataTypes.REAL_VALUED, InputTypes.ID),
      ('datetime', DataTypes.DATE, InputTypes.TIME),
      ('GPP_NT_VUT_REF', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('TA_ERA', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('SW_IN_ERA', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('LW_IN_ERA', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('VPD_ERA', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('P_ERA', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('PA_ERA', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('year', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('day', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('IGBP', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('koppen_main', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('koppen_sub', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('lat', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
      ('long', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
  ]

  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None

  def split_data(self, df, train_sites=None, valid_sites=None, test_sites=None):
    """Splits data frame into training-validation-test data frames.
    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      train_sites: training sites
      valid_sites: validation sites
      test_sites: testing sites

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')
<<<<<<< HEAD
    
=======

    if train_sites is None:
      train_sites = ['US-NR1', 'IT-Lav']

    if valid_sites is None:
      valid_sites = ['ES-LM2', 'US-AR1', 'US-GLE']
      # valid_sites = ["US-Vcp"]

    if test_sites is None:
      test_sites = [ 'US-Seg', 'CA-Cbo', 'FR-Lam']
      # test_sites = ["US-GLE"]

>>>>>>> b9757a19c3aae6d8ca97856d94e1cf0f3b6d2f5a
    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                column_definitions)

    print(f"Raw size: {df.shape} from {df[id_column].unique()}")
    train = df[~df[id_column].isin(valid_sites)]
    train = train[~train[id_column].isin(test_sites)]
    self.set_scalers(train, set_real=True)

    # Use all data for label encoding to handle labels not present in training.
    
    self.identifiers = list(df[id_column].unique())
    self.set_scalers(df, set_real=False)

    # Filter out identifiers not present in training (i.e. cold-started items).
<<<<<<< HEAD
    if not (train_sites is None):
      train = df[df[id_column].isin(train_sites)]
=======
    # train = df[df[id_column].isin(train_sites)]
>>>>>>> b9757a19c3aae6d8ca97856d94e1cf0f3b6d2f5a
    valid = df[df[id_column].isin(valid_sites)]
    test = df[df[id_column].isin(test_sites)]

    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df, set_real=True):
    """Calibrates scalers using the data supplied.

    Label encoding is applied to the entire dataset (i.e. including test),
    so that unseen labels can be handled at run-time.

    Args:
      df: Data to use to calibrate scalers.
      set_real: Whether to fit set real-valued or categorical scalers
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    if set_real:
      # Extract identifiers in case required
      self.identifiers = list(df[id_column].unique())
      print(f"IDs:{self.identifiers}")
      realnum_inputs = utils.extract_cols_from_data_type(
          DataTypes.REAL_VALUED, column_definitions,
          {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET})
      print(f"Real number input: {realnum_inputs}.")

      # Format real scalers
      data = df[realnum_inputs].values
      self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        
    else:
      # Format categorical scalers
      categorical_inputs = utils.extract_cols_from_data_type(
          DataTypes.CATEGORICAL, column_definitions,
          {InputTypes.ID, InputTypes.TIME})
      print(f"Categorical input: {categorical_inputs}")

      categorical_scalers = {}
      num_classes = []
      for col in categorical_inputs:
        # Set all to str so that we don't have mixed integer/string columns
        srs = df[col].apply(str)
        categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
            srs.values)

        num_classes.append(srs.nunique())

      # Set categorical scaler outputs
      self._cat_scalers = categorical_scalers
      self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()

    realnum_inputs = utils.extract_cols_from_data_type(
          DataTypes.REAL_VALUED, column_definitions,
          {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET})

    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Format real inputs
    output[realnum_inputs] = self._real_scalers.transform(df[realnum_inputs].values)

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    # No transformation needed for predictions
    output = predictions.copy()
    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': (7*24) + 6,
        'num_encoder_steps': 7*24,
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.1,
        'hidden_layer_size': 240,
        'learning_rate': 0.001,
        'minibatch_size': 128,
        'max_gradient_norm': 100.,
        'num_heads': 4,
        'stack_size': 1
    }

    return model_params

  def get_num_samples_for_calibration(self):
    """Gets the default number of training and validation samples.

    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.

    Returns:
      Tuple of (training samples, validation samples)
    """
    return 450000, 50000

  def get_column_definition(self):
    """"Formats column definition in order expected by the TFT.

    Modified for Favorita to match column order of original experiment.

    Returns:
      Favorita-specific column definition
    """

    column_definition = self._column_definition

    # Sanity checks first.
    # Ensure only one ID and time column exist
    def _check_single_column(input_type):

      length = len([tup for tup in column_definition if tup[2] == input_type])

      if length != 1:
        raise ValueError('Illegal number of inputs ({}) of type {}'.format(
            length, input_type))

    _check_single_column(InputTypes.ID)
    _check_single_column(InputTypes.TIME)

    identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
    time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]
    real_inputs = [
        tup for tup in column_definition if tup[1] == DataTypes.REAL_VALUED and
        tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

    col_definition_map = {tup[0]: tup for tup in column_definition}
    col_order = [
        'TA_ERA', 'SW_IN_ERA', 'LW_IN_ERA', 'VPD_ERA','P_ERA', 'PA_ERA',
        'year', 'month', 'day', 'hour',
        'IGBP', 'koppen_main', 'koppen_sub',
        'lat', 'long'
    ]
    categorical_inputs = [
        col_definition_map[k] for k in col_order if k in col_definition_map
    ]

    return identifier + time + real_inputs + categorical_inputs
