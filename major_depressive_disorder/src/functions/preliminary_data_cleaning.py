import logging
import json
import pandas as pd
import numpy as np
from dateutil import parser
from src.helper_functions.json_helpers import extract_json_configs
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
logging.basicConfig(level=logging.INFO)

class PreliminaryData():
  
  def __init__(self, data_frame):
    self.data_frame = data_frame
    self.col_missing = [col for col in self.data_frame.columns if len(self.data_frame[self.data_frame[col].isna()]) > 0]
    self.missing_data_config = extract_json_configs('src/configs/missing_data_config.json')
    self.data_type_config = extract_json_configs('src/configs/data_type_config.json')
    self.combined_data_type_config = {**self.data_type_config['data']['clinical'], **self.data_type_config['data']['demographics'], **self.data_type_config['data']['bill_id'], **self.data_type_config['data']['bill_amount']}

  # Main Function

  def prelim_data_cleaning(self, method_continuous='remove', method_categorical='impute_mode', iterative_imputation=False):
    data_frame = self.data_frame
    data_frame = self._handle_date_time(data_frame)
    data_frame = self._combine_same_classes(data_frame)
    if iterative_imputation:
      logging.info('Method of Handling Missing Data: Impute Null Values with Random Forest')
      data_frame = self._impute_rf(data_frame)
    else:
      data_frame = self._handle_missing_data(data_frame, method_continuous, method_categorical)
    data_frame_clean = self._enforce_data_types(data_frame)
    return data_frame_clean
  
  # Sub-Modular Functions

  def harmonize_date(self, date_str):
    return parser.parse(date_str, dayfirst=True).strftime('%Y-%m-%d')

  def _handle_date_time(self, data_frame):
    date_columns = [col for col in data_frame.columns if self.combined_data_type_config[col] == 'date']
    for col in date_columns:
      data_frame[col] = data_frame[col].apply(self.harmonize_date)
      data_frame[col] = pd.to_datetime(data_frame[col], format='%Y-%m-%d')
      # pd.to_datetime(data_frame[col], format='%d/%m/%y', errors='coerce', dayfirst=True)
      # data_frame[col].apply(lambda col: col.replace('-', '/'))
      # data_frame[col] = pd.to_datetime(data_frame[col], format='%d/%m/%y', errors='coerce')
      # data_frame[col].dt.strftime('%d-%m-%y')
    return data_frame

  def _combine_same_classes(self, data_frame):
    general_binary_map = self.general_binary_class_mapper()
    gender_map = self.gender_class_mapper()
    race_map = self.race_class_mapper()
    resident_status_map = self.resident_status_mapper()
    for col in data_frame.columns:
      if col == 'medical_history_hbp':
        data_frame['medical_history_hbp'] = data_frame['medical_history_hbp'].replace(general_binary_map)
      elif col == 'gender':
        data_frame['gender'] = data_frame['gender'].replace(gender_map)
      elif col == 'race':
        data_frame['race'] = data_frame['race'].replace(race_map)
      elif col == 'resident_status':
        data_frame['resident_status'] = data_frame['resident_status'].replace(resident_status_map)
      elif self.combined_data_type_config[col] == 'nominal':
        data_frame[col] = data_frame[col].replace(general_binary_map)
    return data_frame 

  def _enforce_data_types(self, data_frame):
    for col in data_frame.columns:
      if self.combined_data_type_config[col] in ['nominal', 'ordinal']:
        data_frame[col] = data_frame[col].astype(int)
      elif self.combined_data_type_config[col] == 'continuous':
        data_frame[col] = data_frame[col].astype(float)
      elif self.combined_data_type_config[col] == 'date':
        # data_frame[col] = pd.to_datetime(data_frame[col])
        if len([i for i in data_frame[col].dt.day if i > 31]) > 0:
          logging.info(f'Error in days detected: {col}')
        if len([i for i in data_frame[col].dt.month if i > 12]) > 0:
          logging.info(f'Error in days detected: {col}')
        if len([i for i in data_frame[col].dt.year if i > 2024]) > 0:
          logging.info(f'Error in days detected: {col}')
    return data_frame

  def _handle_missing_data(self, data_frame, method_continuous, method_categorical):
    for col in self.col_missing:
      if self.combined_data_type_config[col] == 'continuous':
        if method_continuous == 'remove':
          logging.info(f'Method of Handling Missing Data: Remove Null Values in {col}')
          data_frame = self._remove_null(data_frame,col)
        elif method_continuous == 'impute_mean':
          logging.info(f'Method of Handling Missing Data: Impute Null Values with Mean in {col}')
          data_frame = self._impute_mean(data_frame,col)
        elif method_continuous == 'auto':
          logging.info('Method of Handling Missing Data: Auto')
          logging.info(f'Null values will be remove at {self.missing_data_config["remove"]}% threshold and impute with mean at {self.missing_data_config["replace_mean"]}% threshold, if exceeds, column will be dropped for {col}')
          if len(data_frame[data_frame[col]])/len(data_frame)*100 < self.missing_data_config["remove"]:
            data_frame = self._remove_null(data_frame,col)
          elif len(data_frame[data_frame[col]])/len(data_frame)*100 < self.missing_data_config["replace_mean"]:
            data_frame = self._impute_mean(data_frame,col)
          else:
            data_frame = self._drop(col)

      elif self.combined_data_type_config[col] == 'nominal':
        if method_categorical == 'remove':
          logging.info(f'Method of Handling Missing Data: Remove Null Values in {col}')
          data_frame = self._remove_null(data_frame,col)
        elif method_categorical == 'impute_mode':
          logging.info(f'Method of Handling Missing Data: Impute Null Values with Mode in {col}')
          data_frame = self._impute_mode(data_frame,col)
    return data_frame

  def _remove_null(self, data_frame, col):
    data_frame = data_frame[data_frame[col].notnull()]
    return data_frame
  
  def _impute_mean(self, data_frame, col):
    col_mean = np.mean(data_frame[col])
    data_frame[col] = data_frame[col].fillna(col_mean)
    return data_frame
  
  def _impute_mode(self, data_frame, col):
    col_mode = data_frame[col].mode().iat[0]
    data_frame[col] = data_frame[col].fillna(col_mode)
    return data_frame

  def _drop(self, data_frame, col):
    data_frame = data_frame.drop(col)
    return data_frame

  def _impute_rf(self, data_frame):
    imputable_columns, not_imputable_columns = self._extract_imputable_col(data_frame)
    imputable_data_frame = data_frame[imputable_columns]
    not_imputable_data_frame = data_frame[not_imputable_columns]
    imputer_algorithms = self.imputer_algorithm_map()
    imputer = IterativeImputer(estimator=imputer_algorithms['random_forest'])
    imputed_data_frame = imputer.fit_transform(imputable_data_frame)
    imputed_data_frame = pd.DataFrame(imputed_data_frame, columns=imputable_columns)
    for col in not_imputable_data_frame.columns:
      imputed_data_frame[col] = data_frame[col]
    imputed_data_frame = imputed_data_frame[not_imputable_columns+imputable_columns]
    return imputed_data_frame

  def _extract_imputable_col(self, data_frame):
    imputable_columns = []
    not_imputable_columns = []
    for col in data_frame.columns:
      if self.combined_data_type_config[col] not in ['continuous', 'nominal', 'categorical']:
        not_imputable_columns.append(col)
      else:
        imputable_columns.append(col)
    return imputable_columns, not_imputable_columns

  def imputer_algorithm_map(self):
    return {
      'bayesian_ridge': BayesianRidge(),
      'linear_regression': LinearRegression(),
      'linear_ridge': Ridge(),
      'linear_lasso': Lasso(),
      'linear_elasticnet': ElasticNet(),
      'decision_tree': DecisionTreeRegressor(),
      'random_forest': RandomForestRegressor(),
      'extra_trees': ExtraTreesRegressor(),
      'gradient_boost': GradientBoostingRegressor(),
      'svm': SVR(),
      'k_neighbors': KNeighborsRegressor(),
      'neural_network': MLPRegressor(),
      'gaussian_process': GaussianProcessRegressor()
  }

  def general_binary_class_mapper(self):
    return {
      'No': 0,
      'Yes': 1,
      '0': 0,
      '1': 1
    }
  
  def gender_class_mapper(self):
    return {
      'm': 0,
      'f': 1,
      'Male': 0,
      'Female': 1 
    }
  
  def race_class_mapper(self):
    return {
      'Chinese': 0,
      'chinese': 0,
      'Malay': 1,
      'Indian': 2,
      'India': 2,
      'Others': 3
    }
  
  def resident_status_mapper(self):
    return {
      'Singaporean': 0,
      'Singapore citizen': 0,
      'PR': 1,
      'Foreigner': 2 
    }