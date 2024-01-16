import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from src.helper_functions.json_helpers import extract_json_configs
logging.basicConfig(level=logging.INFO)

class Summary():
  def __init__(self, data_frame, plot=True):
    self.data_frame = data_frame
    self.plot = plot
    data_type_config = extract_json_configs('src/configs/data_type_config.json')
    self.combined_config = {**data_type_config['data']['clinical'], **data_type_config['data']['demographics'], **data_type_config['data']['bill_id'], **data_type_config['data']['bill_amount']}
  # Main Function
    
  def summarize(self):
    logging.info('Checking Missing Data')
    self._check_missing_data()
    logging.info('Checking Duplication')
    self._check_duplicates()
    logging.info('Checking Data Integrity')
    self._check_data_integrity()
    logging.info('Identifying Potential Outliers')
    self._identify_outliers()
    return
  
  # Sub-Modular Functions

  def _check_missing_data(self):
    col_missing = [col for col in self.data_frame.columns if len(self.data_frame[self.data_frame[col].isna()]) > 0]
    for col in col_missing:
      n_records = len(self.data_frame[col])
      n_missing = len(self.data_frame[self.data_frame[col].isna()])
      logging.info(f'Column with missing data: {col} - {n_missing} out of {n_records} ({n_missing/n_records*100} %)')
    return 
  
  def _check_duplicates(self):
    if len(self.data_frame[self.data_frame.duplicated()]) == 0:
      logging.info('No Duplicated Records Detected')
    else: 
      logging.info('Duplicated Records Detected')
    return 
  
  def _check_data_integrity(self):
    for col in self.data_frame.columns:
      if self.combined_config[col] in ['nominal', 'ordinal']:
        if len(self.data_frame[col].value_counts()) > 2:
          logging.info(f'Types of groups found in {col}: {self.data_frame[col].unique()}')
    return

  def _identify_outliers(self):    
    for col in self.data_frame.columns:
      if self.combined_config[col] == 'continuous':
        if self.plot:
          self._plot_histogram(col)
    return 
  
  def _plot_histogram(self, col):
    self.data_frame[col].plot(kind='hist')
    plt.show()
    return 
  
  
