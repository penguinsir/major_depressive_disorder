import logging
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)

class RawDataset():
  def __init__(self):
    self.folder_directory = 'datasets/raw_dataset'
  
  # Main Function
    
  def data_extraction(self):
    logging.info('Extracting Datasets')
    df_bill_amount, df_bill_id, df_clinical, df_demographics = self._retrieve_dataset()
    df_clinical = df_clinical.rename(columns={'id':'patient_id'})
    logging.info('Combining Relevant Datasets')
    df_clinical_combined = self._combine_clinical(df_clinical, df_demographics)
    df_bill_combined = self._combine_bill(df_bill_amount, df_bill_id)
    return df_clinical_combined, df_bill_combined
  
  # Sub-Modular Functions

  def _retrieve_dataset(self):
    df_bill_amount = pd.read_csv(f'{self.folder_directory}/bill_amount.csv')
    df_bill_id = pd.read_csv(f'{self.folder_directory}/bill_id.csv')
    df_clinical = pd.read_csv(f'{self.folder_directory}/clinical_data.csv')
    df_demographics = pd.read_csv(f'{self.folder_directory}/demographics.csv')
    return df_bill_amount, df_bill_id, df_clinical, df_demographics
  
  def _combine_clinical(self, df_clinical_data, df_demographics):
    df_clinical_combined = pd.merge(left=df_clinical_data, right = df_demographics, on='patient_id', how='inner')
    return df_clinical_combined
  
  def _combine_bill(self, df_bill_amount, df_bill_id):
    df_bill_combined = pd.merge(left=df_bill_id, right=df_bill_amount, on='bill_id', how='inner')
    return df_bill_combined