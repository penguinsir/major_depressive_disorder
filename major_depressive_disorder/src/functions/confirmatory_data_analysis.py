import logging
import pingouin as pg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.functions.methods.epidemiology as EpidemiologyMethod
from src.helper_functions.json_helpers import extract_json_configs

logging.basicConfig(level=logging.INFO)

class ConfirmatoryDataAnalysis:
  def __init__(self, data_frame, plot=True):
    self.data_frame = data_frame
    data_importance_config = extract_json_configs('src/configs/data_association_config.json')
    self.importance_combined_config = {**data_importance_config['data']['clinical'], **data_importance_config['data']['demographics'], **data_importance_config['data']['bill_id'], **data_importance_config['data']['bill_amount'], **data_importance_config['data']['additional']}
    data_type_config = extract_json_configs('src/configs/data_type_config.json')
    self.type_combined_config = {**data_type_config['data']['clinical'], **data_type_config['data']['demographics'], **data_type_config['data']['bill_id'], **data_type_config['data']['bill_amount'], **data_type_config['data']['additional']}

  # Main Function

  def run_analysis(self):
    data_frame = self.data_frame
    logging.info('Obtaining Demographics for patients with improved condition and patients with worsened condition')
    df_demographic_continuous, df_demographic_categorical = self._obtain_demographic(data_frame)
    logging.info('Continuous Demographic Information')
    display(df_demographic_continuous)
    logging.info('Categorical Demographic Information')
    display(df_demographic_categorical)
    logging.info('Running Statistical Tests for Demographics')
    df_fisher, df_chi2, df_ttest, df_mannwhitney = self._statistical_tests(data_frame, 'cgis_improve')
    logging.info('Investigating Patterns of SSR prescription')
    logging.info('Are there any treatment which are prescribed only by itself?')
    df_one_treatment_results = self._analyze_one_treatment(data_frame)
    logging.info('SSRI is present in which treatment combination?')
    df_multiple_treatment_results = self._analyze_multiple_treatment(data_frame)
    logging.info('What is the most common combination therapy?')
    df_combination_therapy = self._analyze_comb_therapy(data_frame)
    return data_frame, df_demographic_continuous, df_demographic_categorical, df_fisher, df_chi2, df_ttest, df_mannwhitney, df_one_treatment_results, df_multiple_treatment_results, df_combination_therapy

  # Sub Modular Function

  def _obtain_demographic(self, data_frame):
    df_map = {'df_cgis_status': data_frame[data_frame['cgis_improve']==1], 'df_cgis_worsen': data_frame[data_frame['cgis_improve']==0]}
    df_demographic_continuous = self._obtain_continuous_demo(df_map)
    df_demographic_categorical = self._obtain_categorical_demo(df_map)
    return df_demographic_continuous, df_demographic_categorical
  
  def _obtain_continuous_demo(self, df_map):
    df_demographic_continuous = pd.DataFrame()
    for df_key in df_map.keys():
      df = df_map[df_key]
      for col in self.data_frame.columns:
        if self.type_combined_config[col] == 'continuous':
          if EpidemiologyMethod.run_shapiro_test(df[col]) < 0.05:
            df_demographic_continuous = pd.concat([df_demographic_continuous,pd.DataFrame([{'column': f'{col}_{df_key}','median': df[col].quantile(0.5), 'iqr_lower':df[col].quantile(0.25), 'iqr_upper':df[col].quantile(0.75)}])])
          else:
            df_demographic_continuous = pd.concat([df_demographic_continuous, pd.DataFrame([{'column': f'{col}_{df_key}','mean': df[col].mean(), 'std_dev':df[col].std()}])])
    return df_demographic_continuous
  
  def _obtain_categorical_demo(self, df_map):
    df_demographic_categorical = pd.DataFrame()
    for df_key in df_map.keys():
      df = df_map[df_key]
      for col in self.data_frame.columns:
        if self.type_combined_config[col] in ['nominal', 'ordinal']:
          class_proportions_reformatted = {}
          class_counts = df[col].value_counts().sort_index()
          class_proportions = [{index:class_counts.values[index]/sum(class_counts.values)} for index in list(range(len(class_counts)))]
          n_classes = [{f'n_{index}':class_counts.values[index]} for index in list(range(len(class_counts)))]
          [class_proportions_reformatted.update(class_proportion) for class_proportion in class_proportions]
          [class_proportions_reformatted.update(n_class) for n_class in n_classes]
          class_proportions_reformatted['column'] = f'{col}_{df_key}'
          df_demographic_categorical = pd.concat([df_demographic_categorical, pd.DataFrame([class_proportions_reformatted])])
    return df_demographic_categorical

  def _analyze_one_treatment(self, data_frame):
    df_anx_only = data_frame[(data_frame['trt_anx']==1) & (data_frame['trt_con'] == 0) & (data_frame['trt_adt']==0) & (data_frame['trt_ssr']==0) & (data_frame['trt_the']==0) & (data_frame['trt_oth']==0)]
    df_con_only = data_frame[(data_frame['trt_anx']==0) & (data_frame['trt_con'] == 1) & (data_frame['trt_adt']==0) & (data_frame['trt_ssr']==0) & (data_frame['trt_the']==0) & (data_frame['trt_oth']==0)]
    df_adt_only = data_frame[(data_frame['trt_anx']==0) & (data_frame['trt_con'] == 0) & (data_frame['trt_adt']==1) & (data_frame['trt_ssr']==0) & (data_frame['trt_the']==0) & (data_frame['trt_oth']==0)]
    df_ssr_only = data_frame[(data_frame['trt_anx']==0) & (data_frame['trt_con'] == 0) & (data_frame['trt_adt']==0) & (data_frame['trt_ssr']==1) & (data_frame['trt_the']==0) & (data_frame['trt_oth']==0)]
    df_the_only = data_frame[(data_frame['trt_anx']==0) & (data_frame['trt_con'] == 0) & (data_frame['trt_adt']==0) & (data_frame['trt_ssr']==0) & (data_frame['trt_the']==1) & (data_frame['trt_oth']==0)]
    df_oth_only = data_frame[(data_frame['trt_anx']==0) & (data_frame['trt_con'] == 0) & (data_frame['trt_adt']==0) & (data_frame['trt_ssr']==0) & (data_frame['trt_the']==0) & (data_frame['trt_oth']==1)]
    
    df_one_treatment_results = {}
    df_one_treatment_map = {'anx': df_anx_only, 'con':df_con_only, 'adt':df_adt_only, 'ssr':df_ssr_only, 'the': df_the_only, 'oth':df_oth_only}
    for key in df_one_treatment_map.keys():
      df_one_treatment = df_one_treatment_map[key]
      df_one_treatment_result = {'n_patients': len(df_one_treatment), 'n_improve': len(df_one_treatment[df_one_treatment['cgis_improve']==1])/len(df_one_treatment)}
      df_one_treatment_results[key] = df_one_treatment_result
    df_one_treatment_results = pd.DataFrame(df_one_treatment_results)
    logging.info('Display Results for Single Treatment Therapy')
    display(df_one_treatment_results)
    df_one_treatment_results.iloc[0].plot(kind='bar')
    plt.show()
    return df_one_treatment_results
  
  def _analyze_multiple_treatment(self, data_frame):
    df_multiple_treatment_results = pd.DataFrame()
    df_multiple_treatment_results_clean = pd.DataFrame()
    df_ssr_treatment = data_frame[data_frame['trt_ssr']==1]
    relevant_columns = [col for col in data_frame.columns if self.type_combined_config[col] in ['nominal', 'ordinal']]
    relevant_treatment_columns = [col for col in relevant_columns if col.startswith('trt')]
    relevant_treatment_columns = [col for col in relevant_treatment_columns if col != 'trt_ssr']
    for col in relevant_treatment_columns:
      df_value_counts = df_ssr_treatment[col].value_counts()
      treatment_count = df_value_counts.values[0]
      df_multiple_treatment_results = pd.concat([df_multiple_treatment_results, pd.DataFrame([{col:treatment_count/len(df_ssr_treatment)}])])
    logging.info('Displaying Results for Combination Treatment that includes SSRIs')
    for col in df_multiple_treatment_results.columns:
      df_multiple_treatment_results_clean[col] = [df_multiple_treatment_results[col].dropna().iloc[0]*len(df_ssr_treatment)]
    display(df_multiple_treatment_results_clean)
    df_multiple_treatment_results_clean.iloc[0].plot(kind='bar')  
    plt.show()  
    return df_multiple_treatment_results_clean

  def _analyze_comb_therapy(self, data_frame):
    df_combination_therapy = pd.DataFrame()
    relevant_columns = [col for col in data_frame.columns if self.type_combined_config[col] in ['nominal', 'ordinal']]
    relevant_treatment_columns = [col for col in relevant_columns if col.startswith('trt')]
    df_treatments = data_frame[relevant_treatment_columns]
    df_treatments['combination'] = df_treatments.apply(lambda row: tuple(row), axis=1)
    df_combination_therapy = pd.DataFrame(df_treatments['combination'].value_counts())
    df_combination_therapy_clean = df_combination_therapy[0:7]
    index_mapper =self._index_mapper()
    df_combination_therapy_clean = df_combination_therapy_clean.rename(index=index_mapper)
    logging.info('Displaying Results for Most Common Types of Combination Treatment')
    display(df_combination_therapy_clean)
    df_combination_therapy_clean.plot(kind='barh')
    plt.show()
    return df_combination_therapy
  
  def _statistical_tests(self, data_frame, outcome_variable):
    df_fisher = pd.DataFrame() 
    df_chi2 = pd.DataFrame()
    df_ttest = pd.DataFrame()
    df_mannwhitney = pd.DataFrame()

    relevant_columns = [col for col in data_frame.columns if self.type_combined_config[col] in ['continuous', 'nominal', 'ordinal']]
    for col in relevant_columns:
      if col != outcome_variable:
        if self.type_combined_config[col] == 'nominal':
          contingency_table = pd.crosstab(data_frame[col], data_frame[outcome_variable])
          if (contingency_table < 5).any().any():
            odds_ratio, p_fisher = EpidemiologyMethod.run_fisher_exact_test(contingency_table)
            df_fisher = pd.concat([df_fisher, pd.DataFrame([[col, odds_ratio, p_fisher]], columns=['col', 'odds_ratio','p-value'])])
          else:
            if len(contingency_table)>1:
              odds_ratio, p_chi2 = EpidemiologyMethod.run_chi_square_test(contingency_table)
              df_chi2 = pd.concat([df_chi2, pd.DataFrame([[col, odds_ratio, p_chi2]], columns=['col', 'odds_ratio','p-value'])])

        if self.type_combined_config[col] in ['continuous', 'ordinal']:
          p_value = EpidemiologyMethod.run_shapiro_test(data_frame[col])
          group_improve = data_frame[data_frame[outcome_variable] == 0][col]
          group_worsen = data_frame[data_frame[outcome_variable] == 1][col]
          if p_value> 0.05 or col in ['height', 'weight']:
            p_value = EpidemiologyMethod.run_t_test_test(group_improve, group_worsen)
            df_ttest = pd.concat([df_ttest, pd.DataFrame([[col, np.mean(group_improve),np.mean(group_worsen), p_chi2]], columns=['col', 'group_improve', 'group_worsen', 'p-value'])])
          else:
            p_value = EpidemiologyMethod.run_mannwhitneyu_test(group_improve, group_worsen)
            df_mannwhitney = pd.concat([df_mannwhitney, pd.DataFrame([[col, np.mean(group_improve),np.mean(group_worsen), p_chi2]], columns=['col', 'group_improve', 'group_worsen', 'p-value'])])
    return df_fisher, df_chi2, df_ttest, df_mannwhitney

  def _index_mapper(self):
    return {
      (1, 1, 1, 1, 1, 1):'All Treatments',
      (0, 1, 1, 1, 1, 1):'All Except Anxiolytics',
      (0, 1, 1, 0, 1, 1):'All Except Anxiolytics and SSRI',
      (1, 1, 1, 0, 1, 1):'All Except SSRI',
      (0, 1, 1, 1, 1, 1):'All Except Anxiolytics and Anticonvulsants',
      (0, 0, 1, 1, 1, 1):'All Except Anticonvulsants',
      (1, 0, 1, 1, 1, 1):'All Except Anticonvulsants and SSRI',
      (1, 0, 1, 0, 1, 1):'All Except Anxiolytics, Anticonvulsants and SSRI'
    }




  # Sub Modular Function
  # For continuous only
  # def _partial_correlations(self, data_frame):
  #   fixed_covariates = [col for col in data_frame.columns if self.importance_combined_config[col] == True and self.type_combined_config[col] in ['continuous', 'ordinal']]
  #   covariates = []
  #   for covar_index in list(range(len(fixed_covariates))):
  #     if len(fixed_covariates) > 1:
  #       covariates.append(fixed_covariates[0])
  #       fixed_covariates.remove(fixed_covariates[0])
  #       for index in list(range(len(data_frame.columns))):
  #         if index != len(data_frame.columns) - 1:
  #           if list(data_frame.columns)[index] not in ['patient_id', 'cgis_change'] and list(data_frame.columns)[index+1] not in ['patient_id', 'cgis_change']:
  #             x = list(data_frame.columns)[index]
  #             y = list(data_frame.columns)[index+1]
  #             if x in fixed_covariates:
  #               covariates = fixed_covariates.remove(x)
  #             else:
  #               covariates = fixed_covariates
              
  #             if y in fixed_covariates:
  #               covariates = covariates.remove(y)
  #             else:
  #               covariates = covariates

  #             result = data_frame.partial_corr(x=x, y=y, covar=covariates, method='spearman').round(2)
  #             if result["r"].values[0] > 0.1 or result["p-val"].values[0] < 0.05:
  #               logging.info(f'Analysis Result for {x}: Strength of Association: {result["r"].values[0]}, Statistical Significance: {result["p-val"].values[0]}, 95% CI: {result["CI95%"].values[0]}')
  #               logging.info(f'Covariates Used: {covariates}')
    return 