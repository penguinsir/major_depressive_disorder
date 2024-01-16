import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.helper_functions.json_helpers import extract_json_configs
import src.functions.methods.epidemiology as EpidemiologyMethod

logging.basicConfig(level=logging.INFO)

class ResearchQuestion:
  def __init__(self, data_frame, df_bill_prelim, plot=True, all_results=False):
    self.data_frame = data_frame
    self.df_bill_prelim = df_bill_prelim
    self.plot = plot
    self.all_results= all_results
    data_importance_config = extract_json_configs('src/configs/data_association_config.json')
    self.importance_combined_config = {**data_importance_config['data']['clinical'], **data_importance_config['data']['demographics'], **data_importance_config['data']['bill_id'], **data_importance_config['data']['bill_amount'], **data_importance_config['data']['additional']}
    data_type_config = extract_json_configs('src/configs/data_type_config.json')
    self.type_combined_config = {**data_type_config['data']['clinical'], **data_type_config['data']['demographics'], **data_type_config['data']['bill_id'], **data_type_config['data']['bill_amount'], **data_type_config['data']['additional']}

  # Main Function

  def run_analysis(self):
    data_frame = self.data_frame
    df_comb_therapy = self._filter_ssr_comb_therapy(data_frame)
    logging.info('Obtaining Demographic Informations for Combination Treatment with SSRI and Without SSRI')
    df_demographic_continuous, df_demographic_categorical = self._obtain_demographic(data_frame)
    logging.info('Continuous Demographic Information')
    display(df_demographic_continuous)
    logging.info('Categorical Demographic Information')
    display(df_demographic_categorical)
    logging.info('Are there any factors that influence the incorporation of SSR into combination?')
    df_ssr_comb_fisher, df_ssr_comb_chi2, df_ssr_comb_ttest, df_ssr_comb_mannwhitney, df_comb_therapy = self._analyze_ssr_comb_therapy(data_frame)
    logging.info('Displaying Results')
    if self.all_results:
      self._display_all([df_ssr_comb_fisher, df_ssr_comb_chi2, df_ssr_comb_ttest, df_ssr_comb_mannwhitney])
    logging.info('Running Simple Logistic Regression')
    df_ssr_simple_log_regression = self._regression_analysis(df_comb_therapy, 'cgis_improve', exclude=['trt_adt','trt_the', 'trt_oth'])
    logging.info('Running Multiple Logistic Regression')
    df_ssr_multiple_regression = self._multiple_regression_analysis(df_comb_therapy, 'cgis_improve', exclude=['trt_adt','trt_the', 'trt_oth'])
    # df_ssr_multiple_regression = pd.DataFrame()
    return df_demographic_continuous, df_demographic_categorical, df_ssr_comb_fisher, df_ssr_comb_chi2, df_ssr_comb_ttest, df_ssr_comb_mannwhitney, df_ssr_simple_log_regression, df_ssr_multiple_regression

  # Sub Modular Functions

  def _obtain_demographic(self, data_frame):
    df_map = {'df_ssr_comb': data_frame[data_frame['trt_ssr']==1], 'df_no_ssr_combi': data_frame[data_frame['trt_ssr']==0]}
    df_demographic_continuous = self._obtain_continuous_demo(df_map)
    df_demographic_categorical = self._obtain_categorical_demo(df_map)
    return df_demographic_continuous, df_demographic_categorical

  def _filter_ssr_comb_therapy(self, data_frame):
    df_comb_therapy = data_frame[data_frame['trt_adt']==1]
    df_comb_therapy = df_comb_therapy[df_comb_therapy['trt_the']==1]
    df_comb_therapy = df_comb_therapy[df_comb_therapy['trt_oth']==1]
    return df_comb_therapy

  def _analyze_ssr_comb_therapy(self, data_frame):
    df_comb_therapy = self._filter_ssr_comb_therapy(data_frame)
    df_ssr_comb = df_comb_therapy[df_comb_therapy['trt_ssr']==1]
    df_no_ssr_comb = df_comb_therapy[df_comb_therapy['trt_ssr']==0]
    self._check_differences_ssr(df_ssr_comb, df_no_ssr_comb)
    df_ssr_comb_fisher, df_ssr_comb_chi2, df_ssr_comb_ttest, df_ssr_comb_mannwhitney = self._statistical_tests(df_comb_therapy, 'trt_ssr')
    logging.info('Merging df_bills with df_ssr_comb and df_no_ssr_comb to investigate differences in bills amount')
    df_ssr_comb_bill = self._check_bills(df_ssr_comb, df_no_ssr_comb)
    return df_ssr_comb_fisher, df_ssr_comb_chi2, df_ssr_comb_ttest, df_ssr_comb_mannwhitney, df_comb_therapy
  
  def _check_differences_ssr(self, df_ssr_comb, df_no_ssr_comb):
    comparison_columns = [col for col in self.data_frame.columns if self.type_combined_config[col] in ['continuous', 'nominal', 'ordinal']]
    
    logging.info('Displaying results for medical_history_mood')
    display(df_ssr_comb['medical_history_mood'].value_counts().sort_index())
    display(df_no_ssr_comb['medical_history_mood'].value_counts().sort_index())
    fig, axes = plt.subplots(1, 2)
    axes[0].set_ylim([0, 1000])
    axes[1].set_ylim([0, 1000])
    df_ssr_comb['medical_history_mood'].value_counts().sort_index().plot(ax=axes[0], kind='bar', grid=True)
    df_no_ssr_comb['medical_history_mood'].value_counts().sort_index().plot(ax=axes[1], color='orange', kind='bar', grid=True)
    plt.show()

    logging.info('Displaying results for symptom_4')
    display(df_ssr_comb['symptom_4'].value_counts().sort_index())
    display(df_no_ssr_comb['symptom_4'].value_counts().sort_index())
    fig, axes = plt.subplots(1, 2)
    axes[0].set_ylim([0, 1000])
    axes[1].set_ylim([0, 1000])
    df_ssr_comb['symptom_4'].value_counts().sort_index().plot(ax=axes[0], kind='bar', grid=True)
    df_no_ssr_comb['symptom_4'].value_counts().sort_index().plot(ax=axes[1], kind='bar', color='orange', grid=True)
    plt.show()

    logging.info('Displaying results for cgis_adm')
    display(df_ssr_comb['cgis_adm'].value_counts().sort_index())
    display(df_no_ssr_comb['cgis_adm'].value_counts().sort_index())
    fig, axes = plt.subplots(1, 2)
    axes[0].set_ylim([0, 1000])
    axes[1].set_ylim([0, 1000])
    df_ssr_comb['cgis_adm'].value_counts().sort_index().plot(ax=axes[0], kind='bar', grid=True)
    df_no_ssr_comb['cgis_adm'].value_counts().sort_index().plot(ax=axes[1], kind='bar',color='orange', grid=True)
    plt.show()

    if self.plot:
      for col in comparison_columns:
        if self.type_combined_config[col] == 'continuous':
          fig, axes = plt.subplots(1, 2)
          df_ssr_comb.hist(col, ax=axes[0])
          df_no_ssr_comb.hist(col, ax=axes[1])
          plt.show()
        elif self.type_combined_config[col] in ['ordinal', 'nominal']:
          fig, axes = plt.subplots(1, 2)
          axes[0].set_ylim([0, 1000])
          axes[1].set_ylim([0, 1000])
          df_ssr_comb[col].value_counts().sort_index().plot(ax=axes[0], kind='bar', grid=True)
          df_no_ssr_comb[col].value_counts().sort_index().plot(ax=axes[1], kind='bar', grid=True)
          plt.show()

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
            if abs(odds_ratio) > 1.5 or p_fisher < 0.05:
              logging.info(f'odds ratio: {odds_ratio}, p-value: {p_fisher}')
          else:
            if len(contingency_table)>1:
              odds_ratio, p_chi2 = EpidemiologyMethod.run_chi_square_test(contingency_table)
              if abs(odds_ratio) > 1.5 or p_chi2 < 0.05:
                logging.info(f'odds ratio: {odds_ratio}, p-value: {p_chi2}')
              df_chi2 = pd.concat([df_chi2, pd.DataFrame([[col, odds_ratio, p_chi2]], columns=['col', 'odds_ratio','p-value'])])

        if self.type_combined_config[col] in ['continuous', 'ordinal']:
          p_value = EpidemiologyMethod.run_shapiro_test(data_frame[col])
          group_improve = data_frame[data_frame[outcome_variable] == 0][col]
          group_worsen = data_frame[data_frame[outcome_variable] == 1][col]
          if p_value> 0.05 or col in ['height', 'weight']:
            p_value = EpidemiologyMethod.run_t_test_test(group_improve, group_worsen)
            df_ttest = pd.concat([df_ttest, pd.DataFrame([[col, np.mean(group_improve),np.mean(group_worsen), p_chi2]], columns=['col', 'group_improve', 'group_worsen', 'p-value'])])
            if p_value < 0.05:
              logging.info(f'p_value: {p_value}, mean {col} for improved group: {np.mean(group_improve)}, mean {col} for worsen group: {np.mean(group_worsen)}')    
          else:
            p_value = EpidemiologyMethod.run_mannwhitneyu_test(group_improve, group_worsen)
            df_mannwhitney = pd.concat([df_mannwhitney, pd.DataFrame([[col, np.mean(group_improve),np.mean(group_worsen), p_chi2]], columns=['col', 'group_improve', 'group_worsen', 'p-value'])])
            
    return df_fisher, df_chi2, df_ttest, df_mannwhitney
  
  def _regression_analysis(self, data_frame, outcome, exclude=[]):
    df_simple_log_regression = pd.DataFrame()
    relevant_columns = [col for col in data_frame.columns if self.importance_combined_config[col] == True]
    if outcome in relevant_columns:
      relevant_columns.remove(outcome)
    if len(exclude)>1:
      [relevant_columns.remove(exclude_col) for exclude_col in exclude]
    for col in relevant_columns:
      p, odds_ratio, log_odds_ratio, conf_interval_low, conf_interval_upper = EpidemiologyMethod.run_simple_log_regresion(data_frame, col, outcome)
      df_simple_log_regression = pd.concat([df_simple_log_regression, pd.DataFrame([[col, p, odds_ratio, log_odds_ratio, conf_interval_low, conf_interval_upper]],  columns=['col', 'p', 'odds_ratio', 'log_odds_ratio', 'conf_interval_lower', 'conf_interval_upper'])])
    return df_simple_log_regression
  
  def _multiple_regression_analysis(self, data_frame, outcome, exclude=[]):
    fixed_columns = [col for col in data_frame.columns if self.importance_combined_config[col] == True]
    if outcome in fixed_columns:
      fixed_columns.remove(outcome)
    if len(exclude)>1:
      [fixed_columns.remove(exclude_col) for exclude_col in exclude]
    # relevant_columns = np.random.choice(fixed_columns, size=5)
    p_values, odds_ratios, log_odds_ratio, conf_interval = EpidemiologyMethod.run_multiple_log_regresion(data_frame, fixed_columns, outcome)
    df_multiple_regression = self._compile_multiple_regression_result(p_values, odds_ratios, log_odds_ratio, conf_interval)
    return df_multiple_regression

  def _compile_multiple_regression_result(self, p_values, odds_ratios, log_odds_ratio, conf_interval):
    df_odds_ratio = pd.DataFrame([odds_ratios.to_dict()])
    df_odds_ratio = df_odds_ratio.rename(index={0:'odds_ratio'})
    df_p_value = pd.DataFrame([p_values.to_dict()])
    df_p_value = df_p_value.rename(index={0:'p-value'})
    df_log_odds_ratio = pd.DataFrame([log_odds_ratio.to_dict()])
    df_log_odds_ratio = df_log_odds_ratio.rename(index={0:'log_odds_ratio'})
    df_conf_interval_low = pd.DataFrame([conf_interval[0].to_dict()])
    df_conf_interval_low = df_conf_interval_low.rename(index={0:'conf_interval_low'})
    df_conf_interval_high = pd.DataFrame([conf_interval[1].to_dict()])
    df_conf_interval_high = df_conf_interval_high.rename(index={0:'conf_interval_upper'})
    df_multiple_regression = pd.concat([df_p_value, df_odds_ratio, df_log_odds_ratio, df_conf_interval_low, df_conf_interval_high])
    return df_multiple_regression

  def _check_bills(self, df_ssr_comb, df_no_ssr_comb):
    df_bill_prelim = self.df_bill_prelim
    df_bill = df_bill_prelim.groupby('patient_id')['amount'].sum().reset_index()
    df_bill = df_bill.drop_duplicates('patient_id')
    df_ssr_bill = pd.merge(left=df_ssr_comb, right=df_bill, on='patient_id', how='inner')
    df_no_ssr_bill = pd.merge(left=df_no_ssr_comb, right=df_bill, on='patient_id', how='inner')

    df_ssr_bill['amount_normalized'] = df_ssr_bill['amount']/df_ssr_bill['hospitalization_duration_days']
    df_no_ssr_bill['amount_normalized'] = df_no_ssr_bill['amount']/df_no_ssr_bill['hospitalization_duration_days']

    logging.info('Results for different bills amount')
    logging.info(f"Median: {np.median(df_ssr_bill['amount'].round(2))}")
    df_ssr_bill['amount'].plot(kind='hist', bins=50)
    # plt.xlim(0, 30000)
    plt.show()

    logging.info(f"Median: {np.median(df_no_ssr_bill['amount'].round(2))}")
    df_no_ssr_bill['amount'].plot(kind='hist', bins=50, color='orange')
    # plt.xlim(0, 30000)
    plt.show()

    logging.info(f"Difference in Median: {abs(np.median(df_ssr_bill['amount'].round(2)) - np.median(df_no_ssr_bill['amount'].round(2)))}")
    logging.info(f"Difference in Mean: {abs(np.mean(df_ssr_bill['amount'].round(2)) - np.mean(df_no_ssr_bill['amount'].round(2)))}")

    logging.info('Results for different bills amount (normalized)')
    logging.info(f"Median: {np.median(df_ssr_bill['amount_normalized'].round(2))}")
    df_ssr_bill['amount_normalized'].plot(kind='hist', bins=50)
    plt.show()

    logging.info(f"Median: {np.median(df_no_ssr_bill['amount_normalized'].round(2))}")
    df_no_ssr_bill['amount_normalized'].plot(kind='hist', bins=50, color='orange')
    plt.show()

    logging.info(f"Difference in Median: {abs(np.median(df_ssr_bill['amount_normalized'].round(2)) - np.median(df_no_ssr_bill['amount_normalized'].round(2)))}")
    logging.info(f"Difference in Mean: {abs(np.mean(df_ssr_bill['amount_normalized'].round(2)) - np.mean(df_no_ssr_bill['amount_normalized'].round(2)))}")

    return


  def _display_all(self, list_of_dfs):
    for df in list_of_dfs:
      if len(df)>1:
        display(df)
    return

  def _display_signficant(self, list_of_dfs):
    for df in list_of_dfs:
      if 'p-value' in df.columns:
        df_display = df[df['p-value']<0.05]
        if len(df_display)>1:
          display(df_display)
      elif 'p' in df.columns:
        df_display = df[df['p']<0.05]
        if len(df_display)>1:
          display(df_display)
      else:
        display(df)
    return 

  def _plot_histogram(self, data_frame, col):
    data_frame[col].plot(kind='hist')
    plt.show()
    return 
  
  def _plot_barchart(self, data_frame, col):
    data_frame[col].value_counts().plot(kind='bar', rot=0)
    plt.show()
    return

  