import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from src.helper_functions.json_helpers import extract_json_configs
import src.functions.methods.epidemiology as EpidemiologyMethod
logging.basicConfig(level=logging.INFO)

class ExploratoryDataAnalysis():
  def __init__(self, data_frame, plot=True):
    self.data_frame = data_frame
    self.plot = plot
    self.missing_data_config = extract_json_configs('src/configs/missing_data_config.json')
    data_type_config = extract_json_configs('src/configs/data_type_config.json')
    self.type_combined_config = {**data_type_config['data']['clinical'], **data_type_config['data']['demographics'], **data_type_config['data']['bill_id'], **data_type_config['data']['bill_amount'], **data_type_config['data']['additional']}
    data_importance_config = extract_json_configs('src/configs/data_association_config.json')
    self.importance_combined_config = {**data_importance_config['data']['clinical'], **data_importance_config['data']['demographics'], **data_importance_config['data']['bill_id'], **data_importance_config['data']['bill_amount'], **data_importance_config['data']['additional']}
    
  # Main Function

  def run_analysis(self):
    logging.info('Running Exploratory Data Analysis')
    data_frame = self.data_frame
    logging.info('Adding new variables derived from currently availables ones')
    data_frame = self._add_new_variables(data_frame)
    logging.info('Running Descriptive Analysis')
    self._run_descriptive(data_frame)
    logging.info('Identifying skewed data')
    self._check_normality(data_frame)
    logging.info('Checking imbalance class')
    self._check_imbalance_class(data_frame)
    logging.info('Running Correlation of features/exposures against outcome cgis_change')
    self._run_correlation(data_frame)
    logging.info('Identifying Differences between Groups that showed improvement and Groups that do not')
    self._check_differences(data_frame)
    logging.info('Running Hypothesis Tests for Association with Outcome Variable')
    df_fisher, df_chi2, df_ttest, df_mannwhitney = self._statistical_tests(data_frame, 'cgis_improve')
    logging.info('Running Simple Logistic Regression')
    df_simple_log_regression = self._regression_analysis(data_frame, 'cgis_improve')
    logging.info('Running Multiple Logistic Regression')
    df_multiple_regression = self._multiple_regression_analysis(data_frame, 'cgis_improve')
    return data_frame
  
  # Sub Modular Functions

  def _add_new_variables(self, data_frame):
    if set(['date_of_discharge', 'date_of_admission']).issubset(set(data_frame.keys())):
      logging.info("Creating 'hospitalization_duration' from date_of_admission and date_of_discharge")
      data_frame['hospitalization_duration_days'] = data_frame['date_of_discharge'] - data_frame['date_of_admission']
      data_frame['hospitalization_duration_days'] = data_frame['hospitalization_duration_days'].dt.days
      data_frame = data_frame[[col for col in data_frame.columns if col not in ['date_of_discharge', 'date_of_admission']]]

    if set(['cgis_adm', 'cgis_dis']).issubset(set(data_frame.keys())):
      logging.info("Creating 'cgis_change' from cgis_adm and cgis_dis ")
      data_frame['cgis_change'] = data_frame['cgis_dis'] - data_frame['cgis_adm']
      # data_frame = data_frame[[col for col in data_frame.columns if col not in ['cgis_dis', 'cgis_adm']]]

    if set(['cgis_change']).issubset(set(data_frame.keys())):
      logging.info("Creating 'cgis_improve' from cgis_change")
      data_frame['cgis_improve'] = data_frame['cgis_change'].map(self.map_cgis_change)

    if set(['date_of_birth']).issubset(set(data_frame.keys())):
      logging.info("Creating 'age' from date_of_birth")
      reference_year = 2017
      data_frame['age'] = reference_year - data_frame['date_of_birth'].dt.year
      data_frame = data_frame[[col for col in data_frame.columns if col != 'date_of_birth']]

    return data_frame

  def _run_descriptive(self, data_frame):
    for col in data_frame.columns:
      if self.type_combined_config[col] == 'continuous':
        logging.info(f'Continuous Feature/Exposure: {col}')
        logging.info(f'Mean of: {np.mean(data_frame[col])}, Median: {np.median(data_frame[col])}, Standard Deviation: {np.std(data_frame[col])}')
        if self.plot:
          self._plot_histogram(data_frame, col)

      elif self.type_combined_config[col] in ['nominal', 'ordinal']:
        logging.info(f'Nominal or Ordinal Feature/Exposure: {col}')
        for n in range(data_frame[col].value_counts().count()):
          logging.info(f'Sub class {data_frame[col].value_counts().index[n]} : {data_frame[col].value_counts().values[n]}')
        if self.plot:
          self._plot_barchart(data_frame, col)

  def _check_normality(self, data_frame):
    df_normality_columns = [col for col in data_frame.columns if self.type_combined_config[col] == 'continuous']
    df_normality = data_frame[df_normality_columns]
    for col in df_normality.columns:
      df_normality_plot = df_normality
      logging.info(f'Histogram plot of {col}')
      if self.plot:
        df_normality_plot[col].plot(kind='hist')
        plt.show()
    return

  def _check_imbalance_class(self, data_frame):
    df_imb_class_columns = [col for col in data_frame.columns if self.type_combined_config[col] in ['nominal', 'ordinal']]
    df_imb_class = data_frame[df_imb_class_columns]
    for col in df_imb_class.columns:
      class_counts = df_imb_class[col].value_counts()
      class_proportions = [class_counts.values[index]/sum(class_counts.values) for index in list(range(len(class_counts)))]
      threshold =  1/len(class_counts)
      class_proportions_above_threshold = [class_prop for class_prop in class_proportions if class_prop > threshold * 1.25]
      if len(class_proportions_above_threshold) > 0:
        logging.info(f'Imbalance Class detected: {col}')
        for index in list(range(len(class_counts))):
          logging.info(f'{class_counts.index[index]}:{class_counts.values[index]} (Percentage: {class_counts.values[index]/sum(class_counts.values)})')
        if self.plot:
          sns.barplot(x=class_counts.index, y=class_counts.values)
          plt.show()
    return

  def _check_differences(self, data_frame):
    df_improved = data_frame[data_frame['cgis_improve']==1]
    df_worsen = data_frame[data_frame['cgis_improve']==0]
    comparison_columns = [col for col in data_frame.columns if self.type_combined_config[col] in ['continuous', 'nominal', 'ordinal']]
    if self.plot:
      for col in comparison_columns:
        logging.info(col)
        if self.type_combined_config[col] == 'continuous':
          fig, axes = plt.subplots(1, 2)
          df_improved.hist(col, ax=axes[0])
          df_worsen.hist(col, ax=axes[1])
          plt.show()
        elif self.type_combined_config[col] in ['ordinal', 'nominal']:
          fig, axes = plt.subplots(1, 2)
          # axes[0].set_ylim([0, 3400])
          # axes[1].set_ylim([0, 3400])
          df_improved[col].value_counts().plot(ax=axes[0], kind='bar', grid=True)
          df_worsen[col].value_counts().plot(ax=axes[1], kind='bar', grid=True)
          plt.show()

  def _run_correlation(self, data_frame):
    df_correlation_columns = [col for col in data_frame.columns if self.type_combined_config[col] in ['continuous', 'nominal', 'ordinal']]
    df_correlation = data_frame[df_correlation_columns]
    plt.figure(figsize=(15,15))
    mask = np.triu(np.ones_like(df_correlation.corr(), dtype=bool))
    f, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(df_correlation.corr(), mask=mask, vmax=1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return 
  
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
            if p_value < 0.05:
              logging.info(f'p_value: {p_value}, mean {col} for improved group: {np.mean(group_improve)}, mean {col} for worsen group: {np.mean(group_worsen)}')
          
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
      if p < 0.05 or odds_ratio > 1.2:
        logging.info(f'Simple Log Regression Result for {col}: p-value {p}, log_odds_ratio {log_odds_ratio}, odds_ratio {odds_ratio}, confidence_interval {conf_interval_low}:{conf_interval_upper}')
    return df_simple_log_regression
  
  def _multiple_regression_analysis(self, data_frame, outcome, exclude=[]):
    fixed_columns = [col for col in data_frame.columns if self.importance_combined_config[col] == True]
    if outcome in fixed_columns:
      fixed_columns.remove(outcome)
    if len(exclude)>1:
      [fixed_columns.remove(exclude_col) for exclude_col in exclude]
    # relevant_columns = np.random.choice(fixed_columns, size=5)
    p_values, odds_ratios, log_odds_ratio, conf_interval = EpidemiologyMethod.run_multiple_log_regresion(data_frame, fixed_columns, outcome)
    logging.info(p_values)
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

  def _plot_histogram(self, data_frame, col):
    data_frame[col].plot(kind='hist')
    plt.show()
    return 
  
  def _plot_barchart(self, data_frame, col):
    data_frame[col].value_counts().plot(kind='bar', rot=0)
    plt.show()
    return

  def map_cgis_change(self, value):
    if value > 0:
      return 1
    else:
      return 0

  
