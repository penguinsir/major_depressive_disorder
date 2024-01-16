import logging
import numpy as np
import statsmodels.api as sm
from scipy.stats import fisher_exact, chi2_contingency, ttest_ind, shapiro, mannwhitneyu
logging.basicConfig(level=logging.INFO)

def run_shapiro_test(column):
  statistic, p_value = shapiro(column)
  return p_value

def run_chi_square_test(contingency_table):
  chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
  odds_ratio = (contingency_table.iloc[0, 0] * contingency_table.iloc[1, 1]) / (contingency_table.iloc[0, 1] * contingency_table.iloc[1, 0])
  return odds_ratio, p_chi2

def run_fisher_exact_test(contingency_table):
  odds_ratio, p_fisher = fisher_exact(contingency_table)
  return odds_ratio, p_fisher

def run_t_test_test(group_improve, group_worsen):
  t_stat, p_value = ttest_ind(group_improve, group_worsen)
  return p_value

def run_mannwhitneyu_test(group_improve, group_worsen):
  statistic_mwu, p_value = mannwhitneyu(group_improve, group_worsen, alternative='two-sided')
  return p_value

def run_simple_log_regresion(data_frame, col, outcome):
  model = sm.Logit(data_frame[outcome],sm.add_constant(data_frame[col])) # data_frame[col]) # sm.add_constant(data_frame[col]))
  result = model.fit()
  # logging.info(result.summary())
  log_odds_ratio = result.params[col]
  odds_ratio = np.exp(result.params[col])
  conf_interval_low = np.exp(result.conf_int().loc[col])[0]
  conf_interval_upper = np.exp(result.conf_int().loc[col])[1]
  p = result.pvalues[col]
  return p, odds_ratio, log_odds_ratio, conf_interval_low, conf_interval_upper

def run_multiple_log_regresion(data_frame, relevant_columns, outcome):
  variables = data_frame[relevant_columns]
  outcome = data_frame[outcome]
  model = sm.Logit(outcome, variables)
  result = model.fit(max_iter=1000)
  p_values = result.pvalues
  odds_ratios = result.params
  coefficients = result.params
  log_odds_ratio = coefficients.apply(lambda x: round(x, 4)) 
  conf_interval = np.exp(result.conf_int())
  return p_values, odds_ratios, log_odds_ratio, conf_interval
  
  
