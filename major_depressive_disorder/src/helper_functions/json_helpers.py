import json

def extract_json_configs(file_directory):
  with open(file_directory) as missing_data_config_json:
    return json.load(missing_data_config_json)