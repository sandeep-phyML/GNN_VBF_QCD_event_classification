from datasets import load_data , read_config
import yaml

config = read_config("input_config_2022.yml")
load_data(config , "output_config_2022.yml")