from datasets import load_dataset , read_config
import yaml

config = read_config("input_config_2022.yml")
load_dataset(config , "output_config_2022.yml")