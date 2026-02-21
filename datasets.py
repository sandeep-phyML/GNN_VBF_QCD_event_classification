import uproot 
import numpy as np
import pandas as pd
import torch
import yaml
import os 

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_out_config(config, out_config_path):
    try :
        with open(out_config_path, 'r') as f:
            out_config = yaml.safe_load(f)
        for key , value in config.items():
            if key in out_config:
                if isinstance(out_config[key], dict) and isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key in out_config[key]:
                            out_config[key][sub_key] = sub_value
                        else:
                            out_config[key][sub_key] = sub_value
                else :
                    out_config[key].update(value)
            else:
                out_config[key] = value
    except FileNotFoundError:
        out_config = config
    with open(out_config_path, 'w') as f:
        yaml.dump(out_config, f)

def load_data(config , out_config_path):
    data_frames = []
    out_config = {}
    file_path_config = config["train_mclass_files_labels"]
    branches_to_load = config["weight_features"] + config["extra_features"]
    for node_feature in config["node_features"]:
        branches_to_load += node_feature
    for process_config in file_path_config["file_names_labels"]:
        file_path = os.path.join(file_path_config["folder_path"], process_config["file_name"])
        with uproot.open(file_path) as file:
            tree = file[file_path_config["tree_name"]]
            data = tree.arrays(branches_to_load , library="pd")
            data["label"] = np.full(len(data), process_config["label"])
        out_config[process_config["process_name"]] = {"sample size" : len(data) , "label" : process_config["label"] }
    data = pd.concat(data_frames, ignore_index=True)
    out_config["full-data"] = {"sample size" : len(data) , "nlabel" : np.unique(data["label"]).tolist() }
    update_out_config({"process_name": {"sample size" : len(data) , "label" : process_config["label"] }} , out_config_path)
    return data


