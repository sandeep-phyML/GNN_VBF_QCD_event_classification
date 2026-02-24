import uproot 
import numpy as np
import pandas as pd
import torch
import yaml
import os 
from torch_geometric.data import Data
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

def load_data(config , out_config_path , nsample = None , save_csv = False):
    data_frames = []
    out_config = {}
    file_path_config = config["train_mclass_files_labels"]
    branches_to_load =  config["extra_features"]
    for node_feature in config["node_features"]:
        branches_to_load += node_feature
    for process_config in file_path_config["file_names_labels"]:
        file_path = os.path.join(file_path_config["folder_path"], process_config["file_name"])
        with uproot.open(file_path) as file:
            tree = file[file_path_config["tree_name"]]
            data = tree.arrays(branches_to_load + process_config["weight_features"], library="pd")
            data["label"] = np.full(len(data), process_config["label"])
            data["sample_weight"] = np.prod(data[process_config["weight_features"]], axis=1)
            data = data.drop(columns=process_config["weight_features"])
        out_config[process_config["process_name"]] = {"sample size" : len(data) , "label" : process_config["label"] }
        if nsample :
            replace = nsample > len(data)
            data = data.sample(n=nsample, replace=replace, random_state=42)
        data_frames.append(data)
    data = pd.concat(data_frames, ignore_index=True)
    nan_count = data.isna().sum().sum()
    inf_count = np.isinf(data.to_numpy()).sum()
    data.replace([np.inf, -np.inf], 0).fillna(0)
    for graph_index in range(len(config["node_features"])):
        labels, counts = np.unique(data["label"], return_counts=True)
        data[f"mask_{graph_index}"] = data[config["node_features"][graph_index][0]] >= 0.0
    out_config["full-data"] = {"sample size" : len(data) , "nlabel_counts" : dict(zip(labels.astype(float), counts.astype(float))) , "nan count" : int(nan_count) , "inf count" : int(inf_count)}
    update_out_config(out_config , out_config_path)
    if save_csv :
        replace = 10000 > len(data)
        data.sample(n=10000, replace = replace ,random_state=42).to_csv("test_sample_gnn.csv", index=False)
    return data


def create_graph_data(pd_data , config):
    # create a list of graphs from the test data
    graphs = []
    edges = []
    data = None
    for index, row in pd_data.iterrows():
        graph = []
        for group in config["node_features"]:
            graph.append(row[group].to_list())
        adjacency_matrix = np.zeros((len(graph), len(graph)))
        marks_array = np.array([row[f"mask_{i}"] for i in range(len(graph))])
        for i in range(len(graph)):
            if marks_array[i] != 0:
                adjacency_matrix[i] = marks_array
        graph = np.array(graph)
        
        x = torch.FloatTensor(graph).view(-1, 6)  # Shape: (n_nodes, 6)
        edge_index = torch.nonzero(torch.FloatTensor(adjacency_matrix)).t()  # Shape: (2, num_edges)
        data = Data(x=x, edge_index=edge_index , weight=torch.tensor([row['sample_weight']], dtype=torch.float) , y=torch.tensor([row['label']], dtype=torch.float))
        if index == 0:
            print(f"Graph shape: {graph.shape}, Adjacency Matrix shape: {adjacency_matrix.shape}")
            print(f"x : shape {x.shape} , edge_index : shape {edge_index.shape}")
            print(marks_array)
            print(graphs)
            print(f"Adjacency Matrix: {adjacency_matrix}")
        graphs.append(data)
    return graphs
