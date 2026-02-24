import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

class GNN(nn.Module):
    def __init__(self, num_features):
        super(GNN, self).__init__()
        # Graph convolution layers
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)

        # Dense layers
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolution layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))

        # Global pooling
        x = global_mean_pool(x, batch)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return torch.sigmoid(x)
