import sys
import os
sys.path.append("/afs/cern.ch/user/s/sapradha/VBF_Analysis_Git/GNN_VBF_QCD_event_classification")
from datasets import create_graph_data , load_data , read_config
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from gnn_model import GNN
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from jet_plotting_utils import plot_confusion_matrix, plot_training_history, plot_roc_curve , plot_jet_image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = read_config("input_config_2022.yml")
pd_loaded_data = load_data(config , "output_config_2022.yml" , nsample = 200000 ,save_csv = True )

sys.exit()
train_df, temp_df = train_test_split(
    pd_loaded_data ,
    test_size=0.4,
    random_state=42,
    shuffle=True
)

# Step 2: Split temp into validation (20%) and test (20%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,   # 0.5 of 40% = 20%
    random_state=42,
    shuffle=True
)

print("Print train , test validation split  :" , len(train_df), len(val_df), len(test_df))

X_train_graphs = create_graph_data(train_df , config)
X_test_graphs = create_graph_data(test_df , config)
X_val_graphs  = create_graph_data(val_df , config)

train_loader = DataLoader(X_train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(X_val_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(X_test_graphs, batch_size=32)



model = GNN(num_features=6).to(device)  # 6 feature for each node 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.to(device))
        loss = criterion(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.to(device))
        loss = criterion(out, data.y.view(-1, 1))
        pred = (out > 0.5).float()
        correct += int((pred == data.y.view(-1, 1)).sum())
    return correct / len(loader.dataset), loss


history = {
    'loss': [],
    'val_loss': [],
    'accuracy': [],
    'val_accuracy': []
}


# Training loop
best_acc = 0
for epoch in range(10):
    loss = train()
    train_acc = test(train_loader)[0]
    val_acc, val_loss = test(val_loader)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')


    # Append to history
    history['loss'].append(loss)
    history['val_loss'].append(val_loss)
    history['accuracy'].append(train_acc)
    history['val_accuracy'].append(val_acc)

plot_training_history(history)
# Load best model
model.load_state_dict(torch.load('best_model.pt'))

# Evaluate on test set
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for data in val_loader:
        out = model(data.to(device))
        y_true.extend(data.to('cpu').y.numpy())
        y_pred.extend(out.to('cpu').numpy())

y_pred = np.array([x[0] for x in y_pred])
pred_discrete = np.where(y_pred > 0.5, 1, 0)
# Plot confusion matrix
plot_confusion_matrix(y_true, pred_discrete)

plot_roc_curve(y_true, y_pred)

y_pred_test = []
for data in test_loader:
    with torch.no_grad():
        output = model(data.to(device))
        # could you change the prediction threshold? Would that make it better?
        y_pred_test.extend(output.to('cpu').numpy())

#y_pred_test = np.array([x[0] for x in y_pred_test])

#solution = pd.DataFrame({'id':test_ids, 'label':y_pred_test})
#solution.to_csv('solution.csv', index=False)

