# -*- coding: utf-8 -*-
"""
This program is used to implement link prediction
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, erdos_renyi_graph
from torch_geometric.data import Data

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv3(x, edge_index)

# Training procedure
def train():
    model.train()

    optimizer.zero_grad()
    z = model(data.x, data.train_pos_edge_index)
    
    # Compute loss based on positive and negative edges
    pos_edge_logits = torch.sigmoid((z[data.train_pos_edge_index[0]] * z[data.train_pos_edge_index[1]]).sum(dim=1))
    neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=data.train_pos_edge_index.size(1))
    neg_edge_logits = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))
    
    loss = F.binary_cross_entropy_with_logits(torch.cat([pos_edge_logits, neg_edge_logits]), torch.cat([torch.ones(pos_edge_logits.size(0)), torch.zeros(neg_edge_logits.size(0))]))
    loss.backward()
    optimizer.step()

    return loss.item()

# Testing procedure
def test(edge_index, eval_mode):
    model.eval()

    with torch.no_grad():
        z = model(data.x, edge_index)

    # Evaluate MRR on positive and negative edges
    pos_edge_logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    neg_edge_index = negative_sampling(edge_index=edge_index, num_nodes=data.num_nodes, num_neg_samples=edge_index.size(1))
    neg_edge_logits = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    
    pos_ranks = torch.argsort(torch.cat([pos_edge_logits, neg_edge_logits])).argsort()[:edge_index.size(1)]
    ranks = pos_ranks + 1
    mrr = (1 / ranks).mean().item()

    return mrr

# Randomly generate a graph using Erdos-Renyi model
num_nodes = 100
edge_index = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=0.05)
x = torch.randn((num_nodes, 16))  # Random node features
data = Data(x=x, edge_index=edge_index)

# Split edges for training and testing (for simplicity, we use the same set here)
data.train_pos_edge_index = edge_index
data.test_pos_edge_index = edge_index

# Define a model and optimizer
model = GCN(num_features=16, hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

best_mrr = 0
for epoch in range(1, 101):
    loss = train()
    mrr = test(data.test_pos_edge_index, eval_mode=True)
    if mrr > best_mrr:
        best_mrr = mrr
        best_data = Data(x=data.x, edge_index=data.edge_index)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, MRR: {mrr:.4f}')

print(f'Best MRR: {best_mrr:.4f}')
