# -*- coding: utf-8 -*-
"""
This program is used to Hierarchical graph pooling based on downsampling
"""

import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        
        # First set of 3 graph convolution layers
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        
        # First pooling layer
        self.pool1 = TopKPooling(256, ratio=0.9)
        
        # Second set of 3 graph convolution layers
        self.conv4 = GCNConv(256, 256)
        self.conv5 = GCNConv(256, 512)
        self.conv6 = GCNConv(512, 1024)
        
        # Second pooling layer
        self.pool2 = TopKPooling(1024, ratio=0.9)
        
        # Final classifier
        self.lin = torch.nn.Linear(1024, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First set of graph convolutions with ReLU and dropout
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # First pooling
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        
        # Second set of graph convolutions with ReLU and dropout
        x = F.relu(self.conv4(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv5(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv6(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second pooling
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        
        # Global pooling (readout) and classifier
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        
        return F.log_softmax(x, dim=1)

# Create a random graph using networkx as an example
G = nx.erdos_renyi_graph(n=100, p=0.05)
data = from_networkx(G)

# Assign random node features and labels
data.x = torch.randn(data.num_nodes, 16)
data.y = torch.randint(0, 2, (1,))

# Initialize model, optimizer, and criterion
model = GNN(16, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Print the node weights after training
print(model.state_dict())



