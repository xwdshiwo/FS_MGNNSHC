# -*- coding: utf-8 -*-
"""
This program is used to implement Hierarchical graph pooling based on downsampling
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.data import Data

class GNNpool(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNpool, self).__init__()
        
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

    def forward(self, x, edge_index, batch=None):
        # First set of graph convolutions with ReLU and dropout
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # First pooling
        x, edge_index, _, batch, perm1, _ = self.pool1(x, edge_index, batch=batch)
        
        # Second set of graph convolutions with ReLU and dropout
        x = F.relu(self.conv4(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv5(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv6(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second pooling
        x, edge_index, _, batch, perm2, _ = self.pool2(x, edge_index, batch=batch)
        
        # Global pooling (readout) and classifier
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        
        return F.log_softmax(x, dim=1), perm1, perm2

# # Sample usage
# adj_matrix = torch.rand((100, 100))  # Sample adjacency matrix
# node_features = torch.randn(100, 16)  # Sample node features

# # Convert adjacency matrix to edge index format
# edge_index = (adj_matrix > 0.5).nonzero(as_tuple=False).t()

# # Initialize model, optimizer, and criterion
# model = GNNpool(16, 30)  # Change the out_channels to 30
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = torch.nn.CrossEntropyLoss()

# label = torch.randint(0, 2, (1,))  # Change the range to 0 to 29 for generating labels


# model.train()
# for epoch in range(100):
#     optimizer.zero_grad()
#     out, perm1, perm2 = model(node_features, edge_index)
#     loss = criterion(out, label)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# # Print the node ranks after training
# print("Node ranks after first pooling:", perm1)
# print("Node ranks after second pooling:", perm2)
