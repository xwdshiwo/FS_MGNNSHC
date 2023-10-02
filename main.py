# -*- coding: utf-8 -*-
"""

@author: Weidong Xie
"""
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from MultiGraphFilter import MultiDimensionalGCNLayer
from MultiGraphFilter import MultiDimensionalGCN
from GraphPurification import GraphProcessor
processor = GraphProcessor()
from LinkPrediction import GCN
from LinkPrediction import LinkPrediction
from MultidimensionalNodeEvaluator import FeatureEvaluator
from SupernodeDiscovery import SupernodeDiscovery
from GraphPooling import GNNpool

num_nodes = 30
in_features = 40
out_features = 30
num_dimensions = 2
Graph_filter_layers = 10
# Simulated input features, a simple example, can be modified based on real data
inputs = torch.rand((num_nodes, in_features))
y = np.random.randint(2, size=30)
# The 2-dimensional adjacency matrix is simulated
adjs = [torch.eye(num_nodes), torch.eye(num_nodes)]


# Model for graph filter
model = MultiDimensionalGCN(in_features, out_features, num_dimensions)


# Target output
targets = torch.rand((num_nodes, out_features))

# Graph purification
adjs = processor.filter_graph_adj_matrices(adjs)
# purified_adj_matrices = processor.filter_graph_adj_matrices(adjs)

# Filter
for i in range(Graph_filter_layers):
    outputs = model(inputs, adjs)

lp = LinkPrediction(adjs[0], outputs)

best_mrr = 0
for epoch in range(1, 10):
    loss = lp.train()
    mrr = lp.test()
    if mrr > best_mrr:
        best_mrr = mrr
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, MRR: {mrr:.4f}')

predicted_adj_matrix = lp.predict()
evaluator = FeatureEvaluator(outputs.detach().numpy(), y)
final_ranking = evaluator.get_feature_ranking()
discovery = SupernodeDiscovery(node_features=None, adj_matrix=adjs[0])
labels = discovery.find_clusters()
print('cluster result:',labels)
'''
You can extract the features that need to be analyzed further 
based on the clustering results and feature ranking results,
where all features are assumed to be used for the graph classification task.
'''

# Initialize model, optimizer, and criterion
model2 = GNNpool(30, 30)
optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
y = torch.tensor([1]) # one sample as example
# outputs = outputs.long()
adj = adjs[0].long()
# Training loop
edge_index = (adj > 0.5).nonzero(as_tuple=False).t()
model.train()
torch.autograd.set_detect_anomaly(True)
outputs2 = outputs[:]
for epoch in range(10):
    optimizer.zero_grad()
    out = model2(outputs2, edge_index)
    loss = criterion(out,y)
    loss.backward(retain_graph=True)
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Print the node weights after training
print(model.state_dict())


