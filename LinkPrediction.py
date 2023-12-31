# -*- coding: utf-8 -*-

"""
This program is used to implement link prediction
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, dense_to_sparse
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

class LinkPrediction:
    def __init__(self, adj_matrix, node_features, hidden_channels=16, lr=0.01, weight_decay=0.001):
        self.data = Data(x=node_features, edge_index=dense_to_sparse(adj_matrix)[0])
        self.data.train_pos_edge_index = self.data.edge_index
        self.data.test_pos_edge_index = self.data.edge_index

        self.model = GCN(num_features=node_features.shape[1], hidden_channels=hidden_channels)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self):
        self.model.train()

        self.optimizer.zero_grad()
        z = self.model(self.data.x, self.data.train_pos_edge_index)

        pos_edge_logits = torch.sigmoid((z[self.data.train_pos_edge_index[0]] * z[self.data.train_pos_edge_index[1]]).sum(dim=1))
        neg_edge_index = negative_sampling(edge_index=self.data.train_pos_edge_index, num_nodes=self.data.num_nodes, num_neg_samples=self.data.train_pos_edge_index.size(1))
        neg_edge_logits = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))

        loss = F.binary_cross_entropy_with_logits(torch.cat([pos_edge_logits, neg_edge_logits]), torch.cat([torch.ones(pos_edge_logits.size(0)), torch.zeros(neg_edge_logits.size(0))]))
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss.item()

    def test(self):
        self.model.eval()

        with torch.no_grad():
            z = self.model(self.data.x, self.data.test_pos_edge_index)

        pos_edge_logits = (z[self.data.test_pos_edge_index[0]] * z[self.data.test_pos_edge_index[1]]).sum(dim=1)
        neg_edge_index = negative_sampling(edge_index=self.data.test_pos_edge_index, num_nodes=self.data.num_nodes, num_neg_samples=self.data.test_pos_edge_index.size(1))
        neg_edge_logits = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

        pos_ranks = torch.argsort(torch.cat([pos_edge_logits, neg_edge_logits])).argsort()[:self.data.test_pos_edge_index.size(1)]
        ranks = pos_ranks + 1
        mrr = (1 / ranks).mean().item()

        return mrr

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            z = self.model(self.data.x, self.data.edge_index)
        edge_scores = (z[self.data.edge_index[0]] * z[self.data.edge_index[1]]).sum(dim=1)
        return edge_scores
    
    def save_best_model(self):
        self.best_model_state = self.model.state_dict()
    
    def load_best_model(self):
        self.model.load_state_dict(self.best_model_state)
    
    def predict_new_adj_matrix(self, threshold=0.5):
        edge_scores = self.predict()
        predicted_adj_matrix = (edge_scores > threshold).int().reshape(adj_matrix.shape)
        return predicted_adj_matrix

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            z = self.model(self.data.x, self.data.edge_index)
            
        # 创建所有可能的边对
        row, col = torch.combinations(torch.arange(z.size(0)), 2).t()
        edge_scores = torch.sigmoid((z[row] * z[col]).sum(dim=1))
        
        return edge_scores, row, col
    
    def predict_new_adj_matrix(self):
        self.model.eval()
        num_nodes = self.data.num_nodes
        new_adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        with torch.no_grad():
            z = self.model(self.data.x, self.data.edge_index)
            edge_scores = (z[self.data.edge_index[0]] * z[self.data.edge_index[1]]).sum(dim=1)
            # threshold to decide which edges to add; this can be adjusted based on your needs
            threshold = edge_scores.mean()
            chosen_edges = edge_scores >= threshold
            chosen_row, chosen_col = self.data.edge_index[:, chosen_edges]
            new_adj_matrix[chosen_row, chosen_col] = 1
        return new_adj_matrix

# Example usage:

# num_nodes = 100
# adj_matrix = torch.bernoulli(0.05 * torch.ones((num_nodes, num_nodes)))
# node_features = torch.randn((num_nodes, 16))

# lp = LinkPrediction(adj_matrix, node_features)

# best_mrr = 0
# for epoch in range(1, 101):
#     loss = lp.train()
#     mrr = lp.test()
#     if mrr > best_mrr:
#         best_mrr = mrr
#         lp.save_best_model()
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, MRR: {mrr:.4f}')

# lp.load_best_model()
# predicted_adj_matrix = lp.predict_new_adj_matrix()
# print(predicted_adj_matrix)
