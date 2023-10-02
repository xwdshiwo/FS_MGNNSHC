# -*- coding: utf-8 -*-
"""
This program is used to implement Supernode discovery
Screening should be done based on clustering results combined with node evaluation
"""

import torch
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch_geometric.utils import from_networkx, to_dense_adj
from scipy.sparse.csgraph import laplacian


class SupernodeDiscovery:
    def __init__(self, node_features, adj_matrix):
        self.node_features = node_features
        self.adj_matrix = adj_matrix.numpy() if torch.is_tensor(adj_matrix) else adj_matrix

    def find_clusters(self):
        # Compute the Laplacian matrix
        laplacian_matrix = laplacian(self.adj_matrix, normed=True)

        # Compute the eigenvectors of the Laplacian matrix
        _, eigenvectors = np.linalg.eigh(laplacian_matrix)

        # Determine the optimal number of clusters k using the elbow method
        silhouette_scores = []
        K = range(2, min(self.adj_matrix.shape[0] - 1, 10))
        for k in K:
            kmeans = KMeans(n_clusters=k).fit(eigenvectors[:, :k])
            score = silhouette_score(eigenvectors[:, :k], kmeans.labels_)
            silhouette_scores.append(score)

        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        print(f"Optimal number of clusters: {optimal_k}")

        # Perform spectral clustering using the best k value
        kmeans = KMeans(n_clusters=optimal_k).fit(eigenvectors[:, :optimal_k])
        return kmeans.labels_

# # Example of usage:
# G = nx.karate_club_graph()
# data = from_networkx(G)
# adj_matrix = to_dense_adj(data.edge_index).squeeze()

# discovery = SupernodeDiscovery(node_features=None, adj_matrix=adj_matrix)
# labels = discovery.find_clusters()
# print("Cluster labels:", labels)
