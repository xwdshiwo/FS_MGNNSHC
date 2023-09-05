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
from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_dense_adj
from scipy.sparse.csgraph import laplacian

# Create a graph structure example using networkx
G = nx.karate_club_graph()
data = from_networkx(G)

# Obtain the adjacency matrix of the graph and compute its Laplacian
adj_matrix = to_dense_adj(data.edge_index).squeeze().numpy()
laplacian_matrix = laplacian(adj_matrix, normed=True)

# Compute the eigenvectors of the Laplacian matrix
eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

# Determine the optimal number of clusters k using the elbow method
silhouette_scores = []
K = range(2, min(len(G.nodes()) - 1, 10))  # Ensure the value of k is within a valid range
for k in K:
    kmeans = KMeans(n_clusters=k).fit(eigenvectors[:, :k])
    score = silhouette_score(eigenvectors[:, :k], kmeans.labels_)
    silhouette_scores.append(score)

optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters: {optimal_k}")

# Perform spectral clustering using the best k value
kmeans = KMeans(n_clusters=optimal_k).fit(eigenvectors[:, :optimal_k])
labels = kmeans.labels_

print("Cluster labels:", labels)

