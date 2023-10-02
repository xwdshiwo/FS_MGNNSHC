# -*- coding: utf-8 -*-

'''
This program is used to implement graph purification
'''
import torch
import networkx as nx

class GraphProcessor:
    def __init__(self):
        pass

    def compute_adjacency_matrix(self, graph):
        return torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=torch.float32)

    def svd_based_filtering(self, adj_matrix, top_k_ratio=0.9):
        # SVD decomposition
        U, S, V = torch.svd(adj_matrix)

        # Determine k based on the top_k_ratio
        k = int(S.shape[0] * top_k_ratio)

        # Reconstruct the adjacency matrix using top-k singular values
        reconstructed = torch.mm(U[:, :k], torch.mm(torch.diag_embed(S[:k]), V[:, :k].t()))

        return reconstructed

    def filter_graph_adj_matrices(self, adj_matrices, top_k_ratio=0.9):
        purified_adj_matrices = []
        for adj_matrix in adj_matrices:
            purified_matrix = self.svd_based_filtering(adj_matrix, top_k_ratio)
            purified_adj_matrices.append(purified_matrix)
        return purified_adj_matrices

# # Example usage:

# # Define a list of adjacency matrices for your graphs
# adj_matrices = [
#     GraphProcessor().compute_adjacency_matrix(nx.gnp_random_graph(10, 0.5)),
#     GraphProcessor().compute_adjacency_matrix(nx.gnp_random_graph(10, 0.4))
# ]

# processor = GraphProcessor()
# purified_adj_matrices = processor.filter_graph_adj_matrices(adj_matrices)

# for matrix in purified_adj_matrices:
#     print(matrix)
