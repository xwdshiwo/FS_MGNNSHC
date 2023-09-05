# -*- coding: utf-8 -*-

'''
This program is used to implement graph purification
'''
import torch
import networkx as nx

def compute_adjacency_matrix(graph):
    return torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=torch.float32)

def svd_based_filtering(adj_matrix, top_k_ratio):
    # SVD decomposition
    U, S, V = torch.svd(adj_matrix)
    
    # Determine k based on the top_k_ratio
    k = int(S.shape[0] * top_k_ratio)
    
    # Reconstruct the adjacency matrix using top-k singular values
    reconstructed = torch.mm(U[:, :k], torch.mm(torch.diag_embed(S[:k]), V[:, :k].t()))
    
    return reconstructed

def filter_graphs(graphs, top_k_ratio=0.9):
    purified_adj_matrices = []
    for graph in graphs:
        adj_matrix = compute_adjacency_matrix(graph)
        purified_matrix = svd_based_filtering(adj_matrix, top_k_ratio)
        purified_adj_matrices.append(purified_matrix)
    return purified_adj_matrices

# Example usage:
# Define a list of graphs using NetworkX
graphs = [nx.gnp_random_graph(10, 0.5), nx.gnp_random_graph(10, 0.4)]
purified_adj_matrices = filter_graphs(graphs)

for matrix in purified_adj_matrices:
    print(matrix)

