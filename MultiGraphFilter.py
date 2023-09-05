# -*- coding: utf-8 -*-


'''
This program is used to implement Multi-dimensional graph filter

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiDimensionalGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_dimensions):
        super(MultiDimensionalGCNLayer, self).__init__()
        self.num_dimensions = num_dimensions

        self.Thetas = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_features, out_features)) 
                                       for _ in range(num_dimensions)])
        self.W = nn.Parameter(torch.FloatTensor(out_features, out_features))

        for theta in self.Thetas:
            nn.init.xavier_uniform_(theta)
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, adjs):
        
        # Corresponding to Equations 4 to 8 in the paper


        
        F_djs = [F.relu(torch.mm(input, theta)) for theta in self.Thetas]
        F_gis = [F.relu(torch.mm(input, theta)) for theta in self.Thetas]
        
        F_wdis = [torch.spmm(adj, F_dj) for adj, F_dj in zip(adjs, F_djs)]
        
        beta_gds = [
            torch.trace(theta_g @ self.W @ theta_d.t()) 
            for theta_d in self.Thetas for theta_g in self.Thetas
        ]
        beta_gds = F.softmax(torch.stack(beta_gds, dim=0), dim=0)
        
        F_adis = [
            sum([beta_gd * F_gi for beta_gd, F_gi in zip(beta_gds, F_gis)])
            for _ in range(self.num_dimensions)
        ]
        
        eta = 0.5
        F_is = [eta * F_wdi + (1 - eta) * F_adi for F_wdi, F_adi in zip(F_wdis, F_adis)]
        return sum(F_is)

class MultiDimensionalGCN(nn.Module):
    def __init__(self, in_features, out_features, num_dimensions):
        super(MultiDimensionalGCN, self).__init__()
        self.layer = MultiDimensionalGCNLayer(in_features, out_features, num_dimensions)

    def forward(self, input, adjs):
        return self.layer(input, adjs)

num_nodes = 3
in_features = 4
out_features = 2
num_dimensions = 2

# Simulated input features, a simple example, can be modified based on real data
inputs = torch.rand((num_nodes, in_features))

# The 2-dimensional adjacency matrix is simulated
adjs = [torch.eye(num_nodes), torch.eye(num_nodes)]


# Model
model = MultiDimensionalGCN(in_features, out_features, num_dimensions)
criterion = nn.CrossEntropyLoss()  # 仅作为示例，可以根据需求修改
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Target output
targets = torch.rand((num_nodes, out_features))

# Train
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs, adjs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

