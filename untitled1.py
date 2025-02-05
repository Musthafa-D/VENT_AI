# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:52:55 2024

@author: DiwanMohideen
"""
from graphxai.datasets import ShapeGGen
import torch
from interpret import GNNExplainer_get_explanation_node
from torch_geometric.nn import GINConv
from graphxai.explainers import GNNExplainer
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph, to_networkx, degree
import networkx as nx
from torch_geometric.data import Data

class GNN(torch.nn.Module):
    def __init__(self,input_feat, hidden_channels, classes = 2):
        super(GNN, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin2 = GINConv(self.mlp_gin2)
        self.mlp_gin3 = torch.nn.Linear(hidden_channels, classes)
        self.gin3 = GINConv(self.mlp_gin3)

    def forward(self, x, edge_index):
        # NOTE: our provided testing function assumes no softmax
        #   output from the forward call.
        x = self.gin1(x, edge_index)
        x = x.relu()
        x = self.gin2(x, edge_index)
        x = x.relu()
        x = self.gin3(x, edge_index)
        return x

dataset = ShapeGGen(
    model_layers = 3,
    num_subgraphs = 100,
    subgraph_size = 12,
    prob_connection = 0.1,
    add_sensitive_feature = False
)

model = GNN(dataset.n_features, 64)
data = dataset.get_graph(use_fixed_split=True)
# print(data)

node_idx = data.test_mask.nonzero(as_tuple=True)[0][13]
# print(f"node_idx: {node_idx}")

# print(f"edge_index: {data.edge_index}")
# print(f"shape of edge_index: {data.edge_index.shape}")

pred = model(data.x, data.edge_index)[node_idx,:].argmax(dim=0)
# print(pred)
# print(pred.shape)

gt_exp = dataset.explanations[node_idx]
# print(len(gt_exp))
# gt_exp[0].visualize_node(num_hops = 3, additional_hops = 0, graph_data = data, show = True)

# Function to visualize the entire graph
def visualize_entire_graph(data):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(12, 12))
    nx.draw(G, with_labels=True, node_color='blue', edge_color='gray')
    plt.title("Entire Graph")
    plt.show()

# Function to print node degrees
def print_node_degrees(edge_index):
    degrees = degree(edge_index[0], dtype=torch.long)
    for node, deg in enumerate(degrees):
        print(f"Node {node}: Degree {deg.item()}")

# Visualize subgraphs
def visualize_subgraph(edge_index, node_idx, num_hops):
    subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=True)
    G = to_networkx(Data(edge_index=sub_edge_index), to_undirected=True)
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color='yellow', edge_color='gray')
    plt.title(f"Subgraph for node {node_idx} with {num_hops} hops")
    plt.show()

print("ShapeGGen Dataset:")
visualize_entire_graph(data)

print("Node degrees in ShapeGGen dataset:")
print_node_degrees(data.edge_index)
# Example visualization
visualize_subgraph(data.edge_index, node_idx=5, num_hops=3)

# print("GNNExplainer")
# data = dataset.get_graph(use_fixed_split=True)
# graph_data = data
# device="cuda"
# forward_kwargs={'x': data.x.to(device),
#                         'node_idx': int(node_idx),
#                         'edge_index': data.edge_index.to(device)}
# exp_method = GNNExplainer(model)

# gnn_explainer_exp = exp_method.get_explanation_node(**forward_kwargs)
# fig, ax = plt.subplots(figsize=(10, 7.5))
# gnn_explainer_exp.visualize_node(
#     num_hops=3, additional_hops=0, graph_data=graph_data, 
#     ax=ax, show_node_labels=True, norm_imps=True
# )

# ax.set_title("Node Explanation", fontsize=20, fontweight='semibold')
# plt.show()