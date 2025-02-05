# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:52:55 2024

@author: DiwanMohideen
"""
from graphxai.datasets import ShapeGGen

dataset = ShapeGGen(
    model_layers = 3,
    num_subgraphs = 100,
    subgraph_size = 12,
    prob_connection = 0.1,
    add_sensitive_feature = False
)

data = dataset.get_graph(use_fixed_split=True)
print(data)

node_idx = data.test_mask.nonzero(as_tuple=True)[0][13]
print(node_idx)
print(node_idx.shape)

gt_exp = dataset.explanations[node_idx]
print(gt_exp)
gt_exp[0].visualize_node(num_hops = 3, additional_hops = 0, graph_data = data, show = True)