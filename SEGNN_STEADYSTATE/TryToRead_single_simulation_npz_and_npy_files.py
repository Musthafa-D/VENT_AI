import torch
from torch_geometric.nn import radius_graph, knn_graph, knn

import numpy as np
import plotly.express as px
import os
import sys
print(os.path.abspath("."))
# sys.path.append('../Utility Scripts')

from Utility_functions import print_3D_graph

"""READING NPY AND POINT CLOUD PLOT"""
probes_array = np.load('../.data/100sims_randG(6-10)randN(2-3)/probes/SIM000_probepos.npy')
probes = torch.tensor(probes_array)
print(probes.shape)
print(probes[:10,:])

pos = probes[:,:3].detach().numpy() # first three dimensions are 3D coordinates

fig = px.scatter_3d(x=pos[:,0],y=pos[:,1],z=-1*pos[:,2])
fig.update_traces(marker={'size': 1.5})
fig.update_layout(scene = dict(aspectmode='data'))#, width=600,height=500)
fig.show()

"""EDGE PRODUCTION AND GRAPH PLOT"""
# radius = 0.003
max_neighbours = 10
kneigh = 10

edge_index = knn_graph(probes, kneigh)
# edges = radius_graph(probes, radius)
print(edge_index.shape)

# utility plot function
print_3D_graph(pos, edge_index)

"""MIS preprocessing - first approach"""
# MIS calculation from scikit learn produces noisy values on the outer shell of the surf.vtp, there are two differents schemes here I implemented to solve most of this inconsistency

# 1) I only take nodes belonging to the outer shell (nearest to the surface, smallest SDF possible) and connect among themselves.
# 2) I only take nods belonging to the last two shells. I connect among themselves nodes that belong to different shells and do not connect them if they belong to the same shell. 
#    The nodes in the near-to-outer shell do not have problems in their MIS value as the noisy value happen apporaching the external surface of the original voxelized grid.
"""
spacing = 0.0008
threshold = 2*spacing

# select only nodes near to the surface
indx = (probes[:,3]<threshold)
pos2 = torch.tensor(pos[indx,:])

# create connectivity among surface nodes
kneigh = 10
edge_index = knn_graph(pos2, kneigh)
edges = radius_graph(pos2, radius)

# only select edge-receiving surface nodes, and extract their MIS
edge_reshaped = edge_index.reshape(-1)[:int(sum(indx)*kneigh)]
sdf_reshaped = probes[indx][edge_reshaped][:,4].reshape(-1,kneigh)

def most_frequent(row):
    values, counts = np.unique(row, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]

# for every row, calculate the most occurring MIS value
# the i-th row represents the i-th edge-sending node (nodes belonging to the outer shell)
new_val = np.apply_along_axis(most_frequent, 1, sdf_reshaped)


# change outer shell's MIS values with the ones computed now
probes2 = probes.clone()
probes2[indx,4] = torch.tensor(new_val)

print_3D_graph(pos, edges, color = (probes[:,4])) # original

# print_3D_graph(pos2, edges, color = (probes[indx,4]))
# print_3D_graph(pos2, edge_index, color = (probes2[indx,4]))

print_3D_graph(pos, edges, color = (probes2[:,4])) # processed
"""

"""MIS preprocessing - second approach"""
spacing = 0.0008 # same spacing as data produced by moebius (it coincides with the smallest value that SDF can take)

neigh = 5
kneigh_shells = 10
radius = 0.0005
edges = radius_graph(probes[:,:3], radius)

# selecting near-surface nodes (outer shell)
idx_shell1 = (probes[:,3]<=spacing)#*(probes[:,5]==2)
pos_shell1 = probes[idx_shell1,:3]

# selecting near-to-outer shell nodes
idx_shell2 = (probes[:,3]>spacing)*(probes[:,3]<=3*spacing)#*(probes[:,5]==2)
pos_shell2 = probes[idx_shell2,:3]

# ordered array of the two shells' nodes
shells_pos = torch.cat((pos_shell1,pos_shell2),0)

# creating edges between set1 and set2
edge_indexes = knn(pos_shell2,pos_shell1, kneigh_shells)
# print(edge_indexes.shape)


# storing indexes of edge-receiving nodes
edge_reshaped = edge_indexes.reshape(-1)[int(sum(idx_shell1)*kneigh_shells):]

# get the MIS value for the edge-receiving nodes
sdf_reshaped = probes[idx_shell2][edge_reshaped][:,4].reshape(-1,kneigh_shells)
# print(edge_indexes.shape, edge_reshaped.shape, sdf_reshaped.shape)


def most_frequent(row):
    i = 0
    values, counts = np.unique(row, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]

# for every row, calculate the most occurring MIS value
# the i-th row represents the i-th edge-sending node (nodes belonging to the outer shell)
new_val = np.apply_along_axis(most_frequent, 1, sdf_reshaped)


# change outer shell's MIS values with the ones computed now
probes3 = probes.clone()
probes3[idx_shell1,4] = torch.tensor(new_val)


edges = radius_graph(probes, radius)
outer_edges = knn_graph(pos_shell1, 5)#[:,:(kneigh_shells*len(pos_shell1))]

# indexing on the whole set of points has to be coherent with nodes ordering 
# (knn function with two sets of nodes, orders both indexing from 0)
edge_indexes_plot = edge_indexes.clone()
edge_indexes_plot[1,:] += edge_indexes_plot[0,-1]+1

probes_cat = torch.cat((probes[idx_shell1],probes[idx_shell2]), 0)

print_3D_graph(probes[:,:3], edges, color = probes[:,4]) # original

# print_3D_graph(shells_pos, edge_indexes_plot, color = (probes_cat[:,4]))
# print_3D_graph(pos_shell1, outer_edges, color = (probes3[idx_shell1,4]))

print_3D_graph(probes[:,:3], edges, color = (probes3[:,4])) # processed

"""READING NPZ"""
fields = np.load('../.data/100sims_randG(6-10)randN(2-3)/fields/SIM000_features.npz')
features = []

for j,i in enumerate(fields.files):
    
    features.append(fields[i])
feat = torch.tensor(features)   
print('Feature shape:', feat.shape)

feat = feat.reshape(-1,10,probes.shape[0]) # the shape is [#frames, #phys_quantities, #nodes]
'''
feat = torch.einsum('ijk->ikj', feat) # now physical quantities become last_dim

print('Feature shape', feat.shape)


vel = feat[...,1:4]
pres = feat[...,0]
magvel = torch.norm(vel, dim = -1)
print(vel.shape, magvel.shape)
'''

idx = -1
radius = 0.003
neigh = 10

edges = radius_graph(probes[:2100,:3], radius)
edges = knn_graph(probes[:2100,:3], neigh)

print_3D_graph(probes[:2100,:3], edges, magvel[idx])
# print_3D_graph(probes[:2100,:3], edges, color=feat[idx,:,0])