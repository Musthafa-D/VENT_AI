import torch
from torch_geometric.nn import knn_graph, knn


import numpy as np
from tqdm import tqdm

import os
import glob
import re

# sys.path.append('../Utility Scripts')

from Utility_functions import print_3D_graph

import params

print('NSIM:', params.NSIM)
print('NSTEP:', params.NSTEP)
print('FLUID_PROBES:', params.NFLUID_PROBES)
print('IO_PROBES:', params.NIO_PROBES)
print('NPOS:', params.NPOS)
print('NFEATURES:', params.NFEATURES)
print('DATADIR:', params.DATADIR)

#dati = np.load("../.data/Extracted_data/probes/SIM000_probepos.npy")
dati = np.load("../.data/Dataset_10sims_10G2N/probes/SIM000_probepos.npy")

print(len(dati))
print(dati) 

data = np.load("../.data/Dataset_10sims_10G2N/fields/SIM000_features.npz")
#data = np.load("../.data/Dataset_10sims_Moebius/fields/SIM000_probes.npz")


print(data.files)

for name in data.files:
    print(f"{name}:")
    print(data[name].tolist())
    print(data[name].shape)
    print(len(data[name]))
    print("\n" + "-"*50 + "\n")  

"""READING NPZ AND NPY"""
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
probes_path = os.path.join(params.DATADIR, 'probes')
probes = sorted(glob.glob(os.path.join(probes_path, '*.npy')), key = numericalSort)
print(probes[0])

fields_path = os.path.join(params.DATADIR, 'fields')
fields = sorted(glob.glob(os.path.join(fields_path, '*.npz')), key = numericalSort)
print(fields[0])

# probe_data_list = []

# for data_file in probes:
#     read_points = np.load(data_file)
#     probe_data_list.append(torch.from_numpy(read_points))

# pos_list = probe_data_list
# #print(pos_list)
# print(len(pos_list))
# print(pos_list[0].tolist())
# print(len(pos_list[0]))
# print(pos_list[0])
# features_list = []

# for data_file in fields:
#     read_features = np.load(data_file)
#     num_points = None
#     for _, filedat in enumerate(read_features.files):
#         if num_points is None:
#             num_points = len(read_features[filedat])
#             print(len(read_features[filedat]))
#             feature_tensor = torch.zeros(num_points, 10)
#             print(len(feature_tensor))
#         #print(read_features[filedat])
#         feature_tensor[:, _] = torch.from_numpy(read_features[filedat]).squeeze()
#     features_list.append(feature_tensor)

# feat_list = features_list
# # print(feat_list)
# # print(len(feat_list))
# # print(pos_list[0].tolist())
# # print(len(feat_list[0]))
# # print(feat_list[0])
print(len(probes))
assert params.NSIM == len(probes)

dataset_len = len(probes)  # number of simulations
print(probes)
positions = np.empty((params.NSIM, params.NPOS, params.NFLUID_PROBES + params.NIO_PROBES))
#print(positions)
for i, data in enumerate(probes):
    read_points = np.load(data)
    #print(read_points)
    positions[i,:] = read_points.T
    
pos = torch.from_numpy(np.einsum('ijk->ikj', positions))
print(len(pos))
print(pos[0])
#print(pos[1])
print(pos.shape) # [params.NSIM, params.NFLUID_PROBES + params.NIO_PROBES, params.NPOS]

# features = np.empty((params.NSIM, params.NSIM * NUM_FEATURES, NUM_NODES))
####positions = np.empty((params.NSIM, params.NPOS, params.NFLUID_PROBES + params.NIO_PROBES))
features = np.zeros((params.NSIM, params.NFRAME * params.NFEATURES, params.NFLUID_PROBES + params.NIO_PROBES))
feat = features.reshape((params.NSIM, params.NFRAME, params.NFEATURES, params.NFLUID_PROBES + params.NIO_PROBES))
print(features.shape)
#print(feat)
print('Features shape:', features.shape)     # [1000, 60 (6*10), 2000]
#print(fields)
for i, data in enumerate(tqdm(fields)):

    read_features = np.load(data)
    #print(read_features.files)
    #print(read_features)
    for j, filedat in enumerate(read_features.files): 
        print(i, j, filedat, read_features[filedat].shape, features[i, j, :].shape)
        #features[i, j, :params.NFLUID_PROBES] = read_features[filedat]
        features[i, j, :] = read_features[filedat]
        #print(features[i, j, :params.NFLUID_PROBES])

print(features[0][0].tolist())

#print(features[0][1].tolist()) # vel_x
feat = features.reshape((params.NSIM, params.NFRAME, params.NFEATURES, params.NFLUID_PROBES + params.NIO_PROBES))
print(feat.shape)
print(feat[0][0]) # vel_x
feat = torch.from_numpy(np.einsum('ijkl->ijlk', feat))
print(feat.shape)
print(feat[0][0][0]) # vel_x
#print(len(feat[0][0]))

# new_pos_list = [pos.unsqueeze(0).repeat(params.NFRAME, 1, 1) for pos in pos_list]
# print('Node position and labels data:', [pos.shape for pos in new_pos_list])
# print('Fluid dynamic features   data:', [feat.shape for feat in feat_list])

# dataset = [torch.cat([new_pos, feat.unsqueeze(0)], dim=-1) for new_pos, feat in zip(new_pos_list, feat_list)]

# print('Concatenated total data:', [data.shape for data in dataset])
new_pos = pos.unsqueeze(1).repeat(1, params.NFRAME, 1, 1)
print('Node position and labels data:', new_pos.shape)
print('Fluid dynamic features   data:', feat.shape)

dataset = torch.cat([new_pos, feat], dim = -1)
print(len(dataset))
print(dataset[0][0][13])

print('Concatenated total data:', dataset.shape) # [100,6,2100,16]

# cut to last frame, we are interested in the steady state

print('Data with all time frames:', dataset.shape)     # [1000,6,2100,16] [samples,frames, nodes, tot_features]

new_dataset = dataset[:, -1:, :, :].squeeze(1) # only last frame
print(new_dataset[0][13]) # coord_x, coord_y, coord_z, mis, size, sdf, press, vel_x, vel_y, vel_z, stress_ij

print('Data with only last frame:', new_dataset.shape)    # [samples, 2100, 16]

"""PREPROCESSING OF MIS VALUES"""
# clean up MIS to have smooth values next to wall (due to the staircase-like discretization)

spacing = params.SPACING
kneigh_shells = params.NEIGHBOURS

idx = 1
nodes = new_dataset.shape[1]
edge_index = knn_graph(new_dataset[idx,:,:3], 10)

print_3D_graph(new_dataset[idx,:,:3], edge_index, color = new_dataset[idx,:,4]) # original

for data in tqdm(new_dataset):

    # selecting near-surface nodes (outer shell)
    idx_shell1 = (data[:,3] <= spacing)
    pos_shell1 = data[idx_shell1,:3]

    # selecting near-to-outer shell nodes
    idx_shell2 = (data[:,3]>spacing)*(data[:,3]<=3*spacing)
    pos_shell2 = data[idx_shell2,:3]

    # ordered array of the two shells' nodes
    shells_pos = torch.cat((pos_shell1,pos_shell2),0)

    # creating edges between set1 and set2
    edge_indexes = knn(pos_shell2,pos_shell1, kneigh_shells)

    # storing indexes of edge-receiving nodes
    edge_reshaped = edge_indexes.reshape(-1)[int(sum(idx_shell1)*kneigh_shells):]

    # get the MIS value for the edge-receiving nodes
    sdf_reshaped = data[idx_shell2][edge_reshaped][:,4].reshape(-1,kneigh_shells)

    def most_frequent(row):
        i = 0
        values, counts = np.unique(row, return_counts=True)
        ind = np.argmax(counts)
        return values[ind]

    # for every row, calculate the most occurring MIS value
    # the i-th row represents the i-th edge-sending node (nodes belonging to the outer shell)
    new_val = np.apply_along_axis(most_frequent, 1, sdf_reshaped)

    # change outer shell's MIS values with the ones computed now
    data[idx_shell1,4] = torch.tensor(new_val)

    # indexing on the whole set of points has to be coherent with nodes ordering 
    # (knn function with two sets of nodes, orders both indexing from 0)
    edge_indexes_plot = edge_indexes.clone()
    edge_indexes_plot[1,:] += edge_indexes_plot[0,-1]+1


print_3D_graph(new_dataset[idx,:,:3], edge_index, color = new_dataset[idx,:,4]) # processed

"""READ A GRAPH AND CONSTRUCT EDGE CONNECTIVITY"""
# check if given the big tensor storing all simulations, I am able to recover a specific steady state and its quantities

single_graph = new_dataset[0,...]

i=0
while i < 61:#len(single_graph):
    print(f"{i}th node: {torch.norm(single_graph[i,7:10])}")
    i += 1

neighbours = params.NEIGHBOURS

edge_index = knn_graph(single_graph[..., :3], neighbours)
print(single_graph[1]) # coord_x, coord_y, coord_z, mis, sdf, size, press, vel_x, vel_y, vel_z, stress_ij
# print(single_graph[1])
# print(single_graph[2])
# print(single_graph[3])
# print(single_graph[4])
#magvel = torch.norm(single_graph[..., 4:7], dim=-1)
magvel = torch.norm(single_graph[..., 7:10], dim=-1) # 7:10 are vel_x, vel_y, vel_z
#print(magvel)   

print('Single graph:', single_graph.shape)
print('Edge index  :', edge_index.shape)
print('Velocity magnitude:', magvel.shape)

print_3D_graph(single_graph, edge_index, magvel)

# [(0, -1), (0, 1), (1, 2), (1, 4), (2, 3),
# (4, 5), (3, 6), (3, 8), (5, 10), (5, 12), 
# (6, 7), (8, 9), (10, 11), (12, 13)]

"""CHECKING MIS PREPROCESSING"""
idx = 0 # steady state idx
neighbours = params.NEIGHBOURS
nodes = new_dataset.shape[1]
new_dataset_copy = new_dataset.clone()

# print(new_dataset_copy.shape)

edge_index = knn_graph(new_dataset_copy[idx,:,:3], neighbours)
print('knn_graph:', edge_index.shape)

print_3D_graph(new_dataset_copy[idx,:,:3], edge_index, color = new_dataset_copy[idx,:,4]) # original    
print_3D_graph(new_dataset[idx,:,:3], edge_index, color = new_dataset[idx,:,4]) # processed

counts = []

# checking number of neighbors per node
for i in range(params.NFLUID_PROBES + params.NIO_PROBES):
    count = (edge_index[1,:] == i)
    count = count.sum()
    counts.append(count)

print('edge_index[:10]', edge_index[:10])
print('counts[:10]:', counts[:10])

count = torch.tensor(counts, dtype=torch.float32)
print('Mean:', count.mean(), 'Std:', count.std())

"""SAVING DATASET"""
saved_dataset = new_dataset.reshape(-1, params.NFLUID_PROBES + params.NIO_PROBES, params.NFEATURES + 6).float()
raw_path = os.path.join(params.DATADIR, 'raw')
processed_path = os.path.join(params.DATADIR, 'processed')
dataset_name = 'Dataset.pt'

if not os.path.exists(raw_path):
    os.makedirs(raw_path)

if not os.path.exists(processed_path):
    os.makedirs(processed_path)

torch.save(saved_dataset, os.path.join(raw_path, dataset_name))