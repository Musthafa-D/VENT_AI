import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

import torch_geometric
from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader

import numpy as np
import random
from sklearn.model_selection import train_test_split
import wandb

import gc
import shap

from segnn.segnn import SEGNN
from e3nn.o3 import Irreps, spherical_harmonics
from segnn.balanced_irreps import BalancedIrreps, WeightBalancedIrreps
from segnn.instance_norm import InstanceNorm
import matplotlib.pyplot as plt
# from Utility_functions import Graph_datasetV2 as Graph_dataset

# use it for input features similar to hemodynamics paper
from Utility_functions import Graph_dataset_with_equiv_features, inlet_distance_mask, print_3D_graph, Graph_dataset_with_equiv_features

import params
from utils import explain_node_prediction, explain_graph_prediction, grad_cam_plots, velocity_pressure, compute_shap_values, plot_shap, plot_ind_shap, plot_gnn_explainer, plot_gradcam
from utils_new import get_exp_method
from torch_geometric.data import Data


print('DATADIR', params.DATADIR)
print('NSIM', params.NSIM)
print('BATCH_SIZE', params.BATCH_SIZE)

print('python.__version__', sys.version_info)
print('torch.__version__', torch.__version__)
print('torch_geometric.__version__', torch_geometric.__version__)
print('torch.cuda.is_available()', torch.cuda.is_available())

a = Irreps('1o+2e+3x0e')
b = Irreps('4x1o+2o+2x1e')

c = Irreps()
print("PyTorch has version {}".format(torch.__version__))
print("The linked CUDA version is", torch.version.cuda)

dev = "cuda" if torch.cuda.is_available() else "cpu"
# dev = torch.device("cpu")
# torch.cuda.init()

print('Running on device:', dev)

"""Reproducibility seeds"""
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

"""Hyperparameters"""
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

for args in [
     
        {'epochs': params.EPOCHS,
         'batch_size': params.BATCH_SIZE,          
         'input_size' : params.INPUT_SIZE, 
         'edge_lmax' : params.EDGE_LMAX,
         'node_lmax' : params.NODE_LMAX,
         'hidden_lmax' : params.HIDDEN_LMAX,
         'num_layers' : params.NUM_LAYERS,
         'task' : params.TASK,
         'norm' : params.NORM,
         'output_size' : params.OUTPUT_SIZE,
         'hidden_size': params.HIDDEN_SIZE, 
         'neighbours' : params.NEIGHBOURS,
         'subsample_dataset': params.SUBSAMPLE_DATASET,
         'opt': params.OPT,
         'scheduler': params.SCHEDULER, 
         'learning_rate': params.LEARNING_RATE,
         'device': params.DEVICE,
         'early_stop' : params.EARLY_STOP}
    ]:
        args = objectview(args)

"""Dataset and dataloader"""
# Dataset, change the root path accordingly
dataset = Graph_dataset_with_equiv_features(root = params.DATADIR)  #, inlet_mask = True)
test_sample = dataset[0]
print(test_sample.edge_index)

# 80-10-10 split

# first the dataset is split 80%-20%, with the first portion being the training set

if params.NSIM == 1:
    train_idx = list(range(1))
    val_test_idx = []
elif params.NSIM == 10:  # in this case the test_size cannot be smaller than 2
    train_idx, val_test_idx = train_test_split(range(len(dataset)), test_size = 0.2, shuffle = False)
else:
    train_idx, val_test_idx = train_test_split(range(len(dataset)), test_size = 0.1, shuffle = False)

print(train_idx)
print(val_test_idx)
# the 20% portion is split in half so to have two sets with 10% data each of the original dataset
if params.NSIM == 1:
    val_idx, test_idx = [], []
else:
    val_idx, test_idx = train_test_split(range(len(val_test_idx)), test_size = 0.5, shuffle = False)

# this line can be used to do training on a smaller subset of the training set
# if args.subsample_dataset != 1
train_idx = train_idx[:(len(train_idx))// args.subsample_dataset] 

train_dataset = dataset[train_idx]
val_dataset = dataset[val_idx]

# print(len(train_idx), len(val_idx), len(test_idx))
# print(len(train_idx) / len(dataset), len(val_idx) / len(dataset), len(test_idx) / len(dataset))

# torch_geometric DataLoaders are used for handling lists of graphs
n_works = 0
t_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=n_works)#, persistent_workers=True)
v_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_works)#, persistent_workers=True)

"""Test if graph is loaded correctly"""
print_3D_graph(test_sample.pos.cpu(), edges = None, color = None)

name = 'SEGNN_model'
project = "local_SEGNN_with_inlet_trials"
for sample in t_loader:
    break

print(t_loader.dataset[0].node_attr)

print(sample)
print(len(sample.node_attr[:]))

# checking number of zero entries on node attributes; they are a lot!
print((torch.sum(sample.node_attr,dim=-1)==0).sum())

print_3D_graph(sample[0].pos.cpu(), edges = None, color = None)
print(sample[0].node_attr)

"""Model"""
gc.collect()
torch.cuda.empty_cache()
print()
print(torch.cuda.memory_allocated()*4/(1024**2), "MB")
print(torch.cuda.max_memory_allocated()*4/(1024**2), "MB")
print()
# print(torch.cuda.memory_summary())

gc.collect()
torch.cuda.empty_cache()

num_input_features = sample.x.shape[1]  # 8 if x is [60, 8]
print(f"x shape: {num_input_features}")
num_output_features = sample.y.shape[1]  # 4 if y is [60, 4]
print(f"y shape: {num_output_features}")
num_node_attr = sample.node_attr.shape[1]  # 4 if node_attr is [60, 4]
print(f"node_attr shape: {num_node_attr}")

# assest dimensions automatically from input data dimensions
if num_input_features == 8:
    # x: node features for fluid nodes [distance vector to closest inlet node,
    #                                   distance vector to closest outlet node,
    #                                   SDF (rel_dist to wall)]
    # 2x1o + 0e means 2 vectors (odd parity) and one scalar (even parity)
    input_irreps = Irreps('2x1o + 2x0e') 
elif num_input_features == 6:
    input_irreps = Irreps('2x1o')
else:
    raise ValueError(f"Unexpected dimensions for input features tensor: {num_input_features}")

# y: node features to predict [MIS,vx,vy,vz]
if num_output_features == 4:
    output_irreps = Irreps('0e + 1o')
else:
    raise ValueError(f"Unexpected dimensions for output features tensor: {num_output_features}")

if num_node_attr == 4:
    # node_attr_irreps=Irreps.spherical_harmonics(lmax=args.node_lmax)
    node_attr_irreps = Irreps('0e + 1o') 
else:
    raise ValueError(f"Unexpected dimensions for node attr tensor: {num_node_attr}")


# black box to specify the hidden layer, with some parametrization
hidden_irreps = BalancedIrreps(lmax=args.hidden_lmax, vec_dim=args.hidden_size)


# edge features (relative distance vector and norm)
edge_attr_irreps = Irreps.spherical_harmonics(lmax=args.edge_lmax)

# no additional attributes specified
additional_message_irreps = None # Irreps('1x0e')

model = SEGNN(hidden_irreps=hidden_irreps,
              output_irreps=output_irreps,
              edge_attr_irreps=edge_attr_irreps,
              node_attr_irreps=node_attr_irreps,
              input_irreps=input_irreps,
              task=args.task,
              norm=args.norm,
              num_layers=args.num_layers,
              additional_message_irreps=additional_message_irreps
              )

model.to(dev)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
nparams = count_parameters(model)
print('Number of parameters; ',nparams)

mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
print()
print('Memory (MB) occupied by parameters and buffers: ',mem_bufs, '\t', mem_params)
print('Memory allocated on GPU:', torch.cuda.memory_allocated()*4/(1024**2), "MB")
print('Peak memory allocated on GPU:', torch.cuda.max_memory_allocated()*4/(1024**2), "MB")
# print(torch.cuda.memory_summary())

"""Loss and optimizer"""
#loss_func = nn.MSELoss()
loss_func = nn.L1Loss()
if args.opt == 'Adam':
    opt = optim.Adam(model.parameters(), lr = args.learning_rate)
if args.scheduler == 'ExponentialLR':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=params.GAMMA)
else:
    scheduler = None    

model_name = name + '_lat'+str(args.hidden_size) + '_knn' + \
            str(args.neighbours) + '_bs' + str(args.batch_size) + '_lr' + str(args.learning_rate) + '_ep' + \
            str(args.epochs) + '_dataset' + str(params.DATASET) + '_nparams' + str(nparams) +'.pt'
instance_norm = True

input_norm = InstanceNorm(input_irreps)
edge_norm = InstanceNorm(edge_attr_irreps)
# node_norm = InstanceNorm(node_attr_irreps)
# output_norm = InstanceNorm(output_irreps)
def train(model, loader, opt, loss_func, dev, log = True, mask = False):
     
    model.train()

    training_loss = []       

    edge_index = torch.tensor(np.load(params.DATADIR+"/connectivity.npy"))
    
    for sample in loader:

        #print('SM...sample.x', sample.x.shape)
        # edge_index = knn_graph(sample.pos, args.neighbours, sample.batch)
        
        sample.edge_index = edge_index
        
        # computes relative positions for every node pair
        edge_relativePos = (torch.index_select(sample.pos, 0, edge_index[1]) - torch.index_select(sample.pos, 0, edge_index[0]))
        # computes distances between pairs
        edge_relativeDist = torch.norm(edge_relativePos, dim = -1, keepdim = True) 
        # concatenates relative pos and distance for every edge
        edge_attr = torch.cat([edge_relativeDist, edge_relativePos], dim = -1) 
        
        sample.edge_attr = edge_attr
        # sample.node_attr = sample.pos
        
        if instance_norm:
            sample.x = input_norm(sample.x, sample.batch)
            sample.edge_attr = edge_norm(sample.edge_attr, sample.edge_index[1,:])

        sample = sample.to(dev)
        #print(sample)
        # print(sample.x)
        # print(sample.pos)
        #print(sample.node_attr)
        
        fluid_nodes = torch.tensor(2)
        
        if mask:
            # sample.mask: node next to inlet
            # take mask of nodes far from inlet
            loss_mask = ~sample.mask
        else:
            loss_mask = torch.ones(sample.x.shape[0], dtype=torch.bool)


        opt.zero_grad()
        # print(f"shape of edge_index: {sample.edge_index.shape}")
        # print(f"shape of pos: {sample.pos.shape}")
        # print(f"shape of edge_attr: {sample.edge_attr.shape}")
        # print(f"shape of node_attr: {sample.node_attr.shape}")
        # print(f"shape of batch: {sample.batch.shape}")
        
        # # Inspecting features
        # print(f"input features (x): {sample.x}")
        # print(f"Shape of input features: {sample.x.shape}")
        
        # # Inspecting outputs
        # print(f"Target (y): {sample.y}")
        # print(f"Shape of target: {sample.y.shape}")
        
        # # Inspecting specific features and outputs
        # for i in range(sample.x.shape[1]):
        #     print(f"Feature {i}: {sample.x[:, i]}")
        
        # for i in range(sample.y.shape[1]):
        #     print(f"Target dimension {i} (Class {i}): {sample.y[:, i]}")
    
        pred = model(sample.x, sample.edge_index, pos=sample.pos, edge_attr=sample.edge_attr, 
                     node_attr=sample.node_attr, batch=sample.batch, y=sample.y) 
        # print(f"shape of pred: {pred.shape}")
        #print('train', pred[0])# pred.shape, pred[loss_mask].shape, sample.y.shape)
        loss = loss_func(pred[loss_mask], sample.y[loss_mask]) 
        # print(f"shape of y loss mask: {sample.y[loss_mask].shape}")
        # print(f"shape of pred loss mask: {pred[loss_mask].shape}")

        loss.backward()
        opt.step() 
        
        training_loss.append(loss)
        

        # if log == True:
        #     wandb.log({"train/batch_train_loss": loss})

        # with torch.no_grad():
        #     magvel = torch.norm(pred[mask,1:4], dim=-1)
        #     print('magvel', magvel)
        #     print_3D_graph(sample[0].pos, edges = None, color = magvel)
        
    return sum(training_loss) / len(loader)



def val(model, loader, loss_func, final_epoch, mask = False, calculate_gradcam=False, 
        calculate_node_prediction=False, calculate_shap=False):
    
    model.eval()
    
    with torch.no_grad():
   
        validation_loss = [] 

        edge_index = torch.tensor(np.load(params.DATADIR+"/connectivity.npy"))


        for sample in loader:
            
            # edge_index = knn_graph(sample.pos, args.neighbours, sample.batch)
            sample.edge_index = edge_index
            
            edge_relativePos = (torch.index_select(sample.pos, 0, edge_index[1]) - torch.index_select(sample.pos, 0, edge_index[0]))
            edge_relativeDist = torch.norm(edge_relativePos, dim = -1, keepdim = True) 
            edge_attr = torch.cat([edge_relativeDist, edge_relativePos], dim = -1) 
            
            sample.edge_attr = edge_attr
            # sample.node_attr = sample.pos
            
            if instance_norm:
                sample.x = input_norm(sample.x, sample.batch)
                sample.edge_attr = edge_norm(sample.edge_attr, sample.edge_index[1,:])
            
            sample = sample.to(dev)
            
            fluid_nodes = torch.tensor(2)
            if mask:
                # sample.mask: node next to inlet
                # take mask of nodes far from inlet
                loss_mask = ~sample.mask
            else:
                loss_mask = torch.ones(sample.x.shape[0], dtype=torch.bool)          

            pred = model(sample.x, sample.edge_index, pos=sample.pos, edge_attr=sample.edge_attr, 
                         node_attr=sample.node_attr, batch=sample.batch, y=sample.y)  
            #print('validation', pred[0])#pred.shape, pred[loss_mask].shape, sample.y.shape)
            loss = loss_func(pred[loss_mask], sample.y[loss_mask])  
            validation_loss.append(loss)
        
    # Compute explanations with gradients enabled
    edge_index = torch.tensor(np.load(params.DATADIR+"/connectivity.npy"))
    for idx, sample in enumerate(loader):
        sample.edge_index = edge_index

        edge_relativePos = (torch.index_select(sample.pos, 0, edge_index[1]) - torch.index_select(sample.pos, 0, edge_index[0]))
        edge_relativeDist = torch.norm(edge_relativePos, dim = -1, keepdim = True) 
        edge_attr = torch.cat([edge_relativeDist, edge_relativePos], dim = -1) 

        sample.edge_attr = edge_attr

        if instance_norm:
            sample.x = input_norm(sample.x, sample.batch)
            sample.edge_attr = edge_norm(sample.edge_attr, sample.edge_index[1,:])

        sample = sample.to(dev)
        
        pred = model(sample.x, sample.edge_index, pos=sample.pos, edge_attr=sample.edge_attr, 
                     node_attr=sample.node_attr, batch=sample.batch, y=sample.y) 
        
        node_idx = 0  # Set your desired node index

        if calculate_node_prediction:
            plot_gnn_explainer(model, sample, node_idx, index=idx, epoch=epoch, final_epoch=final_epoch)
            # explain_node_prediction(model, sample, dev, epochs_exp=10, node_idx=40, 
            #                         index=idx, epoch=epoch) # epochs_exp is The number of epochs to train by explainer, node_idx is the Index of the node you want to explain

        if calculate_gradcam:
            plot_gradcam(model, sample, node_idx, index=idx, epoch=epoch)
            # grad_cam_plots(model, sample, model.layers[-1], dev, index=idx, epoch=epoch)
        
        if calculate_shap:
            # Compute SHAP values for node features
            shap_values = compute_shap_values(model, sample, dev)
            plot_shap(shap_values, sample.x.detach().cpu().numpy(), idx, epoch)
            
            for i in range(len(shap_values)):
                plt.figure(figsize=(12, 8))
                plot_ind_shap(shap_values[i], sample.x.detach().cpu().numpy(), i, idx, epoch)

    return sum(validation_loss) / len(loader)


def early_stopping(val_loss, best_loss, counter):
        
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
    else:
        counter += 1
    return counter, best_loss

"""Training and validation"""
samples = 0
best_loss = 10**6
counter = 0
log = True

# wandb.init(  
#       project= project,    
#       config={
#         "epochs": args.epochs,
#         "bs": args.batch_size,
#         "lr": args.learning_rate,
#         "neighbours": args.neighbours,
#         "latent size": args.hidden_size,
#         "model parameters": nparams
#         })

path = os.path.join(params.DATADIR, 'checkpoints', model_name)

# wandb.run.name = model_name

# mask = True
mask = False

for epoch in range(args.epochs):

    # training 
    train_loss = train(model, t_loader, opt, loss_func, dev, log, mask)
    #train_loss = train(model, train_dataset, opt, loss_func, dev, log, mask)
    if scheduler != None: 
        scheduler.step()
    samples += len(train_dataset)

    metrics = {"train/train_loss": train_loss,  
                "train/samples": samples}
    # wandb.log(metrics)    
    
    # validation
    final_epoch = (args.epochs)-1
    val_loss = val(model, v_loader, loss_func, final_epoch, mask, 
                   calculate_gradcam=False, calculate_node_prediction=True, calculate_shap=False)
    # val_loss = val(model, val_dataset, loss_func, mask)

    val_metrics = {"val/val_loss": val_loss,
                   "epoch": epoch+1}

    # wandb.log(val_metrics)
    
    # early stopping
    counter, best_loss = early_stopping(val_loss, best_loss, counter)
    if counter == 0:

        checkpoint = {
            'epoch': epoch+1,
            'model': model,
            'optimizer_state_dict': opt.state_dict(),
            'training_loss': train_loss,
            'validation_loss': val_loss,
            'input_irreps':input_irreps,
            'hidden_irreps': hidden_irreps,
            'output_irreps': output_irreps,
            'edge_attr_irreps': edge_attr_irreps,
            'node_attr_irreps': node_attr_irreps,
            'task': args.task,
            'norm': args.norm,
            'num_layers': args.num_layers,
            'additional_message_irreps': additional_message_irreps
        }


    # if counter >= args.early_stop:        

    #     torch.save(checkpoint, path)
    #     wandb.alert(
    #         title="Early stopping on validation data", 
    #         text=f"Loss {best_loss} was the best result, now is overfitting")
    #     break

    
    print("Epoch " + str(epoch+1) + ": T loss " + str(train_loss) + " V loss " + str(val_loss))

# Run explain_graph_prediction after all epochs
# sample_data = val_dataset[0]
# # print(sample_data)
# edge_index = knn_graph(sample_data.pos, args.neighbours, sample_data.batch).to(dev)
# # explain_graph_prediction(model, sample_data, edge_index, dev, epochs_exp=10) 
        
# torch.save(checkpoint, path)
# wandb.finish()

"""Test after training"""
test_graph = val_dataset[0]
neighbours = args.neighbours
print(args.neighbours)
print(neighbours)

with torch.no_grad():

    # edge_index = knn_graph(test_graph.pos, neighbours, test_graph.batch)
    edge_index = torch.tensor(np.load(params.DATADIR+"/connectivity.npy"))
    test_graph.edge_index = edge_index
    
    edge_relativePos = (torch.index_select(test_graph.pos, 0, edge_index[1]) - torch.index_select(test_graph.pos, 0, edge_index[0]))
    edge_relativeDist = torch.norm(edge_relativePos, dim = -1, keepdim = True) 
    edge_attr = torch.cat([edge_relativeDist, edge_relativePos], dim = -1) 
    
    test_graph.edge_attr = edge_attr
    test_graph = test_graph.to(dev)

    pred = model(test_graph.x, 
                 test_graph.edge_index, 
                 pos=test_graph.pos, 
                 edge_attr=test_graph.edge_attr, 
                 node_attr=test_graph.node_attr, 
                 batch=test_graph.batch, 
                 y=test_graph.y) 


# edges = knn_graph(test_graph.pos, neighbours)
edges = torch.tensor(np.load(params.DATADIR+"/connectivity.npy"))


# print(len(pred))
for i in range(len(pred)):
    print(pred[i].tolist())

colors_magvel = torch.norm(pred[...,1:4], dim=-1)
colors_press = torch.norm(pred[...,[0]], dim=-1) 

print_3D_graph(test_graph.pos.cpu(), edges = edges, color = colors_magvel.cpu())
# print_3D_graph(test_graph.pos.cpu(), edges = edges, color = colors_press.cpu())

import glob
import re
import matplotlib.pyplot as plt

numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

probes_path = os.path.join(params.DATADIR, 'probes')
probes = sorted(glob.glob(os.path.join(probes_path, '*.npy')), key = numericalSort)
# print(probes[0])

probe_data_list = []

for data_file in probes:
    read_points = np.load(data_file)
    probe_data_list.append(torch.from_numpy(read_points))

pos_list = probe_data_list
#print(pos_list)
# print(pos_list[0][...,3])



# model prediction
out = pred
out_vel = torch.norm(out[...,1:4], dim=-1)
out_press = out[...,0]
print(f"out_vel: {out_vel.tolist()}")
print(f"out_press: {out_press.tolist()}")
true_vel = torch.norm(test_graph.y[...,1:4], dim=-1)
true_press = test_graph.y[...,0]
print(f"true_vel: {true_vel.tolist()}")
print(f"true_press: {true_press.tolist()}")

loss_vel = 100*abs((true_vel-out_vel)/true_vel)
loss_press = 100*abs((true_press-out_press)/true_press)
#loss_val = 100*(test_graph.y[...,0:3]-out[...,1:4])/torch.mean(out[...,1:4], test_graph.y[...,0:3])
# print(len(loss_val))
# print(loss_val.tolist())

# ground truth
# test_graph = val_dataset[0]
# print(test_graph.y)
# print(test_graph.y)

# true_mis = test_graph.y[...,3]
true_mis = pos_list[0][...,3]
true_mis = true_mis[:len(loss_vel)]

# print(len(true_mis))
# print(true_mis)

loss_vel = loss_vel[:(len(loss_vel)-1)]
loss_press = loss_press[:(len(loss_press)-1)]
true_mis = true_mis[:(len(true_mis)-1)]


# print(len(loss_val))
velocity_pressure(loss_vel, loss_press, true_mis)
