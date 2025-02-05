import torch
import torch.nn as nn

from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import gc

from plotly.subplots import make_subplots
from segnn.segnn import SEGNN
from e3nn.o3 import Irreps, spherical_harmonics
from segnn.balanced_irreps import BalancedIrreps, WeightBalancedIrreps
from segnn.instance_norm import InstanceNorm

# use it for input features similar to hemodynamics paper
from Utility_functions import print_3D_graph, manual_print_3D_graph, Graph_dataset_with_equiv_features, rotate_graph_coords, translate_graph_coords

import os
import params


def main():
    print('DATADIR', params.DATADIR)
    print('NSIM', params.NSIM)
    print('BATCH_SIZE', params.BATCH_SIZE)

    print('torch.__version__', torch.__version__)
    print('torch.cuda.is_available()', torch.cuda.is_available())
    
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # dev = "cpu"
    print(dev)
    
    """Model"""
    gc.collect()
    torch.cuda.empty_cache()
    
    # change the path accordingly
    path = os.path.join(params.DATADIR, "checkpoints/best_model_.pt")
    print(f"Loading model checkpoint from {str(path)}")
    
    # checkpoint = torch.load(path)
    checkpoint = torch.load(path, map_location=torch.device('cpu')) # my laptop only has integraded graphycs
    
    input_irreps = checkpoint['input_irreps']
    hidden_irreps = checkpoint['hidden_irreps']
    output_irreps = checkpoint['output_irreps']
    edge_attr_irreps = checkpoint['edge_attr_irreps']
    node_attr_irreps = checkpoint['node_attr_irreps']
    task = checkpoint['task']
    norm=checkpoint['norm']
    num_layers=checkpoint['num_layers']
    additional_message_irreps=checkpoint['additional_message_irreps']
    
    model = SEGNN(input_irreps=input_irreps,
                  hidden_irreps=hidden_irreps,
                  output_irreps=output_irreps,
                  edge_attr_irreps=edge_attr_irreps,
                  node_attr_irreps=node_attr_irreps,
                  task=task,
                  norm=norm,
                  num_layers=num_layers,
                  additional_message_irreps=additional_message_irreps
                  )
    
    model = checkpoint['model']
    
    loss_func = nn.MSELoss()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(count_parameters(model))
    
    model.to(dev)
    print(model)
    model.eval()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # change the path accordingly
    path = os.path.join(params.DATADIR, "checkpoints/best_model.pt")
    print(f"Loading model checkpoint from {str(path)}")
    
    # checkpoint = torch.load(path)
    
    input_irreps = checkpoint['input_irreps']
    hidden_irreps = checkpoint['hidden_irreps']
    output_irreps = checkpoint['output_irreps']
    edge_attr_irreps = checkpoint['edge_attr_irreps']
    node_attr_irreps = checkpoint['node_attr_irreps']
    task = checkpoint['task']
    norm=checkpoint['norm']
    num_layers=checkpoint['num_layers']
    additional_message_irreps=checkpoint['additional_message_irreps']
    
    model = SEGNN(input_irreps=input_irreps,
                  hidden_irreps=hidden_irreps,
                  output_irreps=output_irreps,
                  edge_attr_irreps=edge_attr_irreps,
                  node_attr_irreps=node_attr_irreps,
                  task=task,
                  norm=norm,
                  num_layers=num_layers,
                  additional_message_irreps=additional_message_irreps
                  )
    
    model = checkpoint['model']
    
    loss_func = nn.MSELoss()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(count_parameters(model))
    
    model.to(dev)
    model.eval()
    
    # change the root path accordingly
    dataset = Graph_dataset_with_equiv_features(params.DATADIR)
    graph_connectivity = torch.tensor(np.load(params.DATADIR+"/connectivity.npy"))
    print(f"Loading dataset from {str(params.DATADIR)}")
    
    # Dataset split 80-10-10
    num_workers = 8
    
    dataset_length = len(dataset)
    if params.NSIM >= 10:
        train_length, test_length = train_test_split(range(dataset_length), test_size = 0.2, shuffle = False)
        val_length, test_length = train_test_split(range(len(test_length)), test_size = 0.5, shuffle = False)
    else:
        train_length, test_length = list(range(1)), [0]
        val_length, test_length = [0], [0]
        
    test_dataset = dataset[test_length]
    
    loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = num_workers)
    
    """Test loss"""
    outputs = []
    loss = []
    
    # mask = True
    neighbours = params.NEIGHBOURS
    
    fluid_nodes = torch.tensor(2)
    
    # # s is the sample graph
    # for s in tqdm(loader): 
    
    
    #     # edge_index = knn_graph(s.pos, neighbours, s.batch)
    #     edge_index = graph_connectivity
    #     # print(edge_index)
    #     s.edge_index = edge_index
    #     # print(len(s.pos))
    #     edge_relativePos = (torch.index_select(s.pos, 0, edge_index[1]) - torch.index_select(s.pos, 0, edge_index[0]))
    #     edge_relativeDist = torch.norm(edge_relativePos, dim = -1, keepdim = True) 
    #     edge_attr = torch.cat([edge_relativeDist, edge_relativePos], dim = -1) 
    
    #     s.edge_attr = edge_attr
    #     #s.node_attr = s.pos
    #     #print(s.node_attr)
        
    #     s = s.to(dev)
    #     with torch.no_grad():
            
    #         if hasattr(s, "mask"):
    #             mask = s.mask
    #         else:
    #             mask = torch.ones(s.x.shape[0], dtype=torch.bool)
                
    #         out = model(s)
    #         #print(out)
    #         loss_val = loss_func(out[mask], s.y[mask])
    #         #print(loss_val)
            
        
    #     outputs.append(out)
    #     loss.append(loss_val.item())
    #     # print(loss)
    #     # print(len(loss))  
    
    # save = True
    save = True
    
    if save:
        nploss = np.save('loss.npy', loss)
    plt.figure(figsize=(15,8))
    plt.plot(range(len(loss)), loss, marker = 'o', markersize=5,
             linewidth=0)
    plt.title('MSE error across test samples', size = 30)
    plt.grid()
    plt.show()
    
    """Test set plots"""
    loadloss = np.load('loss.npy')

    plt.figure(figsize=(16,8))
    
    plt.plot(range(len(loadloss)), loadloss, marker='o', markersize=5,
             linewidth=1, label='MSE on test sample')
    
    plt.xlabel('Test samples', size=25)
    plt.ylabel('Mean Squared Error (MSE)', size=25)
    plt.title('MSE error across test samples', size=30)
    plt.grid()
    legend = plt.legend()
    legend.get_frame().set_facecolor('lightgray')
    
    print("Number of test samples:", len(loadloss))
    
    plt.show()
    
    """Single test plot"""
    # idx ofgraph (steady state) to plot
    test_graph = test_dataset[0]
    
    #test_graph = s
    with torch.no_grad():
    
        # edge_index = knn_graph(test_graph.pos, neighbours, test_graph.batch)
        edge_index = graph_connectivity
        test_graph.edge_index = edge_index
        
        edge_relativePos = (torch.index_select(test_graph.pos, 0, edge_index[1]) - torch.index_select(test_graph.pos, 0, edge_index[0]))
        edge_relativeDist = torch.norm(edge_relativePos, dim = -1, keepdim = True) 
        edge_attr = torch.cat([edge_relativeDist, edge_relativePos], dim = -1) 
        #print(edge_attr)
        
        test_graph.edge_attr = edge_attr
        test_graph = test_graph.to(dev)
        #test_graph.node_attr = test_graph.pos
    
        # print(test_graph)
        # print(test_graph.node_attr)
        #print(test_graph.x)
    
        pred = model(test_graph.x, test_graph.edge_index, pos=test_graph.pos, edge_attr=test_graph.edge_attr, 
                     node_attr=test_graph.node_attr, batch=test_graph.batch, y=test_graph.y)
        print(pred[0]) # press, vel_x, vel_y, vel_z (=6.3379e-02)
    
        print(test_graph.node_attr)
    
    # remember that to plot the graphs ignoring the ground truth velocities and pressures, you have
    # to plot pred[sample.mask], and not simply the "pred" tensor (which contains predictions on all nodes,
    # included the ones that were masked out during training because true values were used in the input, and 
    # that were not taken care of by backpropagation)
    
    """Ground truth"""
    # edges = knn_graph(test_graph.pos, neighbours)
    edges = graph_connectivity
    # print(len(test_graph.pos))
    # print(test_graph.pos[1:4].tolist())
    # print(test_graph.pos[0].tolist())
    
    # print(test_graph.num_features)
    # print(test_graph.pos)
    
    # print(test_graph.node_attr[..., 0:4])
    
    
    for i in range(5):
        print(test_graph.node_attr[..., 0:4].tolist())
    
    true_magvel = torch.norm(test_graph.cpu().y[...,1:4], dim=-1) # y is ground truth
    true_press = torch.norm(test_graph.cpu().y[...,[0]], dim=-1)
    
    print_3D_graph(test_graph.pos.cpu(), edges = edges, color = true_magvel)
    print_3D_graph(test_graph.pos.cpu(), edges = edges, color = true_press)
    # print_3D_graph(test_graph.pos.cpu(), edges = edges, color = test_graph.edge_index) # edge_index is MIS, wrong for testing purposes
    
    """Prediction"""
    # edges = knn_graph(test_graph.pos, neighbours)
    edges = graph_connectivity
    
    for i in range(5):
        print(pred[i].tolist())
    #print(pred[-1].tolist())
    pred_magvel = torch.norm(pred[...,1:4], dim=-1)
    pred_press = torch.norm(pred[...,[0]], dim=-1) 
    
    print_3D_graph(test_graph.pos.cpu(), edges = edges, color = pred_magvel.cpu())
    print_3D_graph(test_graph.pos.cpu(), edges = edges, color = pred_press.cpu())
    
    fig1_colors = true_magvel.cpu()
    fig1 = manual_print_3D_graph(test_graph.pos.cpu(), edges.cpu(), fig1_colors.numpy()) # manual_print_3D_graph needs fig.show() below
    
    fig2_colors = pred_magvel.cpu()
    fig2 = manual_print_3D_graph(test_graph.pos.cpu(), edges.cpu(), fig2_colors.numpy())
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        subplot_titles=("True vel", "Pred vel"))
    
    for trace in fig1['data']:
        fig.add_trace(trace, row=1, col=1)
    
    for trace in fig2['data']:
        fig.add_trace(trace, row=1, col=2)
    
    fig.show()
    
    """Test equivariance"""
    # print(test_graph)
    # print(pred)
    # print(true_magvel)
    print(len(pred))
    print(len(true_magvel))
    # print_3D_graph(test_graph.pos, edges = edges, color = true_magvel.cpu())
    
    translation_vector = 0 # np.random.randint(0, 30, size=3)
    rotation_angle = 90 # np.random.randint(0, 90) # rotation is anti clockwise
    translated_test_graph = translate_graph_coords(test_graph.cpu(), translation_vector)
    rotated_test_graph = rotate_graph_coords(translated_test_graph.cpu(), rotation_angle, axis="y")
    # print(rotated_test_graph)
    rotated_test_graph = rotated_test_graph.to(dev)
    
    rotated_graph_pred = model(rotated_test_graph.x, rotated_test_graph.edge_index, pos=rotated_test_graph.pos, edge_attr=rotated_test_graph.edge_attr, 
                 node_attr=rotated_test_graph.node_attr, batch=rotated_test_graph.batch, y=rotated_test_graph.y)
    
    rotated_graph_magvel_pred = torch.norm(rotated_graph_pred[...,1:4], dim=-1)
    # print(rotated_graph_pred)
    # print(rotated_graph_magvel_pred)
    print(len(rotated_graph_magvel_pred))
    
    print_3D_graph(rotated_test_graph.pos, edges = edges, color = true_magvel)
    
    equivariance_loss = pred - rotated_graph_pred
    # print(equivariance_loss.tolist())
    print(f"Total error is: {equivariance_loss.sum()}\n")
    
    for row in range(pred.size(0)):
        diff = pred[row, :] - rotated_graph_pred[row, :]
        if (diff != 0).any():
            print(f"On node {row}: {diff.tolist()}")
    
    print(" ")
    print(f"Graph has: {len(pred)} rows")
    
    """Loss analysis"""
    import glob
    import re
    
    numbers = re.compile(r'(\d+)')
    
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    
    probes_path = os.path.join(params.DATADIR, 'probes')
    probes = sorted(glob.glob(os.path.join(probes_path, '*.npy')), key = numericalSort)
    print(probes[0])
    
    probe_data_list = []
    
    for data_file in probes:
        read_points = np.load(data_file)
        probe_data_list.append(torch.from_numpy(read_points))
    
    pos_list = probe_data_list
    #print(pos_list)
    print(pos_list[0][...,3])
    
    # model prediction
    out = pred.cpu()
    out_vel = torch.norm(out[...,1:4], dim=-1).cpu()
    out_press = out[...,0].cpu()
    print(f"out_vel: {out_vel.tolist()}")
    print(f"out_press: {out_press.tolist()}")
    true_vel = torch.norm(test_graph.y[...,1:4], dim=-1).cpu()
    true_press = test_graph.y[...,0].cpu()
    print(f"true_vel: {true_vel.tolist()}")
    print(f"true_press: {true_press.tolist()}")
    
    loss_vel = 100*abs((true_vel-out_vel)/true_vel)
    loss_press = 100*abs((true_press-out_press)/true_press)
    #loss_val = 100*(test_graph.y[...,0:3]-out[...,1:4])/torch.mean(out[...,1:4], test_graph.y[...,0:3])
    # print(len(loss_val))
    # print(loss_val.tolist())
    
    # ground truth
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

    plt.figure(figsize=(15,10))
    
    color = np.arange(len(loss_vel))
    plt.scatter(true_mis, loss_vel, marker = ".", c = color, cmap = 'hsv', s = 10)
    plt.colorbar()
    
    # for i in range(len(loss_vel_z)):
    #     plt.text(true_mis[i], loss_vel_z[i], str(i), fontsize=4, ha='right')
    
    plt.title("Velocity error on test sample", size= 15)
    plt.xlabel("MIS", size=15)
    plt.ylabel("Loss in %", size=15)
    # plt.semilogy()
    plt.grid()
    
    plt.show()
    
    plt.figure(figsize=(15,10))

    color = np.arange(len(loss_press))
    plt.scatter(true_mis, loss_press, marker = ".", c = color, cmap = 'hsv', s = 10)
    plt.colorbar()
    
    # for i in range(len(loss_vel_z)):
    #     plt.text(true_mis[i], loss_vel_z[i], str(i), fontsize=4, ha='right')
    
    plt.title("Pressure error on test sample", size= 15)
    plt.xlabel("MIS", size=15)
    plt.ylabel("Loss in %", size=15)
    # plt.semilogy()
    plt.grid()
    
    plt.show()

if __name__ == '__main__':
    main()