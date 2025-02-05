import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
# from torch_geometric.nn.models import GNNExplainer
# from gradcam import GradCAM
import datetime
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import shap
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data
from inspect import signature
from math import sqrt
from interpret import GNNExplainer_get_explanation_node, GradCAM_get_explanation_node, GNNExplainer_get_explanation_graph
from graphxai.explainers import GNNExplainer, GradCAM


"""Figurestorage class to store figures"""
class FigureStorage:
    def __init__(self, result_folder):
        self.images_folder = os.path.join(result_folder, 'Images')
        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)
        # Create subfolders for 'png' and 'pdf'
        self.png_folder = os.path.join(self.images_folder, 'png')
        self.pdf_folder = os.path.join(self.images_folder, 'pdf')
        if not os.path.exists(self.png_folder):
            os.makedirs(self.png_folder)
        if not os.path.exists(self.pdf_folder):
            os.makedirs(self.pdf_folder)

    def store(self, fig, file_name):
        self.save_as_png(fig, file_name)
        self.save_as_pdf(fig, file_name)

    def save_as_png(self, fig, file_name):
        file_path = os.path.join(self.png_folder, file_name + ".png")
        fig.savefig(file_path)

    def save_as_pdf(self, fig, file_name):
        file_path = os.path.join(self.pdf_folder, file_name + ".pdf")
        fig.savefig(file_path)

#Create a folder with the current timestamp to save the results
desired_folder = "C://Users//DiwanMohideen//Desktop//Files//vent_ai" # choose the desired location
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_folder = os.path.join(desired_folder,'Results_Segnn_SteadyState', now)
if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
figure_storage = FigureStorage(result_folder)

"""Velocity and Pressure error plots"""
def velocity_pressure(loss_vel, loss_press, true_mis):
    fig = plt.figure(figsize=(8, 4), dpi=200)

    color = np.arange(len(loss_vel))
    plt.scatter(true_mis.cpu(), loss_vel.cpu(), marker = ".", c = color, cmap = 'hsv', s = 10)
    plt.colorbar()

    # for i in range(len(loss_vel_z)):
    #     plt.text(true_mis[i], loss_vel_z[i], str(i), fontsize=4, ha='right')

    plt.title("Velocity error on test sample", size= 15)
    plt.xlabel("MIS", size=15)
    plt.ylabel("Loss in %", size=15)
    # plt.semilogy()
    plt.grid()
    figure_storage.store(
            fig, "Velocity error on test sample")
    plt.show()


    fig = plt.figure(figsize=(8, 4), dpi=200)

    color = np.arange(len(loss_press))
    plt.scatter(true_mis.cpu(), loss_press.cpu(), marker = ".", c = color, cmap = 'hsv', s = 10)
    plt.colorbar()

    plt.title("Pressure error on test sample", size= 15)
    plt.xlabel("MIS", size=15)
    plt.ylabel("Loss in %", size=15)
    # plt.semilogy()
    plt.grid()
    figure_storage.store(
            fig, "Pressure error on test sample")
    plt.show()

# Function to generate and visualize GNNExplainer explanation
def plot_gnn_explainer(model, sample, node_idx, num_hops):
    fig, ax = plt.subplots(figsize=(10, 7.5))  # Adjust the figure size as needed
    gnn_explainer_exp_method = GNNExplainer(model)
    
    graph_data = Data(
        x=sample.x, edge_index=sample.edge_index, pos=sample.pos, 
        edge_attr=sample.edge_attr, node_attr=sample.node_attr, 
        batch=sample.batch, y=sample.y
    )
    
    gnn_explainer_exp = GNNExplainer_get_explanation_node(
        gnn_explainer_exp_method,
        model,
        node_idx,
        sample.x,
        sample.edge_index,
        sample.pos, 
        sample.edge_attr, 
        sample.node_attr, 
        sample.batch, 
        sample.y,
        graph_data,
        None,
        num_hops=num_hops ,
        explain_feature=True
    )

    gnn_explainer_exp.visualize_node(
        num_hops=num_hops, additional_hops=0, graph_data=graph_data, 
        ax=ax, show_node_labels=True, norm_imps=True
    )

    ax.set_title("Node Explanation", fontsize=20, fontweight='semibold')
    figure_storage.store(
            fig, "Node Explanation")
    plt.show()
    
    # fig, ax = plt.subplots(figsize=(10, 7.5))  # New figure for the graph explanation
    # gnn_explainer_exp = GNNExplainer_get_explanation_graph(
    #     gnn_explainer_exp_method,
    #     model,
    #     sample.x,
    #     sample.edge_index,
    #     sample.pos, 
    #     sample.edge_attr, 
    #     sample.node_attr, 
    #     sample.batch, 
    #     sample.y
    # )
    
    # G, pos = gnn_explainer_exp.visualize_graph(ax=ax, show=True, show_node_labels=True)

    # ax.set_title("Graph Explanation", fontsize=20, fontweight='semibold')
    # figure_storage.store(
    #         fig, "Graph Explanation")
    # plt.show()

# Function to generate and visualize GradCAM explanation
def plot_gradcam(model, sample, node_idx, num_hops):
    fig, ax = plt.subplots(figsize=(10, 7.5))  # Adjust the figure size as needed
    grad_cam_exp_method = GradCAM(model)
    
    grad_cam_exp = GradCAM_get_explanation_node(
        grad_cam_exp_method,
        model,
        sample.x,
        sample.y,
        node_idx,
        sample.edge_index,
        sample.pos, 
        sample.edge_attr, 
        sample.node_attr, 
        sample.batch,
        None,
        True,
        0
    )

    graph_data = Data(
        x=sample.x, edge_index=sample.edge_index, pos=sample.pos, 
        edge_attr=sample.edge_attr, node_attr=sample.node_attr, 
        batch=sample.batch, y=sample.y
    )

    grad_cam_exp.visualize_node(
        num_hops=num_hops, additional_hops=0, graph_data=graph_data, 
        ax=ax, norm_imps=False
    )

    ax.set_title("GradCAM Explanation", fontsize=20, fontweight='semibold')
    figure_storage.store(
            fig, f"{node_idx} GradCAM Explanation")
    plt.show()
