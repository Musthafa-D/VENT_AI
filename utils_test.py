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

# Define a custom colormap
custom_cmap = LinearSegmentedColormap.from_list("BlYeGn", ["blue", "yellow", "green"])

#Create a folder with the current timestamp to save the results
desired_folder = "C://Users//DiwanMohideen//Desktop//Files//vent_ai" # choose the desired location
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_folder = os.path.join(desired_folder,'Results_Segnn_SteadyState', now)
if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
figure_storage = FigureStorage(result_folder)


def custom_visualize_subgraph(explainer, node_idx, edge_index, edge_mask, y=None,
                       threshold=None, edge_y=None, node_alpha=None,
                       seed=10, **kwargs):
    import networkx as nx
    import matplotlib.pyplot as plt

    assert edge_mask.size(0) == edge_index.size(1)

    if node_idx == -1:
        hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1),
                                          device=edge_mask.device)
        subset = torch.arange(edge_index.max().item() + 1,
                              device=edge_index.device)
        y = None

    else:
        # Only operate on a k-hop subgraph around `node_idx`.
        subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
            node_idx, explainer.num_hops, edge_index, relabel_nodes=True,
            num_nodes=None, flow=explainer.__flow__())

    edge_mask = edge_mask[hard_edge_mask]

    if threshold is not None:
        edge_mask = (edge_mask >= threshold).to(torch.float)

    if y is None:
        y = torch.zeros(edge_index.max().item() + 1,
                        device=edge_index.device)
    else:
        y = y[subset].to(torch.float) / y.max().item()

    if edge_y is None:
        edge_color = ['black'] * edge_index.size(1)
    else:
        colors = list(plt.rcParams['axes.prop_cycle'])
        edge_color = [
            colors[i % len(colors)]['color']
            for i in edge_y[hard_edge_mask]
        ]

    data = Data(edge_index=edge_index, att=edge_mask,
                edge_color=edge_color, y=y, num_nodes=y.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'],
                    edge_attrs=['att', 'edge_color'])
    mapping = {k: i for k, i in enumerate(subset.tolist())}
    G = nx.relabel_nodes(G, mapping)

    node_args = set(signature(nx.draw_networkx_nodes).parameters.keys())
    node_kwargs = {k: v for k, v in kwargs.items() if k in node_args}
    node_kwargs['node_size'] = kwargs.get('node_size') or 800
    node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

    label_args = set(signature(nx.draw_networkx_labels).parameters.keys())
    label_kwargs = {k: v for k, v in kwargs.items() if k in label_args}
    label_kwargs['font_size'] = kwargs.get('font_size') or 10

    pos = nx.kamada_kawai_layout(G)
    ax = plt.gca()
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->",
                alpha=max(data['att'], 0.1),
                color=data['edge_color'],
                shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ))

    if node_alpha is None:
        nx.draw_networkx_nodes(G, pos, node_color=y.tolist(),
                               **node_kwargs)
    else:
        node_alpha_subset = node_alpha[subset]
        assert ((node_alpha_subset >= 0) & (node_alpha_subset <= 1)).all()
        nx.draw_networkx_nodes(G, pos, alpha=node_alpha_subset.tolist(),
                               node_color=y.tolist(), **node_kwargs)

    nx.draw_networkx_labels(G, pos, **label_kwargs)

    return ax, G

def custom_explain_node(explainer, node_idx, x, edge_index, **kwargs):
    explainer.model.eval()
    explainer.__clear_masks__()

    num_nodes = x.size(0)
    num_edges = edge_index.size(1)
    
    print(f"edge_index in old: {edge_index}")
    print(f"edge_index in old shape: {edge_index.shape}")

    # Only operate on a k-hop subgraph around `node_idx`.
    x, edge_index, mapping, hard_edge_mask, subset, kwargs = \
        explainer.__subgraph__(node_idx, x, edge_index, **kwargs)
    
    print(f"edge_index in new: {edge_index}")
    print(f"edge_index in new shape: {edge_index.shape}")

    # Get the initial prediction.
    with torch.no_grad():
        out = explainer.model(x=x, edge_index=edge_index, **kwargs)
        if explainer.return_type == 'regression':
            prediction = out
        else:
            log_logits = explainer.__to_log_prob__(out)
            pred_label = log_logits.argmax(dim=-1)

    explainer.__set_masks__(x, edge_index)
    explainer.to(x.device)

    if explainer.allow_edge_mask:
        parameters = [explainer.node_feat_mask, explainer.edge_mask]
    else:
        parameters = [explainer.node_feat_mask]
    optimizer = torch.optim.Adam(parameters, lr=explainer.lr)

    if explainer.log:  # pragma: no cover
        pbar = tqdm(total=explainer.epochs)
        pbar.set_description(f'Explain node {node_idx}')

    for epoch in range(1, explainer.epochs + 1):
        optimizer.zero_grad()
        h = x * explainer.node_feat_mask.sigmoid()
        out = explainer.model(x=h, edge_index=edge_index, **kwargs)
        if explainer.return_type == 'regression':
            loss = explainer.__loss__(mapping, out, prediction)
        else:
            log_logits = explainer.__to_log_prob__(out)
            loss = explainer.__loss__(mapping, log_logits, pred_label)
        loss.backward(retain_graph=True)
        optimizer.step()

        if explainer.log:  # pragma: no cover
            pbar.update(1)

    if explainer.log:  # pragma: no cover
        pbar.close()

    node_feat_mask = explainer.node_feat_mask.detach().sigmoid()
    if explainer.feat_mask_type == 'individual_feature':
        new_mask = x.new_zeros(num_nodes, x.size(-1))
        new_mask[subset] = node_feat_mask
        node_feat_mask = new_mask
    elif explainer.feat_mask_type == 'scalar':
        new_mask = x.new_zeros(num_nodes, 1)
        new_mask[subset] = node_feat_mask
        node_feat_mask = new_mask
    node_feat_mask = node_feat_mask.squeeze()

    edge_mask = explainer.edge_mask.new_zeros(num_edges)
    edge_mask[hard_edge_mask] = explainer.edge_mask.detach().sigmoid()

    explainer.__clear_masks__()

    return node_feat_mask, edge_mask

def custom_explain_graph(explainer, x, edge_index, **kwargs):
    explainer.model.eval()
    explainer.__clear_masks__()

    # All nodes belong to the same graph
    batch = torch.zeros(x.shape[0], dtype=int, device=x.device)

    # Get the initial prediction
    with torch.no_grad():
        out = explainer.model(x=x, edge_index=edge_index, batch=batch, **kwargs)
        if explainer.return_type == 'regression':
            prediction = out
        else:
            log_logits = explainer.__to_log_prob__(out)
            pred_label = log_logits.argmax(dim=-1)

    explainer.__set_masks__(x, edge_index)
    explainer.to(x.device)
    parameters = [explainer.node_feat_mask]
    if explainer.allow_edge_mask:
        parameters.append(explainer.edge_mask)

    optimizer = torch.optim.Adam(parameters, lr=explainer.lr)

    if explainer.log:  # pragma: no cover
        pbar = tqdm(total=explainer.epochs)
        pbar.set_description('Explain graph')

    for epoch in range(1, explainer.epochs + 1):
        optimizer.zero_grad()
        h = x * explainer.node_feat_mask.sigmoid()
        out = explainer.model(x=h, edge_index=edge_index, batch=batch, **kwargs)
        if explainer.return_type == 'regression':
            loss = explainer.__loss__(-1, out, prediction)
        else:
            log_logits = explainer.__to_log_prob__(out)
            loss = explainer.__loss__(-1, log_logits, pred_label)
        loss = loss.mean()  # Ensure the loss is a scalar, ensures proper gradient computation
        loss.backward()
        optimizer.step()

        if explainer.log:  # pragma: no cover
            pbar.update(1)

    if explainer.log:  # pragma: no cover
        pbar.close()

    node_feat_mask = explainer.node_feat_mask.detach().sigmoid().squeeze()
    edge_mask = explainer.edge_mask.detach().sigmoid()

    explainer.__clear_masks__()
    return node_feat_mask, edge_mask


"""Explain node plots"""
def explain_node_prediction(model, data, dev, epochs_exp, node_idx, index, epoch):
    with torch.no_grad():
        data = data.to(dev)

    y_min, y_max = data.y.min(), data.y.max()
    data.y = (data.y - y_min) / (y_max - y_min)

    explainer = GNNExplainer(model, epochs=epochs_exp, return_type='log_prob')
    
    node_feat_mask, edge_mask = custom_explain_node(
        explainer,
        node_idx, 
        data.x, 
        data.edge_index, 
        pos=data.pos, 
        edge_attr=data.edge_attr, 
        node_attr=data.node_attr, 
        batch=data.batch, 
        y=data.y
    )
    print(f"\nnode_feat_mask, edge_mask\n{node_feat_mask}, {edge_mask}")

    fig = plt.figure(figsize=(10, 6), dpi=200)
    ax, G = custom_visualize_subgraph(explainer, node_idx, data.edge_index, edge_mask, y=data.y)

    plt.title(f"{node_idx}'s Node Explanation for sample {index} of epoch {epoch}")
    figure_storage.store(
            fig, f"{node_idx}'s Node Explanation for sample {index} of epoch {epoch}")
    plt.show()

    print("Ground Truth label for node: ", node_idx, " is ", data.y[node_idx].cpu().numpy())

    # Create a table for the edge information
    edge_info = []
    for (source, target, data) in G.edges(data=True):
        direction = f"{source} -> {target}"
        edge_info.append({
            "Source Node": source,
            "Target Node": target,
            "Edge Thickness (Mask Value)": data['att'],
            "Edge Color": data['edge_color'],
            "Direction": direction
        })

    edge_info_df = pd.DataFrame(edge_info)
    print(edge_info_df)

    # save the table to a CSV file
    edge_info_df.to_csv(os.path.join(result_folder, "Docs", f"node_{node_idx}_explanation_edges_epoch{epoch}.csv"), index=False)

"""Explain graph plot"""
def explain_graph_prediction(model, data, edge_index, dev, epochs_exp):
    model.eval()
    with torch.no_grad():
        data = data.to(dev)
        data.edge_index = edge_index
        edge_relativePos = (torch.index_select(data.pos, 0, data.edge_index[1]) - torch.index_select(data.pos, 0, data.edge_index[0]))
        edge_relativeDist = torch.norm(edge_relativePos, dim = -1, keepdim = True)
        edge_attr = torch.cat([edge_relativeDist, edge_relativePos], dim = -1)
        data.edge_attr = edge_attr

    y_min, y_max = data.y.min(), data.y.max()
    data.y = (data.y - y_min) / (y_max - y_max)

    explainer = GNNExplainer(model, epochs=epochs_exp, return_type='log_prob')
  
    node_feat_mask, edge_mask = custom_explain_graph(
        explainer,
        data.x, 
        data.edge_index, 
        pos=data.pos, 
        edge_attr=data.edge_attr, 
        node_attr=data.node_attr, 
        y=data.y
    )
    print(f"\nnode_feat_mask, edge_mask\n{node_feat_mask}, {edge_mask}")
    
    fig = plt.figure(figsize=(20, 12), dpi=200)
    ax, G = custom_visualize_subgraph(explainer, -1, data.edge_index.cpu(), edge_mask.cpu(), y=data.y.cpu())
   
    plt.title("Graph Explanation")
    figure_storage.store(
            fig, "graph_explanation")
    plt.show()
    
    # Create a table for the edge information
    edge_info = []
    for (source, target, data) in G.edges(data=True):
        direction = f"{source} -> {target}"
        edge_info.append({
            "Source Node": source,
            "Target Node": target,
            "Edge Thickness (Mask Value)": data['att'],
            "Edge Color": data['edge_color'],
            "Direction": direction
        })

    edge_info_df = pd.DataFrame(edge_info)
    # print(edge_info_df)

    # save the table to a CSV file
    edge_info_df.to_csv(os.path.join(result_folder, "Docs", "graph_explanation_edges.csv"), index=False)

"""GradCAM plots"""
def grad_cam_plots(model, data, target_layer, dev, index, epoch):
    data = data.to(dev)

    grad_cam = GradCAM(model, target_layer)
    data.requires_grad = True

    heatmap, activations, weights = grad_cam(data.x, data.edge_index, pos=data.pos, edge_attr=data.edge_attr, node_attr=data.node_attr, batch=data.batch, y=data.y)

    # # Plot the heatmap
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(data.pos[:, 0].detach().cpu().numpy(), data.pos[:, 1].detach().cpu().numpy(), data.pos[:, 2].detach().cpu().numpy(), c=heatmap.detach().cpu().numpy(), cmap='hot', s=20)
    # fig.colorbar(scatter)
    # plt.title(f'Grad-CAM Heatmap for sample {index} of epoch {epoch}')
    # figure_storage.store(
    #         fig, f'Grad-CAM Heatmap for sample {index} of epoch {epoch}')
    # plt.show()
    
    # Plot the heatmap in 2D
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data.pos[:, 0].detach().cpu().numpy(), data.pos[:, 1].detach().cpu().numpy(), c=heatmap.detach().cpu().numpy(), cmap='hot', s=20)
    fig.colorbar(scatter)
    plt.title(f'Grad-CAM Heatmap for sample {index} of epoch {epoch}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    figure_storage.store(
            fig, f'Grad-CAM Heatmap for sample {index} of epoch {epoch}')
    plt.show()

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

"""SHAP plots"""
def model_prediction(x, model, data, device):
    data = data.to(device)
    x = torch.tensor(x, dtype=torch.float32).to(device)
    output = model(x, data.edge_index, pos=data.pos, edge_attr=data.edge_attr, node_attr=data.node_attr, batch=data.batch, y=data.y)
    return output.detach().cpu().numpy()

def compute_shap_values(model, data, device, num_samples=100):
    model.eval()
    data = data.to(device)

    x = data.x.detach().cpu().numpy()
    baseline = np.mean(x, axis=0).reshape(1, -1)
    print(f"Baseline shape: {baseline.shape}")
    print(f"Data shape: {x.shape}")

    explainer = shap.KernelExplainer(lambda x: model_prediction(x, model, data, device), baseline)

    shap_values = explainer.shap_values(x, nsamples=num_samples)
    return shap_values

def plot_shap(shap_values, data_x, index, epoch):
    fig = plt.figure(figsize=(15, 8), dpi=200)
    shap.summary_plot(shap_values, data_x, show=False)
    plt.title(f"SHAP Summary Plot for Node Features of sample {index} of epoch {epoch}", fontsize=10)
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=10)
    # plt.ylabel("Node Features", fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    figure_storage.store(
            fig, f"SHAP Summary Plot for Node Features of sample {index} of epoch {epoch}")
    plt.show()

def plot_ind_shap(shap_value, data_x, i, index, epoch):
    fig = plt.figure(figsize=(15, 8), dpi=200)
    shap.summary_plot(shap_value, data_x, show=False)
    plt.title(f"SHAP Summary Plot for Node Feature of class {i} of sample {index} of epoch {epoch}", fontsize=10)
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=10)
    # plt.ylabel("Node Features", fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    figure_storage.store(
            fig, f"SHAP Summary Plot for Node Feature of class {i} of sample {index} of epoch {epoch}")
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
def plot_gradcam(model, sample, node_idx, index, epoch):
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
        num_hops=3, additional_hops=0, graph_data=graph_data, 
        ax=ax, norm_imps=False
    )

    ax.set_title("GradCAM Explanation of sample {index} of epoch {epoch}", fontsize=20, fontweight='semibold')
    figure_storage.store(
            fig, f"{node_idx} GradCAM Explanation of sample {index} of epoch {epoch}")
    plt.show()
