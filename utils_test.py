import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch_geometric.nn.models import GNNExplainer
from gradcam import GradCAM
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data
from inspect import signature
import vtk

def visualize_3d_graph_vtk(data, edge_mask=None, node_color=None, highlight_node=None):
    """
    Visualize a 3D graph using VTK.
    
    Args:
        data (torch_geometric.data.Data): Input graph data.
        edge_mask (torch.Tensor, optional): Edge mask for highlighting edges.
        node_color (torch.Tensor, optional): Node colors.
        highlight_node (int, optional): Node index to highlight.
    """
    # Initialize VTK rendering environment
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    diamArray = vtk.vtkFloatArray()
    diamArray.SetName('Diameter')
    
    num_nodes = data.pos.size(0)
    
    # Convert PyTorch geometric data to VTK data structures
    for i, pos in enumerate(data.pos):
        points.InsertNextPoint(pos.cpu().numpy())
        if node_color is not None and i < len(node_color):
            diamArray.InsertNextValue(node_color[i].item())
        else:
            diamArray.InsertNextValue(1.0)  # Default diameter value
    
    for i in range(data.edge_index.size(1)):
        line = vtk.vtkLine()
        src, dst = data.edge_index[:, i].cpu().numpy()
        line.GetPointIds().SetId(0, src)
        line.GetPointIds().SetId(1, dst)
        lines.InsertNextCell(line)
    
    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    if node_color is not None:
        polydata.GetPointData().SetScalars(diamArray)
    
    # Setup actor and mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    if node_color is not None:
        mapper.ScalarVisibilityOn()
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # Highlight the node if specified
    if highlight_node is not None:
        sphere = vtk.vtkSphereSource()
        highlight_pos = data.pos[highlight_node].cpu().numpy()
        sphere.SetCenter(highlight_pos)
        sphere.SetRadius(0.05)  # Adjust the radius as needed
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(1, 0, 0)  # Red color

    # Setup renderer and render window
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    
    # Setup render window interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    
    # Add actor to renderer
    renderer.AddActor(actor)
    if highlight_node is not None:
        renderer.AddActor(sphere_actor)
    renderer.SetBackground(0.1, 0.2, 0.4)  # Background color
    
    # Render and start interaction
    renderWindow.Render()
    renderWindowInteractor.Start()

def custom_visualize_subgraph(explainer, node_idx, edge_index, edge_mask, y=None,
                              threshold=None, edge_y=None, node_alpha=None,
                              seed=10, **kwargs):

    assert edge_mask.size(0) == edge_index.size(1)

    if node_idx == -1:
        hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1),
                                          device=edge_mask.device)
        subset = torch.arange(edge_index.max().item() + 1,
                              device=edge_index.device)
        y = None

    else:
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

    if hasattr(data, 'pos') and data.pos is not None:
        pos = {i: data.pos[i].cpu().numpy() for i in G.nodes()}
        is_3d = data.pos.size(1) == 3
    else:
        pos = nx.spring_layout(G, seed=seed)
        is_3d = False

    fig = kwargs.pop('fig', None)
    ax = kwargs.pop('ax', None)
    
    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d' if is_3d else 'rectilinear')

    for source, target, data in G.edges(data=True):
        if is_3d:
            ax.plot([pos[source][0], pos[target][0]],
                    [pos[source][1], pos[target][1]],
                    [pos[source][2], pos[target][2]],
                    color=data['edge_color'], alpha=max(data['att'], 0.1))
        else:
            ax.plot([pos[source][0], pos[target][0]],
                    [pos[source][1], pos[target][1]],
                    color=data['edge_color'], alpha=max(data['att'], 0.1))

    node_args = set(signature(nx.draw_networkx_nodes).parameters.keys())
    node_kwargs = {k: v for k, v in kwargs.items() if k in node_args}
    node_kwargs['s'] = node_kwargs.pop('node_size', 800)
    node_kwargs['cmap'] = node_kwargs.get('cmap', 'cool')

    label_args = set(signature(nx.draw_networkx_labels).parameters.keys())
    label_kwargs = {k: v for k, v in kwargs.items() if k in label_args}
    label_kwargs['font_size'] = label_kwargs.get('font_size', 10)

    node_color = []
    for i in G.nodes():
        if i < y.size(0):
            node_color.append(y[i].item() if y[i].numel() == 1 else y[i].mean().item())
        else:
            print(f"Warning: Node index {i} is out of bounds for y with size {y.size(0)}")
            node_color.append(0)  # or some default value

    if is_3d:
        ax.scatter(*zip(*[pos[n] for n in G.nodes()]), c=node_color, **node_kwargs)
    else:
        ax.scatter(*zip(*[pos[n] for n in G.nodes()]), c=node_color, **node_kwargs)

    return ax, G

def custom_explain_node(explainer, node_idx, x, edge_index, **kwargs):
    explainer.model.eval()
    explainer.__clear_masks__()

    num_nodes = x.size(0)
    num_edges = edge_index.size(1)

    x, edge_index, mapping, hard_edge_mask, subset, kwargs = \
        explainer.__subgraph__(node_idx, x, edge_index, **kwargs)

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

    if explainer.log:
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

        if explainer.log:
            pbar.update(1)

    if explainer.log:
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

    batch = torch.zeros(x.shape[0], dtype=int, device=x.device)

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

    if explainer.log:
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
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        if explainer.log:
            pbar.update(1)

    if explainer.log:
        pbar.close()

    node_feat_mask = explainer.node_feat_mask.detach().sigmoid().squeeze()
    edge_mask = explainer.edge_mask.detach().sigmoid()

    explainer.__clear_masks__()
    return node_feat_mask, edge_mask

"""Explaining the model's node predictions"""
def explain_node_prediction(model, data, dev, plot_dim, epochs_exp, node_idx, index, epoch):
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

    if plot_dim == 3:
        visualize_3d_graph_vtk(data, edge_mask=edge_mask, node_color=node_feat_mask, highlight_node=node_idx)
    else:
        fig = plt.figure(figsize=(10, 6), dpi=200)
        ax, G = explainer.visualize_subgraph(node_idx, data.edge_index.cpu(), edge_mask.cpu(), y=data.y.cpu())
        plt.title(f"{node_idx}'s Node Explanation for sample {index} of epoch {epoch}")
        plt.show()

    print("Ground Truth label for node: ", node_idx, " is ", data.y[node_idx].cpu().numpy())

"""Explaining the model's graph predictions"""
def explain_graph_prediction(model, data, plot_dim, edge_index, dev, epochs_exp):
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

    if plot_dim == 3:
        visualize_3d_graph_vtk(data, edge_mask=edge_mask, node_color=node_feat_mask)
    else:
        fig = plt.figure(figsize=(10, 6), dpi=200)
        ax, G = explainer.visualize_subgraph(-1, data.edge_index.cpu(), edge_mask.cpu(), y=data.y.cpu())
        plt.title("Graph Explanation")
        plt.show()

def grad_cam_plots(model, data, target_layer, dev, index, epoch):
    data = data.to(dev)

    grad_cam = GradCAM(model, target_layer)
    data.requires_grad = True

    heatmap, activations, weights = grad_cam(data.x, data.edge_index, pos=data.pos, edge_attr=data.edge_attr, node_attr=data.node_attr, batch=data.batch, y=data.y)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data.pos[:, 0].detach().cpu().numpy(), data.pos[:, 1].detach().cpu().numpy(), data.pos[:, 2].detach().cpu().numpy(), c=heatmap.detach().cpu().numpy(), cmap='hot', s=20)
    fig.colorbar(scatter)
    plt.title(f'Grad-CAM Heatmap for sample {index} of epoch {epoch}')
    plt.show()
