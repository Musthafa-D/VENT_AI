import torch
from graphxai.utils import Explanation, node_mask_from_edge_mask
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import List, Tuple, Dict
from torch_geometric.data import Data


def graph_loss(log_logits, pred_label, edge_mask, x, edge_index):
    loss = -log_logits[0, pred_label[0]]
    
    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }
    
    EPS = 1e-15
    std = 0.1
    (n, f), E = x.size(), edge_index.size(1)
    node_feat_mask = torch.nn.Parameter(torch.randn(1, f) * std)

    m = edge_mask.sigmoid()
    edge_reduce = getattr(torch, coeffs['edge_reduction'])
    loss = loss + coeffs['edge_size'] * edge_reduce(m)
    ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
    loss = loss + coeffs['edge_ent'] * ent.mean()

    m = node_feat_mask.sigmoid()
    node_feat_reduce = getattr(torch, coeffs['node_feat_reduction'])
    loss = loss + coeffs['node_feat_size'] * node_feat_reduce(m)
    ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
    loss = loss + coeffs['node_feat_ent'] * ent.mean()

    return loss


def GNNExplainer_get_explanation_graph(gnn_explainer,
                                       model,
                                       x: torch.Tensor,                 
                                       edge_index: torch.Tensor,
                                       pos, 
                                       edge_attr, 
                                       node_attr, 
                                       batch, 
                                       y,
                                       lr = 0.01):
    r"""Learns and returns a node feature mask and an edge mask that play a
    crucial role to explain the prediction made by the GNN for a graph.

    Args:
        x (Tensor): The node feature matrix.
        edge_index (LongTensor): The edge indices.
        **kwargs (optional): Additional arguments passed to the GNN module.

    :rtype: (:class:`Tensor`, :class:`Tensor`)
    """

    model.eval()
    gnn_explainer._clear_masks()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPS = 1e-15

    # all nodes belong to same graph
    # batch = torch.zeros(x.shape[0], dtype=int, device=x.device)

    # Get the initial prediction.
    with torch.no_grad():
        # out = self.model(x=x, edge_index=edge_index, **forward_kwargs)
        # # if self.return_type == 'regression':
        # #     prediction = out
        # # else:
        # log_logits = self.__to_log_prob__(out)
        # log_logits = self._predict(x.to(device), edge_index.to(device), forward_kwargs = forward_kwargs, return_type='log_prob')
        log_logits = custom_predict(x, edge_index, model,
                               pos, 
                               edge_attr, 
                               node_attr, 
                               batch, 
                               y,
                               return_type='log_prob')
        pred_label = log_logits.argmax(dim=-1)

    gnn_explainer._set_masks(x, edge_index, edge_mask = None, explain_feature = True, device = x.device)
    #self.to(x.device)
    #if self.allow_edge_mask:
    # self.feature_mask = self.feature_mask.to(x.device)
    # self.edge_mask = self.edge_mask.to(x.device)
    parameters = [gnn_explainer.feature_mask, gnn_explainer.edge_mask]
    #else:
        #parameters = [self.feature_mask]
    #ipdb.set_trace()
    optimizer = torch.optim.Adam(parameters, lr=lr)

    # if self.log:  # pragma: no cover
    #     pbar = tqdm(total=self.epochs)
    #     pbar.set_description('Explain graph')

    def loss_fn(node_idx, log_logits, pred_label):
        # node_idx is -1 for explaining graphs
        # if self.return_type == 'regression':
        #     if node_idx != -1:
        #         loss = torch.cdist(log_logits[node_idx], pred_label[node_idx])
        #     else:
        #         loss = torch.cdist(log_logits, pred_label)
        # else:
        if node_idx != -1:
            loss = -log_logits[node_idx, pred_label[node_idx]]
        else:
            loss = -log_logits[0, pred_label[0]]

        m = gnn_explainer.edge_mask.sigmoid()
        #edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + gnn_explainer.coeff['edge']['size'] * torch.sum(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        #loss = loss + self.coeffs['edge_ent'] * ent.mean()
        loss = loss + gnn_explainer.coeff['edge']['entropy'] * ent.mean()

        m = gnn_explainer.feature_mask.sigmoid()
        #node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        #loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        loss = loss + gnn_explainer.coeff['feature']['size'] * torch.sum(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        #loss = loss + self.coeffs['node_feat_ent'] * ent.mean()
        loss = loss + gnn_explainer.coeff['feature']['entropy'] * ent.mean()

        return loss

    num_epochs = 200 # TODO: make more general
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        h = x * gnn_explainer.feature_mask.sigmoid()
        # out = self.model(x=h, edge_index=edge_index, **forward_kwargs)

        # log_logits = self.__to_log_prob__(out)

        # log_logits = self._predict(h.to(device), edge_index.to(device), forward_kwargs = forward_kwargs, return_type='log_prob')
        log_logits = custom_predict(h.to(device), edge_index.to(device), 
                                  model.to(device), 
                                  pos.to(device), 
                                  edge_attr.to(device), 
                                  node_attr.to(device), 
                                  batch.to(device), 
                                  y.to(device),
                                  return_type='log_prob')

        loss = loss_fn(-1, log_logits, pred_label)
        loss.backward()
        optimizer.step()

    #     if self.log:  # pragma: no cover
    #         pbar.update(1)

    # if self.log:  # pragma: no cover
    #     pbar.close()

    feature_mask = gnn_explainer.feature_mask.detach().sigmoid().squeeze()
    edge_mask = gnn_explainer.edge_mask.detach().sigmoid()

    gnn_explainer._clear_masks()

    node_imp = node_mask_from_edge_mask(
        torch.arange(x.shape[0]).to(x.device), 
        edge_index, 
        (edge_mask > 0.5)) # Make edge mask into discrete and convert to node mask

    exp = Explanation(
        feature_imp = feature_mask,
        node_imp = node_imp.float(),
        edge_imp = edge_mask 
    )

    exp.set_whole_graph(Data(x=x, edge_index=edge_index))

    return exp


def GNNExplainer_get_explanation_node(gnn_explainer,
                                      model,
                                      node_idx: int, 
                                      x: torch.Tensor,                 
                                      edge_index: torch.Tensor,
                                      pos, 
                                      edge_attr, 
                                      node_attr, 
                                      batch, 
                                      y,
                                      graph,
                                      label: torch.Tensor = None,
                                      num_hops: int = None,
                                      explain_feature: bool = True
                                      ):
    """
    Explain a node prediction.

    Args:
        node_idx (int): index of the node to be explained
        edge_index (torch.Tensor, [2 x m]): edge index of the graph
        x (torch.Tensor, [n x d]): node features
        label (torch.Tensor, optional, [n x ...]): labels to explain
            If not provided, we use the output of the model.
        num_hops (int, optional): number of hops to consider
            If not provided, we use the number of graph layers of the GNN.
        explain_feature (bool): whether to compute the feature mask or not
            (:default: :obj:`True`)
        forward_kwargs (dict, optional): additional arguments to model.forward
            beyond x and edge_index

    Returns:
        exp (dict):
            exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
            exp['edge_imp'] (torch.Tensor): k-hop edge importance
        khop_info (4-tuple of torch.Tensor):
            0. the nodes involved in the subgraph
            1. the filtered `edge_index`
            2. the mapping from node indices in `node_idx` to their new location
            3. the `edge_index` mask indicating which edges were preserved
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label = custom_predict(x, edge_index, model,
                           pos, 
                           edge_attr, 
                           node_attr, 
                           batch, 
                           y) # if label is None else label
    num_hops = gnn_explainer.L if num_hops is None else num_hops

    org_eidx = edge_index.clone().to(device)
    
    if node_idx == -1:
        hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1),
                                          device="cpu")
        subset = torch.arange(edge_index.max().item() + 1,
                              device=device)
        x = x[subset].to(device)
        # print(f'subset: {subset}')
        # print(f'edge_index: {edge_index}')
        # print(f'hard_edge_mask: {hard_edge_mask}')
        # print(f'x shape: {x.shape}')
        # print(f'edge_index shape: {edge_index.shape}')
    else:
        khop_info = subset, sub_edge_index, mapping, hard_edge_mask = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True) #num_nodes=x.shape[0])
        sub_x = x[subset].to(device)
        # print(f'subset: {subset}')
        # print(f'sub_edge_index: {sub_edge_index}')
        # print(f'mapping: {mapping}')
        # print(f'hard_edge_mask: {hard_edge_mask}')
        # print(f'sub_x shape: {sub_x.shape}')
        # print(f'sub_edge_index shape: {sub_edge_index.shape}')

    if node_idx != -1:
        gnn_explainer._set_masks(sub_x.to(device), sub_edge_index.to(device), explain_feature=explain_feature)
    else:
        gnn_explainer._set_masks(x.to(device), edge_index.to(device), explain_feature=explain_feature)

    gnn_explainer.model.eval()
    num_epochs = 200

    # Loss function for GNNExplainer's objective
    def loss_fn(log_prob, mask, mask_type):
        # Select the log prob and the label of node_idx
        node_log_prob = log_prob[torch.where(subset==node_idx)].squeeze()
        node_label = label[mapping]
        # Maximize the probability of predicting the label (cross entropy)
        loss = -node_log_prob[node_label].item()
        a = mask.sigmoid()
        # Size regularization
        loss += gnn_explainer.coeff[mask_type]['size'] * torch.sum(a)
        # Element-wise entropy regularization
        # Low entropy implies the mask is close to binary
        entropy = -a * torch.log(a + 1e-15) - (1-a) * torch.log(1-a + 1e-15)
        loss += gnn_explainer.coeff[mask_type]['entropy'] * entropy.mean()
        return loss

    def train(mask, mask_type):
        optimizer = torch.optim.Adam([mask], lr=0.01)
        for epoch in range(1, num_epochs+1):
            optimizer.zero_grad()
            if mask_type == 'feature':
                if node_idx != -1:
                    h = sub_x.to(device) * mask.view(1, -1).sigmoid().to(device)
                else:
                    h = x.to(device) * mask.view(1, -1).sigmoid().to(device)
            else:
                if node_idx != -1:
                    h = sub_x.to(device)
                else:
                    h = x.to(device)
            
            if node_idx != -1:
                # Index edge_attr to match sub_edge_index
                sub_edge_attr = edge_attr[hard_edge_mask].to(device)
            else:
                graph_edge_attr = edge_attr[hard_edge_mask].to(device)
            
            if node_idx != -1:
                log_prob = custom_predict(h.to(device), sub_edge_index.to(device), 
                                          model.to(device), 
                                          pos[subset].to(device), 
                                          sub_edge_attr.to(device), 
                                          node_attr[subset].to(device), 
                                          batch.to(device), 
                                          y.to(device),
                                          return_type='log_prob')
                loss = loss_fn(log_prob, mask, mask_type)
                loss.backward()
                optimizer.step()
            else:
                log_prob = custom_predict(h.to(device), edge_index.to(device), 
                                          model.to(device), 
                                          pos[subset].to(device), 
                                          graph_edge_attr.to(device), 
                                          node_attr[subset].to(device), 
                                          batch.to(device), 
                                          y.to(device),
                                          return_type='log_prob')
                loss = graph_loss(log_prob, label, gnn_explainer.edge_mask, h.to(device), edge_index.to(device))
                loss.backward()
                optimizer.step()

    feat_imp = None
    if explain_feature: # Get a feature mask
        train(gnn_explainer.feature_mask, 'feature')
        feat_imp = gnn_explainer.feature_mask.data.sigmoid()

    train(gnn_explainer.edge_mask, 'edge')
    edge_imp = gnn_explainer.edge_mask.data.sigmoid().to(device)

    # print('pre activation edge_imp:', edge_imp)

    # print('IN GNNEXPLAINER')
    # print('edge imp shape', edge_imp.shape)

    gnn_explainer._clear_masks()

    discrete_edge_mask = (edge_imp > 0.5) # Turn into bool activation because of sigmoid

    if node_idx != -1:
        khop_info = (subset, org_eidx[:,hard_edge_mask], mapping, hard_edge_mask)

        exp = Explanation(
            feature_imp = feat_imp,
            node_imp = node_mask_from_edge_mask(khop_info[0], khop_info[1], edge_mask = discrete_edge_mask),
            edge_imp = discrete_edge_mask.float(),
            node_idx = node_idx,
            graph=graph
        )
    else:
        exp = Explanation(
            feature_imp = feat_imp,
            node_imp = node_mask_from_edge_mask(subset, org_eidx[:,hard_edge_mask], edge_mask = discrete_edge_mask),
            edge_imp = discrete_edge_mask.float(),
            node_idx = node_idx,
            graph=graph
        )
        
    # print('Node importance', exp.node_imp)
    # print('Node importance shape', exp.node_imp.shape)
    
    if node_idx != -1:
        exp.set_enclosing_subgraph(khop_info)
    # else:
    #     khop_info = (subset, org_eidx[:,hard_edge_mask], 0, hard_edge_mask)
    #     exp.set_enclosing_subgraph(khop_info)

    return exp


def GradCAM_get_explanation_node(gradcam_explainer, 
                                 model,
        x: torch.Tensor, 
        y: torch.Tensor, 
        node_idx: int, 
        edge_index: torch.Tensor, 
        pos, 
        edge_attr, 
        node_attr, 
        batch,
        label: int = None,  
        average_variant: bool = True, 
        layer: int = 0
    ) -> Explanation:
    '''
    Explain a node in the given graph
    Args:
        x (torch.Tensor, (n,)): Tensor of node features from the entire graph, with n nodes.
        y (torch.Tensor, (n,)): Ground-truth labels for all n nodes in the graph.
        node_idx (int): node index for which to explain a prediction around
        edge_index (torch.Tensor): Edge_index of entire graph.
        label (int, optional): Label for which to compute Grad-CAM against. If None, computes
            the Grad-CAM with respect to the model's predicted class for this node.
            (default :obj:`None`)
        forward_kwargs (dict, optional): Additional arguments to model.forward 
            beyond x and edge_index. (default: :obj:`None`)
        average_variant (bool, optional): If True, computes the average Grad-CAM across all convolutional
            layers in the model. If False, computes Grad-CAM for `layer`. (default: :obj:`True`)
        layer (int, optional): Layer by which to compute the Grad-CAM. Argument only has an effect if 
            `average_variant == True`. Must be less-than the total number of convolutional layers
            in the model. (default: :obj:`0`)

    :rtype: :class:`graphxai.Explanation`

    Returns:
        exp (:class:`Explanation`): Explanation output from the method.
            Fields are:
            `feature_imp`: :obj:`None`
            `node_imp`: :obj:`torch.Tensor, [nodes_in_khop,]`
            `edge_imp`: :obj:`None`
            `enc_subgraph`: :obj:`graphxai.utils.EnclosingSubgraph`
    '''

    x = x.detach().clone()
    y = y

    x.requires_grad = True

    if label is None:
        pred = custom__forward_pass(model, x, y, edge_index, pos, edge_attr, node_attr, batch)[0][node_idx, :].reshape(1, -1)
        y[node_idx] = pred.argmax(dim=1).item()
    else: # Transform node_idx's label if provided by user
        pred, loss = custom__forward_pass(model, x, y, edge_index, pos, edge_attr, node_attr, batch)
        y[node_idx] = label

    walk_steps, _ = custom_extract_step(model, x, edge_index, pos, edge_attr, node_attr, batch, y, detach=True, split_fc=True)

    khop_info = k_hop_subgraph(node_idx, gradcam_explainer.L, edge_index)
    subgraph_nodes = khop_info[0]

    N = maybe_num_nodes(edge_index, None)
    subgraph_N = len(subgraph_nodes.tolist())

    exp = Explanation(
        node_idx = node_idx
    )
    exp.set_enclosing_subgraph(khop_info)

    if average_variant:
        # Size of all nodes in the subgraph:
        avg_gcam = torch.zeros(subgraph_N)

        for l in range(gradcam_explainer.L):
            # Compute gradients for this layer ahead of time:
            gradients = gradcam_explainer.__grad_by_layer(l)

            for i in range(subgraph_N): # Over all subgraph nodes
                n = subgraph_nodes[i]
                avg_gcam[i] += gradcam_explainer.__get_gCAM_layer(walk_steps, l, n, gradients)

        avg_gcam /= gradcam_explainer.L # Apply average

        exp.node_imp = avg_gcam

    else:
        assert layer < len(walk_steps), "Layer must be an index of convolutional layers"

        gcam = torch.zeros(subgraph_N)
        gradients = gradcam_explainer.__grad_by_layer(layer)
        for i in range(subgraph_N):
            n = subgraph_nodes[i]
            gcam[i] += gradcam_explainer.__get_gCAM_layer(walk_steps, layer, n, gradients)#[0]

        exp.node_imp = gcam

    return exp


def custom_predict(x: torch.Tensor, edge_index: torch.Tensor, model, pos, edge_attr, node_attr, batch, y, explain_graph: bool = False,
             return_type: str = 'label'):
    """
    Get the model's prediction.

    Args:
        x (torch.Tensor, [n x d]): node features
        edge_index (torch.Tensor, [2 x m]): edge index of the graph
        return_type (str): one of ['label', 'prob', 'log_prob']
        forward_kwargs (dict, optional): additional arguments to model.forward
            beyond x and edge_index

    Returns:
        pred (torch.Tensor, [n x ...]): model prediction
    """
    # Compute unnormalized class score
    with torch.no_grad():
        out = model(x, edge_index, pos=pos, edge_attr=edge_attr, 
                     node_attr=node_attr, batch=batch, y=y)  
        if return_type == 'label':
            out = out.argmax(dim=-1)
        elif return_type == 'prob':
            out = F.softmax(out, dim=-1)
        elif return_type == 'log_prob':
            out = F.log_softmax(out, dim=-1)
        else:
            raise ValueError("return_type must be 'label', 'prob', or 'log_prob'")

        if explain_graph:
            out = out.squeeze()

        return out


def custom__forward_pass(model, x, label, edge_index, pos, edge_attr, node_attr, batch):
    x.requires_grad = True # Enforce that x needs gradient

    # Forward pass:
    model.eval()
    pred = model(x, edge_index, pos=pos, edge_attr=edge_attr, 
                 node_attr=node_attr, batch=batch, y=label)
    
    criterion = F.cross_entropy

    loss = criterion(pred, label)
    loss.backward() # Propagate loss backward through network

    return pred, loss


def custom_extract_step(model, x: torch.Tensor, edge_index: torch.Tensor, pos, edge_attr, node_attr, batch, y, detach: bool = True, split_fc: bool = False):
    '''Gets information about every layer in the graph
    Args:

        forward_kwargs (tuple, optional): Additional arguments to model forward call (other than x and edge_index)
            (default: :obj:`None`)
    '''

    layer_extractor = []
    hooks = []

    def register_hook(module: torch.nn.Module):
        if not list(module.children()) or isinstance(module, MessagePassing):
            hooks.append(module.register_forward_hook(forward_hook))

    def forward_hook(module: torch.nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
        # input contains x and edge_index
        if detach:
            layer_extractor.append((module, input[0].clone().detach(), output.clone().detach()))
        else:
            layer_extractor.append((module, input[0], output))

    # --- register hooks ---
    model.apply(register_hook)

    # ADDED: OWEN QUEEN --------------
    _ = model(x, edge_index, pos=pos, edge_attr=edge_attr, 
                 node_attr=node_attr, batch=batch, y=y)
    # --------------------------------
    # Remove hooks:
    for hook in hooks:
        hook.remove()

    # --- divide layer sets ---

    # print('Layer extractor', [layer_extractor[i][0] for i in range(len(layer_extractor))])

    walk_steps = []
    fc_steps = []
    pool_flag = False
    step = {'input': None, 'module': [], 'output': None}
    for layer in layer_extractor:
        if isinstance(layer[0], MessagePassing):
            if step['module']: # Append step that had previously been building
                walk_steps.append(step)

            step = {'input': layer[1], 'module': [], 'output': None}

        elif isinstance(layer[0], GNNPool):
            pool_flag = True
            if step['module']:
                walk_steps.append(step)

            # Putting in GNNPool
            step = {'input': layer[1], 'module': [], 'output': None}

        elif isinstance(layer[0], torch.nn.Linear):
            if step['module']:
                if isinstance(step['module'][0], MessagePassing):
                    walk_steps.append(step) # Append MessagePassing layer to walk_steps
                else: # Always append Linear layers to fc_steps
                    fc_steps.append(step)

            step = {'input': layer[1], 'module': [], 'output': None}

        # Also appends non-trainable layers to step (not modifying input):
        step['module'].append(layer[0])
        step['output'] = layer[2]

    if step['module']:
        if isinstance(step['module'][0], MessagePassing):
            walk_steps.append(step)
        else: # Append anything to FC that is not MessagePassing at its origin
            # Still supports sequential layers
            fc_steps.append(step)
        # print('layer', layer[0])
        # if isinstance(layer[0], MessagePassing) or isinstance(layer[0], GNNPool):
        #     if isinstance(layer[0], GNNPool):
        #         pool_flag = True
        #     if step['module'] and step['input'] is not None:
        #         walk_steps.append(step)
        #     step = {'input': layer[1], 'module': [], 'output': None}
        # if pool_flag and split_fc and isinstance(layer[0], nn.Linear):
        #     if step['module']:
        #         fc_steps.append(step)
        #     step = {'input': layer[1], 'module': [], 'output': None}
        # step['module'].append(layer[0])
        # step['output'] = layer[2]

    for walk_step in walk_steps:
        if hasattr(walk_step['module'][0], 'nn') and walk_step['module'][0].nn is not None:
            # We don't allow any outside nn during message flow process in GINs
            walk_step['module'] = [walk_step['module'][0]]
        elif hasattr(walk_step['module'][0], 'lin') and walk_step['module'][0].lin is not None:
            walk_step['module'] = [walk_step['module'][0]]

    # print('Walk steps', [walk_steps[i]['module'] for i in range(len(walk_steps))])
    # print('fc steps', [fc_steps[i]['module'] for i in range(len(fc_steps))])

    return walk_steps, fc_steps


class GNNPool(torch.nn.Module):
    def __init__(self):
        super().__init__()