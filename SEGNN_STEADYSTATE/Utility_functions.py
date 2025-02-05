import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_geometric
from torch_geometric.data import Data, Dataset
from tqdm import tqdm, trange
import os
import plotly.graph_objects as go
import numpy as np





def translate_graph_coords(graph, translation_vector):
    """ Graph translation function for invariance testing
        - it lets you translate the graph by a given vector
        - translation_vector is a numpy array or list of length 3 
    """
    
    coords = graph.pos
    
    translated_coords = coords + translation_vector
    translated_graph = graph
    translated_graph.pos = translated_coords
    
    return translated_graph





def rotate_graph_coords(graph, angle, axis="z"): # default to 'z' axis
    """ Graph rotation function for equivariance testing
        - it lets you rotate the graph anti clockwise 
        - angle is radians
    """

    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    else:
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    coords = graph.pos
    
    rotated_coords = np.dot(coords, rotation_matrix.T)
    rotated_graph = graph
    rotated_graph.pos = rotated_coords
    
    return rotated_graph





def print_3D_graph(nodes, edges = None, color = "royalblue", rescale = False):
    """ Utility plot function 
        how to use it: print_3D_graph(pos, edge_index, colors) 
    """
    
    if rescale == True:
        cmin = color.min().item()#float()
        cmax = color.max().item()#float()
        # print(cmin.dtype,cmax.dtype)
    else:
        cmin = None
        cmax = None

    trace1 = go.Scatter3d(
        x=nodes[:,0],
        y=nodes[:,1],
        z=nodes[:,2],
        mode = 'markers',
        showlegend = False,
        marker = dict(size=2, color = color, colorscale = 'turbo', showscale = True, cmin = cmin, cmax= cmax)
        )
    
    if edges != None:
            
        x_lines = []
        y_lines = []
        z_lines = []

        for i in range(edges.shape[1]):
            x_lines.append(nodes[edges[0,i],0])
            y_lines.append(nodes[edges[0,i],1])
            z_lines.append(nodes[edges[0,i],2])
            
            x_lines.append(nodes[edges[1,i],0])
            y_lines.append(nodes[edges[1,i],1])
            z_lines.append(nodes[edges[1,i],2])
            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)

    
        trace2 = go.Scatter3d(
        
            x = x_lines,
            y = y_lines,
            z = z_lines,
            mode = 'lines',
            name = 'edge',
            showlegend = False,
            line = dict(width=1, color="grey"),
            opacity=0.5
            )
        data = [trace1,trace2]
        
    else:
        data = trace1
        
    layout = go.Layout(scene = dict(aspectmode='data'), width = 900, height = 800)  #aspectratio=dict(x=1, y=1, z=1)))
    fig = go.Figure(data=data, layout = layout)

        
    #fig.update_scenes(zaxis_autorange="reversed")
    fig.show()





def manual_print_3D_graph(nodes, edges = None, color = "royalblue", rescale = False):
    """ Utility plot function 
        how to use it: print_3D_graph(pos, edge_index, colors) 
        needs fig.show() to plot
    """
    
    if rescale == True:
        cmin = color.min().item()#float()
        cmax = color.max().item()#float()
        # print(cmin.dtype,cmax.dtype)
    else:
        cmin = None
        cmax = None

    trace1 = go.Scatter3d(
        x=nodes[:,0],
        y=nodes[:,1],
        z=nodes[:,2],
        mode = 'markers',
        showlegend = False,
        marker = dict(size=2, color = color, colorscale = 'turbo', showscale = True, cmin = cmin, cmax= cmax)
        )
    
    if edges != None:
            
        x_lines = []
        y_lines = []
        z_lines = []

        for i in range(edges.shape[1]):
            x_lines.append(nodes[edges[0,i],0])
            y_lines.append(nodes[edges[0,i],1])
            z_lines.append(nodes[edges[0,i],2])
            
            x_lines.append(nodes[edges[1,i],0])
            y_lines.append(nodes[edges[1,i],1])
            z_lines.append(nodes[edges[1,i],2])
            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)

    
        trace2 = go.Scatter3d(
        
            x = x_lines,
            y = y_lines,
            z = z_lines,
            mode = 'lines',
            name = 'edge',
            showlegend = False,
            line = dict(width=1, color="grey"),
            opacity=0.5
            )
        data = [trace1,trace2]
        
    else:
        data = trace1
        
    layout = go.Layout(scene = dict(aspectmode='data'), width = 2400, height = 2400)  #aspectratio=dict(x=1, y=1, z=1)))

    # fig.update_scenes(zaxis_autorange="reversed")

    fig = go.Figure(data=data, layout=layout)
    return fig





def node_to_wall_dist(fluid_nodes, proxy_wall_nodes, offset):
  """ DEPRECATED """

  relative_dist = torch.norm((proxy_wall_nodes - fluid_nodes), dim = -1)
  approx_wall_distance = torch.norm(relative_dist+offset, dim=-1)
  
  return approx_wall_distance




def equiv_features(pos, in_features):
  """ check section "3.2 Input features" here : https://arxiv.org/pdf/2302.08780.pdf """

  # define correct mapping to categorical labels
  # "torch.tensor(N)" simply means that I have a torch object wth a single element, that is N
  inlet_nodes = torch.tensor(0)
  #print(inlet_nodes)
  outlet_nodes = torch.tensor(1)
  fluid_nodes = torch.tensor(2)

  # select inlet nodes
  inlet_nodes_mask = (torch.argmax(in_features[:,2:],dim=1)==inlet_nodes) 
  #print(torch.argmax(in_features[:,2:],dim=1).tolist())
  #print(torch.argmax(in_features[:,2:],dim=1).tolist().count(0))
  # select outlet nodes
  outlet_nodes_mask = (torch.argmax(in_features[:,2:],dim=1)==outlet_nodes) 
  # select fluid nodes
  fluid_nodes_mask = (torch.argmax(in_features[:,2:],dim=1)==fluid_nodes) 
  # print(fluid_nodes_mask.tolist())

  # Take the SDF values for all fluid nodes
  fluid_nodes_sdf = in_features[fluid_nodes_mask, 0]

  # Take the MIS values for all fluid nodes
  fluid_nodes_sdf = in_features[fluid_nodes_mask, 1]

  # For every fluid node, we look for its nearest nodes that belong to each of the following categories: inlet, outlet or wall. 
  # Doing so we compute the vector distances. In principle the result should have 9 components (3 vectors x 3 sets) for
  # every node, but wall nodes are not present so SDF values are simply taken as "fluid node-to-wall" distance, for
  # a total of 3+3+1=7 components

  # for every fluid node, find its nearest inlet node
  #print(inlet_nodes_mask)
  #print(inlet_nodes_mask.tolist())
  #print(inlet_nodes_mask.tolist().count(True))
  # print(pos)
  # print(pos[inlet_nodes_mask])
  arg_feat1 = torch.argmin(torch.norm((pos[fluid_nodes_mask,None,:] - pos[inlet_nodes_mask]),dim=-1), dim = -1)
  # for every fluid node, find its nearest outlet node
  arg_feat2 = torch.argmin(torch.norm((pos[fluid_nodes_mask,None,:] - pos[outlet_nodes_mask]),dim=-1), dim = -1)
  # SDF as a substitute for wall component, as wall nodes were not present in the dataset

  feat3 = pos[fluid_nodes_mask, 0].unsqueeze(-1)

  # MIS value as additional scalar that provides local geometrical information
  feat4 = pos[fluid_nodes_mask, 1].unsqueeze(-1)


  feat1 = (pos[arg_feat1]-pos[fluid_nodes_mask])
  feat2 = (pos[arg_feat2]-pos[fluid_nodes_mask])

  # shape should be N x (3+3+1) = [N,7] where N are the nodes of the graph
  # new_input_features = torch.cat((feat1,feat2,feat3), dim = -1) 

  # shape should be N x (3+3+1+1) = [N,8] where N are the nodes of the graph
  new_input_features = torch.cat((feat1,feat2,feat3,feat4), dim = -1) 

  return new_input_features



def inlet_distance_mask(coords_and_labels, num_nodes = 200):
  """This function is used to evaluate node distances with respect to inlet, and select
     a fixed amount of nodes based on smallest inlet distance
  """

  # define correct mapping to categorical labels
  # "torch.tensor(N)" simply means that I have a torch object wth a single element, that is N
  inlet_nodes = torch.tensor(0)
  fluid_nodes = torch.tensor(2)

  pos = coords_and_labels[:,:3]
  labels = coords_and_labels[:,-1]
  
  # select inlet and fluid nodes
  inlet_nodes_mask =  (labels == inlet_nodes)
  fluid_nodes_mask =  (labels == fluid_nodes)
  
  # for every fluid node, find its nearest inlet node
  distances, arg_feat = torch.min(torch.norm((pos[fluid_nodes_mask,None,:] - pos[inlet_nodes_mask]),dim=-1), dim = -1)
  
  # Get the indices of the "num_nodes" smallest distances
  _, indices = torch.topk(distances, k=num_nodes, largest=False)

  # Create a boolean mask with ones at the specified indices
  boolean_mask = torch.zeros_like(coords_and_labels[:,0], dtype=torch.bool)
  boolean_mask[indices] = 1
  boolean_mask = boolean_mask[fluid_nodes_mask]
  # slice to 2000 to only retain fluid nodes indices
  return boolean_mask



class Graph_dataset(Dataset):
  """ Custom PyG dataset class """

  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):

    """ Root: where the dataset should be stored """
    super().__init__(root, transform, pre_transform, pre_filter)
    
  @property
  def raw_file_names(self):
    return 'Dataset.pt'

  @property
  def processed_file_names(self):

    self.data = torch.load(os.path.join(self.raw_dir,'Dataset.pt'))

    return [f'data_{i}.pt' for i in trange(self.data.shape[0])]

  
  def download(self):
    """ not implemented; it acts if property "raw_file_names" doesn't find the specified file """
    pass

  
  def process(self):
    """it acts if property "processed_file_names" doesn't find the specified list of files in given path
       the code is written so that once "process" acts, it isn't executed anymore in future calls to Graph_dataset 
    """

    # "Dataset.pt" has shape [NxM,F] where:
    # - N is the number of nodes in a graph
    # - M is the number of simulations (steady states)
    # - F is the number of features per node
    # Here F has 13 components: [x,y,z,P,Vx,Vy,Vz,Sxx,Syy,Szz,Sxy,Sxz,Syz]
    self.data = torch.load(os.path.join(self.raw_dir,'Dataset.pt'))
    
    for i, graph in enumerate(tqdm(self.data)):

      # 3D coordinates are unpacked in node_pos variable
      node_pos = graph[..., :3].detach().clone()
      # other components correspond to P,V,Stress now stored in "node_features"
      node_features = graph[..., 3:].detach().clone()

      # all previous torch tensors are stored as separate attributes of a "Data" object 
      # (a class that is used to model a graph).
      data = Data(x = node_features, pos = node_pos)

      # every single graph (data object) is saved in a .pt file
      torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

  def len(self):        
    return self.data.shape[0]

  
  def get(self, idx):
    """ Equivalent to __getitem__ in pytorch 
        it retrieves the graph corresponding to the required index 
    """ 

    sample = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))

    return sample



class Graph_datasetV2(Dataset):
  """Differently from base "Graph_dataset", here we also have SDF, MIS and categorical labels """

  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):

    """ Root: where the dataset should be stored """
    super().__init__(root, transform, pre_transform, pre_filter)
    
  @property
  def raw_file_names(self):
    return 'Dataset.pt'

  @property
  def processed_file_names(self):

    self.data = torch.load(os.path.join(self.raw_dir,'Dataset.pt'))

    return [f'data_{i}.pt' for i in trange(self.data.shape[0])]

  
  def download(self):
    """ not implemented; it acts if property "raw_file_names" doesn't find the specified file """
    pass

  
  def process(self):

    """ it acts if property "processed_file_names" doesn't find the specified list of files in given path
        the code is written so that once "process" acts, it isn't executed anymore in future calls to Graph_datasetV2 
    """

    # "Dataset.pt" has shape [NxM,F] where:
    # - N is the number of nodes in a graph
    # - M is the number of simulations (steady states)
    # - F is the number of features per node
    # Here F has 16 components: [x,y,z,SDF,MIS,label,P,Vx,Vy,Vz,Sxx,Syy,Szz,Sxy,Sxz,Syz]
    self.data = torch.load(os.path.join(self.raw_dir,'Dataset.pt'))
    
    for i, graph in enumerate(tqdm(self.data)):

      # 3D coordinates are unpacked in node_pos variable
      node_pos = graph[..., :3].detach().clone()
      # components 3,4 correspond to SDF and MIS, now stored in node_geom_features1
      # we are excluding positions (first three components) from node features
      node_geom_features1 = graph[..., 3:5].detach().clone() 
      # component 5 is the categorical label; 0 for inlet nodes, 1 for outlet nodes and 2 for fluid nodes
      # the labels are then one-hot encoded in 3 classes ("num_classes"): 0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1]
      node_geom_features2 = F.one_hot(graph[..., 5].detach().clone().to(torch.int64), num_classes=3)
      # SDF, MIS and the one-hot encoding are concatenated to create new 5D feature vector with geometric information
      node_geom_features = torch.cat((node_geom_features1, node_geom_features2), dim=-1)
      # physical features that we want to predict as output are sliced separately
      # "6:10" is ignoring stress tensor, only P,V are extracted
      node_phys_features = graph[..., 6:10].detach().clone() 

      # Use simple tensor as placeholder for nodes attributes 
      node_attr = torch.ones(node_pos.shape[0],1)

      # all previous torch tensors are stored as separate attributes of a "Data" object 
      # (a class that is used to model a graph).
      # pos: node coordinates
      # x: input features
      # node_attr: node attributes
      # edge_index: list of edges (already given if connectivity is specified, else has to be computed)
      # y: output features (to predict)
      data = Data(x = node_geom_features, pos = node_pos, node_attr = node_attr, edge_index = None, y = node_phys_features)
      
      torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

  def len(self):        
    return self.data.shape[0]

  
  def get(self, idx):
    """ Equivalent to __getitem__ in pytorch 
        it retrieves the graph corresponding to the required index
    """ 

    sample = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))

    return sample
  

# it implements the method described in the "equiv_features" function above,
# plus some other minor redistribution of features
class Graph_dataset_with_equiv_features(Dataset):

  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, inlet_mask = False):

    """ Root: where the dataset should be stored """
    super().__init__(root, transform, pre_transform, pre_filter)

    self.inlet_mask = inlet_mask
    
  @property
  def raw_file_names(self):
    return 'Dataset.pt'

  @property
  def processed_file_names(self):

    self.data = torch.load(os.path.join(self.raw_dir,'Dataset.pt'))

    return [f'data_{i}.pt' for i in trange(self.data.shape[0])]

  
  def download(self):
    """ not implemented; it acts if property "raw_file_names" doesn't find the specified file """
    pass

  
  def process(self):
    """ it acts if property "processed_file_names" doesn't find the specified list of files in given path
        the code is written so that once "process" acts, it isn't executed anymore in future calls to Graph_dataset_with_equiv_features
    """

    # "Dataset.pt" has shape [NxM,F] where:
    # - N is the number of nodes in a graph
    # - M is the number of simulations (steady states)
    # - F is the number of features per node
    # Here F has 16 components: [x,y,z,SDF,MIS,label,P,Vx,Vy,Vz,Sxx,Syy,Szz,Sxy,Sxz,Syz]
    self.data = torch.load(os.path.join(self.raw_dir,'Dataset.pt'))
    
    for i, graph in enumerate(tqdm(self.data)):

      # 3D coordinates are unpacked in node_pos variable
      node_pos = graph[..., :3].detach().clone()
      # print(f"Node pos: {node_pos.shape}")
      node_mis = graph[..., 4].detach().clone()
      # components 3,4 correspond to SDF and MIS, now stored in node_geom_features1
      # we are excluding positions (first three components) from node features
      node_geom_features1 = graph[..., 3:5].detach().clone()
      # component 5 is the categorical label; 0 for inlet nodes, 1 for outlet nodes and 2 for fluid nodes
      # the labels are then one-hot encoded in 3 classes ("num_classes"): 0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1]
      labels = graph[..., 5].detach().clone().to(torch.int64)
      node_geom_features2 = F.one_hot(labels, num_classes=3)
      #print(f"Labels: {str(labels.tolist())}")
      # here are stored all 3D coordiantes (including inlet and outlet), and categorical labels
      original_pos_plus_labels = torch.cat((node_pos, labels.unsqueeze(-1)), dim = -1)
      # print(f"Original pos plus labels: {original_pos_plus_labels.shape}")
      # SDF, MIS and the one-hot encoding are concatenated to create new 5D feature vector with geometric information
      node_geom_features = torch.cat((node_geom_features1, node_geom_features2), dim=-1)
      # print(f"Node geom features: {node_geom_features.shape}")
      # physical features that we want to predict as output are sliced separately
      # "6:10" is ignoring stress tensor, only P,V are extracted - changed to 7:10 to ignore P too
      node_phys_features = graph[..., 6:10].detach().clone()
      # print(f"Node phys features: {node_phys_features.shape}")
      node_phys_features_and_mis = torch.cat((graph[..., 7:10].detach().clone(), node_mis.unsqueeze(-1)), dim=1)

      # produce equivariant input features vectors with "equiv_features" function (see above)
      # (you do NOT need this preprocessing for SEGNN to work)
      input_features = equiv_features(node_pos, node_geom_features)
      # print(f"Input features: {input_features.shape}")
      #print(len(input_features))
      # print(self.inlet_mask)
      # if self.inlet_mask == False:
      if False:

        # Use simple tensor as placeholder for nodes attributes 
        # node_attr = torch.ones(node_pos.shape[0],1)
        # node_attr = node_geom_features
        
        # this way node attributes are the same type of the node coordinate ([x,y,z], a 3D vector)
        node_attr = node_pos
      else:
        # here ":2000" means only fluid nodes are considered
        num_fluid_nodes = len(input_features)
        #print(num_fluid_nodes)
        padding = torch.zeros_like(node_phys_features[:num_fluid_nodes,:])
        # print(f"Padding: {padding.shape}")
        mask = inlet_distance_mask(original_pos_plus_labels, num_nodes=num_fluid_nodes)
        padding[mask] = node_phys_features[:num_fluid_nodes,...][mask]
        # this way node attributes are the same type of the node_phys features (here [P,vx,vy,vz], a scalar and a 3D vector
        node_attr = padding
        
      # Use simple tensor as placeholder for nodes attributes 
      # node_attr = torch.ones(node_pos.shape[0],1)

      # (unsure about this choice at the moment, have to check it later)
      # node_attr = node_geom_features
      
      # here ":2000" means only fluid nodes are stored (here inlet and outlet nodes
      #  are only used to compute "input features" and then discarded)
      data = Data(x = input_features, pos = node_pos[:num_fluid_nodes,...], node_attr = node_attr[:num_fluid_nodes,...], 
                  edge_index = None, y = node_phys_features[:num_fluid_nodes,...], original_pos_plus_labels = original_pos_plus_labels, mask = mask)
      
      torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

  def len(self):        
    return self.data.shape[0]


  def get(self, idx):
    """ Equivalent to __getitem__ in pytorch 
        it retrieves the graph corresponding to the required index
    """ 

    sample = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))

    return sample





class Normalizer(nn.Module):
  """ Class used to accumulate node features' statistics during training
      this function is deprecated here, it does not take into account the 
      geometrical nature of different components (i.e. NOT equivariant)
  """

  def __init__(self, size, max_accumulations = 10 ** 6, std_epsilon = 1e-8):

    super().__init__()

    self._max_accumulations = max_accumulations
    self.register_buffer('_std_epsilon', torch.tensor([std_epsilon], requires_grad=False))
    
    self.register_buffer('_acc_count', torch.zeros(1, dtype=torch.float32, requires_grad=False))
    self.register_buffer('_num_accumulations', torch.zeros(1, dtype=torch.float32, requires_grad=False))
    self.register_buffer('_acc_sum', torch.zeros(size, dtype=torch.float32, requires_grad=False))
    self.register_buffer('_acc_sum_squared', torch.zeros(size, dtype=torch.float32, requires_grad=False))


  def get_acc_sum(self):
    return self._acc_sum 

  def _accumulate(self, batched_data):
    """Function to perform the accumulation of the batch_data statistics."""
    
    dev = batched_data.device
    count = torch.tensor(batched_data.shape[0], dtype=torch.float32, device=dev)
    count = torch.tensor(batched_data.shape[0], dtype=torch.float32)
    data_sum = torch.sum(batched_data, dim = 0).to(dev)
    squared_data_sum = torch.sum(batched_data ** 2, dim=0)

    self._acc_sum = self._acc_sum.add(data_sum)
    self._acc_sum_squared = self._acc_sum_squared.add(squared_data_sum)
    self._acc_count = self._acc_count.add(count)
    self._num_accumulations = self._num_accumulations.add(1.)


  def _mean(self):

    dev = self._acc_count.device
    safe_count = torch.max(self._acc_count, torch.tensor([1.], device = dev))
    return self._acc_sum / safe_count

  def _std_with_epsilon(self):

    dev = self._acc_count.device
    safe_count = torch.max(self._acc_count, torch.tensor([1.], device=dev))
    std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean()**2)
    return torch.maximum(std, self._std_epsilon)

  def inverse(self, normalized_batch_data):
    """Inverse transformation of the normalizer."""

    return normalized_batch_data * self._std_with_epsilon() + self._mean()

  def forward(self, batched_data, accumulate):
    """Normalizes input data and accumulates statistics."""

    if accumulate and self._num_accumulations < self._max_accumulations:
        # stop accumulating after a million updates, to prevent accuracy/memory issues
        self._accumulate(batched_data)
    return (batched_data - self._mean()) / self._std_with_epsilon()