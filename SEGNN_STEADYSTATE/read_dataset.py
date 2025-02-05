import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os

from Utility_functions import print_3D_graph, Graph_dataset_with_equiv_features

import params

print('DATADIR', params.DATADIR)
print('NSIM', params.NSIM)
print('BATCH_SIZE', params.BATCH_SIZE)
print('torch.__version__', torch.__version__)
print('torch_geometric.__version__', torch_geometric.__version__)
print('torch.cuda.is_available()', torch.cuda.is_available())

# first time ever this function is called, it processes the raw data stored in the ".../raw" folder
# and stores processed data in ".../processed". After first processing, the process function is 
# no longer executed as long as the "processed" folder or the "process" function in Graph_datasetV3 remain unchanged.

dataset = Graph_dataset_with_equiv_features(root = params.DATADIR)

dataloader = DataLoader(dataset, batch_size=params.BATCH_SIZE, shuffle=False)

# taking the first batch
p = next(iter(dataloader)) 

# Databatch: I have #batch_size number of objects. Every object is of type "Data",
# that stores the graph information in its different attributes
# here a graph is a single geometry with its steady state
print(f"p\n{p}") 
print()
# First "Data" object in the batch
print(f"p[0], len(p[0])\n{p[0], len(p[0])}") 
print()
print(f"type(p[0])\n{type(p[0])}")
print()
# .batch attribute (the shape is [#batch_size*nodes_per_graph]) is used to distinguish every graph in the batch
print(f"p[0], p.batch\n{p[0], p.batch}") 