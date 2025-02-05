import torch
import argparse
import os
import numpy as np
from e3nn.o3 import Irreps, spherical_harmonics
from segnn.balanced_irreps import BalancedIrreps, WeightBalancedIrreps
# from segnn.segnn import SEGNN
print(os.getcwd())

filename = 'probesrandompointsfinal.npy'
filename2 = 'probesfinal.npz'

path = os.path.join(os.getcwd(),'..', 'Some_probes_files', filename)
path2 = os.path.join(os.getcwd(),'..', 'Some_probes_files', filename2)

print(path)
print(path2)

pcloud = np.load(path).T
pcloud = torch.tensor(pcloud)

features = np.load(path2)
feats = []
for j,i in enumerate(features.files):
    
    feats.append(features[i])

feat = torch.tensor(np.array(feats)) # first doing list->array and then array->tensor is time saving
feat = feat.reshape(-1,10,2000) # the shape is [#frames, #phys_quantities, #nodes]
feat = torch.einsum('ijk->ikj', feat) # now physical quantities become last_dim

print(feat.shape)

fields = feat[-1] # "steady state" frame
vel = fields[:,1:4]    
# irreps_out = Irreps("2x0e+3x1o+1x3e+3x0e")
# prova = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]
# print(irreps_out,prova)


# prova2 = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]
# print(prova2)

# prova3 = irreps_out.slices()
# print(prova3)
def Augment(irreps_in, fields, graph):

    angles = e3nn.o3.rand_angles()
    node_attr = torch.norm(fields[:,1:4].float(),dim=-1,keepdim=True).repeat(1,3)
    node_attr = spherical_harmonics([*range(edge_lmax+1)], node_attr, False, normalization='component')
    
    pcloudrot = torch.einsum("ij,zj->zi", irreps_in.D_from_angles(*angles), pcloud.float())
    edge_attr_rot = (torch.index_select(pcloudrot, 0, edge_index[1]) - torch.index_select(pcloudrot, 0, edge_index[0])).float()
    fieldsrot = torch.einsum("ij,zj->zi", irreps_in.D_from_angles(*angles), graph.y[:,1:4].float())

    pcloud_rot_sh = spherical_harmonics([*range(edge_lmax+1)], pcloudrot, False, normalization='component')
    edge_attr_rot_sh = spherical_harmonics([*range(edge_lmax+1)], edge_attr_rot, False, normalization='component')
    node_attr_rot_sh = pcloud_rot_sh.detach().clone()
    graph_rot = Batch(x=pcloudrot, pos = pcloudrot, node_attr = node_attr_rot_sh, edge_attr = edge_attr_rot_sh, edge_index = edge_index, batch = batch, y = fieldsrot[:,:4].float())
    rotated_truth = torch.cat((fields[:,0].float().unsqueeze(-1),fieldsrot[:,:4].float()),dim=-1)

    return graph_rot, rotated_truth
import importlib
import segnn.segnn
from segnn import o3_building_blocks
# # Reload the module
importlib.reload(segnn.segnn)
# importlib.reload(o3_building_blocks)

edge_lmax = 3 # e.g. lmax=1 --> 1x0e+1x1o

model = segnn.segnn.SEGNN(input_irreps=Irreps('1o'), # x: node features [x,y,z]
                    hidden_irreps=BalancedIrreps(lmax=1, vec_dim=64),
                    output_irreps=Irreps('0e+1o'), # y: features to predict [P,vx,vy,vz]
                    edge_attr_irreps=Irreps.spherical_harmonics(lmax=edge_lmax),
                    node_attr_irreps=Irreps.spherical_harmonics(lmax=edge_lmax),
                    task = 'node',
                    num_layers=1,
                    additional_message_irreps=None # Irreps('1x0e')
            )

# for some reason, I cannot choose node_attr = None if I don't have at least a scalar (0e) in my input irreps;
# at least one 0e has to come out from the tensor product
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(params)
print()
import torch.nn as nn
import torch.optim as optim
import e3nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph
from tqdm import tqdm


training_loss = []
epochs = 150
augment= False
irreps_in = e3nn.o3.Irreps("1o")

loss_func = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr = 3e-4)

batch = torch.ones_like(pcloud)[:,0]
edge_index = knn_graph(pcloud,k=5)
node_attr = torch.norm(fields[:,1:4].float(),dim=-1,keepdim=True).repeat(1,3)
edge_attr = (torch.index_select(pcloud, 0, edge_index[1]) - torch.index_select(pcloud, 0, edge_index[0])).float()

if edge_lmax >= 1:
    node_attr_sh = spherical_harmonics([*range(edge_lmax+1)], node_attr, False, normalization='component')
    edge_attr_sh = spherical_harmonics([*range(edge_lmax+1)], edge_attr, False, normalization='component')
    pcloud_sh = spherical_harmonics([*range(edge_lmax+1)], pcloud, False, normalization='component')

node_attr_sh = pcloud_sh.float().detach().clone()
graph = Batch(x=pcloud.float(), pos = pcloud.float(), node_attr = node_attr_sh, edge_attr = edge_attr_sh, edge_index = edge_index, batch = batch, y = fields[:,:4].float())

for i in (range(epochs)):

    if augment:
        graph_rot, rotated_truth = Augment(irreps_in, fields, graph)
        out = model(graph_rot)
        loss = loss_func(out, rotated_truth)    
    else:
        print(graph.x[:3,:])
        out = model(graph)
        loss = loss_func(out, fields[:,:4].float()) 
    
    loss.backward()
    opt.step() 
    opt.zero_grad()
    training_loss.append(loss.item())
print(training_loss)
import matplotlib.pyplot as plt

plt.plot(range(len(training_loss)),np.array(training_loss))
# original
# nodattr = torch.norm(fields[:,1:4].float(),dim=-1,keepdim=True).repeat(1,3)
# nodattr = spherical_harmonics([*range(edge_lmax+1)], nodattr, False, normalization='component')

batch = torch.ones_like(pcloud)[:,0]
edge_index = knn_graph(pcloud,k=5)
node_attr = torch.norm(fields[:,1:4].float(),dim=-1,keepdim=True).repeat(1,3)
edge_attr = (torch.index_select(pcloud, 0, edge_index[1]) - torch.index_select(pcloud, 0, edge_index[0])).float()

if edge_lmax >= 1:
    node_attr_sh = spherical_harmonics([*range(edge_lmax+1)], node_attr, False, normalization='component')
    edge_attr_sh = spherical_harmonics([*range(edge_lmax+1)], edge_attr, False, normalization='component')
    pcloud_sh = spherical_harmonics([*range(edge_lmax+1)], pcloud, False, normalization='component')

node_attr_sh = pcloud_sh.float().detach().clone()


graph = Batch(x=pcloud.float(), pos = pcloud.float(), node_attr = node_attr_sh, edge_attr = edge_attr_sh, edge_index = edge_index, batch = batch, y = fields[:,:4].float())

# rotate
angles = e3nn.o3.rand_angles()
irreps_in = e3nn.o3.Irreps("1o")

pcloudrot = torch.einsum("ij,zj->zi", irreps_in.D_from_angles(*angles), pcloud.float())
fieldsrot = torch.einsum("ij,zj->zi", irreps_in.D_from_angles(*angles), graph.y[:,1:4].float())

node_attr_rot = torch.norm(fields[:,1:4].float(),dim=-1,keepdim=True).repeat(1,3)
edge_attr_rot = (torch.index_select(pcloudrot, 0, edge_index[1]) - torch.index_select(pcloudrot, 0, edge_index[0])).float()

node_attr_rot_sh = spherical_harmonics([*range(edge_lmax+1)], node_attr_rot, False, normalization='component')
edge_attr_rot_sh = spherical_harmonics([*range(edge_lmax+1)], edge_attr_rot, False, normalization='component')
pcloudrot_sh = spherical_harmonics([*range(edge_lmax+1)], pcloudrot, False, normalization='component')

node_attr_rot_sh = pcloudrot_sh.detach().clone()

# pcloud_sh_rot = torch.einsum("ij,zj->zi", Irreps.spherical_harmonics(lmax=1).D_from_angles(*angles), pcloud_sh.float())
# node_attr_rot_sh = pcloud_sh_rot.detach().clone()

graph_rot = Batch(x=pcloudrot.float(), pos = pcloud.float(), node_attr = node_attr_rot_sh, edge_attr = edge_attr_rot_sh, edge_index = edge_index, batch = batch, y = fieldsrot[:,:4].float())
print(pcloud[:3,:])
print(pcloudrot[:3,:])


print(node_attr[:3,:])
print(node_attr_sh[:3,:])
print(node_attr_rot_sh[:3,:])
with torch.no_grad():
    model.eval()
    # out = model(graph)
    rot_out = model(graph_rot)
    out = model(graph)

    rotloss = nn.MSELoss()(rot_out[:,1:],fieldsrot)
    newloss = nn.MSELoss()(out[:,1:],graph.y[:,1:])
    equiv_loss = nn.MSELoss()(out[:,1:],rot_out[:,1:])
    rotated_out = torch.einsum("ij,zj->zi", irreps_in.D_from_angles(*angles), out[:,1:])
    equiv_loss2 = nn.MSELoss()(rotated_out[:,:],rot_out[:,1:])
    

print('losses', rotloss, newloss, equiv_loss, equiv_loss2)
print('original\n',graph.y[:4,1:])
print('\n')
print('rotated\n',fieldsrot[:4,:])
print('\n')
print('prediction on original\n',out[:4,1:])
print('\n')
print('prediction on rotated\n',rot_out[:4,1:])
from Utility_functions import print_3D_graph
# print_3D_graph(pcloud,edge_index, fields[:,1])
i = 3
print(out[:4,1:])
print(rot_out[:4,1:])
print(fieldsrot[:4,:])
print_3D_graph(pcloud,edge_index, fields[:,i], rescale = True)
print_3D_graph(pcloud,edge_index, out[:,i], rescale = True)
print_3D_graph(pcloudrot,edge_index, fieldsrot[:,i-1], rescale = True)
print_3D_graph(pcloudrot,edge_index, rot_out[:,i], rescale = True)
input_irreps = Irreps("1o+1e") # xyz coords
node_attr_irreps = Irreps("1o+5x0e") # xyz coords
edge_attr_irreps = Irreps("1o") # relative xyz coords
output_irreps = Irreps("0e+1o") # pressure + velocity components
additional_message_irreps = None




# hidden features represents the max degree of hidden rep
class Args:
    def __init__(self):
        self.subspace_type = "weightbalanced"
        self.hidden_features = 64
        self.lmax_h = 1


args = Args()

# Create hidden irreps

print(input_irreps,hidden_irreps,node_attr_irreps)
if args.subspace_type == "weightbalanced":
    print(args.hidden_features)
    hidden_irreps = WeightBalancedIrreps(
        Irreps("{}x0e".format(args.hidden_features)), node_attr_irreps, sh=True, lmax=args.lmax_h)
    print('hidden_irreps', hidden_irreps)
elif args.subspace_type == "balanced":
    hidden_irreps = BalancedIrreps(args.lmax_h, args.hidden_features, True)
else:
    raise Exception("Subspace type not found")


from e3nn.o3 import FullyConnectedTensorProduct,FullTensorProduct


tensor_prod = FullyConnectedTensorProduct(input_irreps, hidden_irreps, node_attr_irreps)
print(tensor_prod)
tensor_prod.visualize()