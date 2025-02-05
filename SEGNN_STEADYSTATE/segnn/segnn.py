import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from e3nn.nn import BatchNorm
import numpy as np
from e3nn.o3 import Irreps#, sort, simplify

from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate
from .instance_norm import InstanceNorm
from torch_geometric.data import Data

class SEGNN(nn.Module):
    """Steerable E(3) equivariant message passing network"""

    def __init__(
        self,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        num_layers,
        input_irreps = None,
        norm=None,
        pool="avg",
        task="graph",
        additional_message_irreps=None,
    ):
        super().__init__()
        self.task = task
        # Create network, embedding first
        # self.embedding_layer_1 = O3TensorProductSwishGate(
        #     input_irreps, hidden_irreps, node_attr_irreps
        # )
        # self.embedding_layer_2 = O3TensorProduct(
        #     hidden_irreps, hidden_irreps, node_attr_irreps
        # )

        # maps the input irreps to hidden irreps
        self.embedding_layer = O3TensorProduct(
            input_irreps, hidden_irreps, node_attr_irreps
        )




        # Message passing layers.
        layers = []
        for i in range(num_layers):
            layers.append(
                SEGNNLayer(
                    hidden_irreps,
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Prepare for output irreps, since the attrs will disappear after pooling
        # the if condition works only for
        if task == "graph":
            pooled_irreps = (
                (output_irreps * hidden_irreps.num_irreps).simplify().sort().irreps
            )
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, pooled_irreps, node_attr_irreps
            )
            self.post_pool1 = O3TensorProductSwishGate(pooled_irreps, pooled_irreps)
            self.post_pool2 = O3TensorProduct(pooled_irreps, output_irreps)
            self.init_pooler(pool)
        elif task == "node":
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, output_irreps, node_attr_irreps
            )

    def init_pooler(self, pool):
        """Initialise pooling mechanism"""
        # this operation computes the average or sum of node features for each graph in the batch.
        # it aggregates node-level features into a single graph-level representation when the task is graph-related
        if pool == "avg":
            self.pooler = global_mean_pool
        elif pool == "sum":
            self.pooler = global_add_pool

    def catch_isolated_nodes(self, graph):
        """Isolated nodes should also obtain attributes"""
        if (
            graph.contains_isolated_nodes()
            and graph.edge_index.max().item() + 1 != graph.num_nodes
        ):
            # adding attributes to isolated nodes, in the shape of nde_attr.shape[-1]
            nr_add_attr = graph.num_nodes - (graph.edge_index.max().item() + 1)
            add_attr = graph.node_attr.new_tensor(
                np.zeros((nr_add_attr, graph.node_attr.shape[-1])) #node_attr -> graph.node_attr correction (?)
            )
            graph.node_attr = torch.cat((graph.node_attr, add_attr), -2)
        # Trivial irrep value should always be 1 (is automatically so for connected nodes, but isolated nodes are now 0)
        graph.node_attr[:, 0] = 1.0
        
    def forward(self, x, edge_index, **kwargs):
        """SEGNN forward pass"""
        pos = kwargs.get('pos', None)
        edge_attr = kwargs.get('edge_attr', None)
        node_attr = kwargs.get('node_attr', None)
        batch = kwargs.get('batch', None)
        y = kwargs.get('y', None)
    
        graph = Data(x=x, edge_index=edge_index, pos=pos, edge_attr=edge_attr, 
                     node_attr=node_attr, batch=batch, y=y)
    
        self.catch_isolated_nodes(graph)
    
        x = self.embedding_layer(graph.x, graph.node_attr)
    
        for layer in self.layers:
            x = layer(
                x, graph.edge_index, graph.edge_attr, graph.node_attr, graph.batch, kwargs.get('additional_message_features', None)
            )
    
        x = self.pre_pool1(x, graph.node_attr)
        x = self.pre_pool2(x, graph.node_attr)
    
        if self.task == "graph":
            x = self.pooler(x, graph.batch)
            x = self.post_pool1(x)
            x = self.post_pool2(x)
    
        return x


class SEGNNLayer(MessagePassing):
    """E(3) equivariant message passing layer."""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        norm=None,
        additional_message_irreps=None,
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps

        message_input_irreps = (2 * input_irreps + additional_message_irreps).simplify()
        update_input_irreps = (input_irreps + hidden_irreps).simplify()

        self.message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps
        )
        self.update_layer_1 = O3TensorProductSwishGate(
            update_input_irreps, hidden_irreps, node_attr_irreps
        )
        self.update_layer_2 = O3TensorProduct(
            hidden_irreps, hidden_irreps, node_attr_irreps
        )

        self.setup_normalisation(norm)

    def setup_normalisation(self, norm):
        """Set up normalisation, either batch or instance norm"""
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.hidden_irreps)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(self.hidden_irreps)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_attr,
        batch,
        additional_message_features=None,
    ):
        """Propagate messages along edges"""
        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
        )
        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i, x_j, edge_attr, additional_message_features):
        """Create messages"""
        if additional_message_features is None:
            input = torch.cat((x_i, x_j), dim=-1)
        else:
            input = torch.cat((x_i, x_j, additional_message_features), dim=-1)
        
        # here it computes the first nonlinear O3 tensor product from (fi,fj,aij)
        message = self.message_layer_1(input, edge_attr)
        # here it computes the second nonlinear O3 tensor product (mij,aij)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        """Update note features"""
        # here it concatenates (fi, sum_j m_ij)
        input = torch.cat((x, message), dim=-1)

        # first nonlinear node update layer from (fi,mi,ai)
        update = self.update_layer_1(input, node_attr)
        # second nonlinear node update layer from (f'i,ai)
        update = self.update_layer_2(update, node_attr)
        x += update  # Residual connection
        return x