import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, RGCNConv, global_mean_pool
from sklearn.model_selection import train_test_split

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.5, model="gcn"):
        '''
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (2 for binary classification)
            dropout: Dropout probability
        '''
        super(GNNModel, self).__init__()
        if model.lower() == "gcn":
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif model.lower() == "rgcn":
            self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations=6)
            self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=6)
        else: # GAT
            self.conv1 = GATConv(input_dim, hidden_dim, heads=8, dropout=dropout)
            hidden_dim = hidden_dim * 8  # GAT concatenates heads by default
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)
        # Output layer
        self.lin = nn.Linear(hidden_dim, 1) #! CHANGED self.lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.model = model.lower()

    def forward(self, data: Data):
        if self.model != "rgcn":
            x, edge_index, batch = data.x, data.edge_index, data.batch

            # First layer
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Second layer
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            
            # Global pooling
            x = global_mean_pool(x, batch)
            
            # Output layer
            x = self.lin(x)
        else: # RGCN
            x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch

            x = self.conv1(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.conv2(x, edge_index, edge_type)
            x = F.relu(x)

            x = global_mean_pool(x, batch)
            x = self.lin(x)
        
        return x.view(-1)  # [batch_size] shaped output #! CHANGED return x