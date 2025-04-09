from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import torch
import os
import numpy
from .json_functions import JsonFuncs as js
from .graph_gen.graph_generator import generate_one_graph as gengraph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "../../data/final_data")

class GraphDataset(Dataset):
    def __init__(self, database_name, save_graphs=True):
        self.database_name = database_name.lower()
        self.data = js.load_json_array(os.path.join(DB_DIR, f"all_{database_name.lower()}_data_new.json"))
        self.save_graphs = save_graphs

    def __len__(self):
        return len(self.data)
    
    def get_vuln_nonvuln_split(self):
        vuln = 0
        nonvuln = 0
        for i in range(self.__len__()):
            if float(self.data[i]["cvss_score"]) > 0:
                vuln += 1
            else:
                nonvuln += 1

        return vuln, nonvuln

    
    def __getitem__(self, idx):

        #* STEP 1: CREATE GRAPH
        G = gengraph(self.data, idx, self.save_graphs) # save_graphs helps decide if we save the graphs to computer or not

        #* STEP TWO: CREATE GRAPH FEATURES
        # a: node feature matrix
        node_types = {'FunctionCall': 0, 'Variable': 1, 'ControlStructure_if': 2, 'ControlStructure_while': 3, 'ControlStructure_switch': 4, 'ControlStructure_for': 5, 'FunctionDefinition': 6}
        
        node_feature_matrix = torch.zeros((G.number_of_nodes(), len(node_types)), dtype=torch.float)
        node_to_idx = {}
        for index, (node, attrs) in enumerate(G.nodes(data=True)):
            node_to_idx[node] = index
            node_type = attrs.get('type', 'unknown')
            node_feature_matrix[index, node_types[node_type]] = 1.0
        
        # b: edge index
        '''edge_index = []
        for u, v in G.edges():
            # Skip if node indices are not found
            if u not in G or v not in G:
                continue
            edge_index.append([G[u], G[v]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()'''
        edge_index = []
        for u, v in G.edges():
            if u in node_to_idx and v in node_to_idx:
                edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()


        score = float(self.data[idx]["cvss_score"])
        if score == 0.0: label = torch.tensor(0.0, dtype=torch.float) #! CHANGED FROM long TO float
        else: label = torch.tensor(1.0, dtype=torch.float) #! CHANGED FROM long TO float

        data = Data(x=node_feature_matrix, edge_index=edge_index, y=label)
        return data
