from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import os
from gensim.models import Word2Vec
from .json_functions import JsonFuncs as js
import re
from .graph_gen.graph_generator import generate_one_graph as gengraph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "../../data/final_data")
W2V_PATH = os.path.join(BASE_DIR, "../w2v/word2vec_code.model")

def split_name_into_subtokens(name):
    """
    Splits identifiers like 'session_data' or 'getHTTPResponse' into subtokens.
    """
    # Split snake_case
    parts = name.lower().split('_')
    subtokens = []
    for part in parts:
        # Split camelCase and PascalCase
        camel_split = re.findall(r'[a-z]+|[A-Z][a-z]*|[0-9]+', part)
        subtokens.extend([s.lower() for s in camel_split if s])
    return subtokens


class GraphDataset(Dataset):
    def __init__(self, database_name, save_graphs=True, w2v=W2V_PATH):
        self.database_name = database_name.lower()
        self.data = js.load_json_array(os.path.join(DB_DIR, f"all_{database_name.lower()}_data_new.json"))
        self.save_graphs = save_graphs
        if not os.path.exists(w2v):
            print("Building new w2v model")
            self._build_word2vec_model()
        else:
            print("Word2Vec exists. Loading pretrained model...")
            self.w2v = Word2Vec.load(W2V_PATH)
            self.embedding_dim = self.w2v.vector_size

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

    def _build_word2vec_model(self):
        # Collect subtokenized names from all graphs
        all_subtoken_lists = []
        for i in range(len(self.data)):
            G = gengraph(self.data, i, save_graphs=self.save_graphs)
            for node in G.nodes:
                subtokens = split_name_into_subtokens(str(node))
                if subtokens:
                    all_subtoken_lists.append(subtokens)

        # Train Word2Vec
        self.w2v = Word2Vec(sentences=all_subtoken_lists, vector_size=32, window=5, min_count=1, sg=1)
        self.embedding_dim = self.w2v.vector_size
    
    def __getitem__(self, idx):

        #* STEP 1: CREATE GRAPH
        G = gengraph(self.data, idx, save_graphs=self.save_graphs) # save_graphs helps decide if we save the graphs to computer or not

        #* STEP TWO: CREATE GRAPH FEATURES
        # a: node feature matrix
        node_types = {'FunctionCall': 0, 'Variable': 1, 'ControlStructure_if': 2, 'ControlStructure_while': 3, 'ControlStructure_switch': 4, 'ControlStructure_for': 5, 'FunctionDefinition': 6}
        
        node_feature_matrix = torch.zeros((G.number_of_nodes(), len(node_types) + self.embedding_dim))
        node_to_idx = {}
        for index, (node, attrs) in enumerate(G.nodes(data=True)):
            node_to_idx[node] = index

            # One-hot type
            node_type = attrs.get('type', 'unknown')
            if node_type in node_types:
                node_feature_matrix[index, node_types[node_type]] = 1.0

            # Word2Vec embedding of node name subtokens
            subtokens = split_name_into_subtokens(str(node))
            embeddings = []
            for token in subtokens:
                if token in self.w2v.wv:
                    embeddings.append(torch.tensor(self.w2v.wv[token]))
            if embeddings:
                mean_embedding = torch.mean(torch.stack(embeddings), dim=0)
                node_feature_matrix[index, len(node_types):] = mean_embedding

        
        # b: edge index and edge type
        edge_type_map = {
            "declares": 0,
            "calls": 1,
            "contains": 2,
            "used_in_condition": 3,
            "used_as_parameter": 4,
            "used_in_body": 5
        }

        '''edge_index = []
        for u, v in G.edges():
            if u in node_to_idx and v in node_to_idx:
                edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()'''
        edge_index = []
        edge_attr = []

        for u, v, attrs in G.edges(data=True):
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_type_str = attrs.get('type', 'unknown')
            edge_attr.append(edge_type_map.get(edge_type_str, -1))  # handle unknowns safely
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)



        score = float(self.data[idx]["cvss_score"])
        if score == 0.0: label = torch.tensor(0.0, dtype=torch.float) #! CHANGED FROM long TO float
        else: label = torch.tensor(1.0, dtype=torch.float) #! CHANGED FROM long TO float

        data = Data(x=node_feature_matrix, edge_index=edge_index, edge_attr=edge_attr, y=label)
        return data
