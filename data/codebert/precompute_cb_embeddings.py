
import torch
from transformers import RobertaTokenizer, RobertaModel
import os
import json
from tqdm import tqdm
from data.graph_gen.graph_generator import generate_one_graph

def precompute_codebert_embeddings(data_path: str, output_path: str = "data/codebert/codebert_node_embeddings.pt", save_graphs: bool = True):
    """
    Precompute CodeBERT embeddings for all unique nodes in a dataset and save to a .pt file.
    Args:
        dataset_path (str): Path to the dataset JSON file.
        output_path (str): Path to save the embeddings.
        save_graphs (bool): Whether to save the generated graphs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Precomputing embeddings on the device: {device}")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device).eval()

    with open(data_path, "r") as f:
        data = json.load(f)

    seen_nodes = set()
    embedding_dict = {}

    for idx in tqdm(range(len(data)), desc="Precomputing CodeBERT embeddings"):
        G = generate_one_graph(data, idx, save_graphs)

        for node in G.nodes():
            node_str = str(node)
            if node_str in seen_nodes:
                continue
            seen_nodes.add(node_str)

            tokens = tokenizer(node_str, return_tensors="pt", truncation=True, padding="max_length", max_length=10).to(device)
            with torch.no_grad():
                outputs = model(**tokens)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
            embedding_dict[node_str] = embedding

    torch.save(embedding_dict, output_path)
    print(f"✅ Saved {len(embedding_dict)} node embeddings to {output_path}")
