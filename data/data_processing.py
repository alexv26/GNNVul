# ChatGPT Generated
import json
from sklearn.model_selection import train_test_split
import random
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import os
from huggingface_hub import hf_hub_download
from .graph_gen.graph_generator import generate_one_graph as gengraph
from tqdm import tqdm
import pickle

# ChatGPT helped with some of the coding
def subsample_and_split(data, output_dir, target_key="target", safe_ratio=3, upsample_vulnerable=False, downsample_safe=False, downsample_factor=2):
    """
    Function to subsample and split data into training, validation, and test sets.
    
    Parameters:
        data (list): The dataset.
        output_dir (str): Directory to save the output.
        target_key (str): Key for the target variable.
        safe_ratio (int): The ratio of safe to vulnerable entries (for safe downsampling).
        upsample_vulnerable (bool): Whether to upsample vulnerable entries.
        downsample_safe (bool): Whether to downsample safe entries.
    """

    # Make output dir if not exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    # https://www.youtube.com/watch?v=aboZctrHfK8
    indices = list(range(len(data)))
    labels = [item['target'] for item in data]

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, labels, test_size=0.3, stratify=labels, random_state=42
    )
    idx_valid, idx_test, y_valid, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    train_set = [data[i] for i in idx_train]
    valid_set = [data[i] for i in idx_valid]
    test_set = [data[i] for i in idx_test]

    num_vuln_in_train = Counter(y_train)[1]
    num_vuln_in_test = Counter(y_test)[1]
    num_vuln_in_valid = Counter(y_valid)[1]

    print(f"Before split | Train: {len(train_set)}, {(num_vuln_in_train / len(train_set) * 100):.2f}% Vulnerable, Valid: {len(valid_set)}, {(num_vuln_in_test / len(train_set) * 100):.2f}% Vulnerable, Test: {len(test_set)}, {(num_vuln_in_valid / len(train_set) * 100):.2f}% Vulnerable")

    if downsample_safe:
        safe = [item for item in train_set if item[target_key] == 0]
        vuln = [item for item in train_set if item[target_key] == 1]
        random.seed(42)
        safe = random.sample(safe, len(safe)//downsample_factor)
        train_set = vuln + safe
        random.shuffle(train_set)
        print(f"After downsampling: {len(train_set)}")
    
    if upsample_vulnerable:
        X = [[i] for i in range(len(train_set))]  # Just index, required 2D shape
        y = [item[target_key] for item in train_set]

        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)

        # Remap oversampled indices back to dicts
        train_set = [train_set[i[0]] for i in X_resampled]

        count = Counter(y_resampled)
        percent_vulnerable = (count[1] / (count[0] + count[1])) * 100

    # Save to JSON files
    def save(dataset, name):
        with open(f"{output_dir}/{name}.json", 'w') as f:
            json.dump(dataset, f, indent=2)

    save(train_set, "train")
    save(valid_set, "valid")
    save(test_set, "test")

    print(f"Saved: {len(train_set)} train, {len(valid_set)} valid, {len(test_set)} test")

    return train_set, valid_set, test_set

def load_huggingface_datasets():
    # LOAD DATASET

    repo_id = "alexv26/GNNVulDatasets"
    train_name = "train.json"
    test_name = "test.json"
    valid_name = "valid.json"

    # Set your target directory
    target_dir = "data/split_datasets"

    # Create the folder if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Download train
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=train_name,
        local_dir=target_dir,
    )

    # Download test
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=test_name,
        local_dir=target_dir,
    )

    # Download valid
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=valid_name,
        local_dir=target_dir,
    )

def print_split_stats(split_name, split_data):
    total = len(split_data)
    get_label = lambda entry: int(entry["target"])

    vuln = sum(get_label(entry) for entry in split_data)
    nonvuln = total - vuln
    print(f"{split_name} — Total: {total}, Vulnerable: {vuln} ({vuln/total:.2%}), Non-vulnerable: {nonvuln} ({nonvuln/total:.2%})")

def preprocess_graphs(train, test, valid):
    #! NEW: DICT TO SAVE GRAPHS TO SAVE COMPUTER SPACE
    seen_graphs = {}
    complete_dataset = train + test + valid
    for i in tqdm(range(len(complete_dataset)), desc="Preprocessing graphs", unit="graph"):
        idx = complete_dataset[i]["idx"]
        if idx not in seen_graphs:
            G = gengraph(complete_dataset, i, save_graphs=False)  # set save_graphs as desired
            seen_graphs[idx] = G
    return seen_graphs

def save_seengraphs(seen_graphs: dict):
    with open("data/graphs/seen_graphs.pkl", "wb") as f:
        pickle.dump(seen_graphs, f)

def load_seengraphs():
    with open("data/graphs/seen_graphs.pkl", "rb") as f:
        seen_graphs = pickle.load(f)
    return seen_graphs
