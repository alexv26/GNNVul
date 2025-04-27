import json
from huggingface_hub import hf_hub_download
import os

def load_configs():
    configs_file_path = "configs.json"
    with open(configs_file_path, 'r') as file:
        configs: dict = json.load(file)
    return (configs["input_dim"], configs["hidden_dim"], configs["output_dim"], 
            configs["dropout"], configs["l2_reg"], configs["batch_size"], configs["learning_rate"], 
            configs["epochs"], configs["downsample_factor"], configs["patience"])

def load_w2v_from_huggingface():
    repo_id = "alexv26/pretrained_w2v"
    w2v_file1 = "word2vec_code.model"
    w2v_file2 = "word2vec_code.model.syn1neg.npy"
    w2v_file3 = "word2vec_code.model.wv.vectors.npy"

    target_dir = "data/w2v"

    os.makedirs(target_dir, exist_ok=True)

    # Download file 1
    hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=w2v_file1,
        local_dir=target_dir,
    )

    # Download file 2
    hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=w2v_file2,
        local_dir=target_dir,
    )

    # Download file 3
    hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=w2v_file3,
        local_dir=target_dir,
    )


def early_stopping(best_f1, val_f1, patience=5, epochs_without_improvement=0):
    if val_f1 > best_f1:
        best_f1 = val_f1
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        print(f"Epochs without improvement: {epochs_without_improvement}")
        if epochs_without_improvement >= patience:
            print("Early stopping...")
            return True
    return False, epochs_without_improvement
