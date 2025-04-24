import json
from huggingface_hub import hf_hub_download
import os

def load_configs():
    configs_file_path = "configs.json"
    with open(configs_file_path, 'r') as file:
        configs: dict = json.load(file)
    return (configs["input_dim"], configs["hidden_dim"], configs["output_dim"], 
            configs["dropout"], configs["l2_reg"], configs["batch_size"], configs["learning_rate"], 
            configs["epochs"], configs["downsample_factor"], configs["load_existing_model"], 
            configs["save_graphs"], configs["archutecture_type"], configs["roc_implementation"], 
            configs["model_save_path"], configs["visualizations_save_path"], configs["losses_file_path"])

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