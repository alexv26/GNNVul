import json

def load_configs():
    configs_file_path = "configs.json"
    with open(configs_file_path, 'r') as file:
        configs: dict = json.load(file)
    return (configs["input_dim"], configs["hidden_dim"], configs["output_dim"], 
            configs["dropout"], configs["l2_reg"], configs["batch_size"], configs["learning_rate"], 
            configs["epochs"], configs["downsample_factor"], configs["load_existing_model"], 
            configs["save_graphs"], configs["archutecture_type"], configs["roc_implementation"], 
            configs["model_save_path"], configs["visualizations_save_path"], configs["losses_file_path"])