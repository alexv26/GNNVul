import os
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from data.GraphDataset import GraphDataset
from model import GNNModel
from tqdm import tqdm
import json
from data.graph_gen.data_visualization import plot_loss, plot_confusion_matrix, plot_training_history, plot_roc_curve
from sklearn.metrics import classification_report, roc_curve
import argparse
from sklearn.model_selection import train_test_split
import random
from data.w2v.train_word2vec import train_w2v
from gensim.models import Word2Vec
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
W2V_PATH = os.path.join(BASE_DIR, "data/w2v/word2vec_code.model")

# Training loop
def train(model, train_loader, val_loader, optimizer, model_save_path, criterion, device, scheduler=None, roc_implementation = True, losses_file_path="training_losses.json"):
    losses = {'epoch_loss': [], 'batch_loss': []}
    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": []
    }
    best_val_f1 = 0.0

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        batch_losses = []  # List to store batch losses for the current epoch

        # Create a tqdm progress bar for the training batches
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as batch_progress:
            for data in batch_progress:
                data = data.to(device) # So i can use either CPU or GPU depending on machine
                optimizer.zero_grad()  # Zero gradients
                out = model(data)  # Forward pass
                
                # Calculate loss
                loss = criterion(out, data.y)
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights
                
                running_loss += loss.item()
                batch_losses.append(loss.item())

                # Get predictions
                _, predicted = torch.max(out, 1)
                correct += (predicted == data.y).sum().item()
                total += data.y.size(0)
                
                # Update the progress bar with loss and accuracy info
                batch_progress.set_postfix(loss=loss.item(), accuracy=correct/total)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # STORE LOSSES
        losses['epoch_loss'].append(epoch_loss)
        losses['batch_loss'].append(batch_losses)
        # Validation evaluation
        if not roc_implementation:
            val_accuracy, val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
            prec, rec, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average="binary", zero_division=0)
        else:
            # Validation evaluation with ROC analysis
            val_accuracy, val_loss, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, device, roc_implementation)

            # ROC thresholding
            from sklearn.metrics import roc_curve
            import numpy as np

            fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
            plot_roc_curve(val_labels, val_probs, dataset_name=f"Val_Epoch{epoch+1}", save_path="visualizations/roc_val") #! Change save_path
            youden_index = tpr - fpr
            best_thresh = thresholds[np.argmax(youden_index)]

            # Apply adjusted threshold
            adjusted_val_preds = (np.array(val_probs) >= best_thresh).astype(int)

            # Metrics with thresholded predictions
            from sklearn.metrics import precision_recall_fscore_support
            prec, rec, f1, _ = precision_recall_fscore_support(val_labels, adjusted_val_preds, average="binary", zero_division=0)

            # Optionally: print it
            print(f"📈 Epoch {epoch+1}: Val F1 (thresholded @ {best_thresh:.3f}) = {f1:.4f}")

        ##! START NEW
        # Log metrics
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_acc)
        history["val_accuracy"].append(val_accuracy)
        history["val_precision"].append(prec)
        history["val_recall"].append(rec)
        history["val_f1"].append(f1)

        print(f"Validation F1: {f1}")
        # Save best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ New best model saved at epoch {epoch+1} (Val F1: {f1:.4f})")

            if scheduler:
                # Pass validation loss to the scheduler
                scheduler.step(val_loss=val_loss)

    # Save final loss and metrics
    with open(losses_file_path, 'w') as f:
        json.dump(losses, f, indent=2)

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("✅ Training history saved to training_history.json")
    ##! END NEW
    plot_loss(losses)


    with open(losses_file_path, 'w') as f:
        json.dump(losses, f, indent=2)

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Evaluate the model

def evaluate(model, loader, criterion, device, roc_implementation=False):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with tqdm(loader, desc="Testing model accuracy", unit="batch") as batch_progress:
        with torch.no_grad():
            for data in batch_progress:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                running_loss += loss.item()

                probs = torch.softmax(out, dim=1)[:, 1]  # Class 1 probabilities
                _, preds = torch.max(out, dim=1)  # shape: [batch_size]
                labels = data.y.view(-1).long()  # make sure it's 1D and long type
                correct += (preds == labels).sum().item()
                total += data.y.size(0)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

    accuracy = correct / total
    avg_loss = running_loss / len(loader)

    if roc_implementation:
        return accuracy, avg_loss, all_preds, all_labels, all_probs
    return accuracy, avg_loss, all_preds, all_labels



# ChatGPT Generated
def subsample_and_split(data, output_dir, target_key="target", safe_ratio=3, upsample_vulnerable=False, downsample_safe=False):
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
    
    # Separate vulnerable and safe entries
    vulnerable = [entry for entry in data if entry[target_key] == 1]
    safe = [entry for entry in data if entry[target_key] == 0]

    print(f"Original: {len(vulnerable)} vulnerable, {len(safe)} safe")

    # Adjust vulnerable entries (upsample if needed)
    if upsample_vulnerable:
        vulnerable_sample_size = max(len(safe) // safe_ratio, len(vulnerable))  # Ensure at least the number of vulnerable entries
        vulnerable_sampled = vulnerable * (vulnerable_sample_size // len(vulnerable))  # Repeat until size is reached
        # If not an exact multiple, append extra samples
        vulnerable_sampled.extend(random.sample(vulnerable, vulnerable_sample_size % len(vulnerable)))
    else:
        vulnerable_sampled = vulnerable

    # Adjust safe entries (downsample if needed)
    if downsample_safe:
        safe_sample_size = min(len(safe), len(vulnerable_sampled) * safe_ratio)  # Match desired ratio
        safe_sampled = random.sample(safe, safe_sample_size)
    else:
        safe_sampled = safe

    # Separate the data into training, validation, and test sets before combining
    combined_data = vulnerable_sampled + safe_sampled
    random.shuffle(combined_data)  # Shuffle combined data to ensure randomness

    # Stratified split
    y = [entry[target_key] for entry in combined_data]
    train_val, test = train_test_split(combined_data, test_size=0.1, stratify=y, random_state=42)
    y_train_val = [entry[target_key] for entry in train_val]
    train, valid = train_test_split(train_val, test_size=0.1111, stratify=y_train_val, random_state=42)  # 10% of total

    # Save to JSON files
    def save(dataset, name):
        with open(f"{output_dir}/{name}.json", 'w') as f:
            json.dump(dataset, f, indent=2)

    save(train, "train")
    save(valid, "valid")
    save(test, "test")

    print(f"Saved: {len(train)} train, {len(valid)} valid, {len(test)} test")

    return train, valid, test


def print_split_stats(split_name, split_data):
    total = len(split_data)
    get_label = lambda entry: int(entry["target"])

    vuln = sum(get_label(entry) for entry in split_data)
    nonvuln = total - vuln
    print(f"{split_name} — Total: {total}, Vulnerable: {vuln} ({vuln/total:.2%}), Non-vulnerable: {nonvuln} ({nonvuln/total:.2%})")

def load_configs():
    configs_file_path = "configs.json"
    with open(configs_file_path, 'r') as file:
        configs: dict = json.load(file)
    return (configs["input_dim"], configs["hidden_dim"], configs["output_dim"], 
            configs["dropout"], configs["l2_reg"], configs["batch_size"], configs["learning_rate"], 
            configs["epochs"], configs["load_existing_model"], configs["save_graphs"],
            configs["archutecture_type"], configs["roc_implementation"], configs["model_save_path"], 
            configs["visualizations_save_path"], configs["losses_file_path"])


if __name__ == "__main__":
    #* STEP 1: LOAD CONFIGS
    input_dim, hidden_dim, output_dim, dropout, l2_reg, batch_size, learning_rate, epochs, load_existing_model, save_graphs, architecture_type, roc_implementation, model_save_path, visualizations_save_path, losses_file_path = load_configs()
    '''
        load_existing_model: boolean value, decides whether we load saved .pth model or train a new one
        save_graphs: boolean value, decide if we save graphs to computer (for faster runtime), or do not save (for better space efficiency)
        model_save_path: path to where we save our model at the end of training
        visualization_save_path: path to where we store graphs
        losses_file_path: path to where we store loss data
    '''

    #* ARGUMENT PARSING
    parser = argparse.ArgumentParser(description="Train and evaluate GNN model")
    parser.add_argument("--in-dataset", type=str, default="data/databases/complete_dataset.json", help="Path to the complete dataset (default: complete_dataset.json)")
    parser.add_argument("--train-dataset", type=str, default="data/split_datasets/train.json", help="Path to the training dataset split (default: train.json)")
    parser.add_argument("--test-dataset", type=str, default="data/split_datasets/test.json", help="Name of the testing dataset split (default: test.json)")
    parser.add_argument("--valid-dataset", type=str, default="data/split_datasets/valid.json", help="Name of the validation dataset split (default: test.json)")    
    parser.add_argument("--upsample-vulnerable", type=str, default=False, help="Upsample vulnerable entries (default: False)")
    parser.add_argument("--downsample-safe", type=str, default=False, help="Downsample safe entries (default: False)")
    parser.add_argument("--do-data-splitting", type=bool, default=False, help="Does data need to be split or is it already split? (default: False)")
    parser.add_argument("--do-lr-scheduling", type=bool, default=True, help="Adjust learning rate after validation loss plateaus (default: True)")

    args = parser.parse_args()

    # LOAD or CREATE w2v
    if not os.path.exists(W2V_PATH):
        print("Building new w2v model")
        train_w2v(args.in_dataset)
        w2v = Word2Vec.load(W2V_PATH)
    else:
        print("Word2Vec exists. Loading pretrained model...")
        w2v = Word2Vec.load(W2V_PATH)

    '''
    The code below basically handles whether you need to do pre-splitting of data or not. If we do, we a pre-split of the data
    and try to keep it balanced between datasets of vuln/nonvuln.
    '''
    if args.do_data_splitting is False:
        train_dataset = GraphDataset(args.train_dataset, w2v, save_graphs)
        val_dataset = GraphDataset(args.valid_dataset, w2v, save_graphs)
        test_dataset = GraphDataset(args.test_dataset, w2v, save_graphs)
    
    else:    
        print("🚧 Splitting dataset into train/val/test...")
        with open(args.in_dataset, 'r') as f:
            full_data = json.load(f)
        train_data, val_data, test_data = subsample_and_split(full_data, "data/split_datasets", upsample_vulnerable=args.upsample_vulnerable, downsample_safe=args.downsample_safe)

        train_dataset = GraphDataset("data/split_datasets/train.json", w2v, save_graphs)
        val_dataset = GraphDataset("data/split_datasets/valid.json", w2v, save_graphs)
        test_dataset = GraphDataset("data/split_datasets/test.json", w2v, save_graphs)
    
    print_split_stats("Train", train_dataset.get_data())
    print_split_stats("Validation", val_dataset.get_data())
    print_split_stats("Test", test_dataset.get_data())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    if architecture_type == "rgcn": input_dim = train_dataset[0].x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, model=architecture_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    if args.do_lr_scheduling:
        print("The model will adjust learning rate when validation loss plateau's.")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    else:
        scheduler = None

    #* STEP 2: TRAIN MODEL
    if not load_existing_model:
        # SETTING WEIGHTS TO FIX VULN/NONVULN INBALANCE #! STREAMLINE LATER, CHANGED!!!!
        vuln, nonvuln = train_dataset.get_vuln_nonvuln_split()
        print(vuln, nonvuln)
        total = vuln + nonvuln

        # Weight inversely proportional to class frequency
        weight = torch.tensor([
            total / nonvuln,   # weight for class 0 (safe)
            total / vuln       # weight for class 1 (vulnerable)
        ], dtype=torch.float).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)

        train(model, train_loader, val_loader, optimizer, model_save_path=model_save_path, criterion=criterion, device=device, roc_implementation=roc_implementation, losses_file_path="training_losses_GAT.json")    
    else:
        print("Loading existing model")
        model.load_state_dict(torch.load(model_save_path))

    #* STEP 3: TEST MODEL
    if not roc_implementation:
        test_accuracy, _, all_preds_test, all_labels_test = evaluate(model, test_loader, criterion, device)
        # Test results
        plot_confusion_matrix(all_labels_test, all_preds_test, dataset_name="Test", save_path=visualizations_save_path)
        print(classification_report(
            all_labels_test,
            all_preds_test,
            target_names=["Safe", "Vulnerable"],
            zero_division=0  # suppress warnings for undefined metrics
        ))
    else:
        test_accuracy, _, all_preds_test, all_labels_test, all_probs_test = evaluate(
        model, test_loader, criterion, device, roc_implementation
        )

        fpr, tpr, thresholds = roc_curve(all_labels_test, all_probs_test)
        plot_roc_curve(all_labels_test, all_probs_test, dataset_name="Test", save_path=visualizations_save_path)
        youden_index = tpr - fpr
        best_thresh = thresholds[np.argmax(youden_index)]
        print(f"📈 Best threshold (Youden's J): {best_thresh:.4f}")

        # Apply new threshold
        adjusted_preds = (np.array(all_probs_test) >= best_thresh).astype(int)

        # Save confusion matrix with adjusted threshold
        plot_confusion_matrix(all_labels_test, adjusted_preds, dataset_name="Test_ThresholdAdjusted", save_path=visualizations_save_path)

        # Report
        print("\n📊 Adjusted Threshold Performance:")
        print(classification_report(
            all_labels_test,
            adjusted_preds,
            target_names=["Safe", "Vulnerable"],
            zero_division=0
        ))

    #* STEP 4: SAVE PREDICTIONS AND LABELS TO JSON
    results = {
        "test": {
            "predictions": [int(x) for x in all_preds_test],
            "labels": [int(x) for x in all_labels_test],
            "accuracy": float(test_accuracy)
        }
    }

    with open("predictions_and_labels.json", "w") as f:
        json.dump(results, f, indent=2)


    print("Saved predictions and labels to predictions_and_labels.json")
    plot_training_history(history_file_path="training_history.json", save_dir=visualizations_save_path)


# CODE TO RUN ON DIVERSEVUL: python gnn_pipeline.py --train-dataset "data/databases/diversevul_file.json" --do-data-splitting True
# CODE TO RUN ON DEVIGN: python gnn_pipeline.py --train-dataset "data/databases/devign.json" --do-data-splitting True
# ON COMPLETE DATASET: python gnn_pipeline.py --train-dataset "data/databases/complete_dataset.json" --do-data-splitting True