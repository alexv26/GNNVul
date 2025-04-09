import os
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from data.GraphDataset import GraphDataset
from model import GNNModel
from tqdm import tqdm
import json
from data.graph_gen.data_visualization import plot_loss, plot_confusion_matrix, plot_training_history
from sklearn.metrics import classification_report

# Training loop
def train(model, train_loader, val_loader, optimizer, model_save_path, criterion, losses_file_path="training_losses.json"):
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
                optimizer.zero_grad()  # Zero gradients
                out = model(data)  # Forward pass
                
                # Calculate loss
                loss = criterion(out, data.y)
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights
                
                running_loss += loss.item()
                batch_losses.append(loss.item())

                # Get predictions
                predicted = (torch.sigmoid(out) >= 0.5).long() #! CHANGED _, predicted = torch.max(out, 1)
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
        val_accuracy, val_preds, val_labels = evaluate(model, val_loader)
        prec, rec, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average="binary", zero_division=0)
        ##! START NEW
        # Log metrics
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_acc)
        history["val_accuracy"].append(val_accuracy)
        history["val_precision"].append(prec)
        history["val_recall"].append(rec)
        history["val_f1"].append(f1)

        # Save best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ New best model saved at epoch {epoch+1} (Val F1: {f1:.4f})")

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

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with tqdm(loader, desc=f"Testing model accuracy", unit="batch") as batch_progress:
        with torch.no_grad():
            for data in batch_progress:
                out = model(data)
                predicted = (torch.sigmoid(out) >= 0.5).long() #! CHANGED _, predicted = torch.max(out, 1)
                correct += (predicted == data.y).sum().item()
                total += data.y.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
                tqdm.write(f"Raw model outputs: {out[:5]}")

    accuracy = correct / total
    return accuracy, all_preds, all_labels

def load_configs():
    configs_file_path = "configs.json"
    with open(configs_file_path, 'r') as file:
        configs: dict = json.load(file)
    return (configs["input_dim"], configs["hidden_dim"], configs["output_dim"], 
            configs["dropout"], configs["batch_size"], configs["learning_rate"], 
            configs["epochs"], configs["load_existing_model"], configs["save_graphs"],
            configs["model_save_path"], configs["visualizations_save_path"], 
            configs["losses_file_path"])


if __name__ == "__main__":
    #* STEP 1: LOAD CONFIGS
    input_dim, hidden_dim, output_dim, dropout, batch_size, learning_rate, epochs, load_existing_model, save_graphs, model_save_path, visualizations_save_path, losses_file_path = load_configs()
    '''
        load_existing_model: boolean value, decides whether we load saved .pth model or train a new one
        save_graphs: boolean value, decide if we save graphs to computer (for faster runtime), or do not save (for better space efficiency)
        model_save_path: path to where we save our model at the end of training
        visualization_save_path: path to where we store graphs
        losses_file_path: path to where we store loss data
    '''
    if not os.path.exists(visualizations_save_path):
        os.mkdir(visualizations_save_path)
    
    model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, model="gat")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #* STEP 2: TRAIN MODEL
    if not load_existing_model:
        train_dataset = GraphDataset("train")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = GraphDataset("valid")
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        # SETTING WEIGHTS TO FIX VULN/NONVULN INBALANCE #! STREAMLINE LATER, CHANGED!!!!
        vuln, nonvuln = train_dataset.get_vuln_nonvuln_split()
        pos_weight = torch.tensor(nonvuln / vuln, dtype=torch.float)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train(model, train_loader, val_loader, optimizer, model_save_path=model_save_path, criterion=criterion, losses_file_path="training_losses_GAT.json")    
    else:
        print("Loading existing model")
        model.load_state_dict(torch.load(model_save_path))

    #* STEP 3: TEST MODEL
    test_dataset = GraphDataset("test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_accuracy, all_preds_test, all_labels_test = evaluate(model, test_loader)
    # Test results
    plot_confusion_matrix(all_labels_test, all_preds_test, dataset_name="Test", save_path=os.path.join(visualizations_save_path, "test_conf_matr.png"))
    print(classification_report(
        all_labels_test,
        all_preds_test,
        target_names=["Safe", "Vulnerable"],
        zero_division=0  # suppress warnings for undefined metrics
    ))
    print(f"Test Accuracy: {test_accuracy:.4f}")

    #* STEP 4: VALIDATE MODEL
    validation_dataset = GraphDataset("valid")
    val_loader = DataLoader(validation_dataset, batch_size=batch_size)
    val_accuracy, all_preds_val, all_labels_val = evaluate(model, val_loader)
    # Validation results
    print(f"Valid Accuracy: {val_accuracy:.4f}")
    plot_confusion_matrix(all_labels_val, all_preds_val, dataset_name="Validation", save_path=os.path.join(visualizations_save_path, "val_conf_matr.png"))
    print(classification_report(
        all_labels_val,
        all_preds_val,
        target_names=["Safe", "Vulnerable"],
        zero_division=0  # suppress warnings for undefined metrics
    ))

    #* STEP 5: SAVE PREDICTIONS AND LABELS TO JSON
    results = {
        "test": {
            "predictions": [int(x) for x in all_preds_test],
            "labels": [int(x) for x in all_labels_test],
            "accuracy": float(test_accuracy)
        },
        "validation": {
            "predictions": [int(x) for x in all_preds_val],
            "labels": [int(x) for x in all_labels_val],
            "accuracy": float(val_accuracy)
        }
    }

    with open("predictions_and_labels.json", "w") as f:
        json.dump(results, f, indent=2)


    print("Saved predictions and labels to predictions_and_labels.json")
    plot_training_history(history_file_path="training_history.json", save_dir=visualizations_save_path)
