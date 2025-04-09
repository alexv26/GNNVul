import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
import os

def plot_loss(losses, path="loss_curve.png"):
    epoch_losses = losses.get("epoch_loss", [])
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.title("Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved loss curve to {path}")


def plot_confusion_matrix(y_true, y_pred, dataset_name="Test", save_path="conf_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Safe", "Vulnerable"], yticklabels=["Safe", "Vulnerable"])
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {dataset_name.lower()} confusion matrix to {save_path}")

import matplotlib.pyplot as plt

def plot_training_history(history_file_path="training_history.json", save_dir="visualizations"):
    with open(history_file_path, "r") as f:
        history = json.load(f)

    epochs = history["epoch"]

    # === Plot 1: Training Loss ===
    plt.figure()
    plt.plot(epochs, history["train_loss"], marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "train_loss_curve.png"))
    plt.close()

    # === Plot 2: Training vs Validation Accuracy ===
    plt.figure()
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy", marker='o')
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy", marker='x')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()

    # === Plot 3: Validation Precision, Recall, F1 ===
    plt.figure()
    plt.plot(epochs, history["val_precision"], label="Precision", marker='o')
    plt.plot(epochs, history["val_recall"], label="Recall", marker='x')
    plt.plot(epochs, history["val_f1"], label="F1 Score", marker='s')
    plt.title("Validation Metrics per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "val_metrics_curve.png"))
    plt.close()

    print("✅ Training history plots saved to:", save_dir)
