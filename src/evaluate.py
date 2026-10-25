"""Metrics, loss curves and confusion matrix, carried over from the previous exercises."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import FIGURES_DIR


def _save_and_show(fig, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _predict(model, loader, device="cpu"):
    """Return the true and predicted labels for every sample in a loader."""
    model.eval()  # Dropout off
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())  # Highest score wins
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def evaluate_model(model, loader, class_names, device="cpu"):
    """Print accuracy and the per-class precision, recall and F1."""
    y_true, y_pred = _predict(model, loader, device)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    return y_true, y_pred


def plot_loss_curves(train_losses, val_losses, model_name="model"):
    """Plot training and validation loss per epoch, marking the epoch that was kept."""
    epochs = range(1, len(train_losses) + 1)
    best_epoch = int(np.argmin(val_losses)) + 1  # The epoch whose weights the training restored

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, label="Train loss", marker="o")
    ax.plot(epochs, val_losses, label="Validation loss", marker="o")
    ax.axvline(best_epoch, color="grey", linestyle="--", label=f"Best epoch ({best_epoch})")
    ax.set_title(f"Training and validation loss ({model_name})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(list(epochs))  # Only three epochs, so fractional ticks would be meaningless
    ax.legend()
    _save_and_show(fig, f"loss_curve_{model_name}.png")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name="model"):
    """Plot the confusion matrix with counts and the share of each true class."""
    matrix = confusion_matrix(y_true, y_pred)
    row_totals = matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")

    for row in range(len(class_names)):
        for col in range(len(class_names)):
            count = matrix[row, col]
            percent = count / row_totals[row, 0] * 100  # Share of that true class, so the diagonal is recall
            colour = "white" if count > matrix.max() / 2 else "black"
            ax.text(col, row, f"{count}\n{percent:.1f}%", ha="center", va="center", color=colour)

    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({model_name})")
    _save_and_show(fig, f"confusion_matrix_{model_name}.png")
