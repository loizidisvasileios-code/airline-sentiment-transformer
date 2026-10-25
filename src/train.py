"""Fine-tuning loop: AdamW over the unfrozen weights, a warmup-then-decay schedule and per-epoch timing."""

import copy
import time

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from src.config import CHECKPOINT_DIR, RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_optimizer(model, train_loader, num_epochs, learning_rate=2e-5, warmup_ratio=0.1):
    """AdamW over the trainable weights, with a learning rate that warms up and then decays to zero."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],  # Frozen weights would only cost optimizer state
        lr=learning_rate,
    )
    total_steps = len(train_loader) * num_epochs  # The schedule is defined over steps, not epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"AdamW lr={learning_rate} | {total_steps} steps | {warmup_steps} warmup steps")
    return optimizer, scheduler


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=3, grad_clip=1.0, device="cpu", model_name="model"):
    """Fine-tune the model, tracking validation loss every epoch and keeping the best weights."""
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())  # A plain assignment would track later updates

    for epoch in range(num_epochs):
        started = time.perf_counter()
        model.train()
        running_loss = 0.0
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()  # Once per batch, so the decay follows the step schedule above
            running_loss += loss.item() * labels.size(0)  # Weight by batch size, the last batch is smaller
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                labels = labels.to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                running_loss += criterion(logits, labels).item() * labels.size(0)
        val_loss = running_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train loss {train_loss:.4f} | "
            f"Val loss {val_loss:.4f} | {(time.perf_counter() - started) / 60:.1f} min"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)  # Return the best epoch
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, CHECKPOINT_DIR / f"{model_name}.pt")
    print(f"Best validation loss {best_val_loss:.4f} | weights saved to checkpoints/{model_name}.pt")
    return model, train_losses, val_losses
