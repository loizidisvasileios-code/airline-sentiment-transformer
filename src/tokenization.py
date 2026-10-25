"""Label encoding, dataset splitting and DistilBERT tokenization into batched tensors."""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

from src.config import LABEL_COL, MAX_LENGTH_PERCENTILE, MODEL_NAME, RANDOM_SEED, TEXT_COL


def encode_labels(df):
    """Turn the three sentiment names into integers."""
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df[LABEL_COL])
    class_names = list(encoder.classes_)
    print("Label mapping:", {name: index for index, name in enumerate(class_names)})
    return df, class_names


def split_data(df):
    """Split 80/10/10 into train, validation and test."""
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["label"]
    )
    print(f"Train {len(train_df)} | Validation {len(val_df)} | Test {len(test_df)}")
    return train_df, val_df, test_df


def load_tokenizer():
    """Load the WordPiece tokenizer that was trained together with the pretrained weights."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"{MODEL_NAME} | vocabulary {len(tokenizer)} | model limit {tokenizer.model_max_length} tokens")
    return tokenizer


def measure_max_length(texts, tokenizer):
    """Pick the padding length from the training tweets, measured in the model's own subword units."""
    lengths = np.array([len(ids) for ids in tokenizer(list(texts))["input_ids"]])  # [CLS] and [SEP] included
    max_length = int(np.percentile(lengths, MAX_LENGTH_PERCENTILE))
    truncated = (lengths > max_length).sum()
    print(
        f"Token lengths: mean {lengths.mean():.1f} | median {np.median(lengths):.0f} | "
        f"p{MAX_LENGTH_PERCENTILE} {max_length} | max {lengths.max()}"
    )
    print(f"Truncated at {max_length}: {truncated} tweets ({truncated / len(lengths) * 100:.1f}%)")
    return max_length


def _to_dataset(df, tokenizer, max_length):
    """Encode one split into the tensors DistilBERT consumes."""
    encoded = tokenizer(
        list(df[TEXT_COL]),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],  # Marks the real tokens so self-attention ignores the padding
        torch.tensor(df["label"].to_numpy(), dtype=torch.long),
    )


def make_loaders(train_df, val_df, test_df, tokenizer, max_length, batch_size=32):
    """Wrap the three splits in DataLoaders that serve the data in batches."""
    train_loader = DataLoader(_to_dataset(train_df, tokenizer, max_length), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_to_dataset(val_df, tokenizer, max_length), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_to_dataset(test_df, tokenizer, max_length), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
