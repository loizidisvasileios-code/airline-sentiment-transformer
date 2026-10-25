"""The pretrained DistilBERT encoder with a classification head on top."""

from transformers import AutoModelForSequenceClassification

from src.config import FROZEN_LAYERS, MODEL_NAME


def build_model(num_labels, frozen_layers=FROZEN_LAYERS):
    """Load the pretrained weights with a fresh head, then freeze the embeddings and the lowest layers."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    encoder = model.base_model  
    encoder.embeddings.requires_grad_(False)  
    for layer in encoder.transformer.layer[:frozen_layers]:
        layer.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"{MODEL_NAME}: {total:,} parameters | trainable {trainable:,} ({trainable / total * 100:.1f}%) | "
        f"frozen embeddings + layers 0-{frozen_layers - 1} of {len(encoder.transformer.layer)}"
    )
    return model
