"""Central configuration: paths, random seed, column names and model settings."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Tweets.csv"

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

RANDOM_SEED = 42

TEXT_COL = "text"

LABEL_COL = "airline_sentiment"


MODEL_NAME = "distilbert-base-uncased"


MAX_LENGTH_PERCENTILE = 95


FROZEN_LAYERS = 3
