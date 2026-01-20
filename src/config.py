# FairProp Inspector Configuration

# Model Configuration
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 512
NUM_LABELS = 2

# Label Mappings
LABEL2ID = {
    "COMPLIANT": 0,
    "NON_COMPLIANT": 1
}

ID2LABEL = {
    0: "COMPLIANT",
    1: "NON_COMPLIANT"
}

# Training Configuration
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01

# Paths
DEFAULT_MODEL_PATH = "artifacts/model"
DEFAULT_DATA_PATH = "data/processed/seed_data.json"
DEFAULT_OUTPUT_DIR = "artifacts/model"

# Performance Thresholds
TARGET_LATENCY_MS = 20  # Target P95 latency
MIN_ACCURACY = 0.90     # Minimum acceptable accuracy

# Violation Categories
VIOLATION_CATEGORIES = [
    "familial_status",
    "age",
    "religion",
    "race",
    "national_origin",
    "sex",
    "disability",
    "economic"
]
