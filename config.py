import os
import torch

# ======================================================
# Project Root
# ======================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ======================================================
# Dataset
# ======================================================

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "iam_words"
)

IMAGE_FOLDER = os.path.join(
    DATASET_ROOT,
    "words"
)

ANNOTATION_FILE = os.path.join(
    DATASET_ROOT,
    "words.txt"
)

# ======================================================
# Output Folders
# ======================================================

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "logs"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ======================================================
# Image
# ======================================================

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 128
# ======================================================
# Model Parameters
# ======================================================

HIDDEN_SIZE = 256

NUM_CHANNELS = 1
DROPOUT = 0.2

# ======================================================
# Training
# ======================================================

BATCH_SIZE = 64

NUM_EPOCHS = 50

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

EARLY_STOPPING_PATIENCE = 10

SEED = 42

# ======================================================
# DataLoader
# ======================================================

NUM_WORKERS = 2

PIN_MEMORY = torch.cuda.is_available()

# ======================================================
# Device
# ======================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ======================================================
# Models
# ======================================================

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

LAST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "last_model.pth"
)

# ======================================================
# Vocabulary
# ======================================================

VOCAB_PATH = os.path.join(
    CHECKPOINT_DIR,
    "vocabulary.pkl"
)