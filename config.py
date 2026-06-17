# config.py

# -----------------------------
# Dataset
# -----------------------------

NUM_CLASSES = 4

IMAGE_SIZE = 224

# -----------------------------
# Training
# -----------------------------

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

LOCAL_EPOCHS = 2

NUM_ROUNDS = 20

RANDOM_SEED = 42

# -----------------------------
# Federated Learning
# -----------------------------

NUM_CLIENTS = 10

CLIENTS_PER_ROUND = 7

DIRICHLET_ALPHA = 0.5

# -----------------------------
# Differential Privacy
# -----------------------------

MAX_GRAD_NORM = 1.0

NOISE_MULTIPLIER = 1.1

DELTA = 1e-5

# -----------------------------
# FedProx
# -----------------------------

MU = 0.01

# -----------------------------
# FedDyn
# -----------------------------

FEDDYN_ALPHA = 0.01

# -----------------------------
# Personalization
# -----------------------------

PERSONALIZATION_EPOCHS = 3

# -----------------------------
# Async FL / Stragglers
# -----------------------------

CLIENT_DROPOUT_RATE = 0.2

STRAGGLER_RATE = 0.2

# -----------------------------
# Paths
# -----------------------------

DATASET_PATH = "AugmentedAlzheimerDataset"

MODEL_SAVE_DIR = "saved_models"

RESULTS_DIR = "results"