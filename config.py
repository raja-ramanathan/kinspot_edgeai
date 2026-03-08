import torch

# Use GPU for mac
DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')

# PREDICTION CONFIGS
CONFIDENCE_THRESHOLD = 0.85


# directory paths
TRAINING_DIR = 'data/family_photos/train'
MODEL_EMBEDDING_DIR= "embeddings"