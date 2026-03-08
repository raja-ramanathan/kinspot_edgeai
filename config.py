import torch

# Use GPU for mac
DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')

# PREDICTION CONFIGS
CONFIDENCE_THRESHOLD = 0.85