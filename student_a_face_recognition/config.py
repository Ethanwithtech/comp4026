"""
Configuration file for Student A - Face Recognition Model
COMP4026 Group Project: Anonymised Facial Expression Recognition
"""

import os

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ============================================================
# Dataset
# ============================================================
DATASET_NAME = "hereisburak/pins-face-recognition"
# Number of identities to use (None = use all)
# Set a smaller number for faster experimentation
NUM_IDENTITIES = None  # e.g., 50 for quick testing, None for all
# Minimum images per identity to include
MIN_IMAGES_PER_IDENTITY = 5
# Train/val/test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# Model
# ============================================================
# Backbone: 'resnet50', 'inception_resnet_v1', 'mobilenet_v2'
BACKBONE = "resnet50"
# Embedding dimension for face recognition
EMBEDDING_DIM = 512
# Use pretrained weights
PRETRAINED = True
# Loss function: 'arcface', 'cross_entropy'
LOSS_TYPE = "arcface"
# ArcFace parameters
ARCFACE_S = 32.0  # scale
ARCFACE_M = 0.3   # margin (reduced for better convergence)
# Warmup: use CE loss for first N epochs before switching to ArcFace
WARMUP_EPOCHS = 5  # train with CrossEntropy first, then switch to ArcFace

# ============================================================
# Training
# ============================================================
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
LR_SCHEDULER = "cosine"  # 'cosine', 'step'
LR_STEP_SIZE = 10
LR_GAMMA = 0.1
NUM_WORKERS = 4

# ============================================================
# Image preprocessing
# ============================================================
IMAGE_SIZE = 160  # Input image size (160x160 for FaceNet-style)
# Data augmentation
USE_AUGMENTATION = True

# ============================================================
# Evaluation
# ============================================================
# Threshold for face verification (cosine similarity)
VERIFICATION_THRESHOLD = 0.5
# Top-K for identification accuracy
TOP_K = [1, 5]

# ============================================================
# Device
# ============================================================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Create directories
for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
