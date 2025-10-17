"""
Configuration file for MS Detection using ResUNet
"""
import os
import torch

# ===========================
# Paths Configuration
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train', 'images')
TRAIN_MASK_DIR = os.path.join(DATASET_DIR, 'train', 'masks')
VAL_IMAGE_DIR = os.path.join(DATASET_DIR, 'valid', 'images')
VAL_MASK_DIR = os.path.join(DATASET_DIR, 'valid', 'masks')
TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test', 'images')
TEST_MASK_DIR = os.path.join(DATASET_DIR, 'test', 'masks')

# Model checkpoint directory
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Results directory
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===========================
# Model Configuration
# ===========================
MODEL_NAME = 'ResUNet'
IN_CHANNELS = 3  # RGB images
OUT_CHANNELS = 1  # Binary segmentation mask
FILTERS = [64, 128, 256, 512]  # Filter sizes for encoder

# ===========================
# Training Configuration
# ===========================
# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Image preprocessing
IMAGE_SIZE = (256, 256)  # Target size for images
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean for RGB
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std for RGB

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_PROB = 0.5

# ===========================
# Training Settings
# ===========================
# Early stopping
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 1e-4

# Learning rate scheduler
USE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5

# Gradient clipping
USE_GRADIENT_CLIPPING = True
MAX_GRAD_NORM = 1.0

# ===========================
# Evaluation Configuration
# ===========================
EVAL_BATCH_SIZE = 4
THRESHOLD = 0.5  # Threshold for binary segmentation

# ===========================
# Logging Configuration
# ===========================
LOG_INTERVAL = 10  # Log every N batches
SAVE_CHECKPOINT_EVERY = 5  # Save checkpoint every N epochs
SAVE_BEST_MODEL = True

# Random seed for reproducibility
RANDOM_SEED = 42

# ===========================
# Loss Configuration
# ===========================
# Options: 'bce', 'dice', 'focal', 'combined', 'weighted_combined'
# Use 'weighted_combined' for datasets with class imbalance (recommended for MS detection)
LOSS_TYPE = 'weighted_combined'  
DICE_WEIGHT = 0.5
BCE_WEIGHT = 0.5

# Focal Loss parameters (for class imbalance)
FOCAL_ALPHA = 0.25  # Weight for positive class
FOCAL_GAMMA = 2.0   # Focusing parameter

