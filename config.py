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
IN_CHANNELS = 1  # Grayscale images
OUT_CHANNELS = 1  # Binary segmentation mask
FILTERS = [64, 128, 256, 512]  # Filter sizes for encoder

# ===========================
# Training Configuration
# ===========================
# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 8  # Physical batch size (fits in memory)
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 8 * 2 = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Image preprocessing
IMAGE_SIZE = (256, 256)  # Target size for images
NORMALIZE_MEAN = [0.5]  # Mean for grayscale images
NORMALIZE_STD = [0.5]   # Std for grayscale images

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_PROB = 0.5

# ===========================
# Patch-based Training Configuration
# ===========================
USE_PATCH_TRAINING = False  # Enable patch-based training instead of full images
PATCH_SIZE = (256, 256)  # Size of patches to extract
PATCHES_PER_IMAGE = 4  # Number of patches to extract per image
FOREGROUND_PATCH_RATIO = 0.7  # Ratio of patches that should contain foreground (lesions)
MIN_FOREGROUND_RATIO = 0.05  # Minimum foreground ratio in mask to consider a patch as "foreground"

# ===========================
# Oversampling/Undersampling Configuration
# ===========================
USE_CLASS_SAMPLING = False  # Enable oversampling/undersampling based on class
OVERSAMPLE_RARE_CLASS = True  # Oversample images with lesions (rare class)
UNDERSAMPLE_BACKGROUND = True  # Undersample pure-background images
RARE_CLASS_OVERSAMPLE_FACTOR = 3  # How many times to oversample rare-class images
BACKGROUND_UNDERSAMPLE_FACTOR = 0.3  # Keep only this fraction of pure-background images
PURE_BACKGROUND_THRESHOLD = 0.01  # Maximum foreground ratio to consider image as "pure background"

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
# Increased for severe class imbalance (1:246 ratio)
FOCAL_ALPHA = 0.75  # Weight for positive class (increased from 0.25 for severe imbalance)
FOCAL_GAMMA = 3.0   # Focusing parameter (increased from 2.0 for harder examples)

