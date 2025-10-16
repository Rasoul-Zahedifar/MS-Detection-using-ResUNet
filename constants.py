"""
Legacy constants file - for backward compatibility
All configuration is now centralized in config.py
"""
# Import from config for backward compatibility
from config import (
    TRAIN_IMAGE_DIR as train_image_dir,
    TRAIN_MASK_DIR as train_mask_dir,
    VAL_IMAGE_DIR as val_image_dir,
    VAL_MASK_DIR as val_mask_dir,
    TEST_IMAGE_DIR as test_image_dir,
    TEST_MASK_DIR as test_mask_dir
)

# Note: This file is deprecated. Please use config.py directly.
