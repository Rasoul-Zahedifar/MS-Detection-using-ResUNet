"""
Dataset preparation and normalization for MS Detection
Handles loading, preprocessing, and augmentation of 2D medical images
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

import config


class MSDataset(Dataset):
    """
    Dataset class for MS Detection with 2D images
    
    Args:
        image_dir (str): Directory containing input images
        mask_dir (str): Directory containing segmentation masks
        image_size (tuple): Target size for images (height, width)
        augment (bool): Whether to apply data augmentation
    """
    
    def __init__(self, image_dir, mask_dir, image_size=(256, 256), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        
        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png'))])
        
        # Verify that we have matching images and masks
        print(f"Found {len(self.image_files)} images and {len(self.mask_files)} masks in {os.path.basename(image_dir)}")
        
        # Define normalization transform
        self.normalize = transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        )
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        # Find corresponding mask - handle different naming patterns
        mask_name = self._find_matching_mask(self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Apply augmentation if enabled
        if self.augment and random.random() < config.AUGMENTATION_PROB:
            image, mask = self._apply_augmentation(image, mask)
        
        # Resize to target size
        image = TF.resize(image, self.image_size, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=Image.NEAREST)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # Normalize image
        image = self.normalize(image)
        
        # Binarize mask (threshold at 0.5)
        mask = (mask > 0.5).float()
        
        return image, mask
    
    def _find_matching_mask(self, image_filename):
        """
        Find the corresponding mask file for an image
        Handles different naming conventions
        """
        # Try to extract the ID from the image filename
        # Example: "pincdistorted_1003_0.3_-0.3_154012.jpg" -> "154012.jpg"
        parts = image_filename.split('_')
        if len(parts) > 1:
            # Get the last part (should be the ID)
            mask_id = parts[-1]
            if mask_id in self.mask_files:
                return mask_id
        
        # If direct match fails, try to find any mask with matching ID
        for mask_file in self.mask_files:
            if image_filename.endswith(mask_file):
                return mask_file
            # Check if the mask ID is in the image filename
            mask_id = mask_file.split('.')[0]
            if mask_id in image_filename:
                return mask_file
        
        # Default: return the mask at the same index
        return self.mask_files[self.image_files.index(image_filename)]
    
    def _apply_augmentation(self, image, mask):
        """
        Apply random augmentations to image and mask
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation (±15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # Random brightness, contrast, saturation (only for image)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
        
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))
        
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))
        
        return image, mask


def get_data_stats(dataset, num_samples=100):
    """
    Calculate mean and std statistics for a dataset
    
    Args:
        dataset: Dataset object
        num_samples: Number of samples to use for calculation
        
    Returns:
        mean, std: Channel-wise mean and standard deviation
    """
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=min(num_samples, len(dataset)),
        shuffle=False
    )
    
    mean = 0.
    std = 0.
    total_samples = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean, std


# ===========================
# Usage Example
# ===========================
if __name__ == "__main__":
    # Create dataset
    train_dataset = MSDataset(
        config.TRAIN_IMAGE_DIR,
        config.TRAIN_MASK_DIR,
        image_size=config.IMAGE_SIZE,
        augment=True
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # Test loading a sample
    image, mask = train_dataset[0]
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Mask unique values: {torch.unique(mask)}")
