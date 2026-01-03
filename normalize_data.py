"""
Dataset preparation and normalization for MS Detection
Handles loading, preprocessing, and augmentation of 2D medical images
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from scipy import ndimage

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
        assert os.path.exists(mask_path), f"No mask found for: {img_path}"

        # Load images
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        # Apply augmentation if enabled
        if self.augment:
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
    
    def _to_pil_gray(self, x):
        """Convert PIL/ndarray/tensor to PIL Image in mode 'L' (grayscale)."""
        # If it's already a PIL Image, convert to 'L'
        if isinstance(x, Image.Image):
            return x.convert('L')
        # If it's a torch Tensor: assume (1,H,W) or (H,W)
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            # squeeze channel dim if present
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            # If float in [0,1], scale to [0,255]
            if np.issubdtype(arr.dtype, np.floating):
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr, mode='L')
        # If numpy array
        if isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            if np.issubdtype(arr.dtype, np.floating):
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr, mode='L')

        raise TypeError(f"Unsupported image type: {type(x)}")

    def _apply_augmentation(self, image, mask):
        """
        Robust augmentation that returns PIL Images (mode 'L').

        Expected inputs: PIL Image, numpy array, or torch.Tensor.
        Returns: (image_pil, mask_pil)
        """
        # --- convert inputs to PIL grayscale ---
        image_pil = self._to_pil_gray(image)
        mask_pil = self._to_pil_gray(mask)

        # --- Resize first (optional) ---
        # If you prefer augmentation at the target resolution, you can resize here:
        # image_pil = image_pil.resize(self.image_size, resample=Image.BILINEAR)
        # mask_pil  = mask_pil.resize(self.image_size, resample=Image.NEAREST)

        # --- Random horizontal/vertical flips ---
        if random.random() < 0.5:
            image_pil = TF.hflip(image_pil)
            mask_pil = TF.hflip(mask_pil)
        if random.random() < 0.5:
            image_pil = TF.vflip(image_pil)
            mask_pil = TF.vflip(mask_pil)

        # --- Random rotation (use small angles; use nearest for mask) ---
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            # torchvision supports interpolation argument in recent versions
            try:
                image_pil = TF.rotate(image_pil, angle, interpolation=Image.BILINEAR)
                mask_pil = TF.rotate(mask_pil, angle, interpolation=Image.NEAREST)
            except TypeError:
                # older torchvision uses 'resample' keyword
                image_pil = TF.rotate(image_pil, angle, resample=Image.BILINEAR)
                mask_pil = TF.rotate(mask_pil, angle, resample=Image.NEAREST)

        # --- Intensity augmentations (image only) ---
        if random.random() < 0.5:
            image_pil = TF.adjust_brightness(image_pil, random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            image_pil = TF.adjust_contrast(image_pil, random.uniform(0.8, 1.2))

        # Return PIL images — the rest of your pipeline will call TF.resize and TF.to_tensor
        return image_pil, mask_pil


class MSPatchDataset(Dataset):
    """
    Patch-based dataset for MS Detection
    Extracts patches centered on lesions and oversamples patches containing foreground
    
    Args:
        image_dir (str): Directory containing input images
        mask_dir (str): Directory containing segmentation masks
        patch_size (tuple): Size of patches to extract (height, width)
        patches_per_image (int): Number of patches to extract per image
        foreground_patch_ratio (float): Ratio of patches that should contain foreground
        min_foreground_ratio (float): Minimum foreground ratio to consider patch as "foreground"
        augment (bool): Whether to apply data augmentation
    """
    
    def __init__(self, image_dir, mask_dir, patch_size=(256, 256), 
                 patches_per_image=4, foreground_patch_ratio=0.7,
                 min_foreground_ratio=0.05, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.foreground_patch_ratio = foreground_patch_ratio
        self.min_foreground_ratio = min_foreground_ratio
        self.augment = augment
        
        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png'))])
        
        print(f"Found {len(self.image_files)} images and {len(self.mask_files)} masks in {os.path.basename(image_dir)}")
        
        # Pre-compute patch indices for all images
        self.patch_indices = []
        self._precompute_patches()
        
        # Define normalization transform
        self.normalize = transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        )
    
    def _precompute_patches(self):
        """Pre-compute all patch locations from all images"""
        print("Pre-computing patch locations...")
        
        for img_idx, img_file in enumerate(self.image_files):
            # Load mask to find lesion locations
            mask_name = self._find_matching_mask(img_file)
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            if not os.path.exists(mask_path):
                continue
            
            mask = Image.open(mask_path).convert('L')
            mask_array = np.array(mask)
            mask_binary = (mask_array > 127).astype(np.uint8)
            
            # Find lesion centers
            lesion_centers = self._find_lesion_centers(mask_binary)
            
            # Calculate number of foreground vs background patches
            num_foreground = int(self.patches_per_image * self.foreground_patch_ratio)
            num_background = self.patches_per_image - num_foreground
            
            # Generate foreground patches (centered on lesions)
            for _ in range(num_foreground):
                if len(lesion_centers) > 0:
                    # Pick a random lesion center
                    center = random.choice(lesion_centers)
                    self.patch_indices.append((img_idx, center, True))
                else:
                    # No lesions found, use random patch
                    center = self._get_random_center(mask_array.shape)
                    self.patch_indices.append((img_idx, center, False))
            
            # Generate background patches (random locations)
            for _ in range(num_background):
                center = self._get_random_center(mask_array.shape)
                # Check if this patch would contain foreground
                has_foreground = self._check_patch_foreground(mask_binary, center)
                self.patch_indices.append((img_idx, center, has_foreground))
        
        print(f"Generated {len(self.patch_indices)} patches from {len(self.image_files)} images")
    
    def _find_lesion_centers(self, mask_binary):
        """Find centers of lesions in the mask"""
        # Label connected components
        labeled_mask, num_features = ndimage.label(mask_binary)
        
        centers = []
        for i in range(1, num_features + 1):
            # Get coordinates of this component
            coords = np.where(labeled_mask == i)
            if len(coords[0]) > 0:
                # Calculate centroid
                center_y = int(np.mean(coords[0]))
                center_x = int(np.mean(coords[1]))
                centers.append((center_y, center_x))
        
        return centers
    
    def _get_random_center(self, image_shape):
        """Get a random center point for patch extraction"""
        h, w = image_shape
        patch_h, patch_w = self.patch_size
        
        # Ensure patch fits within image
        max_y = max(0, h - patch_h)
        max_x = max(0, w - patch_w)
        
        if max_y <= 0 or max_x <= 0:
            # Image is smaller than patch, use center
            return (h // 2, w // 2)
        
        center_y = random.randint(patch_h // 2, max_y + patch_h // 2)
        center_x = random.randint(patch_w // 2, max_x + patch_w // 2)
        
        return (center_y, center_x)
    
    def _check_patch_foreground(self, mask_binary, center):
        """Check if a patch centered at 'center' contains enough foreground"""
        patch_h, patch_w = self.patch_size
        center_y, center_x = center
        
        # Calculate patch bounds
        y1 = max(0, center_y - patch_h // 2)
        y2 = min(mask_binary.shape[0], center_y + patch_h // 2)
        x1 = max(0, center_x - patch_w // 2)
        x2 = min(mask_binary.shape[1], center_x + patch_w // 2)
        
        # Extract patch
        patch = mask_binary[y1:y2, x1:x2]
        
        if patch.size == 0:
            return False
        
        # Calculate foreground ratio
        foreground_ratio = np.sum(patch > 0) / patch.size
        return foreground_ratio >= self.min_foreground_ratio
    
    def _find_matching_mask(self, image_filename):
        """Find the corresponding mask file for an image"""
        parts = image_filename.split('_')
        if len(parts) > 1:
            mask_id = parts[-1]
            if mask_id in self.mask_files:
                return mask_id
        
        for mask_file in self.mask_files:
            if image_filename.endswith(mask_file):
                return mask_file
            mask_id = mask_file.split('.')[0]
            if mask_id in image_filename:
                return mask_file
        
        # Default: try to find by index
        try:
            img_idx = self.image_files.index(image_filename)
            if img_idx < len(self.mask_files):
                return self.mask_files[img_idx]
        except ValueError:
            pass
        
        # Last resort: return first mask (shouldn't happen in practice)
        if len(self.mask_files) > 0:
            return self.mask_files[0]
        raise ValueError(f"No mask found for image: {image_filename}")
    
    def _crop_patch(self, image, mask, center):
        """Crop a patch from image and mask centered at 'center'"""
        patch_h, patch_w = self.patch_size
        center_y, center_x = center
        
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask)
        else:
            mask_array = mask
        
        # Calculate patch bounds
        y1 = max(0, center_y - patch_h // 2)
        y2 = min(img_array.shape[0], center_y + patch_h // 2)
        x1 = max(0, center_x - patch_w // 2)
        x2 = min(img_array.shape[1], center_x + patch_w // 2)
        
        # Crop
        img_patch = img_array[y1:y2, x1:x2]
        mask_patch = mask_array[y1:y2, x1:x2]
        
        # Pad if necessary
        if img_patch.shape[0] < patch_h or img_patch.shape[1] < patch_w:
            pad_h = max(0, patch_h - img_patch.shape[0])
            pad_w = max(0, patch_w - img_patch.shape[1])
            img_patch = np.pad(img_patch, ((0, pad_h), (0, pad_w)), mode='constant')
            mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode='constant')
        
        # Convert back to PIL
        img_pil = Image.fromarray(img_patch, mode='L')
        mask_pil = Image.fromarray(mask_patch, mode='L')
        
        return img_pil, mask_pil
    
    def _to_pil_gray(self, x):
        """Convert PIL/ndarray/tensor to PIL Image in mode 'L' (grayscale)."""
        if isinstance(x, Image.Image):
            return x.convert('L')
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            if np.issubdtype(arr.dtype, np.floating):
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr, mode='L')
        if isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            if np.issubdtype(arr.dtype, np.floating):
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr, mode='L')
        raise TypeError(f"Unsupported image type: {type(x)}")
    
    def _apply_augmentation(self, image, mask):
        """Apply augmentation to patch"""
        image_pil = self._to_pil_gray(image)
        mask_pil = self._to_pil_gray(mask)
        
        # Random horizontal/vertical flips
        if random.random() < 0.5:
            image_pil = TF.hflip(image_pil)
            mask_pil = TF.hflip(mask_pil)
        if random.random() < 0.5:
            image_pil = TF.vflip(image_pil)
            mask_pil = TF.vflip(mask_pil)
        
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            try:
                image_pil = TF.rotate(image_pil, angle, interpolation=Image.BILINEAR)
                mask_pil = TF.rotate(mask_pil, angle, interpolation=Image.NEAREST)
            except TypeError:
                image_pil = TF.rotate(image_pil, angle, resample=Image.BILINEAR)
                mask_pil = TF.rotate(mask_pil, angle, resample=Image.NEAREST)
        
        # Intensity augmentations (image only)
        if random.random() < 0.5:
            image_pil = TF.adjust_brightness(image_pil, random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            image_pil = TF.adjust_contrast(image_pil, random.uniform(0.8, 1.2))
        
        return image_pil, mask_pil
    
    def __len__(self):
        return len(self.patch_indices)
    
    def __getitem__(self, idx):
        img_idx, center, _ = self.patch_indices[idx]
        
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.image_files[img_idx])
        mask_name = self._find_matching_mask(self.image_files[img_idx])
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load images
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        # Crop patch
        image, mask = self._crop_patch(image, mask, center)
        
        # Apply augmentation if enabled
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        
        # Resize to target size (in case patch was smaller)
        image = TF.resize(image, self.patch_size, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, self.patch_size, interpolation=Image.NEAREST)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # Normalize image
        image = self.normalize(image)
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        return image, mask


def analyze_image_classes(image_dir, mask_dir):
    """
    Analyze images to determine which are rare-class (with lesions) and pure-background
    
    Returns:
        tuple: (rare_class_indices, background_indices, all_indices)
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png'))])
    
    rare_class_indices = []
    background_indices = []
    
    for idx, img_file in enumerate(image_files):
        # Find matching mask
        mask_name = None
        parts = img_file.split('_')
        if len(parts) > 1:
            mask_id = parts[-1]
            if mask_id in mask_files:
                mask_name = mask_id
        
        if mask_name is None:
            for mask_file in mask_files:
                if img_file.endswith(mask_file):
                    mask_name = mask_file
                    break
                mask_id = mask_file.split('.')[0]
                if mask_id in img_file:
                    mask_name = mask_file
                    break
        
        if mask_name is None:
            continue
        
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            continue
        
        # Load and analyze mask
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        mask_binary = (mask_array > 127).astype(np.uint8)
        
        # Calculate foreground ratio
        foreground_ratio = np.sum(mask_binary > 0) / mask_binary.size
        
        if foreground_ratio <= config.PURE_BACKGROUND_THRESHOLD:
            background_indices.append(idx)
        else:
            rare_class_indices.append(idx)
    
    all_indices = list(range(len(image_files)))
    
    return rare_class_indices, background_indices, all_indices


def create_weighted_sampler(dataset, rare_class_indices, background_indices):
    """
    Create a WeightedRandomSampler for oversampling rare-class and undersampling background
    
    Args:
        dataset: Dataset object
        rare_class_indices: List of indices for images with lesions
        background_indices: List of indices for pure-background images
    
    Returns:
        WeightedRandomSampler
    """
    # Create weight array
    weights = np.ones(len(dataset))
    
    # Oversample rare-class images
    if config.OVERSAMPLE_RARE_CLASS:
        for idx in rare_class_indices:
            if idx < len(weights):
                weights[idx] = config.RARE_CLASS_OVERSAMPLE_FACTOR
    
    # Undersample background images
    if config.UNDERSAMPLE_BACKGROUND:
        for idx in background_indices:
            if idx < len(weights):
                weights[idx] = config.BACKGROUND_UNDERSAMPLE_FACTOR
    
    # Convert to tensor
    weights = torch.DoubleTensor(weights)
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)


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
