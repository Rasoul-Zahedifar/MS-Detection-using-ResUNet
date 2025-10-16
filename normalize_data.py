import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from skimage import exposure

# ------------------------------
# Helper Functions
# ------------------------------

def resample_image(img, target_shape=(128, 128, 128)):
    """Resample 3D image to target shape using zoom"""
    factors = [t/s for t, s in zip(target_shape, img.shape)]
    img_resampled = zoom(img, factors, order=1)  # linear interpolation
    return img_resampled

def normalize_image(img, method='zscore'):
    """Normalize image intensities"""
    img = img.astype(np.float32)
    if method == 'zscore':
        mean = np.mean(img)
        std = np.std(img)
        if std == 0:
            std = 1e-8
        img = (img - mean) / std
    elif method == 'minmax':
        img = (img - img.min()) / (img.max() - img.min())
    return img

def crop_or_pad(img, target_shape=(128, 128, 128)):
    """Crop or pad image to target shape"""
    result = np.zeros(target_shape, dtype=img.dtype)
    slices = []
    for i in range(3):
        if img.shape[i] < target_shape[i]:
            start = (target_shape[i] - img.shape[i]) // 2
            slices.append(slice(0, img.shape[i]))
        else:
            start = (img.shape[i] - target_shape[i]) // 2
            slices.append(slice(start, start + target_shape[i]))
    result_slices = tuple(slice(0, s.stop-s.start) for s in slices)
    result[result_slices] = img[tuple(slices)]
    return result

# ------------------------------
# Dataset Class
# ------------------------------

class MSPreparedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_shape=(128,128,128), norm_method='zscore'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.target_shape = target_shape
        self.norm_method = norm_method

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Clean and prepare
        image = resample_image(image, self.target_shape)
        mask = resample_image(mask, self.target_shape)
        image = normalize_image(image, self.norm_method)
        image = crop_or_pad(image, self.target_shape)
        mask = crop_or_pad(mask, self.target_shape)

        # Convert to tensors and add channel dimension
        image = torch.tensor(np.expand_dims(image, axis=0), dtype=torch.float32)
        mask = torch.tensor(np.expand_dims(mask, axis=0), dtype=torch.float32)

        return image, mask

# ------------------------------
# Usage Example
# ------------------------------

if __name__ == "__main__":
    image_dir = "/path/to/MS/images"
    mask_dir = "/path/to/MS/masks"

    dataset = MSPreparedDataset(image_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for imgs, masks in loader:
        print("Images:", imgs.shape, "Masks:", masks.shape)
        break
