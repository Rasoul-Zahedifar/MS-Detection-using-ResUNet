"""
Data fetching and DataLoader creation for MS Detection
Provides convenient interface for loading train/val/test datasets
"""
from torch.utils.data import DataLoader
from normalize_data import MSDataset
import config


class MSDataFetcher:
    """
    Handles creation and management of datasets and dataloaders
    for training, validation, and testing
    """
    
    def __init__(self, 
                 batch_size=None, 
                 image_size=None,
                 use_augmentation=None,
                 num_workers=4):
        """
        Initialize data fetcher
        
        Args:
            batch_size (int): Batch size for dataloaders (default: from config)
            image_size (tuple): Target image size (default: from config)
            use_augmentation (bool): Apply augmentation to training data (default: from config)
            num_workers (int): Number of workers for data loading
        """
        self.batch_size = batch_size or config.BATCH_SIZE
        self.image_size = image_size or config.IMAGE_SIZE
        self.use_augmentation = use_augmentation if use_augmentation is not None else config.USE_AUGMENTATION
        self.num_workers = num_workers
        
        self.datasets = {}
        self.loaders = {}
        
        self._prepare_datasets()
        self._prepare_loaders()
    
    def _prepare_datasets(self):
        """Create dataset objects for train/val/test splits"""
        print("Preparing datasets...")
        
        # Training dataset with augmentation
        self.datasets['train'] = MSDataset(
            config.TRAIN_IMAGE_DIR,
            config.TRAIN_MASK_DIR,
            image_size=self.image_size,
            augment=self.use_augmentation
        )
        
        # Validation dataset without augmentation
        self.datasets['val'] = MSDataset(
            config.VAL_IMAGE_DIR,
            config.VAL_MASK_DIR,
            image_size=self.image_size,
            augment=False
        )
        
        # Test dataset without augmentation
        self.datasets['test'] = MSDataset(
            config.TEST_IMAGE_DIR,
            config.TEST_MASK_DIR,
            image_size=self.image_size,
            augment=False
        )
        
        print(f"Train: {len(self.datasets['train'])} samples")
        print(f"Val: {len(self.datasets['val'])} samples")
        print(f"Test: {len(self.datasets['test'])} samples")
    
    def _prepare_loaders(self):
        """Create DataLoader objects for train/val/test splits"""
        # Training loader with shuffling
        self.loaders['train'] = DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if config.DEVICE.type == 'cuda' else False,
            drop_last=True  # Drop last incomplete batch
        )
        
        # Validation loader without shuffling
        self.loaders['val'] = DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if config.DEVICE.type == 'cuda' else False
        )
        
        # Test loader without shuffling
        self.loaders['test'] = DataLoader(
            self.datasets['test'],
            batch_size=config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if config.DEVICE.type == 'cuda' else False
        )
    
    def get_loader(self, split='train'):
        """
        Get DataLoader for specified split
        
        Args:
            split (str): One of 'train', 'val', or 'test'
            
        Returns:
            DataLoader object
        """
        if split not in self.loaders:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(self.loaders.keys())}")
        return self.loaders[split]
    
    def get_dataset(self, split='train'):
        """
        Get Dataset for specified split
        
        Args:
            split (str): One of 'train', 'val', or 'test'
            
        Returns:
            Dataset object
        """
        if split not in self.datasets:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(self.datasets.keys())}")
        return self.datasets[split]
    
    def get_batch_count(self, split='train'):
        """
        Get number of batches for specified split
        
        Args:
            split (str): One of 'train', 'val', or 'test'
            
        Returns:
            Number of batches
        """
        return len(self.get_loader(split))


# ===========================
# Usage Example
# ===========================
if __name__ == "__main__":
    # Create data fetcher
    data_fetcher = MSDataFetcher()
    
    # Get train loader
    train_loader = data_fetcher.get_loader('train')
    print(f"\nTrain loader: {len(train_loader)} batches")
    
    # Test loading a batch
    for images, masks in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Masks: {masks.shape}")
        print(f"  Device: {images.device}")
        break
