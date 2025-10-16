# fetch_data.py
from constants import train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, test_image_dir, test_mask_dir
from normalize_data import MSPreparedDataset
from torch.utils.data import DataLoader

class MSDataFetcher:
    def __init__(self, batch_size=2, target_shape=(128,128,128), norm_method='zscore'):
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.norm_method = norm_method
        self.datasets = {}
        self.loaders = {}

        self._prepare_datasets()
        self._prepare_loaders()

    def _prepare_datasets(self):
        self.datasets['train'] = MSPreparedDataset(train_image_dir, train_mask_dir,
                                                    self.target_shape, self.norm_method)
        self.datasets['val']   = MSPreparedDataset(val_image_dir, val_mask_dir,
                                                    self.target_shape, self.norm_method)
        self.datasets['test']  = MSPreparedDataset(test_image_dir, test_mask_dir,
                                                    self.target_shape, self.norm_method)

    def _prepare_loaders(self):
        self.loaders['train'] = DataLoader(self.datasets['train'], batch_size=self.batch_size, shuffle=True)
        self.loaders['val']   = DataLoader(self.datasets['val'], batch_size=self.batch_size, shuffle=False)
        self.loaders['test']  = DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=False)

    def get_loader(self, split='train'):
        return self.loaders.get(split, None)