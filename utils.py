"""
Utility functions for MS Detection
Includes loss functions, metrics, visualization, and helper functions
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import config


# ===========================
# Loss Functions
# ===========================
# All loss functions expect LOGITS as input (raw model outputs before sigmoid)

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    Expects logits as input, applies sigmoid internally
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (B, 1, H, W) - no sigmoid applied
            targets: Binary ground truth (B, 1, H, W) - values in [0, 1]
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Expects logits as input, applies sigmoid internally
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (B, 1, H, W) - no sigmoid applied
            targets: Binary ground truth (B, 1, H, W) - values in [0, 1]
        """
        # Use BCEWithLogitsLoss for numerical stability
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Calculate pt for focal term
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Calculate alpha_t
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1 - self.alpha, device=logits.device)
        )
        
        # Focal loss formula
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice Loss
    Expects logits as input
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()  # Handles sigmoid internally
        self.dice = DiceLoss()
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (B, 1, H, W) - no sigmoid applied
            targets: Binary ground truth (B, 1, H, W) - values in [0, 1]
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class WeightedCombinedLoss(nn.Module):
    """
    Combined Focal + Dice Loss (best for severe class imbalance)
    Expects logits as input
    """
    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.25, gamma=2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (B, 1, H, W) - no sigmoid applied
            targets: Binary ground truth (B, 1, H, W) - values in [0, 1]
        """
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


def get_loss_function(loss_type='combined'):
    """
    Get loss function by type
    All loss functions expect logits as input
    
    Args:
        loss_type: One of 'bce', 'dice', 'focal', 'combined', 'weighted_combined'
    
    Returns:
        Loss function module
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'focal':
        return FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
    elif loss_type == 'combined':
        return CombinedLoss(config.BCE_WEIGHT, config.DICE_WEIGHT)
    elif loss_type == 'weighted_combined':
        return WeightedCombinedLoss(
            focal_weight=0.5,
            dice_weight=0.5,
            alpha=config.FOCAL_ALPHA,
            gamma=config.FOCAL_GAMMA
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ===========================
# Metrics
# ===========================
# All metric functions expect LOGITS as input and apply sigmoid internally

def dice_coefficient(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Calculate Dice coefficient
    
    Args:
        logits: Raw model outputs (B, 1, H, W) - no sigmoid applied
        targets: Binary ground truth (B, 1, H, W)
        threshold: Threshold for binarization
        smooth: Smoothing factor
    
    Returns:
        float: Dice coefficient
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds = preds.view(preds.size(0), -1)
    targs = targets.view(targets.size(0), -1)
    intersection = (preds * targs).sum(1)
    dice = (2 * intersection + smooth) / (preds.sum(1) + targs.sum(1) + smooth)
    return dice.mean().item()


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Calculate IoU (Jaccard) score
    
    Args:
        logits: Raw model outputs (B, 1, H, W) - no sigmoid applied
        targets: Binary ground truth (B, 1, H, W)
        threshold: Threshold for binarization
        smooth: Smoothing factor
    
    Returns:
        float: IoU score
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float().view(-1)
    targs = targets.view(-1)
    intersection = (preds * targs).sum()
    union = preds.sum() + targs.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def pixel_accuracy(logits, targets, threshold=0.5):
    """
    Calculate pixel-wise accuracy
    
    Args:
        logits: Raw model outputs (B, 1, H, W) - no sigmoid applied
        targets: Binary ground truth (B, 1, H, W)
        threshold: Threshold for binarization
    
    Returns:
        float: Pixel accuracy
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == targets).sum()
    total = targets.numel()
    return (correct / (total + 1e-6)).item()


def sensitivity_specificity(logits, targets, threshold=0.5):
    """
    Calculate sensitivity (recall/TPR) and specificity (TNR)
    
    Args:
        logits: Raw model outputs (B, 1, H, W) - no sigmoid applied
        targets: Binary ground truth (B, 1, H, W)
        threshold: Threshold for binarization
    
    Returns:
        tuple: (sensitivity, specificity)
    """
    # Apply sigmoid and binarize predictions
    probs = torch.sigmoid(logits)
    predictions = (probs > threshold).float()
    
    # Flatten tensors
    predictions = predictions.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    
    # Calculate metrics
    sensitivity = tp / (tp + fn + 1e-6)  # True Positive Rate / Recall
    specificity = tn / (tn + fp + 1e-6)  # True Negative Rate
    
    return sensitivity, specificity


def calculate_metrics(logits, targets, threshold=0.5):
    """
    Calculate all metrics
    
    Args:
        logits: Raw model outputs (B, 1, H, W) - no sigmoid applied
        targets: Binary ground truth (B, 1, H, W)
        threshold: Threshold for binarization
    
    Returns:
        dict: Dictionary of metrics
    """
    sens, spec = sensitivity_specificity(logits, targets, threshold)
    return {
        'dice': dice_coefficient(logits, targets, threshold),
        'iou': iou_score(logits, targets, threshold),
        'accuracy': pixel_accuracy(logits, targets, threshold),
        'sensitivity': sens,
        'specificity': spec,
    }


# ===========================
# Visualization
# ===========================

def visualize_predictions(images, masks, predictions, num_samples=4, save_path=None):
    """
    Visualize images, ground truth masks, and predictions
    
    Args:
        images (torch.Tensor): Input images (B, C, H, W)
        masks (torch.Tensor): Ground truth masks (B, 1, H, W)
        predictions (torch.Tensor): Predicted masks (B, 1, H, W)
        num_samples (int): Number of samples to visualize
        save_path (str): Path to save the figure (optional)
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Convert to numpy and denormalize image
        image = images[i].cpu().numpy()
        if image.shape[0] == 1:
            # Grayscale image
            image = image[0]  # Remove channel dimension
            image = image * config.NORMALIZE_STD[0] + config.NORMALIZE_MEAN[0]
        else:
            # RGB image
            image = image.transpose(1, 2, 0)
            image = image * np.array(config.NORMALIZE_STD) + np.array(config.NORMALIZE_MEAN)
        image = np.clip(image, 0, 1)
        
        mask = masks[i, 0].cpu().numpy()
        pred = torch.sigmoid(predictions[i, 0]).cpu().detach().numpy()
        
        # Plot image
        axes[i, 0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Plot ground truth mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot prediction
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        p = Path(save_path)
        plt.savefig(p, bbox_inches='tight', dpi=150)
        plt.savefig(p.with_suffix('.svg'), bbox_inches='tight', format='svg')
        print(f"Saved visualization to {save_path} and {p.with_suffix('.svg')}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and metrics)
    
    Args:
        history (dict): Training history containing losses and metrics
        save_path (str): Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Dice coefficient
    if 'train_dice' in history and 'val_dice' in history:
        axes[1].plot(history['train_dice'], label='Train Dice')
        axes[1].plot(history['val_dice'], label='Val Dice')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Coefficient')
        axes[1].set_title('Training and Validation Dice Score')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        p = Path(save_path)
        plt.savefig(p, bbox_inches='tight', dpi=150)
        plt.savefig(p.with_suffix('.svg'), bbox_inches='tight', format='svg')
        print(f"Saved training history to {save_path} and {p.with_suffix('.svg')}")
    else:
        plt.show()
    
    plt.close()


# ===========================
# Model Utilities
# ===========================

def save_checkpoint(model, optimizer, epoch, loss, metrics, filepath):
    """
    Save model checkpoint
    
    Args:
        model (nn.Module): PyTorch model
        optimizer (torch.optim.Optimizer): Optimizer
        epoch (int): Current epoch
        loss (float): Current loss
        metrics (dict): Current metrics
        filepath (str): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model checkpoint
    
    Args:
        model (nn.Module): PyTorch model
        optimizer (torch.optim.Optimizer): Optimizer
        filepath (str): Path to checkpoint file
        device (torch.device): Device to load model to
    
    Returns:
        tuple: (model, optimizer, epoch, loss, metrics)
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from epoch {epoch}")
    
    return model, optimizer, epoch, loss, metrics

def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Set deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """
    
    def __init__(self, patience=7, min_delta=0, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            verbose (bool): Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# ===========================
# Usage Example
# ===========================
if __name__ == "__main__":
    # Test loss functions
    predictions = torch.rand(4, 1, 256, 256)
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
    # Test different losses
    bce_loss = nn.BCELoss()(predictions, targets)
    dice_loss = DiceLoss()(predictions, targets)
    combined_loss = CombinedLoss()(predictions, targets)
    
    print("Loss Functions:")
    print(f"  BCE Loss: {bce_loss.item():.4f}")
    print(f"  Dice Loss: {dice_loss.item():.4f}")
    print(f"  Combined Loss: {combined_loss.item():.4f}")
    
    # Test metrics
    metrics = calculate_metrics(predictions, targets)
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key.capitalize()}: {value:.4f}")
