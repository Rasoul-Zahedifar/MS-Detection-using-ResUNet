"""
Diagnostic script to understand class imbalance and loss behavior
Run this to see what's happening with your dataset and loss function
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
import config
from normalize_data import MSDataset
from utils import get_loss_function, calculate_metrics
import matplotlib.pyplot as plt

def analyze_dataset_imbalance():
    """Analyze class imbalance in the dataset"""
    print("=" * 60)
    print("ANALYZING DATASET CLASS IMBALANCE")
    print("=" * 60)
    
    # Load training dataset
    train_dataset = MSDataset(
        config.TRAIN_IMAGE_DIR,
        config.TRAIN_MASK_DIR,
        image_size=config.IMAGE_SIZE,
        augment=False
    )
    
    # Sample a few masks to analyze
    num_samples = min(50, len(train_dataset))
    
    total_pixels = 0
    lesion_pixels = 0
    
    print(f"\nAnalyzing {num_samples} samples...")
    
    for i in range(num_samples):
        _, mask = train_dataset[i]
        total_pixels += mask.numel()
        lesion_pixels += mask.sum().item()
    
    background_pixels = total_pixels - lesion_pixels
    
    lesion_ratio = lesion_pixels / total_pixels * 100
    background_ratio = background_pixels / total_pixels * 100
    imbalance_ratio = background_pixels / (lesion_pixels + 1e-6)
    
    print(f"\nResults:")
    print(f"  Total pixels:      {total_pixels:,}")
    print(f"  Background pixels: {background_pixels:,} ({background_ratio:.2f}%)")
    print(f"  Lesion pixels:     {lesion_pixels:,} ({lesion_ratio:.2f}%)")
    print(f"  Imbalance ratio:   1:{imbalance_ratio:.1f}")
    print()
    print(f"⚠️  This means for every 1 lesion pixel, there are {imbalance_ratio:.0f} background pixels!")
    
    return lesion_ratio, imbalance_ratio


def test_loss_functions():
    """Test how different loss functions behave with imbalanced data"""
    print("\n" + "=" * 60)
    print("TESTING LOSS FUNCTION BEHAVIOR")
    print("=" * 60)
    
    # Simulate realistic predictions
    batch_size = 4
    H, W = 256, 256
    
    # Scenario 1: Model predicts all zeros (common early problem)
    print("\n1. MODEL PREDICTS ALL ZEROS (BAD - no learning)")
    print("-" * 60)
    
    targets = torch.rand(batch_size, 1, H, W)
    targets = (targets > 0.99).float()  # ~1% lesions (realistic)
    predictions_zeros = torch.zeros_like(targets) + 0.01  # Almost zero
    
    print(f"   Ground truth has {targets.sum().item():.0f} lesion pixels out of {targets.numel()}")
    print(f"   Predictions are all near zero")
    
    # Test different losses
    bce_loss = torch.nn.BCELoss()(predictions_zeros, targets)
    dice_loss = get_loss_function('dice')(predictions_zeros, targets)
    focal_loss = get_loss_function('focal')(predictions_zeros, targets)
    weighted_loss = get_loss_function('weighted_combined')(predictions_zeros, targets)
    
    metrics = calculate_metrics(predictions_zeros, targets)
    
    print(f"\n   Losses:")
    print(f"     BCE Loss:            {bce_loss.item():.4f}")
    print(f"     Dice Loss:           {dice_loss.item():.4f}")
    print(f"     Focal Loss:          {focal_loss.item():.4f}")
    print(f"     Weighted Combined:   {weighted_loss.item():.4f}")
    print(f"\n   Metrics:")
    print(f"     Accuracy: {metrics['accuracy']:.4f} (HIGH but meaningless!)")
    print(f"     Dice:     {metrics['dice']:.4f} (LOW - model is bad)")
    print(f"     IoU:      {metrics['iou']:.4f} (LOW - model is bad)")
    print(f"     Sensitivity: {metrics['sensitivity']:.4f} (LOW - misses lesions)")
    
    # Scenario 2: Model starts detecting something
    print("\n2. MODEL DETECTS SOME LESIONS (BETTER - learning)")
    print("-" * 60)
    
    predictions_better = torch.rand(batch_size, 1, H, W) * 0.3
    # Add some high predictions where lesions are
    predictions_better = predictions_better + targets * 0.5
    predictions_better = torch.clamp(predictions_better, 0, 1)
    
    print(f"   Predictions now have some signal in lesion regions")
    
    bce_loss_better = torch.nn.BCELoss()(predictions_better, targets)
    dice_loss_better = get_loss_function('dice')(predictions_better, targets)
    focal_loss_better = get_loss_function('focal')(predictions_better, targets)
    weighted_loss_better = get_loss_function('weighted_combined')(predictions_better, targets)
    
    metrics_better = calculate_metrics(predictions_better, targets)
    
    print(f"\n   Losses (should be LOWER than scenario 1):")
    print(f"     BCE Loss:            {bce_loss_better.item():.4f} (Δ: {bce_loss.item() - bce_loss_better.item():+.4f})")
    print(f"     Dice Loss:           {dice_loss_better.item():.4f} (Δ: {dice_loss.item() - dice_loss_better.item():+.4f})")
    print(f"     Focal Loss:          {focal_loss_better.item():.4f} (Δ: {focal_loss.item() - focal_loss_better.item():+.4f})")
    print(f"     Weighted Combined:   {weighted_loss_better.item():.4f} (Δ: {weighted_loss.item() - weighted_loss_better.item():+.4f})")
    print(f"\n   Metrics (should be HIGHER):")
    print(f"     Accuracy: {metrics_better['accuracy']:.4f} (Δ: {metrics_better['accuracy'] - metrics['accuracy']:+.4f})")
    print(f"     Dice:     {metrics_better['dice']:.4f} (Δ: {metrics_better['dice'] - metrics['dice']:+.4f})")
    print(f"     IoU:      {metrics_better['iou']:.4f} (Δ: {metrics_better['iou'] - metrics['iou']:+.4f})")
    print(f"     Sensitivity: {metrics_better['sensitivity']:.4f} (Δ: {metrics_better['sensitivity'] - metrics['sensitivity']:+.4f})")


def recommend_focal_params(lesion_ratio, imbalance_ratio):
    """Recommend focal loss parameters based on class imbalance"""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR YOUR DATASET")
    print("=" * 60)
    
    # Calculate recommended alpha (weight for positive class)
    # For severe imbalance, increase alpha
    if imbalance_ratio > 200:
        recommended_alpha = 0.75
        severity = "SEVERE"
    elif imbalance_ratio > 100:
        recommended_alpha = 0.5
        severity = "HIGH"
    elif imbalance_ratio > 50:
        recommended_alpha = 0.35
        severity = "MODERATE"
    else:
        recommended_alpha = 0.25
        severity = "MILD"
    
    print(f"\nYour class imbalance: {severity} (1:{imbalance_ratio:.0f})")
    print(f"Current Focal Alpha: {config.FOCAL_ALPHA}")
    print(f"Current Focal Gamma: {config.FOCAL_GAMMA}")
    
    print(f"\n📊 Recommendations:")
    print(f"   1. Increase FOCAL_ALPHA to {recommended_alpha} (from {config.FOCAL_ALPHA})")
    print(f"      - This gives more weight to rare lesion pixels")
    print(f"      - Alpha = {recommended_alpha} means lesions get {recommended_alpha:.0%} importance")
    
    print(f"\n   2. Optionally increase FOCAL_GAMMA to 3.0 (from {config.FOCAL_GAMMA})")
    print(f"      - This makes the model focus even more on hard examples")
    print(f"      - Higher gamma = more aggressive focusing")
    
    print(f"\n   3. Monitor the RIGHT metrics:")
    print(f"      ✅ Watch: Dice Coefficient (should reach >0.5)")
    print(f"      ✅ Watch: IoU (should reach >0.4)")
    print(f"      ✅ Watch: Sensitivity (should reach >0.7)")
    print(f"      ❌ IGNORE: Accuracy (will stay ~99% regardless)")
    
    print(f"\n   4. Training tips:")
    print(f"      - Train for at least 50-100 epochs")
    print(f"      - Dice will start low (~0.1) and slowly improve")
    print(f"      - If Dice doesn't improve after 20 epochs, increase alpha more")
    print(f"      - Check visual predictions to see if lesions are detected")
    
    print(f"\n💡 Quick Fix - Add to config.py:")
    print(f"   FOCAL_ALPHA = {recommended_alpha}")
    print(f"   FOCAL_GAMMA = 3.0")


def main():
    print("\n🔍 MS DETECTION TRAINING DIAGNOSTIC TOOL\n")
    
    # Analyze dataset
    lesion_ratio, imbalance_ratio = analyze_dataset_imbalance()
    
    # Test loss functions
    test_loss_functions()
    
    # Provide recommendations
    recommend_focal_params(lesion_ratio, imbalance_ratio)
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Update config.py with recommended parameters")
    print("   2. Delete old checkpoints: rm -rf checkpoints/*")
    print("   3. Retrain: python main.py --mode train")
    print("   4. Monitor Dice/IoU, NOT accuracy!")
    print()


if __name__ == "__main__":
    main()

