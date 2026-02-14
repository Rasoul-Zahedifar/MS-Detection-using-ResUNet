"""
Sliding Window Inference for Patch-Based Models
Handles train-test mismatch by evaluating full images using overlapping patches
"""
import torch
import numpy as np
from typing import Tuple, Optional


def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: Tuple[int, int] = (256, 256),
    stride: Optional[Tuple[int, int]] = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    Perform sliding window inference on a full image using patch-based model
    
    This function:
    1. Divides the image into overlapping patches
    2. Runs the model on each patch
    3. Stitches predictions back together with averaging in overlap regions
    
    Args:
        model: Trained model (expects logits output)
        image: Input image tensor (1, C, H, W) or (C, H, W)
        patch_size: Size of patches to extract (height, width)
        stride: Stride between patches (default: patch_size // 2 for 50% overlap)
        device: Device to run inference on
    
    Returns:
        torch.Tensor: Full prediction map (1, 1, H, W) - logits
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Handle input dimensions
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Add batch dimension
    
    batch_size, channels, img_h, img_w = image.shape
    assert batch_size == 1, "Sliding window inference only supports batch_size=1"
    
    # Default stride: 50% overlap
    if stride is None:
        stride = (patch_size[0] // 2, patch_size[1] // 2)
    
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride
    
    # Calculate number of patches needed
    n_patches_h = max(1, (img_h - patch_h) // stride_h + 1)
    n_patches_w = max(1, (img_w - patch_w) // stride_w + 1)
    
    # Initialize output arrays
    # We'll accumulate predictions and counts for averaging in overlap regions
    predictions = torch.zeros((1, 1, img_h, img_w), device=device, dtype=torch.float32)
    counts = torch.zeros((1, 1, img_h, img_w), device=device, dtype=torch.float32)
    
    # Extract and process patches
    with torch.no_grad():
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch coordinates
                start_h = i * stride_h
                start_w = j * stride_w
                
                # Ensure patch doesn't go out of bounds
                if start_h + patch_h > img_h:
                    start_h = img_h - patch_h
                if start_w + patch_w > img_w:
                    start_w = img_w - patch_w
                
                end_h = start_h + patch_h
                end_w = start_w + patch_w
                
                # Extract patch
                patch = image[:, :, start_h:end_h, start_w:end_w]
                
                # Run model on patch (returns logits)
                patch_pred = model(patch.to(device))
                
                # Accumulate prediction and count
                predictions[:, :, start_h:end_h, start_w:end_w] += patch_pred
                counts[:, :, start_h:end_h, start_w:end_w] += 1
    
    # Average predictions in overlapping regions
    # Avoid division by zero (shouldn't happen, but just in case)
    predictions = predictions / (counts + 1e-8)
    
    return predictions


def evaluate_with_sliding_window(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    patch_size: Tuple[int, int] = (256, 256),
    stride: Optional[Tuple[int, int]] = None,
    device: torch.device = None,
    calculate_metrics_fn=None
) -> dict:
    """
    Evaluate model on dataset using sliding window inference
    
    Args:
        model: Trained model
        dataloader: DataLoader with full images (not patches)
        patch_size: Patch size used during training
        stride: Stride for sliding window (default: 50% overlap)
        device: Device to run on
        calculate_metrics_fn: Function to calculate metrics (default: from utils)
    
    Returns:
        dict: Dictionary with average metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    if calculate_metrics_fn is None:
        from utils import calculate_metrics
        calculate_metrics_fn = calculate_metrics
    
    model.eval()
    
    all_metrics = {
        'dice': [],
        'iou': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': []
    }
    
    print(f"Evaluating with sliding window inference:")
    print(f"  Patch size: {patch_size}")
    print(f"  Stride: {stride if stride else (patch_size[0]//2, patch_size[1]//2)} (50% overlap)")
    
    from tqdm import tqdm
    for images, masks in tqdm(dataloader, desc="Sliding window evaluation"):
        # Process each image in batch separately (sliding window expects single images)
        for i in range(images.shape[0]):
            image = images[i:i+1]  # Keep batch dimension
            mask = masks[i:i+1]
            
            # Run sliding window inference
            prediction = sliding_window_inference(
                model=model,
                image=image,
                patch_size=patch_size,
                stride=stride,
                device=device
            )
            
            # Calculate metrics
            metrics = calculate_metrics_fn(prediction, mask.to(device))
            
            # Store metrics
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    # Calculate average metrics
    avg_metrics = {
        key: np.mean(values) for key, values in all_metrics.items()
    }
    
    # Calculate standard deviations
    std_metrics = {
        key: np.std(values) for key, values in all_metrics.items()
    }
    
    return avg_metrics, std_metrics, all_metrics


# ===========================
# Usage Example
# ===========================
if __name__ == "__main__":
    # Test sliding window inference
    print("Testing sliding window inference...")
    
    # Create dummy model and image
    from ResUNet_model import ResUNet
    
    model = ResUNet(
        in_channels=1,
        out_channels=1,
        filters=[64, 128, 256, 512]
    )
    
    # Test with different image sizes
    test_sizes = [(512, 512), (768, 768), (1024, 1024)]
    
    for img_h, img_w in test_sizes:
        image = torch.randn(1, 1, img_h, img_w)
        
        print(f"\nTesting {img_h}x{img_w} image:")
        prediction = sliding_window_inference(
            model=model,
            image=image,
            patch_size=(256, 256),
            stride=(128, 128)
        )
        
        print(f"  Input shape: {image.shape}")
        print(f"  Output shape: {prediction.shape}")
        assert prediction.shape == (1, 1, img_h, img_w), "Output shape mismatch!"
        print(f"  ✓ Shape check passed")
    
    print("\n✓ All tests passed!")
