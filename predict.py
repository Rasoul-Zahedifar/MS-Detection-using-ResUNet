"""
Inference script for MS Detection using ResUNet
Allows prediction on custom images (single image or folder)
Uses the same visualize_predictions function as evaluate.py for consistency
"""
import torch
import numpy as np
import os
import sys
import argparse
from PIL import Image
from torchvision import transforms
import json

# Add current directory to path for imports (works in both Colab and local)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from ResUNet_model import ResUNet
from utils import load_checkpoint, set_seed, visualize_predictions


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        device (torch.device): Device to load model to
    
    Returns:
        nn.Module: Loaded model
    """
    model = ResUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        filters=config.FILTERS
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer
    model, _, epoch, _, metrics = load_checkpoint(
        model, optimizer, checkpoint_path, device
    )
    
    print(f"Loaded model from epoch {epoch}")
    if metrics:
        print(f"Checkpoint metrics: {metrics}")
    
    model.eval()
    return model


def preprocess_image(image_path):
    """
    Load and preprocess a single image
    
    Args:
        image_path (str): Path to image file
    
    Returns:
        tuple: (processed tensor, original PIL image, original size)
    """
    # Load image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    original_size = image.size  # (width, height)
    
    # Define transforms (same as in evaluate.py data pipeline)
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    # Apply transforms
    image_tensor = transform(image)
    
    return image_tensor, image, original_size


def predict_single_image(model, image_tensor, device):
    """
    Predict segmentation mask for a single image
    
    Args:
        model (nn.Module): Trained model
        image_tensor (torch.Tensor): Preprocessed image tensor
        device (torch.device): Device to run inference on
    
    Returns:
        torch.Tensor: Predicted mask (1, 1, H, W) - raw logits before sigmoid
    """
    with torch.no_grad():
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Forward pass - return raw logits (visualize_predictions applies sigmoid)
        output = model(image_tensor)
    
    return output


def predict_on_image(model, image_path, output_dir, device, threshold=0.5):
    """
    Run prediction on a single image and save results using visualize_predictions
    Automatically looks for ground truth mask in parallel masks folder
    
    Args:
        model (nn.Module): Trained model
        image_path (str): Path to input image
        output_dir (str): Directory to save results
        device (torch.device): Device to run inference on
        threshold (float): Threshold for binary mask
    
    Returns:
        dict: Statistics about the prediction
    """
    # Preprocess image
    image_tensor, _, _ = preprocess_image(image_path)
    
    # Predict - returns raw logits
    prediction = predict_single_image(model, image_tensor, device)
    
    # Get image name and directory
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_filename = os.path.basename(image_path)
    image_dir = os.path.dirname(image_path)
    
    # Automatically find ground truth mask
    # Look for masks folder at same level as images folder
    ground_truth_path = None
    if 'images' in image_dir:
        # Replace 'images' with 'masks' in the path
        mask_dir = image_dir.replace('/images', '/masks').replace('\\images', '\\masks')
        potential_mask = os.path.join(mask_dir, image_filename)
        if os.path.exists(potential_mask):
            ground_truth_path = potential_mask
    
    # Load ground truth if found, otherwise create placeholder
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt_image = Image.open(ground_truth_path).convert('L')
        gt_transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor()
        ])
        ground_truth = gt_transform(gt_image)
        # Normalize mask to [0, 1] in case it's in [0, 255]
        if ground_truth.max() > 1.0:
            ground_truth = ground_truth / 255.0
        print(f"  ✓ Loaded ground truth mask (min: {ground_truth.min():.3f}, max: {ground_truth.max():.3f})")
    else:
        # Create placeholder ground truth (will show as black in visualization)
        ground_truth = torch.zeros(1, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])
        print("  ℹ No ground truth found (prediction only mode)")
    
    # Add batch dimension to match visualize_predictions expected format
    images_batch = image_tensor.unsqueeze(0)  # (1, C, H, W)
    masks_batch = ground_truth.unsqueeze(0)   # (1, 1, H, W)
    predictions_batch = prediction            # Already (1, 1, H, W)
    
    # Use visualize_predictions from utils.py (same as evaluate.py)
    vis_path = os.path.join(output_dir, f"{image_name}_prediction.png")
    visualize_predictions(
        images=images_batch,
        masks=masks_batch,
        predictions=predictions_batch,
        num_samples=1,
        save_path=vis_path
    )
    
    # Calculate statistics
    pred_sigmoid = torch.sigmoid(prediction[0, 0]).cpu().numpy()
    binary_mask = (pred_sigmoid > threshold).astype(np.float32)
    lesion_pixels = binary_mask.sum()
    total_pixels = binary_mask.size
    lesion_percentage = (lesion_pixels / total_pixels) * 100
    
    stats = {
        'image_name': image_name,
        'lesion_pixels': int(lesion_pixels),
        'total_pixels': int(total_pixels),
        'lesion_percentage': float(lesion_percentage),
        'max_probability': float(pred_sigmoid.max()),
        'mean_probability': float(pred_sigmoid.mean()),
        'has_lesions': bool(lesion_pixels > 0)
    }
    
    return stats


def predict_on_folder(model, input_folder, output_dir, device, threshold=0.5):
    """
    Run prediction on all images in a folder
    Automatically looks for ground truth masks in parallel masks folder
    
    Args:
        model (nn.Module): Trained model
        input_folder (str): Path to input folder
        output_dir (str): Directory to save results
        device (torch.device): Device to run inference on
        threshold (float): Threshold for binary mask
    
    Returns:
        list: List of statistics for all predictions
    """
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return []
    
    print(f"Found {len(image_files)} images")
    
    # Check if masks folder exists
    if 'images' in input_folder:
        mask_folder = input_folder.replace('/images', '/masks').replace('\\images', '\\masks')
        if os.path.exists(mask_folder):
            print(f"Found masks folder: {mask_folder}")
        else:
            print("No masks folder found (prediction only mode)")
    else:
        print("Not in 'images' folder structure (prediction only mode)")
    
    all_stats = []
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"\nProcessing [{idx}/{len(image_files)}]: {image_file}")
        
        image_path = os.path.join(input_folder, image_file)
        
        try:
            stats = predict_on_image(model, image_path, output_dir, device, threshold)
            all_stats.append(stats)
            
            print(f"  Lesion coverage: {stats['lesion_percentage']:.2f}%")
            print(f"  Max probability: {stats['max_probability']:.4f}")
            
        except Exception as e:
            print(f"  Error processing {image_file}: {str(e)}")
            continue
    
    return all_stats


def main():
    """
    Main function for inference
    """
    parser = argparse.ArgumentParser(
        description='MS Detection Inference - Predict on custom images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on a single image
  python predict.py --input path/to/image.png --checkpoint checkpoints/best_by_dice.pth
  
  # Predict on a folder of images (auto-detects masks if in parallel masks folder)
  python predict.py --input dataset/test/images/ --checkpoint checkpoints/best_by_dice.pth
  
  # Specify custom output directory and threshold
  python predict.py --input dataset/test/images/ --checkpoint checkpoints/best_by_dice.pth --output my_results --threshold 0.6

Note: Ground truth masks are automatically loaded if they exist in a parallel 'masks' folder
      (e.g., images in 'dataset/test/images/' → masks in 'dataset/test/masks/' with same filenames)
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image or folder containing images'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for predictions (default: results/<model_name>_predictions/)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binary segmentation (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Determine output directory
    if args.output is None:
        checkpoint_filename = os.path.basename(args.checkpoint)
        model_name = os.path.splitext(checkpoint_filename)[0]
        output_dir = os.path.join(config.RESULTS_DIR, f"{model_name}_predictions")
    else:
        output_dir = args.output
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("MS Detection - Inference Mode")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {config.DEVICE}")
    print("Ground truth: Auto-detect from parallel masks folder")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, config.DEVICE)
    
    # Check if input is file or folder
    if os.path.isfile(args.input):
        print("\nRunning prediction on single image...")
        stats = predict_on_image(model, args.input, output_dir, config.DEVICE, 
                                args.threshold)
        all_stats = [stats]
        
        print("\nPrediction Statistics:")
        print("=" * 80)
        print(f"Lesion pixels: {stats['lesion_pixels']}/{stats['total_pixels']}")
        print(f"Lesion coverage: {stats['lesion_percentage']:.2f}%")
        print(f"Max probability: {stats['max_probability']:.4f}")
        print(f"Mean probability: {stats['mean_probability']:.4f}")
        print(f"Has lesions: {stats['has_lesions']}")
        print("=" * 80)
        
    elif os.path.isdir(args.input):
        print("\nRunning prediction on folder...")
        all_stats = predict_on_folder(model, args.input, output_dir, config.DEVICE, 
                                      args.threshold)
        
        if all_stats:
            print("\nSummary Statistics:")
            print("=" * 80)
            print(f"Total images processed: {len(all_stats)}")
            
            images_with_lesions = sum(1 for s in all_stats if s['has_lesions'])
            print(f"Images with lesions: {images_with_lesions}/{len(all_stats)} ({100*images_with_lesions/len(all_stats):.1f}%)")
            
            avg_coverage = np.mean([s['lesion_percentage'] for s in all_stats])
            print(f"Average lesion coverage: {avg_coverage:.2f}%")
            
            avg_max_prob = np.mean([s['max_probability'] for s in all_stats])
            print(f"Average max probability: {avg_max_prob:.4f}")
            print("=" * 80)
    else:
        raise ValueError(f"Input path does not exist: {args.input}")
    
    # Save statistics to JSON
    stats_path = os.path.join(output_dir, 'prediction_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=4)
    print(f"\nStatistics saved to {stats_path}")
    
    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

