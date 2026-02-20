"""
Evaluation script for MS Detection using ResUNet
Evaluates trained model on test dataset and generates visualizations
"""
import torch
import numpy as np
from tqdm import tqdm
import os
import sys
import json

# Add current directory to path for imports (works in both Colab and local)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from ResUNet_model import ResUNet
from fetch_data import MSDataFetcher
from utils import (
    calculate_metrics,
    dice_coefficient,
    load_checkpoint,
    visualize_predictions,
    set_seed
)
from sliding_window_inference import (
    sliding_window_inference,
    evaluate_with_sliding_window
)


class Evaluator:
    """
    Evaluator class for MS Detection model
    """
    
    def __init__(self, model, test_loader, device):
        """
        Initialize evaluator
        
        Args:
            model (nn.Module): Trained ResUNet model
            test_loader (DataLoader): Test data loader
            device (torch.device): Device to evaluate on
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.eval()
    
    def evaluate(self, save_visualizations=True, num_vis_samples=10):
        """
        Evaluate model on test dataset
        Automatically uses sliding window inference if patch training was used
        
        Args:
            save_visualizations (bool): Whether to save visualization samples
            num_vis_samples (int): Number of samples to visualize
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print("=" * 60)
        print("Evaluating model on test dataset...")
        
        # Check if we should use sliding window inference
        use_sliding_window = getattr(config, 'USE_PATCH_TRAINING', False)
        
        if use_sliding_window:
            print("⚠️  PATCH TRAINING DETECTED")
            print("   Using sliding window inference to match training distribution")
            patch_size = getattr(config, 'PATCH_SIZE', (256, 256))
            stride = (patch_size[0] // 2, patch_size[1] // 2)  # 50% overlap
            print(f"   Patch size: {patch_size}, Stride: {stride}")
        
        print("=" * 60)
        
        all_metrics = {
            'dice': [],
            'iou': [],
            'accuracy': [],
            'sensitivity': [],
            'specificity': []
        }
        
        # For visualization
        vis_images = []
        vis_masks = []
        vis_predictions = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            
            for batch_idx, (images, masks) in enumerate(pbar):
                # Move data to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                if use_sliding_window:
                    # Process each image separately with sliding window
                    batch_predictions = []
                    for i in range(images.shape[0]):
                        image = images[i:i+1]
                        
                        # Run sliding window inference
                        prediction = sliding_window_inference(
                            model=self.model,
                            image=image,
                            patch_size=patch_size,
                            stride=stride,
                            device=self.device
                        )
                        batch_predictions.append(prediction)
                    
                    outputs = torch.cat(batch_predictions, dim=0)
                else:
                    # Standard forward pass
                    outputs = self.model(images)
                
                # Calculate metrics for batch
                batch_metrics = calculate_metrics(outputs, masks)
                
                # Store metrics
                for key in all_metrics:
                    all_metrics[key].append(batch_metrics[key])
                
                # Update progress bar
                pbar.set_postfix({
                    'dice': f"{batch_metrics['dice']:.4f}",
                    'iou': f"{batch_metrics['iou']:.4f}"
                })
                
                # Save samples for visualization
                if save_visualizations and len(vis_images) < num_vis_samples:
                    samples_to_take = min(
                        images.shape[0],
                        num_vis_samples - len(vis_images)
                    )
                    vis_images.append(images[:samples_to_take].cpu())
                    vis_masks.append(masks[:samples_to_take].cpu())
                    vis_predictions.append(outputs[:samples_to_take].cpu())
        
        # Calculate average metrics
        avg_metrics = {
            key: np.mean(values) for key, values in all_metrics.items()
        }
        
        # Calculate standard deviations
        std_metrics = {
            key: np.std(values) for key, values in all_metrics.items()
        }
        
        # Print results
        print("\n" + "=" * 60)
        print("Evaluation Results:")
        if use_sliding_window:
            print("(Using Sliding Window Inference)")
        print("=" * 60)
        for key in avg_metrics:
            print(f"{key.capitalize():15s}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
        print("=" * 60)
        
        # Save metrics to file
        results_path = os.path.join(config.RESULTS_DIR, 'test_results.json')
        results = {
            'average_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'all_metrics': {key: [float(v) for v in values] 
                          for key, values in all_metrics.items()},
            'evaluation_method': 'sliding_window' if use_sliding_window else 'standard'
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {results_path}")
        
        # Save visualizations
        if save_visualizations and vis_images:
            vis_images = torch.cat(vis_images, dim=0)
            vis_masks = torch.cat(vis_masks, dim=0)
            vis_predictions = torch.cat(vis_predictions, dim=0)
            
            vis_path = os.path.join(config.RESULTS_DIR, 'test_predictions.png')
            visualize_predictions(
                vis_images,
                vis_masks,
                vis_predictions,
                num_samples=min(num_vis_samples, vis_images.shape[0]),
                save_path=vis_path
            )
        
        return avg_metrics, std_metrics
    
    def evaluate_single_sample(self, image, mask, visualize=True):
        """
        Evaluate a single sample
        
        Args:
            image (torch.Tensor): Input image
            mask (torch.Tensor): Ground truth mask
            visualize (bool): Whether to visualize the result
        
        Returns:
            tuple: (prediction, metrics)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Add batch dimension if needed
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if mask.dim() == 3:
                mask = mask.unsqueeze(0)
            
            # Move to device
            image = image.to(self.device)
            mask = mask.to(self.device)
            
            # Forward pass
            prediction = self.model(image)
            
            # Calculate metrics
            metrics = calculate_metrics(prediction, mask)
            
            # Visualize if requested
            if visualize:
                visualize_predictions(image, mask, prediction, num_samples=1)
            
            return prediction, metrics
    
    def predict_batch(self, images):
        """
        Predict segmentation masks for a batch of images
        
        Args:
            images (torch.Tensor): Batch of images (B, C, H, W)
        
        Returns:
            torch.Tensor: Predicted masks (B, 1, H, W)
        """
        self.model.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            predictions = self.model(images)
        
        return predictions
    
    def get_prediction_statistics(self):
        """
        Calculate statistics about predictions (e.g., lesion coverage)
        Uses sliding window if patch training was used
        
        Returns:
            dict: Statistics dictionary
        """
        stats = {
            'total_samples': 0,
            'samples_with_lesions': 0,
            'avg_lesion_coverage': 0.0,
            'predictions_with_lesions': 0,
            'avg_predicted_coverage': 0.0
        }
        
        # Check if we should use sliding window
        use_sliding_window = getattr(config, 'USE_PATCH_TRAINING', False)
        if use_sliding_window:
            patch_size = getattr(config, 'PATCH_SIZE', (256, 256))
            stride = (patch_size[0] // 2, patch_size[1] // 2)
        
        with torch.no_grad():
            for images, masks in tqdm(self.test_loader, desc='Calculating stats'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Get predictions
                if use_sliding_window:
                    batch_predictions = []
                    for i in range(images.shape[0]):
                        image = images[i:i+1]
                        prediction = sliding_window_inference(
                            model=self.model,
                            image=image,
                            patch_size=patch_size,
                            stride=stride,
                            device=self.device
                        )
                        batch_predictions.append(prediction)
                    outputs = torch.cat(batch_predictions, dim=0)
                else:
                    outputs = self.model(images)
                
                # Binarize predictions
                pred_binary = (torch.sigmoid(outputs) > config.THRESHOLD).float()
                
                # Update statistics
                batch_size = images.shape[0]
                stats['total_samples'] += batch_size
                
                for i in range(batch_size):
                    # Check if ground truth has lesions
                    mask_sum = masks[i].sum().item()
                    if mask_sum > 0:
                        stats['samples_with_lesions'] += 1
                        coverage = mask_sum / masks[i].numel()
                        stats['avg_lesion_coverage'] += coverage
                    
                    # Check if prediction has lesions
                    pred_sum = pred_binary[i].sum().item()
                    if pred_sum > 0:
                        stats['predictions_with_lesions'] += 1
                        coverage = pred_sum / pred_binary[i].numel()
                        stats['avg_predicted_coverage'] += coverage
        
        # Calculate averages
        if stats['samples_with_lesions'] > 0:
            stats['avg_lesion_coverage'] /= stats['samples_with_lesions']
        if stats['predictions_with_lesions'] > 0:
            stats['avg_predicted_coverage'] /= stats['predictions_with_lesions']
        
        return stats


def evaluate_and_save_best_samples(
    checkpoint_path=None,
    split='test',
    num_best=10,
    rank_by='dice',
    save_path=None
):
    """
    Run the best model (by loss or dice) on the test set, rank samples by per-sample
    Dice, and save only the top num_best samples in a vertical row (same layout as
    mode --evaluate: Input | Ground Truth | Prediction).

    Goes through all samples to find the best ones by accuracy (Dice), then stops
    and saves only those — unlike evaluate which saves the first N samples.

    Args:
        checkpoint_path (str): Path to checkpoint. If None, uses best_by_dice.pth
            or best_by_loss.pth according to rank_by.
        split (str): Dataset split ('test', 'val', or 'train').
        num_best (int): Number of best samples to save (default 10).
        rank_by (str): Which best checkpoint to load when checkpoint_path is None:
            'dice' -> checkpoints/best_by_dice.pth, 'loss' -> checkpoints/best_by_loss.pth.
            Samples are always ranked by per-sample Dice (higher = better).
        save_path (str): Path for the output image. If None, saved to
            results/<model_name>/test_predictions_best10.png (and .svg).

    Returns:
        tuple: (avg_metrics, std_metrics) over full split; and the list of (dice, index)
            for the saved samples.
    """
    set_seed(config.RANDOM_SEED)

    if checkpoint_path is None:
        if rank_by == 'dice':
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_by_dice.pth')
        else:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_by_loss.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_filename = os.path.basename(checkpoint_path)
    model_name = os.path.splitext(checkpoint_filename)[0]
    model_results_dir = os.path.join(config.RESULTS_DIR, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    original_results_dir = config.RESULTS_DIR
    config.RESULTS_DIR = model_results_dir

    use_sliding_window = getattr(config, 'USE_PATCH_TRAINING', False)
    if use_sliding_window:
        patch_size = getattr(config, 'PATCH_SIZE', (256, 256))
        stride = (patch_size[0] // 2, patch_size[1] // 2)

    print("Loading data...")
    data_fetcher = MSDataFetcher(
        batch_size=config.EVAL_BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        use_augmentation=False
    )
    test_loader = data_fetcher.get_loader(split)

    print("Loading model...")
    model = ResUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        filters=config.FILTERS,
        use_transformer=getattr(config, 'USE_TRANSFORMER', True),
        transformer_layers=getattr(config, 'TRANSFORMER_LAYERS', 2),
        transformer_heads=getattr(config, 'TRANSFORMER_HEADS', 8),
        transformer_dim=getattr(config, 'TRANSFORMER_DIM', None)
    ).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    model, _, epoch, _, metrics = load_checkpoint(
        model, optimizer, checkpoint_path, config.DEVICE
    )
    model.eval()

    print(f"Finding top {num_best} samples by Dice (checkpoint: {checkpoint_filename})...")
    # (dice_score, image, mask, prediction) per sample
    samples = []
    all_metrics = {'dice': [], 'iou': [], 'accuracy': [], 'sensitivity': [], 'specificity': []}

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Scoring samples')
        for images, masks in pbar:
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)
            if use_sliding_window:
                batch_predictions = []
                for i in range(images.shape[0]):
                    pred = sliding_window_inference(
                        model=model,
                        image=images[i:i+1],
                        patch_size=patch_size,
                        stride=stride,
                        device=config.DEVICE
                    )
                    batch_predictions.append(pred)
                outputs = torch.cat(batch_predictions, dim=0)
            else:
                outputs = model(images)

            for i in range(images.shape[0]):
                dice = dice_coefficient(
                    outputs[i:i+1], masks[i:i+1],
                    threshold=config.THRESHOLD
                )
                samples.append((
                    dice,
                    images[i:i+1].cpu(),
                    masks[i:i+1].cpu(),
                    outputs[i:i+1].cpu()
                ))
            batch_metrics = calculate_metrics(outputs, masks)
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])
            pbar.set_postfix({'samples': len(samples), 'dice': f"{batch_metrics['dice']:.4f}"})

    # Sort by Dice descending (best first), take top num_best
    samples.sort(key=lambda x: x[0], reverse=True)
    top = samples[:num_best]
    if not top:
        config.RESULTS_DIR = original_results_dir
        raise ValueError("No samples in dataset.")

    vis_images = torch.cat([x[1] for x in top], dim=0)
    vis_masks = torch.cat([x[2] for x in top], dim=0)
    vis_predictions = torch.cat([x[3] for x in top], dim=0)
    dice_scores = [x[0] for x in top]

    if save_path is None:
        save_path = os.path.join(
            config.RESULTS_DIR,
            f'test_predictions_best{num_best}.png'
        )
    from pathlib import Path
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    visualize_predictions(
        vis_images,
        vis_masks,
        vis_predictions,
        num_samples=len(top),
        save_path=str(p)
    )
    print(f"Saved top {num_best} samples (by Dice) to {save_path} and {p.with_suffix('.svg')}")

    # Optionally save dice scores for the best samples
    best_scores_path = os.path.join(config.RESULTS_DIR, f'best_{num_best}_dice_scores.json')
    with open(best_scores_path, 'w') as f:
        json.dump({'dice_scores': dice_scores, 'num_best': num_best}, f, indent=2)
    print(f"Dice scores for best samples saved to {best_scores_path}")

    avg_metrics = {key: float(np.mean(values)) for key, values in all_metrics.items()}
    std_metrics = {key: float(np.std(values)) for key, values in all_metrics.items()}
    config.RESULTS_DIR = original_results_dir
    return avg_metrics, std_metrics, list(zip(dice_scores, range(len(top))))


def evaluate_model(checkpoint_path=None, split='test'):
    """
    Main evaluation function
    
    Args:
        checkpoint_path (str): Path to model checkpoint (default: best_model.pth)
        split (str): Dataset split to evaluate on ('test', 'val', or 'train')
    
    Returns:
        dict: Evaluation metrics
    """
    # Set random seed for reproducibility
    set_seed(config.RANDOM_SEED)
    
    # Default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Extract model name from checkpoint path and create results subdirectory
    checkpoint_filename = os.path.basename(checkpoint_path)
    model_name = os.path.splitext(checkpoint_filename)[0]  # Remove .pth extension
    model_results_dir = os.path.join(config.RESULTS_DIR, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    
    # Update config.RESULTS_DIR temporarily for this evaluation
    original_results_dir = config.RESULTS_DIR
    config.RESULTS_DIR = model_results_dir
    
    # Create data loader
    print("Loading data...")
    data_fetcher = MSDataFetcher(
        batch_size=config.EVAL_BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        use_augmentation=False
    )
    
    test_loader = data_fetcher.get_loader(split)
    
    # Create model
    print("\nLoading model...")
    model = ResUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        filters=config.FILTERS,
        use_transformer=getattr(config, 'USE_TRANSFORMER', True),
        transformer_layers=getattr(config, 'TRANSFORMER_LAYERS', 2),
        transformer_heads=getattr(config, 'TRANSFORMER_HEADS', 8),
        transformer_dim=getattr(config, 'TRANSFORMER_DIM', None)
    ).to(config.DEVICE)
    
    # Load checkpoint
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer
    model, _, epoch, _, metrics = load_checkpoint(
        model, optimizer, checkpoint_path, config.DEVICE
    )
    
    print(f"Loaded model from epoch {epoch}")
    if metrics:
        print(f"Checkpoint metrics: {metrics}")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=config.DEVICE
    )
    
    # Evaluate the model
    avg_metrics, std_metrics = evaluator.evaluate(
        save_visualizations=True,
        num_vis_samples=10
    )
    
    # Get prediction statistics
    print("\nCalculating prediction statistics...")
    stats = evaluator.get_prediction_statistics()
    
    print("\nPrediction Statistics:")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples with lesions (GT): {stats['samples_with_lesions']}")
    print(f"Average lesion coverage (GT): {stats['avg_lesion_coverage']:.4f}")
    print(f"Samples with lesions (Pred): {stats['predictions_with_lesions']}")
    print(f"Average predicted coverage: {stats['avg_predicted_coverage']:.4f}")
    print("=" * 60)
    
    # Restore original RESULTS_DIR
    config.RESULTS_DIR = original_results_dir
    
    return avg_metrics, std_metrics


# ===========================
# Usage Example
# ===========================
if __name__ == "__main__":
    # Evaluate the best model on test set
    avg_metrics, std_metrics = evaluate_model()
    
    print("\nEvaluation completed successfully!")
