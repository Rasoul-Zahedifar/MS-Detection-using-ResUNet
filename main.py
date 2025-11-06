"""
Main entry point for MS Detection using ResUNet
Provides command-line interface for training, evaluation, and inference
"""
import argparse
import os
import sys

# Add current directory to path for imports (works in both Colab and local)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def parse_args():
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='MS Detection using ResUNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model from scratch
  python main.py --mode train
  
  # Resume training from a checkpoint
  python main.py --mode train --resume checkpoints/checkpoint_epoch_10.pth
  
  # Evaluate the best model on test set
  python main.py --mode evaluate
  
  # Evaluate a specific checkpoint
  python main.py --mode evaluate --checkpoint checkpoints/best_model.pth
  
  # Evaluate on validation set
  python main.py --mode evaluate --split val
  
  # Predict on a single image
  python main.py --mode predict --input path/to/image.png --checkpoint checkpoints/best_by_dice.pth
  
  # Predict on a folder of images
  python main.py --mode predict --input path/to/images/ --checkpoint checkpoints/best_by_loss.pth
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'predict', 'info'],
        required=True,
        help='Mode to run: train, evaluate, predict, or info'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for evaluation (default: best_model.pth)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default='test',
        help='Dataset split to evaluate on (default: test)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Batch size (default: {config.BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f'Number of epochs (default: {config.NUM_EPOCHS})'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help=f'Learning rate (default: {config.LEARNING_RATE})'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input image or folder for prediction mode'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for predictions'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binary segmentation in prediction mode (default: 0.5)'
    )
    
    return parser.parse_args()


def print_config_info():
    """
    Print configuration information
    """
    print("=" * 80)
    print("MS Detection using ResUNet - Configuration")
    print("=" * 80)
    
    print("\n[PATHS]")
    print(f"  Dataset directory:    {config.DATASET_DIR}")
    print(f"  Checkpoint directory: {config.CHECKPOINT_DIR}")
    print(f"  Results directory:    {config.RESULTS_DIR}")
    
    print("\n[MODEL]")
    print(f"  Architecture:         {config.MODEL_NAME}")
    print(f"  Input channels:       {config.IN_CHANNELS}")
    print(f"  Output channels:      {config.OUT_CHANNELS}")
    print(f"  Filters:              {config.FILTERS}")
    
    print("\n[TRAINING]")
    print(f"  Device:               {config.DEVICE}")
    print(f"  Batch size:           {config.BATCH_SIZE}")
    print(f"  Number of epochs:     {config.NUM_EPOCHS}")
    print(f"  Learning rate:        {config.LEARNING_RATE}")
    print(f"  Weight decay:         {config.WEIGHT_DECAY}")
    print(f"  Image size:           {config.IMAGE_SIZE}")
    print(f"  Loss type:            {config.LOSS_TYPE}")
    
    print("\n[DATA AUGMENTATION]")
    print(f"  Use augmentation:     {config.USE_AUGMENTATION}")
    print(f"  Augmentation prob:    {config.AUGMENTATION_PROB}")
    
    print("\n[OPTIMIZATION]")
    print(f"  Use LR scheduler:     {config.USE_LR_SCHEDULER}")
    if config.USE_LR_SCHEDULER:
        print(f"  LR scheduler patience: {config.LR_SCHEDULER_PATIENCE}")
        print(f"  LR scheduler factor:   {config.LR_SCHEDULER_FACTOR}")
    print(f"  Gradient clipping:    {config.USE_GRADIENT_CLIPPING}")
    if config.USE_GRADIENT_CLIPPING:
        print(f"  Max gradient norm:    {config.MAX_GRAD_NORM}")
    
    print("\n[EARLY STOPPING]")
    print(f"  Patience:             {config.EARLY_STOPPING_PATIENCE}")
    print(f"  Min delta:            {config.MIN_DELTA}")
    
    print("\n[DATASET INFO]")
    # Check if dataset exists
    if os.path.exists(config.TRAIN_IMAGE_DIR):
        train_count = len([f for f in os.listdir(config.TRAIN_IMAGE_DIR) 
                          if f.endswith(('.jpg', '.png'))])
        print(f"  Train images:         {train_count}")
    
    if os.path.exists(config.VAL_IMAGE_DIR):
        val_count = len([f for f in os.listdir(config.VAL_IMAGE_DIR) 
                        if f.endswith(('.jpg', '.png'))])
        print(f"  Validation images:    {val_count}")
    
    if os.path.exists(config.TEST_IMAGE_DIR):
        test_count = len([f for f in os.listdir(config.TEST_IMAGE_DIR) 
                         if f.endswith(('.jpg', '.png'))])
        print(f"  Test images:          {test_count}")
    
    print("\n" + "=" * 80)


def main():
    """
    Main function
    """
    args = parse_args()
    
    # Override config with command-line arguments if provided
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    
    # Execute based on mode
    if args.mode == 'info':
        print_config_info()
    
    elif args.mode == 'train':
        from train import train_model
        
        print_config_info()
        print("\n" + "=" * 80)
        print("Starting Training")
        print("=" * 80 + "\n")
        
        try:
            history = train_model(resume_from=args.resume)
            print("\n" + "=" * 80)
            print("Training completed successfully!")
            print("=" * 80)
            
        except KeyboardInterrupt:
            print("\n" + "=" * 80)
            print("Training interrupted by user")
            print("=" * 80)
            sys.exit(0)
        
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"Training failed with error: {str(e)}")
            print("=" * 80)
            raise
    
    elif args.mode == 'evaluate':
        from evaluate import evaluate_model
        
        print_config_info()
        print("\n" + "=" * 80)
        print("Starting Evaluation")
        print("=" * 80 + "\n")
        
        try:
            avg_metrics, std_metrics = evaluate_model(
                checkpoint_path=args.checkpoint,
                split=args.split
            )
            
            print("\n" + "=" * 80)
            print("Evaluation completed successfully!")
            print("=" * 80)
            
        except FileNotFoundError as e:
            print("\n" + "=" * 80)
            print(f"Error: {str(e)}")
            print("Please train the model first or specify a valid checkpoint path.")
            print("=" * 80)
            sys.exit(1)
        
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"Evaluation failed with error: {str(e)}")
            print("=" * 80)
            raise
    
    elif args.mode == 'predict':
        from predict import load_model, predict_on_image, predict_on_folder
        
        if args.input is None:
            print("\n" + "=" * 80)
            print("Error: --input argument is required for prediction mode")
            print("Please provide a path to an image or folder using --input")
            print("=" * 80)
            sys.exit(1)
        
        if args.checkpoint is None:
            print("\n" + "=" * 80)
            print("Error: --checkpoint argument is required for prediction mode")
            print("Please provide a path to a model checkpoint using --checkpoint")
            print("=" * 80)
            sys.exit(1)
        
        if not os.path.exists(args.input):
            print("\n" + "=" * 80)
            print(f"Error: Input path does not exist: {args.input}")
            print("=" * 80)
            sys.exit(1)
        
        if not os.path.exists(args.checkpoint):
            print("\n" + "=" * 80)
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            print("=" * 80)
            sys.exit(1)
        
        # Determine output directory
        if args.output is None:
            checkpoint_filename = os.path.basename(args.checkpoint)
            model_name = os.path.splitext(checkpoint_filename)[0]
            output_dir = os.path.join(config.RESULTS_DIR, f"{model_name}_predictions")
        else:
            output_dir = args.output
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 80)
        print("Starting Prediction")
        print("=" * 80)
        print(f"Input: {args.input}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Output directory: {output_dir}")
        print(f"Threshold: {args.threshold}")
        print("=" * 80 + "\n")
        
        try:
            import numpy as np
            import json
            
            # Load model
            print("Loading model...")
            model = load_model(args.checkpoint, config.DEVICE)
            
            # Check if input is file or folder
            if os.path.isfile(args.input):
                print("\nRunning prediction on single image...")
                stats = predict_on_image(model, args.input, output_dir, config.DEVICE, args.threshold)
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
                all_stats = predict_on_folder(model, args.input, output_dir, config.DEVICE, args.threshold)
                
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
            
            # Save statistics to JSON
            stats_path = os.path.join(output_dir, 'prediction_statistics.json')
            with open(stats_path, 'w') as f:
                json.dump(all_stats, f, indent=4)
            print(f"\nStatistics saved to {stats_path}")
            
            print("\n" + "=" * 80)
            print("Prediction completed successfully!")
            print(f"Results saved to: {output_dir}")
            print("=" * 80)
            
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"Prediction failed with error: {str(e)}")
            print("=" * 80)
            raise


if __name__ == "__main__":
    main()

