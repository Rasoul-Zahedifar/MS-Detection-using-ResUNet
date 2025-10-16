"""
Main entry point for MS Detection using ResUNet
Provides command-line interface for training, evaluation, and inference
"""
import argparse
import os
import sys

import config
from train import train_model
from evaluate import evaluate_model


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
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'info'],
        required=True,
        help='Mode to run: train, evaluate, or info'
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


if __name__ == "__main__":
    main()

