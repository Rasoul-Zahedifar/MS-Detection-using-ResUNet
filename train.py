"""
Training script for MS Detection using ResUNet
Handles model training with validation and checkpoint saving
"""
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import sys
import time

# Add current directory to path for imports (works in both Colab and local)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from ResUNet_model import ResUNet, count_parameters
from fetch_data import MSDataFetcher
from utils import (
    get_loss_function, 
    calculate_metrics, 
    save_checkpoint,
    load_checkpoint,
    set_seed,
    EarlyStopping,
    plot_training_history
)


class Trainer:
    """
    Trainer class for MS Detection model
    """
    
    def __init__(self, model, train_loader, val_loader, optimizer, 
                 criterion, device, config_dict=None):
        """
        Initialize trainer
        
        Args:
            model (nn.Module): ResUNet model
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            optimizer (torch.optim.Optimizer): Optimizer
            criterion: Loss function
            device (torch.device): Device to train on
            config_dict (dict): Configuration dictionary (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rate': []
        }
        
        # Learning rate scheduler
        if config.USE_LR_SCHEDULER:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.LR_SCHEDULER_FACTOR,
                patience=config.LR_SCHEDULER_PATIENCE
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.MIN_DELTA,
            verbose=True
        )
    
    def train_epoch(self):
        """
        Train for one epoch
        
        Returns:
            tuple: (average_loss, average_dice)
        """
        self.model.train()
        
        running_loss = 0.0
        running_dice = 0.0
        
        # Gradient accumulation steps
        accumulation_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        
        # Initialize gradients to zero at start of epoch
        self.optimizer.zero_grad()
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, masks)
            
            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights only every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if config.USE_GRADIENT_CLIPPING:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        config.MAX_GRAD_NORM
                    )
                
                # Update weights
                self.optimizer.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = calculate_metrics(outputs, masks)
            
            # Update statistics (use unnormalized loss for display)
            running_loss += loss.item() * accumulation_steps
            running_dice += metrics['dice']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'dice': f"{metrics['dice']:.4f}",
                'eff_bs': f"{config.BATCH_SIZE * accumulation_steps}"
            })
        
        # CRITICAL: Handle leftover accumulated gradients at end of epoch
        # If total batches is not divisible by accumulation_steps,
        # we need to apply the remaining gradients
        if len(self.train_loader) % accumulation_steps != 0:
            if config.USE_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    config.MAX_GRAD_NORM
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Calculate averages
        avg_loss = running_loss / len(self.train_loader)
        avg_dice = running_dice / len(self.train_loader)
        
        return avg_loss, avg_dice
    
    def validate(self):
        """
        Validate the model
        
        Returns:
            tuple: (average_loss, average_dice, all_metrics)
        """
        self.model.eval()
        
        running_loss = 0.0
        running_metrics = {
            'dice': 0.0,
            'iou': 0.0,
            'accuracy': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0
        }
        
        # Track prediction statistics to detect model collapse
        total_positive_preds = 0
        total_pixels = 0
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(pbar):
                # Move data to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # MONITORING: Check prediction distribution (first few epochs)
                if self.current_epoch < 3 and batch_idx == 0:
                    with torch.no_grad():
                        probs = torch.sigmoid(outputs)
                        print(f"\n[Sanity Check - Epoch {self.current_epoch + 1}]")
                        print(f"  Output logits range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
                        print(f"  Output probs range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")
                        print(f"  Positive prediction rate (@0.5): {(probs > 0.5).float().mean().item():.4f}")
                        print(f"  Ground truth positive rate: {masks.mean().item():.4f}")
                
                # Track predictions for collapse detection
                probs = torch.sigmoid(outputs)
                total_positive_preds += (probs > 0.5).sum().item()
                total_pixels += outputs.numel()
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                metrics = calculate_metrics(outputs, masks)
                
                # Update statistics
                running_loss += loss.item()
                for key in running_metrics:
                    running_metrics[key] += metrics[key]
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{metrics['dice']:.4f}"
                })
        
        # Calculate averages
        avg_loss = running_loss / len(self.val_loader)
        avg_metrics = {key: value / len(self.val_loader) 
                      for key, value in running_metrics.items()}
        
        # CRITICAL: Check for model collapse (predicting all zeros)
        positive_pred_rate = total_positive_preds / total_pixels
        if positive_pred_rate < 0.001:  # Less than 0.1% positive predictions
            print(f"\n{'='*60}")
            print(f"⚠️  WARNING: POTENTIAL MODEL COLLAPSE DETECTED!")
            print(f"{'='*60}")
            print(f"  Positive prediction rate: {positive_pred_rate:.6f} ({positive_pred_rate*100:.4f}%)")
            print(f"  The model is predicting mostly background (zeros).")
            print(f"  This may indicate:")
            print(f"    - Class imbalance is too severe")
            print(f"    - Learning rate is too low")
            print(f"    - Loss function weights need adjustment")
            print(f"  Recommendations:")
            print(f"    - Increase FOCAL_ALPHA (current: {config.FOCAL_ALPHA})")
            print(f"    - Increase FOCAL_GAMMA (current: {config.FOCAL_GAMMA})")
            print(f"    - Increase learning rate")
            print(f"    - Use more aggressive patch sampling")
            print(f"{'='*60}\n")
        
        return avg_loss, avg_metrics['dice'], avg_metrics
    
    def train(self, num_epochs):
        """
        Train the model for multiple epochs
        
        Args:
            num_epochs (int): Number of epochs to train
        """
        print("=" * 60)
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss, train_dice = self.train_epoch()
            
            # Validate
            val_loss, val_dice, val_metrics = self.validate()
            if val_loss < self.best_val_loss - 1e-6:
                self.best_val_loss = val_loss
                save_checkpoint(self.model, self.optimizer, epoch + 1, val_loss, val_metrics,
                                os.path.join(config.CHECKPOINT_DIR, 'best_by_loss.pth'))
                print(f"  >> Saved best_by_loss model (ValLoss: {val_loss:.4f})")

            # Save best by DICE (higher is better now)
            if val_dice > self.best_val_dice + 1e-6:
                self.best_val_dice = val_dice
                save_checkpoint(self.model, self.optimizer, epoch + 1, val_loss, val_metrics,
                                os.path.join(config.CHECKPOINT_DIR, 'best_by_dice.pth'))
                print(f"  >> Saved best_by_dice model (Dice: {val_dice:.4f})")
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            print(f"  Val IoU: {val_metrics['iou']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Sensitivity: {val_metrics['sensitivity']:.4f}, Specificity: {val_metrics['specificity']:.4f}")
            
            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save best model
            if config.SAVE_BEST_MODEL:
                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    self.best_val_loss = val_loss
                    
                    best_model_path = os.path.join(
                        config.CHECKPOINT_DIR, 
                        'best_model.pth'
                    )
                    save_checkpoint(
                        self.model, 
                        self.optimizer, 
                        epoch + 1,
                        val_loss, 
                        val_metrics,
                        best_model_path
                    )
                    print(f"  >> Saved best model (Dice: {val_dice:.4f})")
            
            # Save checkpoint periodically
            if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
                checkpoint_path = os.path.join(
                    config.CHECKPOINT_DIR,
                    f'checkpoint_epoch_{epoch + 1}.pth'
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    val_loss,
                    val_metrics,
                    checkpoint_path
                )
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
            
            print("-" * 60)
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("=" * 60)
        print(f"Training completed in {elapsed_time / 60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation Dice: {self.best_val_dice:.4f}")
        print("=" * 60)
        
        # Save final model
        final_model_path = os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
        save_checkpoint(
            self.model,
            self.optimizer,
            num_epochs,
            val_loss,
            val_metrics,
            final_model_path
        )
        
        # Plot training history
        history_plot_path = os.path.join(config.RESULTS_DIR, 'training_history.png')
        plot_training_history(self.history, history_plot_path)
        
        return self.history


def train_model(resume_from=None):
    """
    Main training function
    
    Args:
        resume_from (str): Path to checkpoint to resume from (optional)
    
    Returns:
        dict: Training history
    """
    # Set random seed for reproducibility
    set_seed(config.RANDOM_SEED)
    
    # Create data loaders
    print("Loading data...")
    data_fetcher = MSDataFetcher(
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        use_augmentation=config.USE_AUGMENTATION,
        use_patch_training=config.USE_PATCH_TRAINING,
        use_class_sampling=config.USE_CLASS_SAMPLING
    )
    
    train_loader = data_fetcher.get_loader('train')
    val_loader = data_fetcher.get_loader('val')
    
    # Create model
    print("\nInitializing model...")
    model = ResUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        filters=config.FILTERS
    ).to(config.DEVICE)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create loss function
    criterion = get_loss_function(config.LOSS_TYPE)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from checkpoint: {resume_from}")
        model, optimizer, start_epoch, _, _ = load_checkpoint(
            model, optimizer, resume_from, config.DEVICE
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=config.DEVICE
    )
    
    # Train the model
    history = trainer.train(config.NUM_EPOCHS - start_epoch)
    
    return history


# ===========================
# Usage Example
# ===========================
if __name__ == "__main__":
    # Train the model
    history = train_model()
    
    print("\nTraining completed successfully!")
