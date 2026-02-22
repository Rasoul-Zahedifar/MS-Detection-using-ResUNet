#!/usr/bin/env python3
"""
Pre-Training Validation Script
Checks if your MS Detection setup is correct before starting training
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_check(passed, message):
    symbol = f"{Colors.GREEN}✓{Colors.END}" if passed else f"{Colors.RED}✗{Colors.END}"
    print(f"  {symbol} {message}")


def print_warning(message):
    print(f"  {Colors.YELLOW}⚠{Colors.END}  {message}")


def print_info(message):
    print(f"  {Colors.BLUE}ℹ{Colors.END}  {message}")


def check_imports():
    """Check if all required packages are importable"""
    print_header("CHECKING IMPORTS")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow',
        'tqdm': 'tqdm'
    }
    
    all_passed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print_check(True, f"{name} installed")
        except ImportError:
            print_check(False, f"{name} NOT installed")
            all_passed = False
    
    return all_passed


def check_config():
    """Check configuration settings"""
    print_header("CHECKING CONFIGURATION")
    
    try:
        import config
        print_check(True, "config.py loaded")
    except Exception as e:
        print_check(False, f"Failed to load config.py: {e}")
        return False
    
    # Check critical settings
    issues = []
    warnings = []
    
    # Check device
    if hasattr(config, 'DEVICE'):
        device_type = str(config.DEVICE)
        print_info(f"Device: {device_type}")
        if 'cuda' in device_type and not torch.cuda.is_available():
            warnings.append("CUDA device specified but not available")
    
    # Check loss type
    if hasattr(config, 'LOSS_TYPE'):
        loss_type = config.LOSS_TYPE
        print_info(f"Loss type: {loss_type}")
        if loss_type not in ['bce', 'dice', 'focal', 'combined', 'weighted_combined']:
            issues.append(f"Unknown loss type: {loss_type}")
        elif loss_type == 'bce':
            warnings.append("Using pure BCE loss - not recommended for class imbalance")
    
    # Check for class imbalance handling
    if hasattr(config, 'USE_PATCH_TRAINING'):
        if config.USE_PATCH_TRAINING:
            print_check(True, "Patch training enabled (good for class imbalance)")
        else:
            warnings.append("Patch training disabled - may struggle with class imbalance")
    
    if hasattr(config, 'LOSS_TYPE'):
        if config.LOSS_TYPE in ['focal', 'weighted_combined']:
            print_check(True, f"Using {config.LOSS_TYPE} loss (good for class imbalance)")
        else:
            warnings.append(f"Using {config.LOSS_TYPE} - consider focal/weighted_combined for severe imbalance")
    
    # Check focal loss parameters
    if hasattr(config, 'FOCAL_ALPHA') and hasattr(config, 'FOCAL_GAMMA'):
        alpha = config.FOCAL_ALPHA
        gamma = config.FOCAL_GAMMA
        print_info(f"Focal loss: alpha={alpha}, gamma={gamma}")
        
        if alpha < 0.5:
            warnings.append(f"FOCAL_ALPHA={alpha} is low for 1:246 imbalance (try 0.75+)")
        if gamma < 2.5:
            warnings.append(f"FOCAL_GAMMA={gamma} is low for hard examples (try 3.0+)")
    
    # Check gradient accumulation
    if hasattr(config, 'GRADIENT_ACCUMULATION_STEPS'):
        steps = config.GRADIENT_ACCUMULATION_STEPS
        batch_size = config.BATCH_SIZE if hasattr(config, 'BATCH_SIZE') else 8
        effective_bs = batch_size * steps
        print_info(f"Gradient accumulation: {steps} steps (effective batch size: {effective_bs})")
    
    # Print warnings and issues
    for warning in warnings:
        print_warning(warning)
    
    for issue in issues:
        print_check(False, issue)
    
    return len(issues) == 0


def check_model():
    """Check if model can be instantiated"""
    print_header("CHECKING MODEL")
    
    try:
        import config
        from ResUNet_model import ResUNet, count_parameters
        
        model = ResUNet(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            filters=config.FILTERS
        )
        
        print_check(True, "Model instantiated successfully")
        
        param_count = count_parameters(model)
        print_info(f"Model parameters: {param_count:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, config.IN_CHANNELS, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        
        print_check(True, f"Forward pass successful: {dummy_input.shape} → {output.shape}")
        
        # Check output range (should be logits, not probabilities)
        out_min = output.min().item()
        out_max = output.max().item()
        
        if -10 < out_min < 10 and -10 < out_max < 10:
            print_check(True, f"Output looks like logits: [{out_min:.2f}, {out_max:.2f}]")
        elif 0 <= out_min and out_max <= 1:
            print_check(False, "Output looks like probabilities! Model should output logits!")
            return False
        else:
            print_warning(f"Unusual output range: [{out_min:.2f}, {out_max:.2f}]")
        
        return True
        
    except Exception as e:
        print_check(False, f"Model check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_loss_functions():
    """Check if loss functions work correctly"""
    print_header("CHECKING LOSS FUNCTIONS")
    
    try:
        from utils import get_loss_function
        import config
        
        # Test loss function
        loss_fn = get_loss_function(config.LOSS_TYPE)
        print_check(True, f"Loss function '{config.LOSS_TYPE}' created")
        
        # Create dummy data (logits and targets)
        # IMPORTANT: Set requires_grad=True to test gradient flow
        logits = torch.randn(2, 1, 64, 64, requires_grad=True)  # Enable gradients
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        # Compute loss
        loss = loss_fn(logits, targets)
        
        print_check(True, f"Loss computation successful: {loss.item():.4f}")
        
        # Check if loss is reasonable
        if torch.isnan(loss) or torch.isinf(loss):
            print_check(False, "Loss is NaN or Inf!")
            return False
        
        if loss.item() < 0:
            print_check(False, f"Loss is negative: {loss.item()}")
            return False
        
        # Check if loss requires gradients (should work now with requires_grad=True input)
        if not loss.requires_grad:
            print_warning("Loss does not require gradients (but this is OK - model outputs will)")
        else:
            print_check(True, "Loss requires gradients correctly")
        
        # Test backward pass
        try:
            loss.backward()
            print_check(True, "Backward pass successful (gradients flow correctly)")
        except Exception as e:
            print_check(False, f"Backward pass failed: {e}")
            return False
        
        print_check(True, "Loss function working correctly")
        return True
        
    except Exception as e:
        print_check(False, f"Loss function check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_metrics():
    """Check if metric functions work correctly"""
    print_header("CHECKING METRICS")
    
    try:
        from utils import calculate_metrics
        
        # Create dummy data
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        # Calculate metrics
        metrics = calculate_metrics(logits, targets)
        
        print_check(True, "Metrics computation successful")
        
        # Check metric values
        required_metrics = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
        for metric in required_metrics:
            if metric not in metrics:
                print_check(False, f"Missing metric: {metric}")
                return False
            
            value = metrics[metric]
            if not (0 <= value <= 1):
                print_check(False, f"{metric} out of range: {value}")
                return False
        
        print_info(f"Sample metrics: Dice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}")
        print_check(True, "All metrics in valid range [0, 1]")
        
        return True
        
    except Exception as e:
        print_check(False, f"Metrics check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_directories():
    """Check if data directories exist"""
    print_header("CHECKING DATA DIRECTORIES")
    
    try:
        import config
        
        directories = {
            'Train Images': config.TRAIN_IMAGE_DIR,
            'Train Masks': config.TRAIN_MASK_DIR,
            'Val Images': config.VAL_IMAGE_DIR,
            'Val Masks': config.VAL_MASK_DIR,
            'Test Images': config.TEST_IMAGE_DIR,
            'Test Masks': config.TEST_MASK_DIR,
        }
        
        all_exist = True
        for name, path in directories.items():
            exists = os.path.exists(path)
            print_check(exists, f"{name}: {path}")
            if not exists:
                all_exist = False
            elif exists:
                # Count files
                files = list(Path(path).glob('*'))
                image_files = [f for f in files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
                print_info(f"  Found {len(image_files)} images")
        
        if not all_exist:
            print_warning("Some data directories are missing!")
            print_warning("Please ensure your dataset is properly organized")
        
        return all_exist
        
    except Exception as e:
        print_check(False, f"Data directory check failed: {e}")
        return False


def check_sliding_window():
    """Check sliding window inference"""
    print_header("CHECKING SLIDING WINDOW INFERENCE")
    
    try:
        import config
        from sliding_window_inference import sliding_window_inference
        from ResUNet_model import ResUNet
        
        # Create dummy model
        model = ResUNet(
            in_channels=1,
            out_channels=1,
            filters=[64, 128, 256, 512]
        )
        model.eval()
        
        # Test with different image sizes
        test_image = torch.randn(1, 1, 512, 512)
        
        with torch.no_grad():
            prediction = sliding_window_inference(
                model=model,
                image=test_image,
                patch_size=(256, 256),
                stride=(128, 128)
            )
        
        print_check(True, "Sliding window inference working")
        print_info(f"Input: {test_image.shape} → Output: {prediction.shape}")
        
        if test_image.shape[2:] != prediction.shape[2:]:
            print_check(False, "Output shape mismatch!")
            return False
        
        print_check(True, "Output shape matches input")
        
        # Check if using patch training
        if hasattr(config, 'USE_PATCH_TRAINING') and config.USE_PATCH_TRAINING:
            print_check(True, "Patch training enabled - sliding window will be used in evaluation")
        else:
            print_info("Patch training disabled - standard evaluation will be used")
        
        return True
        
    except Exception as e:
        print_check(False, f"Sliding window check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks"""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{'MS DETECTION - PRE-TRAINING VALIDATION':^70}{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}")
    
    checks = [
        ("Imports", check_imports),
        ("Configuration", check_config),
        ("Model", check_model),
        ("Loss Functions", check_loss_functions),
        ("Metrics", check_metrics),
        ("Data Directories", check_data_directories),
        ("Sliding Window", check_sliding_window),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_check(False, f"Check '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        print_check(result, name)
    
    print(f"\n{Colors.BLUE}Results: {passed}/{total} checks passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}✓ ALL CHECKS PASSED!{Colors.END}")
        print(f"{Colors.GREEN}  You're ready to start training.{Colors.END}")
        print(f"\n  Run: {Colors.BLUE}python main.py --mode train{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.RED}✗ SOME CHECKS FAILED{Colors.END}")
        print(f"{Colors.RED}  Please fix the issues above before training.{Colors.END}")
        print(f"\n  See CRITICAL_FIXES_GUIDE.py for help.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
