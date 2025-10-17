#!/bin/bash

# Script to retrain the model with fixed class imbalance parameters
# Run this after the configuration has been updated

echo "=========================================="
echo "MS Detection - Retrain with Fixed Config"
echo "=========================================="
echo ""

# Check if old checkpoints exist
if [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
    echo "⚠️  Old checkpoints found!"
    echo ""
    read -p "Do you want to backup and clean them? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Backup old checkpoints
        timestamp=$(date +%Y%m%d_%H%M%S)
        echo "📦 Backing up to checkpoints_old_${timestamp}..."
        mv checkpoints "checkpoints_old_${timestamp}"
        mkdir checkpoints
        
        if [ -d "results" ]; then
            echo "📦 Backing up results to results_old_${timestamp}..."
            mv results "results_old_${timestamp}"
            mkdir results
        fi
        echo "✅ Backup complete!"
    else
        echo "⚠️  Continuing with existing checkpoints..."
    fi
    echo ""
fi

# Show current configuration
echo "Current Configuration:"
python -c "
import config
print(f'  Loss Type: {config.LOSS_TYPE}')
print(f'  Focal Alpha: {config.FOCAL_ALPHA}')
print(f'  Focal Gamma: {config.FOCAL_GAMMA}')
print(f'  Epochs: {config.NUM_EPOCHS}')
print(f'  Batch Size: {config.BATCH_SIZE}')
"
echo ""

# Ask for number of epochs
read -p "Number of epochs (default=100): " epochs
epochs=${epochs:-100}
echo ""

echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""
echo "💡 What to watch:"
echo "   ✅ Dice Coefficient (should increase to >0.5)"
echo "   ✅ IoU (should increase to >0.4)"
echo "   ✅ Sensitivity (should increase to >0.7)"
echo "   ❌ Accuracy (ignore - will stay ~99%)"
echo ""
echo "Press Ctrl+C to stop training at any time"
echo ""
sleep 3

# Start training
python main.py --mode train --epochs $epochs

# Show results
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training Complete!"
    echo "=========================================="
    echo ""
    echo "📊 Check results:"
    echo "   - Training plots: results/training_history.png"
    echo "   - Best model: checkpoints/best_model.pth"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Evaluate: python main.py --mode evaluate"
    echo "   2. Check visual results: xdg-open results/test_predictions.png"
    echo ""
else
    echo ""
    echo "❌ Training failed! Check the error messages above."
    echo ""
fi

