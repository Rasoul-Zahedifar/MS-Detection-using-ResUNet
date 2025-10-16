# Quick Setup Guide

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirement.txt
```

Or install with specific versions:
```bash
pip install torch torchvision numpy Pillow matplotlib tqdm scikit-learn
```

### 2. Verify Installation

Check if everything is set up correctly:
```bash
python main.py --mode info
```

This will display:
- Configuration parameters
- Dataset statistics
- Model architecture details

### 3. Quick Start Training

**Basic training (default parameters):**
```bash
python main.py --mode train
```

**Custom training:**
```bash
python main.py --mode train --batch-size 16 --epochs 100 --lr 0.0001
```

### 4. Evaluation

After training, evaluate the model:
```bash
python main.py --mode evaluate
```

## Expected Output

### Training
- Checkpoints saved in: `checkpoints/`
- Training history plot: `results/training_history.png`
- Best model: `checkpoints/best_model.pth`

### Evaluation
- Test metrics: `results/test_results.json`
- Prediction visualizations: `results/test_predictions.png`

## Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce batch size in config.py or use command line:
```bash
python main.py --mode train --batch-size 4
```

### Issue: Dataset not found
**Solution:** Verify your dataset structure:
```
dataset/
├── train/
│   ├── images/
│   └── masks/
├── valid/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### Issue: Module not found
**Solution:** Install missing dependencies:
```bash
pip install -r requirement.txt
```

## Testing Individual Components

**Test model architecture:**
```bash
python ResUNet_model.py
```

**Test data loading:**
```bash
python fetch_data.py
```

**Test dataset preparation:**
```bash
python normalize_data.py
```

**Test utilities:**
```bash
python utils.py
```

## Configuration

All configuration is in `config.py`. Key parameters:

- `BATCH_SIZE`: 8 (reduce if GPU memory is limited)
- `NUM_EPOCHS`: 50
- `LEARNING_RATE`: 1e-4
- `IMAGE_SIZE`: (256, 256)
- `LOSS_TYPE`: 'combined' (BCE + Dice)

Modify these as needed for your hardware and dataset.

## Need Help?

1. Check the main README.md for detailed documentation
2. Run `python main.py --mode info` to see current configuration
3. Check training logs in the terminal output
4. Review saved results in `results/` directory

