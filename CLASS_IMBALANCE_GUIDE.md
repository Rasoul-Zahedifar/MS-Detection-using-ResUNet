# Understanding and Fixing Class Imbalance in MS Detection

## 🔍 The Problem

Your dataset has **severe class imbalance**:

```
Background pixels: 99.63%
Lesion pixels:      0.37%
Ratio:              1:270
```

This means for every 1 lesion pixel, there are 270 background pixels!

### Why This Causes Issues

1. **High Accuracy is Misleading**: The model achieves 99% accuracy by simply predicting "no lesion" everywhere
2. **Model Learns Nothing**: It never actually learns to detect lesions
3. **Validation Loss Decreases**: Loss can decrease while the model predicts all zeros
4. **False Sense of Progress**: Metrics look good but the model is useless

### Visual Example

```
Ground Truth:  [0,0,0,0,0,0,1,0,0,0]  (10% lesions)
Bad Prediction: [0,0,0,0,0,0,0,0,0,0]  (0% lesions)
Accuracy: 90%! ✓ (but missed the lesion!)

Good Prediction: [0,0,0,0,0,0,1,0,0,0]  (10% lesions)  
Accuracy: 100% ✓ (and found the lesion!)
```

---

## ✅ Solutions Implemented

### 1. **Focal Loss** (NEW!)

Focal Loss addresses class imbalance by:
- **Down-weighting easy examples** (abundant background pixels)
- **Focusing on hard examples** (rare lesion pixels)

Formula: `FL = -α(1-p)^γ * log(p)`
- `α = 0.25`: More weight on positive class (lesions)
- `γ = 2.0`: Focus on hard examples

### 2. **Weighted Combined Loss** (RECOMMENDED)

Combines:
- **Focal Loss**: Handles class imbalance
- **Dice Loss**: Focuses on overlap/segmentation quality

This is now the default in your `config.py`!

### 3. **Better Metrics to Monitor**

Instead of just accuracy, focus on:

| Metric | What It Measures | Good Value |
|--------|-----------------|------------|
| **Dice Coefficient** | Overlap between prediction and ground truth | > 0.7 |
| **IoU (Jaccard)** | Intersection over Union | > 0.6 |
| **Sensitivity (Recall)** | % of actual lesions detected | > 0.8 |
| **Specificity** | % of background correctly identified | > 0.95 |
| **Pixel Accuracy** | ⚠️ Misleading with imbalance | Ignore |

---

## 🎯 How to Use

### Option 1: Use New Default (Recommended)

The config is already updated! Just train normally:

```bash
python main.py --mode train
```

This now uses **Focal Loss + Dice Loss** which handles class imbalance much better.

### Option 2: Try Different Loss Functions

You can experiment with different loss functions in `config.py`:

```python
# In config.py, change LOSS_TYPE to one of:

LOSS_TYPE = 'weighted_combined'  # ✅ BEST for class imbalance (Focal + Dice)
LOSS_TYPE = 'dice'               # Good for segmentation
LOSS_TYPE = 'focal'              # Good for imbalance
LOSS_TYPE = 'combined'           # Original (BCE + Dice)
LOSS_TYPE = 'bce'                # ❌ Bad for imbalance
```

### Option 3: Adjust Focal Loss Parameters

Fine-tune the focal loss in `config.py`:

```python
# More aggressive focus on lesions
FOCAL_ALPHA = 0.75  # From 0.25 (higher = more weight on lesions)
FOCAL_GAMMA = 3.0   # From 2.0 (higher = more focus on hard examples)
```

---

## 📊 What to Monitor During Training

### ❌ DON'T Focus On:
- **Pixel Accuracy** (will be ~99% even if model is terrible)

### ✅ DO Focus On:
1. **Dice Coefficient** (should improve from ~0 to >0.5)
2. **Sensitivity** (should improve from ~0 to >0.7)
3. **IoU** (should improve from ~0 to >0.5)
4. **Visual Inspection** (check `results/training_history.png`)

### Good Training Progression

```
Epoch 1:
  Train Dice: 0.05  (barely detecting anything)
  Val Dice: 0.03

Epoch 10:
  Train Dice: 0.35  (starting to learn)
  Val Dice: 0.28

Epoch 30:
  Train Dice: 0.65  (good detection)
  Val Dice: 0.58

Epoch 50:
  Train Dice: 0.75  (excellent!)
  Val Dice: 0.68
```

---

## 🔬 Verify the Fix

Run this to check if the loss is working:

```bash
python -c "
import torch
from utils import WeightedCombinedLoss

# Simulated data with class imbalance
# Scenario 1: Model predicts all zeros (bad)
predictions_bad = torch.zeros(1, 1, 256, 256)
targets = torch.rand(1, 1, 256, 256) > 0.99  # 1% lesions
targets = targets.float()

# Scenario 2: Model predicts something (better)
predictions_good = torch.rand(1, 1, 256, 256) * 0.5

loss_fn = WeightedCombinedLoss()

loss_bad = loss_fn(predictions_bad, targets)
loss_good = loss_fn(predictions_good, targets)

print(f'Loss (all zeros): {loss_bad.item():.4f}')
print(f'Loss (some detection): {loss_good.item():.4f}')
print(f'Improvement: {(loss_bad - loss_good).item():.4f}')
"
```

---

## 🚀 Training Recommendations

### 1. **Start Fresh** (Recommended)

If you've already trained with the old loss, start over:

```bash
# Backup old checkpoints
mv checkpoints checkpoints_old
mv results results_old

# Train with new loss function
python main.py --mode train
```

### 2. **Use More Epochs**

With Focal Loss, the model needs more time to learn:

```bash
python main.py --mode train --epochs 100
```

### 3. **Monitor Dice, Not Accuracy**

Look at the training output:
```
Epoch 20/100
  Train Loss: 0.4234, Train Dice: 0.5521  ← Watch this!
  Val Loss: 0.4587, Val Dice: 0.4876      ← And this!
  Val IoU: 0.3821, Val Accuracy: 0.9965   ← Ignore accuracy
```

### 4. **Check Visual Results**

After training:
```bash
python main.py --mode evaluate
xdg-open results/test_predictions.png
```

Look for:
- ✅ Lesions are highlighted in predictions
- ✅ Predictions match ground truth reasonably well
- ❌ All black predictions = model failed

---

## 📈 Expected Results

### With Old Loss (BCE + Dice):
```
Accuracy: 99.6%    ✓ (but meaningless)
Dice: 0.05         ✗ (model predicts mostly zeros)
Sensitivity: 0.02  ✗ (misses 98% of lesions)
```

### With New Loss (Focal + Dice):
```
Accuracy: 99.5%    ~ (still high but less meaningful)
Dice: 0.65         ✓ (good overlap with ground truth)
Sensitivity: 0.75  ✓ (finds 75% of lesions)
IoU: 0.52          ✓ (reasonable detection)
```

---

## 🛠️ Advanced Techniques (Optional)

If Focal Loss still doesn't work well enough, try:

### 1. **Weighted Sampling**

Sample batches to include more lesion-heavy images.

### 2. **Data Augmentation**

Add more aggressive augmentation for rare lesions.

### 3. **Two-Stage Training**

1. Train on lesion-heavy images first
2. Fine-tune on full dataset

### 4. **Post-Processing**

Apply morphological operations to clean up predictions.

---

## 📝 Summary

| Problem | Solution | Status |
|---------|----------|--------|
| Class imbalance (1:270) | Focal Loss | ✅ Implemented |
| Misleading accuracy | Use Dice/IoU instead | ✅ Already tracked |
| Model predicts all zeros | Weighted Combined Loss | ✅ Now default |

**You're ready to train with the improved loss function!**

```bash
python main.py --mode train --epochs 50
```

Monitor **Dice coefficient** instead of accuracy, and you should see real improvements!

