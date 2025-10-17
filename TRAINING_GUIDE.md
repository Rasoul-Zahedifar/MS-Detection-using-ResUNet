# Complete Training Guide - Fixed for Class Imbalance

## 🎯 Quick Answer to Your Question

**Why is validation accuracy 0.99 from the beginning?**

Your dataset has **severe class imbalance**:
- 99.63% background pixels
- 0.37% lesion pixels (1:270 ratio)

The model achieves 99% accuracy by simply predicting "no lesion" everywhere - it's not actually learning to detect lesions!

**The Fix:** I've implemented Focal Loss which penalizes the model for missing rare lesions, forcing it to actually learn detection.

---

## ✅ What I've Fixed

### 1. Added Focal Loss
A specialized loss function that:
- Down-weights easy examples (abundant background)
- Up-weights hard examples (rare lesions)
- Forces the model to focus on detecting lesions

### 2. Updated Default Configuration
Changed `config.py` to use `weighted_combined` loss:
- Combines Focal Loss (for class imbalance)
- With Dice Loss (for segmentation quality)

### 3. Added Documentation
- `CLASS_IMBALANCE_GUIDE.md` - Detailed explanation
- `class_imbalance_summary.txt` - Quick reference

---

## 🚀 How to Train Now

### Step 1: Verify GPU Setup (Optional but Recommended)

```bash
# Check if GPU is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# If you have GPU, install GPU version:
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Verify Configuration

```bash
python -c "import config; print(f'Loss: {config.LOSS_TYPE}'); print(f'Device: {config.DEVICE}')"
```

Should show:
```
Loss: weighted_combined
Device: cuda  (or cpu if no GPU)
```

### Step 3: Train the Model

**Recommended: Start with fewer epochs to test:**
```bash
python main.py --mode train --epochs 10
```

**Full training:**
```bash
python main.py --mode train --epochs 50
```

**With GPU and larger batch size:**
```bash
python main.py --mode train --epochs 50 --batch-size 16
```

### Step 4: Monitor the RIGHT Metrics

During training, you'll see:
```
Epoch 5/50
  Train Loss: 0.4234, Train Dice: 0.3521  ← WATCH THIS
  Val Loss: 0.4587, Val Dice: 0.2876      ← AND THIS
  Val IoU: 0.2821, Val Accuracy: 0.9965   ← IGNORE ACCURACY
  Sensitivity: 0.4234, Specificity: 0.9923
```

**What to monitor:**
- ✅ **Dice Coefficient** - Should improve from ~0.1 to >0.6
- ✅ **IoU** - Should improve from ~0.05 to >0.5
- ✅ **Sensitivity** - Should improve from ~0.1 to >0.7
- ❌ **Accuracy** - Will stay at ~99%, ignore it!

### Step 5: Evaluate

After training:
```bash
python main.py --mode evaluate
```

Check the visual results:
```bash
xdg-open results/test_predictions.png
```

Look for:
- ✅ Lesions highlighted in predictions
- ✅ Reasonable match with ground truth
- ❌ All black = model failed (shouldn't happen with Focal Loss)

---

## 📊 Expected Training Progress

### Good Training Pattern:

```
Epoch 1:   Dice: 0.05 → 0.10   (Model starting to detect something)
Epoch 5:   Dice: 0.20 → 0.25   (Learning lesion patterns)
Epoch 10:  Dice: 0.35 → 0.40   (Decent detection)
Epoch 20:  Dice: 0.50 → 0.55   (Good detection)
Epoch 40:  Dice: 0.65 → 0.70   (Excellent!)
```

**Key indicators the fix is working:**
1. Dice coefficient actually INCREASES over time
2. Sensitivity improves (not stuck at ~0)
3. Visual predictions show lesions (not all black)

### Bad Training Pattern (old loss):

```
Epoch 1:   Dice: 0.02   (Predicting mostly zeros)
Epoch 10:  Dice: 0.03   (Still mostly zeros)
Epoch 50:  Dice: 0.05   (Never learned to detect lesions)
```

---

## 🔧 Troubleshooting

### Issue 1: Dice coefficient stays near 0

**Cause:** Loss function might not be strong enough.

**Solution:** Increase Focal Loss focus in `config.py`:
```python
FOCAL_ALPHA = 0.75  # From 0.25 (more weight on lesions)
FOCAL_GAMMA = 3.0   # From 2.0 (more focus on hard examples)
```

### Issue 2: Training is too slow

**With CPU:**
```python
# In config.py:
BATCH_SIZE = 4              # From 8
IMAGE_SIZE = (128, 128)     # From (256, 256)
NUM_EPOCHS = 20             # From 50
```

**With GPU:**
```python
# In config.py:
BATCH_SIZE = 16  # or 32 if enough VRAM
```

### Issue 3: CUDA out of memory

```bash
python main.py --mode train --batch-size 4
```

Or reduce image size in `config.py`:
```python
IMAGE_SIZE = (192, 192)  # From (256, 256)
```

---

## 📈 Performance Expectations

### With the Fix (Focal Loss):

| Metric | Epoch 1 | Epoch 20 | Epoch 50 | Target |
|--------|---------|----------|----------|--------|
| Dice | 0.05 | 0.45 | 0.65 | >0.60 |
| IoU | 0.03 | 0.30 | 0.50 | >0.50 |
| Sensitivity | 0.10 | 0.60 | 0.75 | >0.70 |
| Specificity | 0.99 | 0.98 | 0.97 | >0.95 |

### Training Time:

**CPU:**
- ~11 seconds per batch
- ~30 minutes per epoch
- ~25 hours for 50 epochs

**GPU (RTX 3080):**
- ~1-2 seconds per batch
- ~3-5 minutes per epoch
- ~2-4 hours for 50 epochs

---

## 🎓 Understanding the Metrics

### Dice Coefficient (Most Important!)
- Measures overlap between prediction and ground truth
- **Range:** 0 to 1 (higher is better)
- **Target:** >0.6 for good detection
- **Why:** Not affected by class imbalance

### IoU (Jaccard Index)
- Intersection over Union
- **Range:** 0 to 1 (higher is better)
- **Target:** >0.5 for reasonable detection
- **Why:** Standard metric for segmentation

### Sensitivity (Recall)
- Percentage of actual lesions detected
- **Range:** 0 to 1 (higher is better)
- **Target:** >0.7 (finding 70%+ of lesions)
- **Why:** Critical for medical diagnosis

### Specificity
- Percentage of background correctly identified
- **Range:** 0 to 1 (higher is better)
- **Target:** >0.95 (not over-predicting)
- **Why:** Avoids false alarms

### ❌ Pixel Accuracy (MISLEADING!)
- With 99.6% background, always ~99%
- **Ignore this metric!**
- **Why:** Meaningless with class imbalance

---

## 📝 Complete Training Commands

### Quick Test (10 epochs, fast):
```bash
python main.py --mode train --epochs 10 --batch-size 8
```

### Full Training (CPU):
```bash
python main.py --mode train --epochs 50
```

### Full Training (GPU):
```bash
python main.py --mode train --epochs 50 --batch-size 16
```

### Resume if Interrupted:
```bash
python main.py --mode train --resume checkpoints/checkpoint_epoch_20.pth
```

### Evaluate:
```bash
python main.py --mode evaluate
```

### View Results:
```bash
cat results/test_results.json
xdg-open results/test_predictions.png
xdg-open results/training_history.png
```

---

## ✨ Summary

1. **Problem Identified:** Severe class imbalance (1:270 ratio)
2. **Root Cause:** Model can be "99% accurate" by predicting all zeros
3. **Solution Implemented:** Focal Loss + Dice Loss (weighted_combined)
4. **How to Verify:** Watch Dice coefficient improve (should reach >0.6)
5. **Ready to Train:** Just run `python main.py --mode train`

**The fix is already in place - just start training and monitor Dice coefficient instead of accuracy!**

---

## 📚 Further Reading

- `CLASS_IMBALANCE_GUIDE.md` - Detailed technical explanation
- `class_imbalance_summary.txt` - Quick reference
- `README.md` - General project documentation
- `TEST_RESULTS.md` - Code verification results

Good luck with training! 🚀

