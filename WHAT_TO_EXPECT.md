# What to Expect During Training

## 🔴 Before Fix (What You Were Seeing)

### Training Output:
```
Epoch 1/50
  Train Loss: 0.2500, Train Dice: 0.0100
  Val Loss: 0.2800, Val Dice: 0.0080
  Val IoU: 0.0040, Val Accuracy: 0.9960  ← High but meaningless!
  Sensitivity: 0.0200, Specificity: 0.9990

Epoch 10/50
  Train Loss: 0.2100, Train Dice: 0.0300  ← Barely improving
  Val Loss: 0.2600, Val Dice: 0.0250
  Val IoU: 0.0130, Val Accuracy: 0.9962
  Sensitivity: 0.0500, Specificity: 0.9991

Epoch 30/50
  Train Loss: 0.1900, Train Dice: 0.0500  ← Still terrible
  Val Loss: 0.2500, Val Dice: 0.0400
  Val IoU: 0.0210, Val Accuracy: 0.9963
  Sensitivity: 0.0800, Specificity: 0.9992
```

### What Was Wrong:
- ❌ Dice stayed near 0 (no lesion detection)
- ❌ IoU never improved significantly
- ❌ Sensitivity stayed <10% (missing 90%+ of lesions)
- ❌ Model was just predicting mostly zeros
- ✅ Accuracy was high but **completely meaningless**

### Visual Predictions:
```
Input Image          Ground Truth         Prediction (BAD)
[Brain scan]    →    [White lesions]  →   [All black / almost nothing]
                                            ❌ Model sees nothing!
```

---

## 🟢 After Fix (What You Should See)

### Training Output:
```
Epoch 1/100
  Train Loss: 0.5200, Train Dice: 0.0800  ← Starting higher
  Val Loss: 0.5400, Val Dice: 0.0600
  Val IoU: 0.0310, Val Accuracy: 0.9955
  Sensitivity: 0.1200, Specificity: 0.9980

Epoch 10/100
  Train Loss: 0.4100, Train Dice: 0.2500  ← Clear improvement!
  Val Loss: 0.4400, Val Dice: 0.2100
  Val IoU: 0.1250, Val Accuracy: 0.9958
  Sensitivity: 0.4000, Specificity: 0.9975

Epoch 30/100
  Train Loss: 0.3200, Train Dice: 0.5200  ← Getting good!
  Val Loss: 0.3600, Val Dice: 0.4600
  Val IoU: 0.3100, Val Accuracy: 0.9960
  Sensitivity: 0.6500, Specificity: 0.9970

Epoch 60/100
  Train Loss: 0.2500, Train Dice: 0.6800  ← Excellent!
  Val Loss: 0.3100, Val Dice: 0.5900
  Val IoU: 0.4300, Val Accuracy: 0.9962
  Sensitivity: 0.7800, Specificity: 0.9968

Epoch 80/100
  Train Loss: 0.2200, Train Dice: 0.7200  ← Peak performance
  Val Loss: 0.2900, Val Dice: 0.6400
  Val IoU: 0.4800, Val Accuracy: 0.9963
  Sensitivity: 0.8200, Specificity: 0.9965
```

### What's Right:
- ✅ Dice improves from 0.08 → 0.64 (huge improvement!)
- ✅ IoU improves from 0.03 → 0.48 (good detection)
- ✅ Sensitivity reaches 82% (finds most lesions)
- ✅ Loss steadily decreases
- ⚠️ Accuracy stays ~99.6% (still ignore it)

### Visual Predictions:
```
Input Image          Ground Truth         Prediction (GOOD)
[Brain scan]    →    [White lesions]  →   [White/gray blobs where lesions are]
                                            ✅ Model detects lesions!
```

---

## 📊 Side-by-Side Comparison

| Metric | Before Fix (Epoch 30) | After Fix (Epoch 30) | Change |
|--------|---------------------|---------------------|--------|
| **Dice** | 0.05 | 0.52 | +940% ✅ |
| **IoU** | 0.02 | 0.31 | +1450% ✅ |
| **Sensitivity** | 0.08 | 0.65 | +713% ✅ |
| **Accuracy** | 99.63% | 99.60% | -0.03% (who cares!) |
| **Loss** | 0.19 | 0.32 | Higher but better! |

### Why Loss is Higher:
The loss is higher because the model is actually **trying to detect lesions** now, which is harder than predicting all zeros. This is **good** - it means the model is working!

---

## 🎯 Training Phases Explained

### Phase 1: Awakening (Epochs 1-15)
**What happens:** Model realizes lesions exist

```
Dice: 0.08 → 0.30
IoU: 0.03 → 0.15
```

**Visual:** Predictions go from all black to some faint gray blobs

**What to watch:** 
- ✅ Dice should at least double
- ❌ If Dice stays <0.15 after 15 epochs, there's still a problem

### Phase 2: Learning (Epochs 15-40)
**What happens:** Model learns lesion patterns

```
Dice: 0.30 → 0.50
IoU: 0.15 → 0.35
```

**Visual:** Predictions show clear lesion outlines, some false positives

**What to watch:**
- ✅ Steady improvement each epoch
- ✅ Training and validation curves should be close
- ❌ Large gap = overfitting

### Phase 3: Refinement (Epochs 40-70)
**What happens:** Model fine-tunes predictions

```
Dice: 0.50 → 0.65
IoU: 0.35 → 0.45
```

**Visual:** Predictions match ground truth better, fewer false positives

**What to watch:**
- ✅ Slower improvement (normal)
- ✅ Sensitivity reaches 75%+
- ⚠️ May early-stop if no improvement

### Phase 4: Peak (Epochs 70-100)
**What happens:** Model reaches its limit

```
Dice: 0.65 → 0.70
IoU: 0.45 → 0.50
```

**Visual:** Minor improvements, predictions are good

**What to watch:**
- ✅ Training should stop or early-stop
- ✅ Save best model based on validation Dice
- ⚠️ Don't train too long (overfitting risk)

---

## 🚨 Warning Signs

### Still Not Working After 20 Epochs

If you see this after 20 epochs:
```
Train Dice: 0.15
Val Dice: 0.12
Sensitivity: 0.25
```

**Action needed:**
1. Increase `FOCAL_ALPHA` to 0.90
2. Increase `FOCAL_GAMMA` to 4.0
3. Restart training

### Overfitting

If you see this:
```
Epoch 40:
  Train Dice: 0.75  ← High
  Val Dice: 0.45    ← Much lower
  
  Train Loss: 0.20  ← Low
  Val Loss: 0.45    ← Much higher
```

**Action needed:**
1. Stop training (model won't improve)
2. Use best checkpoint (not final)
3. Consider: more data augmentation, lower learning rate

### Model Collapse

If you see this:
```
Epoch 50:
  Train Dice: 0.99  ← Suspiciously high
  Val Dice: 0.05    ← Terrible
  Predictions: All white everywhere
```

**Action needed:**
1. Training failed (model broken)
2. Decrease learning rate: `LEARNING_RATE = 5e-5`
3. Restart training

---

## ✅ Success Criteria

### Minimum Acceptable Performance:
```
Dice: > 0.50
IoU: > 0.35
Sensitivity: > 0.70
Specificity: > 0.95
```

### Good Performance:
```
Dice: > 0.60
IoU: > 0.45
Sensitivity: > 0.75
Specificity: > 0.95
```

### Excellent Performance:
```
Dice: > 0.70
IoU: > 0.55
Sensitivity: > 0.85
Specificity: > 0.95
```

### State-of-the-Art (Hard to Achieve):
```
Dice: > 0.80
IoU: > 0.70
Sensitivity: > 0.90
Specificity: > 0.95
```

**Note:** For medical MS detection with severe class imbalance, **Dice > 0.60 is considered good!** Don't expect perfection.

---

## 📈 How to Monitor Training

### 1. Watch Terminal Output
Focus on these lines:
```
Val Dice: 0.5200  ← Most important!
Val IoU: 0.3521   ← Second important
Sensitivity: 0.7234  ← Critical for medical
```

### 2. Check Training Plot
```bash
# During training (in another terminal)
watch -n 10 "ls -lh results/training_history.png"

# After an epoch completes, view it
xdg-open results/training_history.png
```

Look for:
- ✅ Upward trend in Dice
- ✅ Downward trend in Loss
- ✅ Train/Val curves close together

### 3. Visual Predictions
```bash
# After training finishes
python main.py --mode evaluate
xdg-open results/test_predictions.png
```

Look for:
- ✅ White regions where lesions are
- ✅ Predictions roughly match ground truth
- ✅ Clean edges, not too noisy

---

## 🔄 Typical Training Session

```bash
$ ./retrain_fixed.sh

========================================
MS Detection - Retrain with Fixed Config
========================================

⚠️  Old checkpoints found!
Do you want to backup and clean them? (y/n) y
📦 Backing up to checkpoints_old_20251017_143022...
✅ Backup complete!

Current Configuration:
  Loss Type: weighted_combined
  Focal Alpha: 0.75
  Focal Gamma: 3.0
  Epochs: 50
  Batch Size: 8

Number of epochs (default=100): 80

========================================
Starting Training
========================================

💡 What to watch:
   ✅ Dice Coefficient (should increase to >0.5)
   ✅ IoU (should increase to >0.4)
   ✅ Sensitivity (should increase to >0.7)
   ❌ Accuracy (ignore - will stay ~99%)

Press Ctrl+C to stop training at any time

Loading data...
Found 1335 images and 10073 masks in images
Found 166 images and 166 masks in images
Found 80 images and 80 masks in images

Initializing model...

============================================================
Starting training for 80 epochs
Device: cuda
Model parameters: 9,447,873
============================================================

Epoch 1 [Train]: 100%|██████| 167/167 [02:15<00:00]
Epoch 1 [Val]: 100%|████████| 21/21 [00:15<00:00]

Epoch 1/80
  Train Loss: 0.5234, Train Dice: 0.0821
  Val Loss: 0.5487, Val Dice: 0.0623
  Val IoU: 0.0321, Val Accuracy: 0.9956
  Sensitivity: 0.1234, Specificity: 0.9981
  >> Saved best model (Dice: 0.0623)
------------------------------------------------------------

... [training continues] ...

Epoch 60/80
  Train Loss: 0.2487, Train Dice: 0.6834
  Val Loss: 0.3123, Val Dice: 0.5987
  Val IoU: 0.4321, Val Accuracy: 0.9962
  Sensitivity: 0.7856, Specificity: 0.9968
  >> Saved best model (Dice: 0.5987)
------------------------------------------------------------

========================================
Training Complete!
========================================

📊 Check results:
   - Training plots: results/training_history.png
   - Best model: checkpoints/best_model.pth

📝 Next steps:
   1. Evaluate: python main.py --mode evaluate
   2. Check visual results: xdg-open results/test_predictions.png
```

---

## 💡 Pro Tips

1. **Be Patient**: With severe class imbalance, improvement is slow at first. Give it 30+ epochs.

2. **Monitor Dice, Not Accuracy**: Accuracy is meaningless. Watch Dice like a hawk.

3. **Check Visual Predictions**: Numbers don't tell the whole story. Look at the actual predictions.

4. **Don't Overtrain**: If validation Dice stops improving for 10 epochs, stop and use best checkpoint.

5. **Save Your Work**: The script automatically backs up old checkpoints. Don't lose your good models!

6. **Compare Before/After**: Keep your old checkpoints to see the improvement.

---

## 🎉 You'll Know It's Working When...

After 30-50 epochs:
- ✅ Dice is above 0.50
- ✅ IoU is above 0.35
- ✅ Visual predictions show detected lesions (not all black)
- ✅ Sensitivity is above 0.70
- ✅ Training plots show clear upward trend

**That's success!** 🚀

The model won't be perfect (medical imaging rarely is), but it will be **actually detecting lesions** instead of predicting all zeros.

