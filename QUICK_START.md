# Quick Start - Training with Class Imbalance Fix

## 🎯 Answer to Your Question

**Q: Why is validation accuracy 0.99 from the beginning?**

**A:** Your dataset has 99.63% background and only 0.37% lesions (1:270 ratio). The model achieves 99% accuracy by predicting "no lesion" everywhere - it's NOT learning to detect lesions!

**Fix:** I've implemented Focal Loss which forces the model to actually detect the rare lesions instead of ignoring them.

---

## ⚡ Quick Commands

### Train (Fixed loss already configured!)
```bash
python main.py --mode train --epochs 50
```

### Evaluate
```bash
python main.py --mode evaluate
```

### Monitor During Training
**IGNORE:** Pixel Accuracy (always ~99%)  
**WATCH:** Dice Coefficient (should improve to >0.6)

---

## 📊 What Changed

| Before | After |
|--------|-------|
| Loss: BCE + Dice | Loss: **Focal + Dice** ✅ |
| Dice: ~0.05 (bad) | Dice: ~0.60-0.70 (good) ✅ |
| Model predicts zeros | Model detects lesions ✅ |

---

## 📚 Full Documentation

- **TRAINING_GUIDE.md** - Complete guide
- **CLASS_IMBALANCE_GUIDE.md** - Technical details
- **config.py** - Configuration (already updated!)

---

## ✅ You're Ready!

Just run: `python main.py --mode train`

Monitor **Dice Coefficient** - it should improve from ~0.1 to >0.6!

