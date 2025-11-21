# MS Detection using ResUNet

A deep learning-based medical image segmentation project for Multiple Sclerosis (MS) lesion detection using Residual U-Net (ResUNet) architecture.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)

## 🔍 Overview

This project implements a ResUNet model for automated detection and segmentation of Multiple Sclerosis lesions in medical images. The ResUNet architecture combines the strengths of U-Net for segmentation with residual connections for improved gradient flow and better feature learning.

### Key Features

- **ResUNet Architecture**: Combines U-Net with residual blocks for better performance
- **Modular Design**: Clean, well-organized code with separate modules for each functionality
- **Comprehensive Metrics**: Dice coefficient, IoU, accuracy, sensitivity, and specificity
- **Data Augmentation**: Built-in augmentation for improved generalization
- **Visualization**: Automatic generation of prediction visualizations
- **Checkpointing**: Automatic saving of best models and periodic checkpoints
- **Early Stopping**: Prevents overfitting with configurable patience
- **Learning Rate Scheduling**: Adaptive learning rate adjustment

## 📁 Project Structure

```
MS-Detection-using-ResUNet/
├── config.py              # Configuration parameters
├── constants.py           # Legacy constants (deprecated)
├── main.py               # Main entry point
├── train.py              # Training logic
├── evaluate.py           # Evaluation logic
├── ResUNet_model.py      # Model architecture
├── normalize_data.py     # Dataset preparation and normalization
├── fetch_data.py         # Data loading utilities
├── utils.py              # Helper functions (metrics, visualization, etc.)
├── requirement.txt       # Python dependencies
├── README.md            # This file
├── dataset/             # Dataset directory
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── valid/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── checkpoints/         # Saved model checkpoints
└── results/            # Training results and visualizations
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd MS-Detection-using-ResUNet
```

2. Install dependencies:
```bash
pip install -r requirement.txt
```

3. Verify installation:
```bash
python main.py --mode info
```

## 📊 Dataset

The project expects the dataset to be organized in the following structure:

```
dataset/
├── train/
│   ├── images/       # Training images (.jpg or .png)
│   └── masks/        # Training masks (.jpg or .png)
├── valid/
│   ├── images/       # Validation images
│   └── masks/        # Validation masks
└── test/
    ├── images/       # Test images
    └── masks/        # Test masks
```

### Dataset Format

- **Images**: RGB images (JPEG or PNG format)
- **Masks**: Binary segmentation masks (grayscale, JPEG or PNG format)
- Masks should have the same filename as their corresponding images

## 💻 Usage

### 1. View Configuration

Display current configuration and dataset statistics:

```bash
python main.py --mode info
```

### 2. Training

Train the model from scratch:

```bash
python main.py --mode train
```

Train with custom parameters:

```bash
python main.py --mode train --batch-size 16 --epochs 100 --lr 0.0001
```

Resume training from a checkpoint:

```bash
python main.py --mode train --resume checkpoints/checkpoint_epoch_10.pth
```

### 3. Evaluation

Evaluate the best model on test set:

```bash
python main.py --mode evaluate
```

Evaluate a specific checkpoint:

```bash
python main.py --mode evaluate --checkpoint checkpoints/best_model.pth
```

Evaluate on validation set:

```bash
python main.py --mode evaluate --split val
```

### 4. Individual Module Usage

You can also run individual modules directly:

**Train only:**
```bash
python train.py
```

**Evaluate only:**
```bash
python evaluate.py
```

**Test model architecture:**
```bash
python ResUNet_model.py
```

**Test data loading:**
```bash
python fetch_data.py
```

## 🏗️ Model Architecture

### ResUNet

The ResUNet model consists of:

**Encoder (Downsampling Path):**
- 4 residual blocks with max pooling
- Captures hierarchical features at different scales
- Filter sizes: [64, 128, 256, 512]

**Bottleneck:**
- 1 residual block with 1024 filters
- Captures the most abstract features

**Decoder (Upsampling Path):**
- 4 transposed convolutions with skip connections
- Reconstructs segmentation mask at original resolution
- Combines low-level and high-level features

**Residual Blocks:**
- Two 3×3 convolutions with batch normalization and ReLU
- Skip connection for gradient flow
- Helps training deeper networks

### Loss Function

Combined loss function:
- **Binary Cross-Entropy (BCE)**: Pixel-wise classification
- **Dice Loss**: Overlap-based metric for segmentation
- Weighted combination (configurable)

## ⚙️ Configuration

Key configuration parameters in `config.py`:

### Model Parameters
- `IN_CHANNELS`: 1 
- `OUT_CHANNELS`: 1 (binary mask)
- `FILTERS`: [64, 128, 256, 512]

### Training Parameters
- `BATCH_SIZE`: 8
- `NUM_EPOCHS`: 50
- `LEARNING_RATE`: 1e-4
- `WEIGHT_DECAY`: 1e-5
- `IMAGE_SIZE`: (256, 256)

### Loss Configuration
- `LOSS_TYPE`: 'combined' ('bce', 'dice', or 'combined')
- `BCE_WEIGHT`: 0.5
- `DICE_WEIGHT`: 0.5

### Augmentation
- `USE_AUGMENTATION`: True
- `AUGMENTATION_PROB`: 0.5

### Optimization
- `USE_LR_SCHEDULER`: True
- `USE_GRADIENT_CLIPPING`: True
- `EARLY_STOPPING_PATIENCE`: 10

## 📈 Results

After training, results are saved in the `results/` directory:

- `training_history.png`: Loss and Dice score curves
- `test_predictions.png`: Sample predictions on test set
- `test_results.json`: Detailed evaluation metrics

### Results Analysis & Visualization

To generate comprehensive visualizations, plots, tables, and reports from your test results, run:

```bash
python3 analyze_results.py
```

This will automatically generate:
- **10+ visualization plots** (distributions, comparisons, correlations, etc.)
- **Statistical tables** (CSV format)
- **Comprehensive markdown report** with analysis and recommendations

#### Generated Files:

**📁 `results/visualizations/`** - All plots and tables:
- `comprehensive_report.png` - **Main multi-panel report** ⭐ (use this for presentations!)
- `metric_comparison_bar.png` - Model comparison bar chart
- `metric_distributions.png` - Distribution histograms for all metrics
- `metric_boxplots.png` - Box plots showing metric variations
- `correlation_heatmap.png` - Correlation matrix between metrics
- `scatter_plots.png` - Relationship plots between metrics
- `performance_categories.png` - Performance tier distribution
- `percentile_analysis.png` - Percentile breakdown
- `model_comparison.csv` - Detailed comparison table
- `statistical_summary.csv` - Statistical summary

**📄 `results/RESULTS_REPORT.md`** - Detailed markdown report with:
- Executive summary
- Model comparison (best by Dice vs best by Loss)
- Detailed metrics analysis with interpretations
- Statistical analysis (distributions, correlations, outliers)
- Performance categories breakdown
- Key findings and insights
- Recommendations for improvement

#### Individual Analysis Scripts:

If needed, you can also run components separately:
```bash
# Generate only visualizations
python3 visualize_results.py

# Generate only markdown report
python3 generate_report.py
```

### Evaluation Metrics

The model is evaluated using:
- **Dice Coefficient**: Overlap between prediction and ground truth (0=no overlap, 1=perfect)
- **IoU (Jaccard Index)**: Intersection over union
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Sensitivity (Recall)**: True positive rate - ability to detect lesions
- **Specificity**: True negative rate - ability to avoid false positives

## 🔧 Customization

### Modify Model Architecture

Edit `ResUNet_model.py` to change the network architecture:

```python
model = ResUNet(
    in_channels=1,
    out_channels=1,
    filters=[32, 64, 128, 256]  # Smaller model
)
```

### Change Hyperparameters

Edit `config.py` or pass command-line arguments:

```bash
python main.py --mode train --batch-size 8 --epochs 100 --lr 0.0001
```

### Custom Loss Function

Add new loss functions in `utils.py` and update `config.LOSS_TYPE`.

## 📝 Code Quality

The codebase follows best practices:

- **Modular Design**: Each file has a specific responsibility
- **Documentation**: Comprehensive docstrings and comments
- **Type Hints**: Clear function signatures
- **Error Handling**: Robust error checking
- **Configurability**: Easy to customize via config file
- **Reproducibility**: Random seed setting for consistent results

## 🐛 Troubleshooting

### Common Issues

**1. CUDA out of memory:**
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `IMAGE_SIZE`

**2. Dataset not found:**
- Verify dataset directory structure
- Check paths in `config.py`

**3. Poor performance:**
- Increase training epochs
- Enable data augmentation
- Try different loss functions
- Adjust learning rate

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- ResUNet architecture inspired by medical image segmentation research
- U-Net paper: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- ResNet paper: [He et al., 2015](https://arxiv.org/abs/1512.03385)

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Happy Training! 🚀**
