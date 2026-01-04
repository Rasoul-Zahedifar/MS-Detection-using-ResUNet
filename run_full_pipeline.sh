#!/bin/bash

# Full Pipeline Script for MS Detection using ResUNet
# This script trains the model, evaluates it, and generates comprehensive analysis

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log file name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/pipeline_${TIMESTAMP}.log"

# Start logging - redirect all output to both terminal and log file
# This captures both stdout and stderr
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MS Detection - Full Pipeline${NC}"
echo -e "${BLUE}Log file: ${LOG_FILE}${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Training
echo -e "${GREEN}[STEP 1/4] Training the model...${NC}"
echo -e "${BLUE}========================================${NC}"
python main.py --mode train --batch-size 8 --epochs 100 --lr 0.0001

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Training failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Training completed successfully!${NC}"
echo ""

# Step 2: Evaluate best_by_dice checkpoint
echo -e "${GREEN}[STEP 2/4] Evaluating best_by_dice model on test set...${NC}"
echo -e "${BLUE}========================================${NC}"

if [ ! -f "checkpoints/best_by_dice.pth" ]; then
    echo -e "${RED}Error: best_by_dice.pth checkpoint not found!${NC}"
    exit 1
fi

python main.py --mode evaluate --checkpoint checkpoints/best_by_dice.pth --split test

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Evaluation of best_by_dice model failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Evaluation of best_by_dice model completed!${NC}"
echo ""

# Step 3: Evaluate best_by_loss checkpoint
echo -e "${GREEN}[STEP 3/4] Evaluating best_by_loss model on test set...${NC}"
echo -e "${BLUE}========================================${NC}"

if [ ! -f "checkpoints/best_by_loss.pth" ]; then
    echo -e "${YELLOW}Warning: best_by_loss.pth checkpoint not found, skipping...${NC}"
else
    python main.py --mode evaluate --checkpoint checkpoints/best_by_loss.pth --split test

    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: Evaluation of best_by_loss model failed, continuing...${NC}"
    else
        echo ""
        echo -e "${GREEN}✓ Evaluation of best_by_loss model completed!${NC}"
    fi
fi

echo ""

# Step 4: Analyze results and generate report
echo -e "${GREEN}[STEP 4/4] Analyzing results and generating comprehensive report...${NC}"
echo -e "${BLUE}========================================${NC}"

python main.py --mode analyze

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Analysis failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Analysis completed successfully!${NC}"
echo ""

# Final summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}PIPELINE COMPLETED SUCCESSFULLY!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Generated Files:${NC}"
echo ""
echo -e "📊 Checkpoints:"
echo "   • checkpoints/best_by_dice.pth"
echo "   • checkpoints/best_by_loss.pth"
echo "   • checkpoints/final_model.pth"
echo ""
echo -e "📈 Evaluation Results:"
echo "   • results/best_by_dice/test_results.json"
echo "   • results/best_by_dice/test_predictions.png"
if [ -f "checkpoints/best_by_loss.pth" ]; then
    echo "   • results/best_by_loss/test_results.json"
    echo "   • results/best_by_loss/test_predictions.png"
fi
echo ""
echo -e "📄 Analysis & Reports:"
echo "   • results/RESULTS_REPORT.md"
echo "   • results/visualizations/comprehensive_report.png"
echo "   • results/visualizations/*.png (various visualizations)"
echo "   • results/visualizations/*.csv (statistical summaries)"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "   1. View the comprehensive report: cat results/RESULTS_REPORT.md"
echo "   2. Check visualizations: ls -lh results/visualizations/"
echo "   3. Compare model performance in the report"
echo ""
echo -e "${YELLOW}Log File:${NC}"
echo "   • ${LOG_FILE}"
echo ""
echo -e "${BLUE}========================================${NC}"

