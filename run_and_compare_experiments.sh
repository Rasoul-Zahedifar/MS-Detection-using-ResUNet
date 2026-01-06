#!/bin/bash

# Shell script to run all experiments and then compare them
# This script orchestrates the full experiment workflow

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default values
EXPERIMENTS_DIR="experiments"
SKIP_COMPLETED=false
MAX_EXPERIMENTS=""
FILTER=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiments-dir)
            EXPERIMENTS_DIR="$2"
            shift 2
            ;;
        --skip-completed)
            SKIP_COMPLETED=true
            shift
            ;;
        --max-experiments)
            MAX_EXPERIMENTS="$2"
            shift 2
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --experiments-dir DIR    Directory to save experiment results (default: experiments)"
            echo "  --skip-completed         Skip experiments that have already been completed"
            echo "  --max-experiments N      Maximum number of experiments to run (for testing)"
            echo "  --filter PATTERN         Filter experiments by name pattern"
            echo "  --help                   Show this help message"
            echo ""
            echo "This script will:"
            echo "  1. Run all experiments with different configurations"
            echo "  2. Compare all experiment results"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log file name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/experiments_${TIMESTAMP}.log"

# Start logging - redirect all output to both terminal and log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MS Detection - Full Experiment Workflow${NC}"
echo -e "${BLUE}Log file: ${LOG_FILE}${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Build the command for run_experiments.py
RUN_EXP_CMD="python run_experiments.py --experiments-dir $EXPERIMENTS_DIR"

if [ "$SKIP_COMPLETED" = true ]; then
    RUN_EXP_CMD="$RUN_EXP_CMD --skip-completed"
fi

if [ -n "$MAX_EXPERIMENTS" ]; then
    RUN_EXP_CMD="$RUN_EXP_CMD --max-experiments $MAX_EXPERIMENTS"
fi

if [ -n "$FILTER" ]; then
    RUN_EXP_CMD="$RUN_EXP_CMD --filter $FILTER"
fi

# Step 1: Run all experiments
echo -e "${GREEN}[STEP 1/2] Running all experiments...${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${CYAN}Command: $RUN_EXP_CMD${NC}"
echo ""

$RUN_EXP_CMD

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Running experiments failed!${NC}"
    echo -e "${YELLOW}Note: You can still run comparison on existing experiments${NC}"
    echo -e "${YELLOW}      using: python compare_experiments.py --experiments-dir $EXPERIMENTS_DIR${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All experiments completed!${NC}"
echo ""

# Step 2: Compare experiments
echo -e "${GREEN}[STEP 2/2] Comparing experiment results...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

COMPARE_CMD="python compare_experiments.py --experiments-dir $EXPERIMENTS_DIR"
echo -e "${CYAN}Command: $COMPARE_CMD${NC}"
echo ""

$COMPARE_CMD

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Comparison failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Comparison completed!${NC}"
echo ""

# Final summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}FULL WORKFLOW COMPLETED SUCCESSFULLY!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Generated Files:${NC}"
echo ""
echo -e "📊 Experiments:"
echo "   • $EXPERIMENTS_DIR/ (all experiment results)"
echo ""
echo -e "📈 Comparison Results:"
echo "   • $EXPERIMENTS_DIR/comparison/experiments_summary.csv"
echo "   • $EXPERIMENTS_DIR/comparison/experiments_summary.md"
echo "   • $EXPERIMENTS_DIR/comparison/metric_comparison.png"
echo "   • $EXPERIMENTS_DIR/comparison/configuration_impact.png"
echo "   • $EXPERIMENTS_DIR/comparison/correlation_heatmap.png"
echo "   • $EXPERIMENTS_DIR/comparison/loss_type_comparison.png"
echo "   • $EXPERIMENTS_DIR/comparison/COMPARISON_REPORT.md"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "   1. View the comparison report: cat $EXPERIMENTS_DIR/comparison/COMPARISON_REPORT.md"
echo "   2. Check visualizations: ls -lh $EXPERIMENTS_DIR/comparison/*.png"
echo "   3. View summary table: cat $EXPERIMENTS_DIR/comparison/experiments_summary.md"
echo ""
echo -e "${YELLOW}Log File:${NC}"
echo "   • ${LOG_FILE}"
echo ""
echo -e "${BLUE}========================================${NC}"

