#!/usr/bin/env python3
"""
Generate a comprehensive markdown report of test results
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


class ReportGenerator:
    """Generate comprehensive markdown reports"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.load_results()
    
    def load_results(self):
        """Load all result files"""
        # Load best_by_dice results
        dice_path = self.results_dir / 'best_by_dice' / 'test_results.json'
        if dice_path.exists():
            with open(dice_path, 'r') as f:
                self.dice_results = json.load(f)
        else:
            self.dice_results = None
        
        # Load best_by_loss results
        loss_path = self.results_dir / 'best_by_loss' / 'test_results.json'
        if loss_path.exists():
            with open(loss_path, 'r') as f:
                self.loss_results = json.load(f)
        else:
            self.loss_results = None
        
        # Load prediction statistics
        pred_stats_path = self.results_dir / 'best_by_dice_predictions' / 'prediction_statistics.json'
        if pred_stats_path.exists():
            with open(pred_stats_path, 'r') as f:
                self.pred_stats = json.load(f)
        else:
            self.pred_stats = None
    
    def generate_report(self):
        """Generate comprehensive markdown report"""
        lines = []
        
        # Header
        lines.append("# MS Detection using ResUNet - Test Results Report\n")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")
        
        # Table of Contents
        lines.append("## Table of Contents\n")
        lines.append("1. [Executive Summary](#executive-summary)")
        lines.append("2. [Model Comparison](#model-comparison)")
        lines.append("3. [Detailed Metrics Analysis](#detailed-metrics-analysis)")
        lines.append("4. [Statistical Analysis](#statistical-analysis)")
        lines.append("5. [Performance Categories](#performance-categories)")
        lines.append("6. [Key Findings](#key-findings)")
        lines.append("7. [Recommendations](#recommendations)")
        lines.append("\n---\n")
        
        # Executive Summary
        lines.append("## Executive Summary\n")
        
        if self.dice_results:
            dice_mean = self.dice_results['average_metrics']['dice']
            iou_mean = self.dice_results['average_metrics']['iou']
            acc_mean = self.dice_results['average_metrics']['accuracy']
            sens_mean = self.dice_results['average_metrics']['sensitivity']
            spec_mean = self.dice_results['average_metrics']['specificity']
            
            n_samples = len(self.dice_results['all_metrics']['dice'])
            
            lines.append(f"This report presents the evaluation results of the ResUNet model for Multiple Sclerosis (MS) lesion detection. ")
            lines.append(f"The model was tested on **{n_samples} samples** from the test dataset.\n")
            
            lines.append("\n### Key Metrics (Best by Dice Model)\n")
            lines.append(f"- **Dice Coefficient:** {dice_mean:.4f}")
            lines.append(f"- **IoU (Jaccard Index):** {iou_mean:.4f}")
            lines.append(f"- **Accuracy:** {acc_mean:.4f} ({acc_mean*100:.2f}%)")
            lines.append(f"- **Sensitivity (Recall):** {sens_mean:.4f}")
            lines.append(f"- **Specificity:** {spec_mean:.4f}")
            lines.append("\n")
            
            # Performance assessment
            if dice_mean >= 0.7:
                performance = "**Excellent**"
                emoji = "🟢"
            elif dice_mean >= 0.5:
                performance = "**Good**"
                emoji = "🟡"
            elif dice_mean >= 0.3:
                performance = "**Moderate**"
                emoji = "🟠"
            else:
                performance = "**Needs Improvement**"
                emoji = "🔴"
            
            lines.append(f"### Overall Performance: {emoji} {performance}\n")
            
            if spec_mean > 0.99:
                lines.append("✅ The model shows **very high specificity** (>99%), indicating excellent ability to correctly identify non-lesion regions.\n")
            
            if acc_mean > 0.99:
                lines.append("✅ **High accuracy** (>99%) demonstrates strong overall performance, though this may be influenced by class imbalance.\n")
            
            if dice_mean < 0.3:
                lines.append("⚠️ The **low Dice score** suggests challenges in lesion segmentation, particularly for small or difficult lesions.\n")
        
        lines.append("\n---\n")
        
        # Model Comparison
        lines.append("## Model Comparison\n")
        
        if self.dice_results and self.loss_results:
            lines.append("Two model checkpoints were evaluated:\n")
            lines.append("- **Best by Dice:** Model checkpoint with highest validation Dice score")
            lines.append("- **Best by Loss:** Model checkpoint with lowest validation loss\n")
            
            lines.append("\n### Comparative Performance\n")
            lines.append("| Metric | Best by Dice | Best by Loss | Difference | Winner |\n")
            lines.append("|--------|--------------|--------------|------------|--------|\n")
            
            metrics = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
            for metric in metrics:
                dice_val = self.dice_results['average_metrics'][metric]
                loss_val = self.loss_results['average_metrics'][metric]
                diff = dice_val - loss_val
                winner = "Dice 🏆" if dice_val > loss_val else "Loss 🏆" if loss_val > dice_val else "Tie"
                
                lines.append(f"| {metric.capitalize()} | {dice_val:.4f} | {loss_val:.4f} | {diff:+.4f} | {winner} |\n")
            
            lines.append("\n### Key Observations\n")
            
            dice_better = sum([1 for m in metrics if self.dice_results['average_metrics'][m] > self.loss_results['average_metrics'][m]])
            
            if dice_better >= 3:
                lines.append(f"- The **Best by Dice** model outperforms on {dice_better}/5 metrics\n")
            else:
                lines.append(f"- The **Best by Loss** model outperforms on {5-dice_better}/5 metrics\n")
            
            # Check if differences are significant
            dice_diff = self.dice_results['average_metrics']['dice'] - self.loss_results['average_metrics']['dice']
            if abs(dice_diff) > 0.01:
                lines.append(f"- There is a **notable difference** in Dice scores ({abs(dice_diff):.4f})\n")
            else:
                lines.append(f"- The models show **similar performance** (Dice difference: {abs(dice_diff):.4f})\n")
        
        lines.append("\n---\n")
        
        # Detailed Metrics Analysis
        lines.append("## Detailed Metrics Analysis\n")
        
        if self.dice_results:
            metrics = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
            
            for metric in metrics:
                data = self.dice_results['all_metrics'][metric]
                mean_val = np.mean(data)
                std_val = np.std(data)
                min_val = np.min(data)
                max_val = np.max(data)
                median_val = np.median(data)
                
                lines.append(f"### {metric.upper()}\n")
                lines.append(f"- **Mean:** {mean_val:.4f}")
                lines.append(f"- **Standard Deviation:** {std_val:.4f}")
                lines.append(f"- **Median:** {median_val:.4f}")
                lines.append(f"- **Range:** [{min_val:.4f}, {max_val:.4f}]")
                lines.append(f"- **Coefficient of Variation:** {(std_val/mean_val)*100:.2f}%\n")
                
                # Interpretation
                if metric == 'dice':
                    if mean_val >= 0.7:
                        lines.append("**Interpretation:** Excellent segmentation performance.\n")
                    elif mean_val >= 0.5:
                        lines.append("**Interpretation:** Good segmentation with room for improvement.\n")
                    elif mean_val >= 0.3:
                        lines.append("**Interpretation:** Moderate performance, significant improvement needed.\n")
                    else:
                        lines.append("**Interpretation:** Poor performance, major improvements required.\n")
                
                elif metric == 'sensitivity':
                    if mean_val < 0.5:
                        lines.append("⚠️ **Low sensitivity** indicates the model misses many lesions (high false negative rate).\n")
                    else:
                        lines.append("✅ **Good sensitivity** shows the model detects most lesions effectively.\n")
                
                elif metric == 'specificity':
                    if mean_val > 0.95:
                        lines.append("✅ **Excellent specificity** means very few false positives.\n")
                    else:
                        lines.append("⚠️ **Moderate specificity** indicates some false positive predictions.\n")
                
                lines.append("\n")
        
        lines.append("---\n")
        
        # Statistical Analysis
        lines.append("## Statistical Analysis\n")
        
        if self.dice_results:
            lines.append("### Distribution Characteristics\n")
            
            dice_scores = np.array(self.dice_results['all_metrics']['dice'])
            
            from scipy import stats
            skewness = stats.skew(dice_scores)
            kurtosis = stats.kurtosis(dice_scores)
            
            lines.append(f"- **Sample Size:** {len(dice_scores)}")
            lines.append(f"- **Skewness:** {skewness:.4f}")
            lines.append(f"- **Kurtosis:** {kurtosis:.4f}\n")
            
            if abs(skewness) < 0.5:
                lines.append("The distribution is approximately **symmetric**.\n")
            elif skewness > 0.5:
                lines.append("The distribution is **right-skewed** (tail extends toward higher values).\n")
            else:
                lines.append("The distribution is **left-skewed** (tail extends toward lower values).\n")
            
            if kurtosis > 0:
                lines.append("The distribution has **heavier tails** than a normal distribution (leptokurtic).\n")
            elif kurtosis < 0:
                lines.append("The distribution has **lighter tails** than a normal distribution (platykurtic).\n")
            
            lines.append("\n### Percentile Analysis\n")
            percentiles = [10, 25, 50, 75, 90]
            lines.append("| Percentile | Dice Score |\n")
            lines.append("|------------|------------|\n")
            for p in percentiles:
                val = np.percentile(dice_scores, p)
                lines.append(f"| {p}th | {val:.4f} |\n")
            
            lines.append("\n")
        
        lines.append("---\n")
        
        # Performance Categories
        lines.append("## Performance Categories\n")
        
        if self.dice_results:
            dice_scores = np.array(self.dice_results['all_metrics']['dice'])
            
            excellent = np.sum(dice_scores >= 0.7)
            good = np.sum((dice_scores >= 0.5) & (dice_scores < 0.7))
            fair = np.sum((dice_scores >= 0.3) & (dice_scores < 0.5))
            poor = np.sum((dice_scores >= 0.1) & (dice_scores < 0.3))
            failed = np.sum(dice_scores < 0.1)
            
            total = len(dice_scores)
            
            lines.append("Samples are categorized based on their Dice scores:\n")
            lines.append("\n| Category | Dice Range | Count | Percentage |\n")
            lines.append("|----------|------------|-------|------------|\n")
            lines.append(f"| 🟢 Excellent | ≥ 0.7 | {excellent} | {excellent/total*100:.1f}% |\n")
            lines.append(f"| 🔵 Good | 0.5 - 0.7 | {good} | {good/total*100:.1f}% |\n")
            lines.append(f"| 🟡 Fair | 0.3 - 0.5 | {fair} | {fair/total*100:.1f}% |\n")
            lines.append(f"| 🟠 Poor | 0.1 - 0.3 | {poor} | {poor/total*100:.1f}% |\n")
            lines.append(f"| 🔴 Failed | < 0.1 | {failed} | {failed/total*100:.1f}% |\n")
            
            lines.append("\n### Analysis\n")
            
            success_rate = (excellent + good) / total * 100
            lines.append(f"- **Success Rate** (Excellent + Good): {success_rate:.1f}%\n")
            
            if success_rate >= 70:
                lines.append("✅ The model performs well on the majority of samples.\n")
            elif success_rate >= 50:
                lines.append("⚠️ The model shows mixed performance across samples.\n")
            else:
                lines.append("❌ The model struggles with most samples.\n")
            
            if failed / total > 0.3:
                lines.append(f"⚠️ **High failure rate** ({failed/total*100:.1f}%) suggests the model has difficulty with many test cases.\n")
        
        lines.append("\n---\n")
        
        # Key Findings
        lines.append("## Key Findings\n")
        
        if self.dice_results:
            lines.append("### Strengths\n")
            
            spec = self.dice_results['average_metrics']['specificity']
            acc = self.dice_results['average_metrics']['accuracy']
            
            if spec > 0.99:
                lines.append("1. **Excellent Specificity:** The model achieves >99% specificity, demonstrating strong ability to avoid false positives.\n")
            
            if acc > 0.99:
                lines.append("2. **High Overall Accuracy:** Accuracy exceeds 99%, though this may be influenced by class imbalance.\n")
            
            dice_scores = np.array(self.dice_results['all_metrics']['dice'])
            if np.max(dice_scores) > 0.8:
                lines.append(f"3. **Peak Performance:** The model achieves excellent results on some samples (max Dice: {np.max(dice_scores):.4f}).\n")
            
            lines.append("\n### Weaknesses\n")
            
            sens = self.dice_results['average_metrics']['sensitivity']
            dice_mean = self.dice_results['average_metrics']['dice']
            
            if sens < 0.5:
                lines.append(f"1. **Low Sensitivity:** Average sensitivity of {sens:.4f} indicates many lesions are missed (high false negative rate).\n")
            
            if dice_mean < 0.3:
                lines.append(f"2. **Low Dice Score:** Mean Dice of {dice_mean:.4f} suggests poor overlap between predictions and ground truth.\n")
            
            failed = np.sum(dice_scores < 0.1)
            if failed > 20:
                lines.append(f"3. **High Failure Rate:** {failed} samples ({failed/len(dice_scores)*100:.1f}%) have Dice scores below 0.1.\n")
            
            std_dice = np.std(dice_scores)
            if std_dice > 0.2:
                lines.append(f"4. **High Variability:** Large standard deviation ({std_dice:.4f}) indicates inconsistent performance across samples.\n")
        
        lines.append("\n---\n")
        
        # Recommendations
        lines.append("## Recommendations\n")
        
        if self.dice_results:
            lines.append("Based on the analysis, here are recommendations for improvement:\n")
            
            sens = self.dice_results['average_metrics']['sensitivity']
            dice_mean = self.dice_results['average_metrics']['dice']
            
            if sens < 0.5:
                lines.append("\n### 1. Improve Sensitivity\n")
                lines.append("- **Adjust Class Weights:** Increase the weight for the lesion class to penalize false negatives more heavily.\n")
                lines.append("- **Focal Loss:** Implement focal loss to focus on hard-to-detect lesions.\n")
                lines.append("- **Data Augmentation:** Add more augmentation techniques to increase diversity of lesion appearances.\n")
            
            if dice_mean < 0.3:
                lines.append("\n### 2. Address Low Dice Scores\n")
                lines.append("- **Loss Function:** Consider using Dice loss or Tversky loss instead of BCE.\n")
                lines.append("- **Post-processing:** Implement morphological operations to refine predictions.\n")
                lines.append("- **Multi-scale Training:** Train the model to recognize lesions at different scales.\n")
            
            lines.append("\n### 3. Handle Class Imbalance\n")
            lines.append("- **Weighted Sampling:** Oversample images with lesions during training.\n")
            lines.append("- **Hard Negative Mining:** Focus on difficult negative samples.\n")
            lines.append("- **Two-stage Detection:** Use a detection network followed by segmentation.\n")
            
            lines.append("\n### 4. Model Architecture\n")
            lines.append("- **Deeper Networks:** Consider using deeper architectures like UNet++, UNet 3+, or Attention U-Net.\n")
            lines.append("- **Pre-trained Encoders:** Use pre-trained encoders (e.g., ResNet, EfficientNet) for better feature extraction.\n")
            lines.append("- **Multi-task Learning:** Add auxiliary tasks like lesion detection or classification.\n")
            
            lines.append("\n### 5. Training Strategy\n")
            lines.append("- **Learning Rate Scheduling:** Experiment with different learning rate schedules.\n")
            lines.append("- **Longer Training:** Increase the number of epochs if the model hasn't converged.\n")
            lines.append("- **Curriculum Learning:** Start with easier samples and gradually increase difficulty.\n")
        
        lines.append("\n---\n")
        
        # Footer
        lines.append("## Appendix\n")
        lines.append("\n### Metric Definitions\n")
        lines.append("- **Dice Coefficient:** Measures overlap between prediction and ground truth (0 = no overlap, 1 = perfect overlap)\n")
        lines.append("- **IoU (Jaccard Index):** Similar to Dice but more sensitive to size differences\n")
        lines.append("- **Accuracy:** Percentage of correctly classified pixels\n")
        lines.append("- **Sensitivity (Recall):** Percentage of actual lesion pixels correctly identified\n")
        lines.append("- **Specificity:** Percentage of actual non-lesion pixels correctly identified\n")
        
        lines.append("\n### Files Generated\n")
        lines.append("- `comprehensive_report.png` - Multi-panel visualization\n")
        lines.append("- `metric_comparison_bar.png` - Bar chart comparing models\n")
        lines.append("- `metric_distributions.png` - Distribution plots for each metric\n")
        lines.append("- `metric_boxplots.png` - Box plots for all metrics\n")
        lines.append("- `correlation_heatmap.png` - Correlation matrix between metrics\n")
        lines.append("- `scatter_plots.png` - Scatter plots showing metric relationships\n")
        lines.append("- `performance_categories.png` - Pie chart of performance categories\n")
        lines.append("- `percentile_analysis.png` - Percentile distribution\n")
        lines.append("- `model_comparison.csv` - Detailed model comparison table\n")
        lines.append("- `statistical_summary.csv` - Comprehensive statistics\n")
        
        lines.append("\n---\n")
        lines.append(f"\n*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n")
        
        return '\n'.join(lines)
    
    def save_report(self, output_path='results/RESULTS_REPORT.md'):
        """Save the report to a file"""
        report = self.generate_report()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {output_path.absolute()}")
        return output_path


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Results Report')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Path to results directory')
    parser.add_argument('--output', type=str, default='results/RESULTS_REPORT.md',
                       help='Output markdown file path')
    
    args = parser.parse_args()
    
    generator = ReportGenerator(results_dir=args.results_dir)
    generator.save_report(output_path=args.output)
    
    print("\n✓ Report generation complete!")


if __name__ == '__main__':
    main()

