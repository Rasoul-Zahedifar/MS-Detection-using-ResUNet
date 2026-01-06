#!/usr/bin/env python3
"""
Compare results from multiple experiments
Generates comprehensive comparison plots, tables, and analysis
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse

# Add current directory to path
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ExperimentComparator:
    """Compare results from multiple experiments"""
    
    def __init__(self, experiments_dir='experiments'):
        self.experiments_dir = Path(experiments_dir)
        self.comparison_dir = self.experiments_dir / 'comparison'
        self.comparison_dir.mkdir(exist_ok=True)
        
        self.experiments_data = []
        self.load_experiments()
    
    def load_experiments(self):
        """Load all experiment results"""
        print("Loading experiment results...")
        
        for exp_dir in sorted(self.experiments_dir.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name == 'comparison':
                continue
            
            metadata_path = exp_dir / 'experiment_metadata.json'
            if not metadata_path.exists():
                continue
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if metadata.get('status') != 'completed':
                continue
            
            # Load results
            results_path = exp_dir / 'results' / 'best_by_dice' / 'test_results.json'
            if not results_path.exists():
                print(f"Warning: No results found for {exp_dir.name}")
                continue
            
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Combine metadata and results
            exp_data = {
                'experiment_name': metadata['experiment_name'],
                'config': metadata['config'],
                'results': results,
                'experiment_dir': str(exp_dir)
            }
            
            self.experiments_data.append(exp_data)
        
        print(f"Loaded {len(self.experiments_data)} completed experiments")
    
    def extract_metrics(self):
        """Extract metrics from all experiments into a DataFrame"""
        rows = []
        
        for exp in self.experiments_data:
            config = exp['config']
            results = exp['results']
            
            # Extract average metrics
            avg_metrics = results.get('average_metrics', {})
            
            row = {
                'experiment': exp['experiment_name'],
                'USE_TRANSFORMER': config['USE_TRANSFORMER'],
                'USE_AUGMENTATION': config['USE_AUGMENTATION'],
                'USE_PATCH_TRAINING': config['USE_PATCH_TRAINING'],
                'USE_CLASS_SAMPLING': config['USE_CLASS_SAMPLING'],
                'LOSS_TYPE': config['LOSS_TYPE'],
                'dice_score': avg_metrics.get('dice_score', np.nan),
                'iou': avg_metrics.get('iou', np.nan),
                'precision': avg_metrics.get('precision', np.nan),
                'recall': avg_metrics.get('recall', np.nan),
                'f1_score': avg_metrics.get('f1_score', np.nan),
                'accuracy': avg_metrics.get('accuracy', np.nan),
                'specificity': avg_metrics.get('specificity', np.nan),
                'sensitivity': avg_metrics.get('sensitivity', np.nan),
            }
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def create_summary_table(self, df):
        """Create a summary table of all experiments"""
        print("\nCreating summary table...")
        
        # Sort by dice score (primary metric)
        df_sorted = df.sort_values('dice_score', ascending=False)
        
        # Save as CSV
        csv_path = self.comparison_dir / 'experiments_summary.csv'
        df_sorted.to_csv(csv_path, index=False)
        print(f"Saved summary table to {csv_path}")
        
        # Create formatted markdown table
        md_path = self.comparison_dir / 'experiments_summary.md'
        with open(md_path, 'w') as f:
            f.write("# Experiments Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Top 10 Experiments by Dice Score\n\n")
            
            # Format top 10
            top10 = df_sorted.head(10)
            f.write("| Rank | Experiment | Transformer | Aug | Patch | Sampling | Loss | Dice | IoU | Precision | Recall | F1 |\n")
            f.write("|------|------------|-------------|-----|-------|----------|------|------|-----|-----------|--------|----|\n")
            
            for idx, (_, row) in enumerate(top10.iterrows(), 1):
                f.write(f"| {idx} | {row['experiment']} | "
                       f"{'✓' if row['USE_TRANSFORMER'] else '✗'} | "
                       f"{'✓' if row['USE_AUGMENTATION'] else '✗'} | "
                       f"{'✓' if row['USE_PATCH_TRAINING'] else '✗'} | "
                       f"{'✓' if row['USE_CLASS_SAMPLING'] else '✗'} | "
                       f"{row['LOSS_TYPE']} | "
                       f"{row['dice_score']:.4f} | "
                       f"{row['iou']:.4f} | "
                       f"{row['precision']:.4f} | "
                       f"{row['recall']:.4f} | "
                       f"{row['f1_score']:.4f} |\n")
            
            f.write("\n## Full Results\n\n")
            f.write("See `experiments_summary.csv` for complete data.\n")
        
        print(f"Saved markdown summary to {md_path}")
        
        return df_sorted
    
    def plot_metric_comparison(self, df):
        """Create comparison plots for different metrics"""
        print("\nCreating metric comparison plots...")
        
        metrics = ['dice_score', 'iou', 'precision', 'recall', 'f1_score', 'accuracy']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Sort by metric value
            df_sorted = df.sort_values(metric, ascending=True)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(df_sorted))
            colors = plt.cm.viridis(df_sorted[metric] / df_sorted[metric].max())
            
            bars = ax.barh(y_pos, df_sorted[metric], color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_sorted['experiment'], fontsize=8)
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=7)
        
        plt.tight_layout()
        plot_path = self.comparison_dir / 'metric_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved metric comparison plot to {plot_path}")
    
    def plot_configuration_impact(self, df):
        """Analyze impact of each configuration parameter"""
        print("\nAnalyzing configuration impact...")
        
        config_params = ['USE_TRANSFORMER', 'USE_AUGMENTATION', 
                        'USE_PATCH_TRAINING', 'USE_CLASS_SAMPLING', 'LOSS_TYPE']
        metrics = ['dice_score', 'iou', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(len(config_params), len(metrics), 
                                figsize=(20, 16))
        
        for param_idx, param in enumerate(config_params):
            for metric_idx, metric in enumerate(metrics):
                ax = axes[param_idx, metric_idx]
                
                if param == 'LOSS_TYPE':
                    # Box plot for loss types
                    loss_types = df['LOSS_TYPE'].unique()
                    data_to_plot = [df[df['LOSS_TYPE'] == lt][metric].values 
                                   for lt in loss_types]
                    bp = ax.boxplot(data_to_plot, labels=loss_types, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')
                else:
                    # Bar plot for boolean parameters
                    true_vals = df[df[param] == True][metric].values
                    false_vals = df[df[param] == False][metric].values
                    
                    x = [0, 1]
                    means = [np.mean(false_vals), np.mean(true_vals)]
                    stds = [np.std(false_vals), np.std(true_vals)]
                    
                    bars = ax.bar(x, means, yerr=stds, capsize=5, 
                                 color=['lightcoral', 'lightgreen'])
                    ax.set_xticks(x)
                    ax.set_xticklabels(['False', 'True'])
                    ax.set_ylabel(metric.replace('_', ' ').title())
                    ax.set_title(f'{param} vs {metric}')
                    ax.grid(axis='y', alpha=0.3)
                    
                    # Add value labels
                    for bar, mean, std in zip(bars, means, stds):
                        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.01,
                               f'{mean:.3f}', ha='center', fontsize=8)
                
                if param_idx == 0:
                    ax.set_title(f'{param}\n{metric}', fontsize=9)
                else:
                    ax.set_title(metric, fontsize=9)
        
        plt.tight_layout()
        plot_path = self.comparison_dir / 'configuration_impact.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved configuration impact plot to {plot_path}")
    
    def plot_correlation_heatmap(self, df):
        """Create correlation heatmap between metrics"""
        print("\nCreating correlation heatmap...")
        
        metrics = ['dice_score', 'iou', 'precision', 'recall', 'f1_score', 
                  'accuracy', 'specificity', 'sensitivity']
        
        corr_matrix = df[metrics].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Metric Correlation Heatmap')
        plt.tight_layout()
        
        plot_path = self.comparison_dir / 'correlation_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved correlation heatmap to {plot_path}")
    
    def plot_loss_type_comparison(self, df):
        """Compare different loss types"""
        print("\nComparing loss types...")
        
        metrics = ['dice_score', 'iou', 'precision', 'recall', 'f1_score']
        loss_types = df['LOSS_TYPE'].unique()
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            data = [df[df['LOSS_TYPE'] == lt][metric].values for lt in loss_types]
            
            bp = ax.boxplot(data, labels=loss_types, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} by Loss Type')
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plot_path = self.comparison_dir / 'loss_type_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved loss type comparison to {plot_path}")
    
    def create_detailed_report(self, df):
        """Create a detailed comparison report"""
        print("\nCreating detailed report...")
        
        report_path = self.comparison_dir / 'COMPARISON_REPORT.md'
        
        with open(report_path, 'w') as f:
            f.write("# Experiments Comparison Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total experiments compared: {len(df)}\n\n")
            
            # Best performing experiments
            f.write("## Best Performing Experiments\n\n")
            for metric in ['dice_score', 'iou', 'precision', 'recall', 'f1_score']:
                best = df.loc[df[metric].idxmax()]
                f.write(f"### Best {metric.replace('_', ' ').title()}\n\n")
                f.write(f"- **Experiment**: {best['experiment']}\n")
                f.write(f"- **Score**: {best[metric]:.4f}\n")
                f.write(f"- **Configuration**:\n")
                f.write(f"  - Transformer: {best['USE_TRANSFORMER']}\n")
                f.write(f"  - Augmentation: {best['USE_AUGMENTATION']}\n")
                f.write(f"  - Patch Training: {best['USE_PATCH_TRAINING']}\n")
                f.write(f"  - Class Sampling: {best['USE_CLASS_SAMPLING']}\n")
                f.write(f"  - Loss Type: {best['LOSS_TYPE']}\n\n")
            
            # Statistical summary
            f.write("## Statistical Summary\n\n")
            f.write("### Mean Metrics by Configuration Parameter\n\n")
            
            for param in ['USE_TRANSFORMER', 'USE_AUGMENTATION', 
                         'USE_PATCH_TRAINING', 'USE_CLASS_SAMPLING']:
                f.write(f"#### {param}\n\n")
                summary = df.groupby(param)[['dice_score', 'iou', 'precision', 
                                            'recall', 'f1_score']].mean()
                f.write(summary.to_markdown())
                f.write("\n\n")
            
            # Loss type comparison
            f.write("### Loss Type Comparison\n\n")
            loss_summary = df.groupby('LOSS_TYPE')[['dice_score', 'iou', 
                                                    'precision', 'recall', 
                                                    'f1_score']].agg(['mean', 'std'])
            f.write(loss_summary.to_markdown())
            f.write("\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            best_dice = df.loc[df['dice_score'].idxmax()]
            f.write(f"Based on the analysis, the best performing configuration is:\n\n")
            f.write(f"- **Experiment**: {best_dice['experiment']}\n")
            f.write(f"- **Dice Score**: {best_dice['dice_score']:.4f}\n")
            f.write(f"- **Configuration**:\n")
            f.write(f"  - Transformer: {best_dice['USE_TRANSFORMER']}\n")
            f.write(f"  - Augmentation: {best_dice['USE_AUGMENTATION']}\n")
            f.write(f"  - Patch Training: {best_dice['USE_PATCH_TRAINING']}\n")
            f.write(f"  - Class Sampling: {best_dice['USE_CLASS_SAMPLING']}\n")
            f.write(f"  - Loss Type: {best_dice['LOSS_TYPE']}\n\n")
        
        print(f"Saved detailed report to {report_path}")
    
    def generate_all(self):
        """Generate all comparison visualizations and reports"""
        if not self.experiments_data:
            print("No completed experiments found!")
            return
        
        print(f"\n{'='*80}")
        print("EXPERIMENTS COMPARISON")
        print(f"{'='*80}\n")
        
        # Extract metrics
        df = self.extract_metrics()
        
        # Generate all visualizations and reports
        self.create_summary_table(df)
        self.plot_metric_comparison(df)
        self.plot_configuration_impact(df)
        self.plot_correlation_heatmap(df)
        self.plot_loss_type_comparison(df)
        self.create_detailed_report(df)
        
        print(f"\n{'='*80}")
        print("COMPARISON COMPLETE!")
        print(f"{'='*80}\n")
        print(f"All comparison files saved to: {self.comparison_dir}")
        print("\nGenerated files:")
        for file in sorted(self.comparison_dir.glob('*')):
            size = file.stat().st_size / 1024  # KB
            print(f"  • {file.name} ({size:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description='Compare results from multiple experiments'
    )
    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='experiments',
        help='Directory containing experiment results (default: experiments)'
    )
    
    args = parser.parse_args()
    
    comparator = ExperimentComparator(experiments_dir=args.experiments_dir)
    comparator.generate_all()


if __name__ == '__main__':
    main()
