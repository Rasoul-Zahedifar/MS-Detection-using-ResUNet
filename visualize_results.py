#!/usr/bin/env python3
"""
Comprehensive Results Visualization and Analysis Script
For MS Detection using ResUNet

This script generates various plots, tables, and statistical analyses
from the test results stored in the results folder.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.gridspec import GridSpec
from scipy import stats
import pandas as pd

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

# Define colors
COLOR_DICE = '#2E86AB'
COLOR_LOSS = '#A23B72'
COLOR_PRIMARY = '#06A77D'
COLOR_SECONDARY = '#F18F01'

class ResultsVisualizer:
    """Class to handle all visualization and analysis tasks"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'visualizations'
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_results()
        
    def load_results(self):
        """Load all result JSON files"""
        print("Loading results...")
        
        # Load best_by_dice results
        dice_results_path = self.results_dir / 'best_by_dice' / 'test_results.json'
        if dice_results_path.exists():
            with open(dice_results_path, 'r') as f:
                self.dice_results = json.load(f)
            print(f"✓ Loaded best_by_dice results")
        else:
            self.dice_results = None
            print("✗ No best_by_dice results found")
            
        # Load best_by_loss results
        loss_results_path = self.results_dir / 'best_by_loss' / 'test_results.json'
        if loss_results_path.exists():
            with open(loss_results_path, 'r') as f:
                self.loss_results = json.load(f)
            print(f"✓ Loaded best_by_loss results")
        else:
            self.loss_results = None
            print("✗ No best_by_loss results found")
            
        # Load prediction statistics
        pred_stats_path = self.results_dir / 'best_by_dice_predictions' / 'prediction_statistics.json'
        if pred_stats_path.exists():
            with open(pred_stats_path, 'r') as f:
                self.pred_stats = json.load(f)
            print(f"✓ Loaded prediction statistics")
        else:
            self.pred_stats = None
            print("✗ No prediction statistics found")
    
    def create_metric_comparison_table(self):
        """Create a comparison table between best_by_dice and best_by_loss models"""
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        if self.dice_results and self.loss_results:
            metrics = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
            
            # Create DataFrame
            data = []
            for metric in metrics:
                dice_mean = self.dice_results['average_metrics'][metric]
                dice_std = self.dice_results['std_metrics'][metric]
                loss_mean = self.loss_results['average_metrics'][metric]
                loss_std = self.loss_results['std_metrics'][metric]
                
                data.append({
                    'Metric': metric.capitalize(),
                    'Best by Dice (Mean ± Std)': f"{dice_mean:.4f} ± {dice_std:.4f}",
                    'Best by Loss (Mean ± Std)': f"{loss_mean:.4f} ± {loss_std:.4f}",
                    'Difference': f"{(dice_mean - loss_mean):.4f}",
                    'Better Model': 'Dice' if dice_mean > loss_mean else 'Loss'
                })
            
            df = pd.DataFrame(data)
            
            # Print table
            print(df.to_string(index=False))
            
            # Save to CSV
            csv_path = self.output_dir / 'model_comparison.csv'
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Table saved to: {csv_path}")
            
            # Create visual comparison
            self._plot_metric_comparison(df)
            
            return df
        else:
            print("Insufficient data for comparison")
            return None
    
    def _plot_metric_comparison(self, df):
        """Create bar chart comparing metrics"""
        metrics = ['Dice', 'IoU', 'Accuracy', 'Sensitivity', 'Specificity']
        
        dice_means = []
        loss_means = []
        
        for metric in metrics:
            dice_val = float(self.dice_results['average_metrics'][metric.lower()])
            loss_val = float(self.loss_results['average_metrics'][metric.lower()])
            dice_means.append(dice_val)
            loss_means.append(loss_val)
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, dice_means, width, label='Best by Dice', 
                       color=COLOR_DICE, alpha=0.8)
        bars2 = ax.bar(x + width/2, loss_means, width, label='Best by Loss',
                       color=COLOR_LOSS, alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        save_path = self.output_dir / 'metric_comparison_bar.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {save_path}")
        plt.close()
    
    def create_distribution_plots(self):
        """Create distribution plots for each metric"""
        print("\nGenerating metric distribution plots...")
        
        if not self.dice_results:
            print("No data available for distribution plots")
            return
        
        metrics = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = self.dice_results['all_metrics'][metric]
            
            # Create histogram with KDE
            ax.hist(data, bins=50, alpha=0.7, color=COLOR_PRIMARY, edgecolor='black')
            
            # Add KDE line
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(min(data), max(data), 100)
            ax2 = ax.twinx()
            ax2.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            ax2.set_ylabel('Density', fontsize=10)
            
            # Add statistics
            mean_val = np.mean(data)
            median_val = np.median(data)
            std_val = np.std(data)
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            
            ax.set_xlabel(f'{metric.capitalize()} Score', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{metric.capitalize()} Distribution\n(μ={mean_val:.3f}, σ={std_val:.3f})', 
                        fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(alpha=0.3)
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        plt.tight_layout()
        save_path = self.output_dir / 'metric_distributions.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Distribution plots saved to: {save_path}")
        plt.close()
    
    def create_box_plots(self):
        """Create box plots for metric comparison"""
        print("\nGenerating box plots...")
        
        if not self.dice_results:
            return
        
        metrics = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
        data_to_plot = [self.dice_results['all_metrics'][m] for m in metrics]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bp = ax.boxplot(data_to_plot, labels=[m.capitalize() for m in metrics],
                       patch_artist=True, notch=True, showmeans=True)
        
        # Customize box plots
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize other elements
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Metric Distribution Comparison (Best by Dice Model)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'metric_boxplots.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Box plots saved to: {save_path}")
        plt.close()
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap between metrics"""
        print("\nGenerating correlation heatmap...")
        
        if not self.dice_results:
            return
        
        metrics = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
        
        # Create correlation matrix
        data_dict = {m.capitalize(): self.dice_results['all_metrics'][m] for m in metrics}
        df = pd.DataFrame(data_dict)
        corr_matrix = df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Metric Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        save_path = self.output_dir / 'correlation_heatmap.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Correlation heatmap saved to: {save_path}")
        plt.close()
        
        # Print correlation insights
        print("\nKey Correlations:")
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    print(f"  • {metrics[i].capitalize()} ↔ {metrics[j].capitalize()}: {corr:.3f}")
    
    def create_performance_scatter(self):
        """Create scatter plots showing relationships between metrics"""
        print("\nGenerating scatter plots...")
        
        if not self.dice_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        # Define scatter plot pairs
        pairs = [
            ('dice', 'iou', 'Dice vs IoU'),
            ('sensitivity', 'specificity', 'Sensitivity vs Specificity'),
            ('dice', 'accuracy', 'Dice vs Accuracy'),
            ('iou', 'sensitivity', 'IoU vs Sensitivity')
        ]
        
        for idx, (metric1, metric2, title) in enumerate(pairs):
            ax = axes[idx]
            x_data = self.dice_results['all_metrics'][metric1]
            y_data = self.dice_results['all_metrics'][metric2]
            
            # Scatter plot
            ax.scatter(x_data, y_data, alpha=0.6, s=50, color=COLOR_PRIMARY, edgecolors='black', linewidth=0.5)
            
            # Add regression line
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x_data), max(x_data), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
            
            # Calculate correlation
            corr = np.corrcoef(x_data, y_data)[0, 1]
            
            ax.set_xlabel(metric1.capitalize(), fontsize=11, fontweight='bold')
            ax.set_ylabel(metric2.capitalize(), fontsize=11, fontweight='bold')
            ax.set_title(f'{title}\n(r={corr:.3f})', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'scatter_plots.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Scatter plots saved to: {save_path}")
        plt.close()
    
    def create_statistical_summary(self):
        """Create comprehensive statistical summary"""
        print("\n" + "="*80)
        print("STATISTICAL SUMMARY")
        print("="*80)
        
        if not self.dice_results:
            return
        
        metrics = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
        
        summary_data = []
        for metric in metrics:
            data = self.dice_results['all_metrics'][metric]
            
            stats_dict = {
                'Metric': metric.capitalize(),
                'Mean': np.mean(data),
                'Median': np.median(data),
                'Std Dev': np.std(data),
                'Min': np.min(data),
                'Max': np.max(data),
                'Q1': np.percentile(data, 25),
                'Q3': np.percentile(data, 75),
                'IQR': np.percentile(data, 75) - np.percentile(data, 25),
                'Skewness': stats.skew(data),
                'Kurtosis': stats.kurtosis(data)
            }
            summary_data.append(stats_dict)
        
        df = pd.DataFrame(summary_data)
        
        # Print table
        print("\nDescriptive Statistics:")
        print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        
        # Save to CSV
        csv_path = self.output_dir / 'statistical_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Statistical summary saved to: {csv_path}")
        
        return df
    
    def create_percentile_analysis(self):
        """Create percentile-based analysis"""
        print("\nGenerating percentile analysis...")
        
        if not self.dice_results:
            return
        
        dice_scores = self.dice_results['all_metrics']['dice']
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = [np.percentile(dice_scores, p) for p in percentiles]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(percentiles, percentile_values, color=COLOR_PRIMARY, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for i, (p, v) in enumerate(zip(percentiles, percentile_values)):
            ax.text(p, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Percentile', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
        ax.set_title('Dice Score Percentile Analysis', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'percentile_analysis.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Percentile analysis saved to: {save_path}")
        plt.close()
    
    def create_performance_categories(self):
        """Categorize predictions into performance tiers"""
        print("\nGenerating performance category analysis...")
        
        if not self.dice_results:
            return
        
        dice_scores = np.array(self.dice_results['all_metrics']['dice'])
        
        # Define categories
        excellent = np.sum(dice_scores >= 0.7)
        good = np.sum((dice_scores >= 0.5) & (dice_scores < 0.7))
        fair = np.sum((dice_scores >= 0.3) & (dice_scores < 0.5))
        poor = np.sum((dice_scores >= 0.1) & (dice_scores < 0.3))
        failed = np.sum(dice_scores < 0.1)
        
        categories = ['Excellent\n(≥0.7)', 'Good\n(0.5-0.7)', 'Fair\n(0.3-0.5)', 
                     'Poor\n(0.1-0.3)', 'Failed\n(<0.1)']
        counts = [excellent, good, fair, poor, failed]
        percentages = [c/len(dice_scores)*100 for c in counts]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        colors_cat = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C', '#95A5A6']
        bars = ax1.bar(categories, counts, color=colors_cat, alpha=0.8, edgecolor='black')
        
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax1.set_title('Performance Category Distribution', fontsize=14, fontweight='bold', pad=20)
        ax1.grid(axis='y', alpha=0.3)
        
        # Pie chart
        explode = (0.05, 0.05, 0.05, 0.05, 0.05)
        wedges, texts, autotexts = ax2.pie(counts, labels=categories, autopct='%1.1f%%',
                                           colors=colors_cat, explode=explode,
                                           shadow=True, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax2.set_title('Performance Distribution', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        save_path = self.output_dir / 'performance_categories.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Performance categories saved to: {save_path}")
        plt.close()
        
        # Print summary
        print("\nPerformance Category Summary:")
        print(f"  • Excellent (≥0.7):   {excellent:3d} samples ({percentages[0]:5.1f}%)")
        print(f"  • Good (0.5-0.7):     {good:3d} samples ({percentages[1]:5.1f}%)")
        print(f"  • Fair (0.3-0.5):     {fair:3d} samples ({percentages[2]:5.1f}%)")
        print(f"  • Poor (0.1-0.3):     {poor:3d} samples ({percentages[3]:5.1f}%)")
        print(f"  • Failed (<0.1):      {failed:3d} samples ({percentages[4]:5.1f}%)")
        print(f"  • Total:              {len(dice_scores)} samples")
    
    def create_comprehensive_report(self):
        """Create a comprehensive PDF-style report figure"""
        print("\nGenerating comprehensive report...")
        
        if not self.dice_results:
            return
        
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(6, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        metrics = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
        colors_metrics = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        # Row 1: Metric distributions (histograms)
        for idx, (metric, color) in enumerate(zip(metrics, colors_metrics)):
            ax = fig.add_subplot(gs[0, idx % 3])
            if idx < 3:
                data = self.dice_results['all_metrics'][metric]
                ax.hist(data, bins=30, alpha=0.7, color=color, edgecolor='black')
                mean_val = np.mean(data)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2)
                ax.set_title(f'{metric.capitalize()}\nMean: {mean_val:.3f}', 
                           fontsize=10, fontweight='bold')
                ax.grid(alpha=0.3)
        
        # Row 2: Remaining distributions
        for idx, (metric, color) in enumerate(zip(metrics[3:], colors_metrics[3:])):
            ax = fig.add_subplot(gs[1, idx])
            data = self.dice_results['all_metrics'][metric]
            ax.hist(data, bins=30, alpha=0.7, color=color, edgecolor='black')
            mean_val = np.mean(data)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2)
            ax.set_title(f'{metric.capitalize()}\nMean: {mean_val:.3f}', 
                       fontsize=10, fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Row 3: Box plots
        ax = fig.add_subplot(gs[2, :])
        data_to_plot = [self.dice_results['all_metrics'][m] for m in metrics]
        bp = ax.boxplot(data_to_plot, labels=[m.capitalize() for m in metrics],
                       patch_artist=True, notch=True)
        for patch, color in zip(bp['boxes'], colors_metrics):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title('Metric Distribution Comparison', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Row 4: Correlation heatmap
        ax = fig.add_subplot(gs[3, :2])
        data_dict = {m.capitalize(): self.dice_results['all_metrics'][m] for m in metrics}
        df = pd.DataFrame(data_dict)
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
        
        # Row 4, Col 3: Performance categories pie
        ax = fig.add_subplot(gs[3, 2])
        dice_scores = np.array(self.dice_results['all_metrics']['dice'])
        excellent = np.sum(dice_scores >= 0.7)
        good = np.sum((dice_scores >= 0.5) & (dice_scores < 0.7))
        fair = np.sum((dice_scores >= 0.3) & (dice_scores < 0.5))
        poor = np.sum((dice_scores >= 0.1) & (dice_scores < 0.3))
        failed = np.sum(dice_scores < 0.1)
        
        categories = ['Excellent', 'Good', 'Fair', 'Poor', 'Failed']
        counts = [excellent, good, fair, poor, failed]
        colors_cat = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C', '#95A5A6']
        
        ax.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors_cat, startangle=90)
        ax.set_title('Performance Categories', fontsize=12, fontweight='bold')
        
        # Row 5: Scatter plots
        pairs = [('dice', 'iou'), ('sensitivity', 'specificity'), ('dice', 'accuracy')]
        for idx, (m1, m2) in enumerate(pairs):
            ax = fig.add_subplot(gs[4, idx])
            x_data = self.dice_results['all_metrics'][m1]
            y_data = self.dice_results['all_metrics'][m2]
            ax.scatter(x_data, y_data, alpha=0.5, s=30, color=COLOR_PRIMARY)
            corr = np.corrcoef(x_data, y_data)[0, 1]
            ax.set_xlabel(m1.capitalize(), fontsize=9)
            ax.set_ylabel(m2.capitalize(), fontsize=9)
            ax.set_title(f'{m1.capitalize()} vs {m2.capitalize()}\nr={corr:.3f}', 
                       fontsize=10, fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Row 6: Summary statistics table
        ax = fig.add_subplot(gs[5, :])
        ax.axis('tight')
        ax.axis('off')
        
        # Create table data
        table_data = [['Metric', 'Mean', 'Std', 'Min', 'Max', 'Median']]
        for metric in metrics:
            data = self.dice_results['all_metrics'][metric]
            row = [
                metric.capitalize(),
                f'{np.mean(data):.4f}',
                f'{np.std(data):.4f}',
                f'{np.min(data):.4f}',
                f'{np.max(data):.4f}',
                f'{np.median(data):.4f}'
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.15, 0.17, 0.17, 0.17, 0.17, 0.17])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(6):
            table[(0, i)].set_facecolor('#34495E')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(6):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ECF0F1')
        
        # Add title
        fig.suptitle('MS Detection ResUNet - Comprehensive Results Report', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        save_path = self.output_dir / 'comprehensive_report.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✓ Comprehensive report saved to: {save_path}")
        plt.close()
    
    def create_outlier_analysis(self):
        """Identify and analyze outliers"""
        print("\n" + "="*80)
        print("OUTLIER ANALYSIS")
        print("="*80)
        
        if not self.dice_results:
            return
        
        dice_scores = np.array(self.dice_results['all_metrics']['dice'])
        
        # Calculate IQR
        Q1 = np.percentile(dice_scores, 25)
        Q3 = np.percentile(dice_scores, 75)
        IQR = Q3 - Q1
        
        # Define outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_low = np.where(dice_scores < lower_bound)[0]
        outliers_high = np.where(dice_scores > upper_bound)[0]
        
        print(f"\nDice Score Statistics:")
        print(f"  Q1 (25th percentile): {Q1:.4f}")
        print(f"  Q3 (75th percentile): {Q3:.4f}")
        print(f"  IQR: {IQR:.4f}")
        print(f"  Lower bound: {lower_bound:.4f}")
        print(f"  Upper bound: {upper_bound:.4f}")
        
        print(f"\nOutliers Detected:")
        print(f"  • Low outliers (< {lower_bound:.4f}): {len(outliers_low)} samples")
        print(f"  • High outliers (> {upper_bound:.4f}): {len(outliers_high)} samples")
        
        if len(outliers_low) > 0:
            print(f"\n  Bottom 5 scores:")
            bottom_indices = np.argsort(dice_scores)[:5]
            for idx in bottom_indices:
                print(f"    Sample {idx}: Dice = {dice_scores[idx]:.4f}")
        
        if len(outliers_high) > 0:
            print(f"\n  Top 5 scores:")
            top_indices = np.argsort(dice_scores)[-5:][::-1]
            for idx in top_indices:
                print(f"    Sample {idx}: Dice = {dice_scores[idx]:.4f}")
    
    def generate_all(self):
        """Generate all visualizations and analyses"""
        print("\n" + "="*80)
        print("MS DETECTION RESUNET - RESULTS VISUALIZATION")
        print("="*80)
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print("\nGenerating all visualizations and analyses...\n")
        
        # Generate all visualizations
        self.create_metric_comparison_table()
        self.create_statistical_summary()
        self.create_distribution_plots()
        self.create_box_plots()
        self.create_correlation_heatmap()
        self.create_performance_scatter()
        self.create_percentile_analysis()
        self.create_performance_categories()
        self.create_outlier_analysis()
        self.create_comprehensive_report()
        
        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE")
        print("="*80)
        print(f"\nAll results saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob('*')):
            print(f"  • {file.name}")
        print("\n")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize MS Detection Results')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Path to results directory (default: results)')
    
    args = parser.parse_args()
    
    # Create visualizer and generate all plots
    visualizer = ResultsVisualizer(results_dir=args.results_dir)
    visualizer.generate_all()


if __name__ == '__main__':
    main()

