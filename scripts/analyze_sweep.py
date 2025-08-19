"""Analysis script for parameter sweep results with mask support."""

# python scripts/analyze_sweep.py results/parameter_sweep_20250729_194442/

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import argparse

plt.style.use('default')
sns.set_palette("husl")


class SweepAnalyzer:
    """Analyze parameter sweep results with mask support."""
    
    def __init__(self, results_dir: str):
        """Initialize analyzer."""
        self.results_dir = results_dir
        self.results_df = None
        self.successful_df = None
        self.plots_dir = os.path.join(results_dir, "analysis_plots")
        self.swept_parameters = []
        self.has_mask_comparison = False
        self.mask_analysis = {}
        
        os.makedirs(self.plots_dir, exist_ok=True)
        self.load_results()
        self.find_swept_parameters()
        self.analyze_mask_experiments()
    
    def load_results(self):
        """Load results from CSV file."""
        results_path = os.path.join(self.results_dir, "results.csv")
        
        if not os.path.exists(results_path):
            raise FileNotFoundError("Results file not found: {}".format(results_path))
        
        self.results_df = pd.read_csv(results_path)
        self.successful_df = self.results_df[self.results_df['status'] == 'success'].copy()
        
        print("Loaded {} experiments".format(len(self.results_df)))
        print("Successful: {} ({:.1f}%)".format(
            len(self.successful_df), len(self.successful_df)/len(self.results_df)*100))
    
    def find_swept_parameters(self):
        """Find which parameters were varied."""
        possible_params = [
            'd_model', 'num_heads', 'num_transformer_blocks', 'dropout_rate',
            'learning_rate', 'batch_size', 'vertex_dense_units', 'dff',
            'lr_reduction_factor', 'use_attention_mask', 'use_spatial_features',
            'use_jet_features', 'use_cell_jet_matching'
        ]
        
        for param in possible_params:
            if param in self.results_df.columns and self.results_df[param].nunique() > 1:
                self.swept_parameters.append(param)
    
    def analyze_mask_experiments(self):
        """Analyze mask-related experiments."""
        if 'use_attention_mask' in self.successful_df.columns:
            mask_values = self.successful_df['use_attention_mask'].unique()
            
            if len(mask_values) > 1:
                self.has_mask_comparison = True
                self.mask_analysis['comparison_type'] = 'mask_vs_traditional'
                
                # Analyze mask vs traditional performance
                mask_results = self.successful_df[self.successful_df['use_attention_mask'] == True]
                trad_results = self.successful_df[self.successful_df['use_attention_mask'] == False]
                
                self.mask_analysis['mask_count'] = len(mask_results)
                self.mask_analysis['traditional_count'] = len(trad_results)
                
                if len(mask_results) > 0 and len(trad_results) > 0:
                    self.mask_analysis['mask_best'] = mask_results['best_val_loss'].min()
                    self.mask_analysis['mask_mean'] = mask_results['best_val_loss'].mean()
                    self.mask_analysis['traditional_best'] = trad_results['best_val_loss'].min()
                    self.mask_analysis['traditional_mean'] = trad_results['best_val_loss'].mean()
                    
                    # Calculate improvement
                    improvement = (self.mask_analysis['traditional_best'] - self.mask_analysis['mask_best']) / self.mask_analysis['traditional_best'] * 100
                    mean_improvement = (self.mask_analysis['traditional_mean'] - self.mask_analysis['mask_mean']) / self.mask_analysis['traditional_mean'] * 100
                    
                    self.mask_analysis['best_improvement'] = improvement
                    self.mask_analysis['mean_improvement'] = mean_improvement
                    
            elif mask_values[0] == True:
                self.mask_analysis['comparison_type'] = 'mask_only'
            else:
                self.mask_analysis['comparison_type'] = 'traditional_only'
        else:
            self.mask_analysis['comparison_type'] = 'no_mask_param'
    
    def print_parameter_summary(self):
        """Print summary of swept parameters with mask information."""
        print("\n" + "="*70)
        print("PARAMETER SWEEP ANALYSIS SUMMARY")
        print("="*70)
        print("Results directory: {}".format(os.path.basename(self.results_dir)))
        
        if self.swept_parameters:
            print("\nSwept parameters:")
            for param in self.swept_parameters:
                values = sorted(self.results_df[param].unique())
                print("  {}: {}".format(param, values))
        else:
            print("No parameter variations found.")
        
        # Print mask analysis summary
        self.print_mask_summary()
        
        print("="*70)
    
    def print_mask_summary(self):
        """Print mask-specific analysis summary."""
        print("\nMask Analysis:")
        
        if self.mask_analysis['comparison_type'] == 'mask_vs_traditional':
            print("  Type: Mask vs Traditional Comparison")
            print("  Mask experiments: {}".format(self.mask_analysis['mask_count']))
            print("  Traditional experiments: {}".format(self.mask_analysis['traditional_count']))
            
            if 'best_improvement' in self.mask_analysis:
                print("  Best model improvement: {:.2f}%".format(self.mask_analysis['best_improvement']))
                print("  Average improvement: {:.2f}%".format(self.mask_analysis['mean_improvement']))
                
        elif self.mask_analysis['comparison_type'] == 'mask_only':
            print("  Type: Mask Optimization Only")
            print("  All experiments use attention masks")
            
        elif self.mask_analysis['comparison_type'] == 'traditional_only':
            print("  Type: Traditional Models Only")
            print("  No attention masks used")
            
        else:
            print("  Type: No mask parameter variation")
    
    def print_best_results(self, top_n=5):
        """Print best results with mask information."""
        if len(self.successful_df) == 0:
            print("No successful experiments to analyze.")
            return
        
        best_df = self.successful_df.nsmallest(top_n, 'best_val_loss')
        
        print("\nTOP {} RESULTS:".format(top_n))
        print("-" * 50)
        
        for i, (_, row) in enumerate(best_df.iterrows()):
            print("{}. {}".format(i+1, row['experiment_id']))
            print("   Val Loss: {:.6f}".format(row['best_val_loss']))
            print("   Val MAE:  {:.6f}".format(row['best_val_mae']))
            
            # Show mask information if available
            if 'use_attention_mask' in row:
                mask_status = "Mask" if row['use_attention_mask'] else "Traditional"
                print("   Type: {}".format(mask_status))
            
            # Show swept parameter values
            param_str = ", ".join(["{}={}".format(p, row[p]) for p in self.swept_parameters if p in row])
            if param_str:
                print("   Params: {}".format(param_str))
            print()
    
    def analyze_parameter_importance(self):
        """Analyze parameter importance with mask consideration."""
        if not self.swept_parameters or len(self.successful_df) == 0:
            return {}
        
        correlations = {}
        
        # Separate analysis for mask and traditional if both exist
        if self.has_mask_comparison:
            print("\nPARAMETER IMPORTANCE ANALYSIS:")
            print("-" * 40)
            
            mask_df = self.successful_df[self.successful_df['use_attention_mask'] == True]
            trad_df = self.successful_df[self.successful_df['use_attention_mask'] == False]
            
            if len(mask_df) > 1:
                print("For MASK models:")
                mask_correlations = {}
                for param in self.swept_parameters:
                    if param != 'use_attention_mask' and param in mask_df.columns:
                        corr = abs(mask_df[param].corr(mask_df['best_val_loss']))
                        if not np.isnan(corr):
                            mask_correlations[param] = corr
                
                mask_importance = dict(sorted(mask_correlations.items(), key=lambda x: x[1], reverse=True))
                for param, score in mask_importance.items():
                    print("  {}: {:.4f}".format(param, score))
            
            if len(trad_df) > 1:
                print("\nFor TRADITIONAL models:")
                trad_correlations = {}
                for param in self.swept_parameters:
                    if param != 'use_attention_mask' and param in trad_df.columns:
                        corr = abs(trad_df[param].corr(trad_df['best_val_loss']))
                        if not np.isnan(corr):
                            trad_correlations[param] = corr
                
                trad_importance = dict(sorted(trad_correlations.items(), key=lambda x: x[1], reverse=True))
                for param, score in trad_importance.items():
                    print("  {}: {:.4f}".format(param, score))
        
        # Overall analysis
        print("\nOVERALL PARAMETER IMPORTANCE:")
        print("-" * 30)
        for param in self.swept_parameters:
            if param in self.successful_df.columns:
                corr = abs(self.successful_df[param].corr(self.successful_df['best_val_loss']))
                if not np.isnan(corr):
                    correlations[param] = corr
        
        importance = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
        for param, score in importance.items():
            print("  {}: {:.4f}".format(param, score))
        
        return importance
    
    def plot_mask_comparison(self):
        """Plot mask vs traditional comparison if applicable."""
        if not self.has_mask_comparison:
            return
        
        mask_df = self.successful_df[self.successful_df['use_attention_mask'] == True]
        trad_df = self.successful_df[self.successful_df['use_attention_mask'] == False]
        
        if len(mask_df) == 0 or len(trad_df) == 0:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Performance distribution comparison
        ax1 = axes[0]
        ax1.hist(mask_df['best_val_loss'], alpha=0.7, label='Mask Models', bins=15, color='blue')
        ax1.hist(trad_df['best_val_loss'], alpha=0.7, label='Traditional Models', bins=15, color='orange')
        ax1.set_xlabel('Best Validation Loss')
        ax1.set_ylabel('Count')
        ax1.set_title('Performance Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot comparison
        ax2 = axes[1]
        data_for_box = [mask_df['best_val_loss'].values, trad_df['best_val_loss'].values]
        labels = ['Mask', 'Traditional']
        bp = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('orange')
        ax2.set_ylabel('Best Validation Loss')
        ax2.set_title('Performance Box Plot Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Best performance by parameter
        if 'd_model' in self.swept_parameters:
            ax3 = axes[2]
            
            # Group by d_model and calculate best performance
            mask_grouped = mask_df.groupby('d_model')['best_val_loss'].min()
            trad_grouped = trad_df.groupby('d_model')['best_val_loss'].min()
            
            x_pos = np.arange(len(mask_grouped.index))
            width = 0.35
            
            ax3.bar(x_pos - width/2, mask_grouped.values, width, label='Mask', alpha=0.8, color='blue')
            ax3.bar(x_pos + width/2, trad_grouped.values, width, label='Traditional', alpha=0.8, color='orange')
            
            ax3.set_xlabel('d_model')
            ax3.set_ylabel('Best Validation Loss')
            ax3.set_title('Best Performance by d_model')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(mask_grouped.index)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No d_model parameter\nto compare', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Parameter Comparison')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'mask_vs_traditional_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("Mask comparison plot saved to: {}".format(save_path))
    
    def plot_parameter_vs_performance(self):
        """Plot parameters vs performance with mask differentiation."""
        if not self.swept_parameters:
            return
        
        n_params = len(self.swept_parameters)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, param in enumerate(self.swept_parameters):
            ax = axes[i]
            
            if self.has_mask_comparison and param != 'use_attention_mask':
                # Plot mask and traditional separately
                mask_df = self.successful_df[self.successful_df['use_attention_mask'] == True]
                trad_df = self.successful_df[self.successful_df['use_attention_mask'] == False]
                
                if len(mask_df) > 0:
                    ax.scatter(mask_df[param], mask_df['best_val_loss'], 
                             alpha=0.7, label='Mask', color='blue', s=50)
                if len(trad_df) > 0:
                    ax.scatter(trad_df[param], trad_df['best_val_loss'], 
                             alpha=0.7, label='Traditional', color='orange', s=50)
                ax.legend()
            else:
                ax.scatter(self.successful_df[param], self.successful_df['best_val_loss'],
                         alpha=0.7, s=50)
            
            ax.set_xlabel(param)
            ax.set_ylabel('Best Validation Loss')
            ax.set_title('{} vs Performance'.format(param))
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_params, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'parameter_vs_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("Parameter plots saved to: {}".format(save_path))
    
    def plot_performance_distribution(self):
        """Plot performance distribution with mask information."""
        if len(self.successful_df) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Validation loss distribution
        if self.has_mask_comparison:
            mask_df = self.successful_df[self.successful_df['use_attention_mask'] == True]
            trad_df = self.successful_df[self.successful_df['use_attention_mask'] == False]
            
            if len(mask_df) > 0:
                ax1.hist(mask_df['best_val_loss'], bins=20, alpha=0.7, 
                        label='Mask Models', color='blue', edgecolor='black')
            if len(trad_df) > 0:
                ax1.hist(trad_df['best_val_loss'], bins=20, alpha=0.7, 
                        label='Traditional Models', color='orange', edgecolor='black')
            ax1.legend()
        else:
            ax1.hist(self.successful_df['best_val_loss'], bins=20, alpha=0.7, 
                    color='blue', edgecolor='black')
        
        ax1.set_xlabel('Best Validation Loss')
        ax1.set_ylabel('Count')
        ax1.set_title('Validation Loss Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Training time vs performance
        if 'training_time' in self.successful_df.columns:
            if self.has_mask_comparison:
                mask_df = self.successful_df[self.successful_df['use_attention_mask'] == True]
                trad_df = self.successful_df[self.successful_df['use_attention_mask'] == False]
                
                if len(mask_df) > 0:
                    ax2.scatter(mask_df['training_time'], mask_df['best_val_loss'],
                              alpha=0.7, label='Mask', color='blue')
                if len(trad_df) > 0:
                    ax2.scatter(trad_df['training_time'], trad_df['best_val_loss'],
                              alpha=0.7, label='Traditional', color='orange')
                ax2.legend()
            else:
                ax2.scatter(self.successful_df['training_time'], self.successful_df['best_val_loss'],
                          alpha=0.7, color='blue')
            
            ax2.set_xlabel('Training Time (seconds)')
            ax2.set_ylabel('Best Validation Loss')
            ax2.set_title('Training Time vs Performance')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No training time data available', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Training Time Analysis')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'performance_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("Performance distribution saved to: {}".format(save_path))
    
    def generate_recommendations(self):
        """Generate recommendations with mask-aware analysis."""
        if len(self.successful_df) == 0:
            return {}
        
        recommendations = {'general': {}, 'mask_specific': {}}
        
        # General best experiment
        best_exp = self.successful_df.loc[self.successful_df['best_val_loss'].idxmin()]
        recommendations['general']['best_experiment'] = {
            'id': best_exp['experiment_id'],
            'val_loss': best_exp['best_val_loss'],
            'parameters': {param: best_exp[param] for param in self.swept_parameters if param in best_exp}
        }
        
        # Mask-specific recommendations
        if self.has_mask_comparison:
            mask_df = self.successful_df[self.successful_df['use_attention_mask'] == True]
            trad_df = self.successful_df[self.successful_df['use_attention_mask'] == False]
            
            if len(mask_df) > 0:
                best_mask = mask_df.loc[mask_df['best_val_loss'].idxmin()]
                recommendations['mask_specific']['best_mask_experiment'] = {
                    'id': best_mask['experiment_id'],
                    'val_loss': best_mask['best_val_loss'],
                    'parameters': {param: best_mask[param] for param in self.swept_parameters if param in best_mask and param != 'use_attention_mask'}
                }
            
            if len(trad_df) > 0:
                best_trad = trad_df.loc[trad_df['best_val_loss'].idxmin()]
                recommendations['mask_specific']['best_traditional_experiment'] = {
                    'id': best_trad['experiment_id'],
                    'val_loss': best_trad['best_val_loss'],
                    'parameters': {param: best_trad[param] for param in self.swept_parameters if param in best_trad and param != 'use_attention_mask'}
                }
            
            # Recommendation based on comparison
            if 'best_improvement' in self.mask_analysis:
                if self.mask_analysis['best_improvement'] > 2:  # >2% improvement
                    recommendations['mask_specific']['recommendation'] = "Use attention masks - significant improvement observed"
                elif self.mask_analysis['best_improvement'] > 0:
                    recommendations['mask_specific']['recommendation'] = "Use attention masks - modest improvement observed"
                else:
                    recommendations['mask_specific']['recommendation'] = "Traditional models perform equally well or better"
        
        return recommendations
    
    def save_summary_report(self):
        """Save comprehensive summary report with mask analysis."""
        report_path = os.path.join(self.results_dir, "summary_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("PARAMETER SWEEP ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("Experiment Overview:\n")
            f.write("Total experiments: {}\n".format(len(self.results_df)))
            f.write("Successful: {}\n".format(len(self.successful_df)))
            f.write("Success rate: {:.1f}%\n\n".format(len(self.successful_df)/len(self.results_df)*100))
            
            if self.swept_parameters:
                f.write("Swept parameters:\n")
                for param in self.swept_parameters:
                    values = sorted(self.results_df[param].unique())
                    f.write("  {}: {}\n".format(param, values))
                f.write("\n")
            
            # Mask analysis section
            f.write("Mask Analysis:\n")
            f.write("-" * 20 + "\n")
            if self.mask_analysis['comparison_type'] == 'mask_vs_traditional':
                f.write("Type: Mask vs Traditional Comparison\n")
                f.write("Mask experiments: {}\n".format(self.mask_analysis['mask_count']))
                f.write("Traditional experiments: {}\n".format(self.mask_analysis['traditional_count']))
                if 'best_improvement' in self.mask_analysis:
                    f.write("Best model improvement: {:.2f}%\n".format(self.mask_analysis['best_improvement']))
                    f.write("Average improvement: {:.2f}%\n".format(self.mask_analysis['mean_improvement']))
            else:
                f.write("Type: {}\n".format(self.mask_analysis['comparison_type']))
            f.write("\n")
            
            if len(self.successful_df) > 0:
                best_exp = self.successful_df.loc[self.successful_df['best_val_loss'].idxmin()]
                f.write("Overall Best Results:\n")
                f.write("Best experiment: {}\n".format(best_exp['experiment_id']))
                f.write("Best val_loss: {:.6f}\n".format(best_exp['best_val_loss']))
                f.write("Best val_mae: {:.6f}\n".format(best_exp['best_val_mae']))
                
                if 'use_attention_mask' in best_exp:
                    mask_status = "Mask-enabled" if best_exp['use_attention_mask'] else "Traditional"
                    f.write("Model type: {}\n".format(mask_status))
                
                f.write("\nBest parameters:\n")
                for param in self.swept_parameters:
                    if param in best_exp:
                        f.write("  {}: {}\n".format(param, best_exp[param]))
        
        print("Summary report saved to: {}".format(report_path))
    
    def run_complete_analysis(self):
        """Run complete analysis with mask support."""
        print("Running sweep analysis with mask support...")
        
        self.print_parameter_summary()
        self.print_best_results()
        self.analyze_parameter_importance()
        
        print("\nGenerating plots...")
        if self.has_mask_comparison:
            self.plot_mask_comparison()
        self.plot_parameter_vs_performance()
        self.plot_performance_distribution()
        
        recommendations = self.generate_recommendations()
        self.save_summary_report()
        
        print("\nAnalysis complete! Results in: {}".format(self.plots_dir))
        return recommendations


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze parameter sweep results with mask support')
    parser.add_argument('results_dir', type=str, help='Directory containing sweep results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print("Error: Results directory not found: {}".format(args.results_dir))
        return 1
    
    analyzer = SweepAnalyzer(args.results_dir)
    recommendations = analyzer.run_complete_analysis()
    
    # Print final recommendations
    if recommendations and 'general' in recommendations:
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        if 'best_experiment' in recommendations['general']:
            best = recommendations['general']['best_experiment']
            print("Overall best configuration: {}".format(best['id']))
            print("Validation loss: {:.6f}".format(best['val_loss']))
            print("Parameters:")
            for param, value in best['parameters'].items():
                print("  {}: {}".format(param, value))
        
        if 'mask_specific' in recommendations and recommendations['mask_specific']:
            print("\nMask-Specific Analysis:")
            mask_rec = recommendations['mask_specific']
            
            if 'best_mask_experiment' in mask_rec:
                print("Best mask model: {} (loss: {:.6f})".format(
                    mask_rec['best_mask_experiment']['id'],
                    mask_rec['best_mask_experiment']['val_loss']))
            
            if 'best_traditional_experiment' in mask_rec:
                print("Best traditional model: {} (loss: {:.6f})".format(
                    mask_rec['best_traditional_experiment']['id'],
                    mask_rec['best_traditional_experiment']['val_loss']))
            
            if 'recommendation' in mask_rec:
                print("\nRecommendation: {}".format(mask_rec['recommendation']))
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
