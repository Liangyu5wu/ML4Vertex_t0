"""Analysis script for parameter sweep results."""

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
    """Analyze parameter sweep results."""
    
    def __init__(self, results_dir: str):
        """Initialize analyzer."""
        self.results_dir = results_dir
        self.results_df = None
        self.successful_df = None
        self.plots_dir = os.path.join(results_dir, "analysis_plots")
        self.swept_parameters = []
        
        os.makedirs(self.plots_dir, exist_ok=True)
        self.load_results()
        self.find_swept_parameters()
    
    def load_results(self):
        """Load results from CSV file."""
        results_path = os.path.join(self.results_dir, "results.csv")
        
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        self.results_df = pd.read_csv(results_path)
        self.successful_df = self.results_df[self.results_df['status'] == 'success'].copy()
        
        print(f"Loaded {len(self.results_df)} experiments")
        print(f"Successful: {len(self.successful_df)} ({len(self.successful_df)/len(self.results_df)*100:.1f}%)")
    
    def find_swept_parameters(self):
        """Find which parameters were varied."""
        possible_params = [
            'd_model', 'num_heads', 'num_transformer_blocks', 'dropout_rate',
            'learning_rate', 'batch_size', 'vertex_dense_units', 'dff',
            'lr_reduction_factor'
        ]
        
        for param in possible_params:
            if param in self.results_df.columns and self.results_df[param].nunique() > 1:
                self.swept_parameters.append(param)
    
    def print_parameter_summary(self):
        """Print summary of swept parameters."""
        print(f"\n{'='*60}")
        print("PARAMETER SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"Results directory: {os.path.basename(self.results_dir)}")
        
        if self.swept_parameters:
            print(f"\nSwept parameters:")
            for param in self.swept_parameters:
                values = sorted(self.results_df[param].unique())
                print(f"  {param}: {values}")
        else:
            print("No parameter variations found.")
        print(f"{'='*60}")
    
    def print_best_results(self, top_n: int = 5):
        """Print best results."""
        if len(self.successful_df) == 0:
            print("No successful experiments to analyze.")
            return
        
        best_df = self.successful_df.nsmallest(top_n, 'best_val_loss')
        
        print(f"\nTOP {top_n} RESULTS:")
        print("-" * 40)
        
        for i, (_, row) in enumerate(best_df.iterrows()):
            print(f"{i+1}. {row['experiment_id']}")
            print(f"   Val Loss: {row['best_val_loss']:.6f}")
            print(f"   Val MAE:  {row['best_val_mae']:.6f}")
            
            # Show swept parameter values
            param_str = ", ".join([f"{p}={row[p]}" for p in self.swept_parameters if p in row])
            if param_str:
                print(f"   Params: {param_str}")
            print()
    
    def analyze_parameter_importance(self) -> Dict[str, float]:
        """Analyze parameter importance using correlation."""
        if not self.swept_parameters or len(self.successful_df) == 0:
            return {}
        
        correlations = {}
        for param in self.swept_parameters:
            if param in self.successful_df.columns:
                corr = abs(self.successful_df[param].corr(self.successful_df['best_val_loss']))
                correlations[param] = corr
        
        # Sort by importance
        importance = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
        
        print(f"\nPARAMETER IMPORTANCE:")
        print("-" * 30)
        for param, score in importance.items():
            print(f"  {param}: {score:.4f}")
        
        return importance
    
    def plot_parameter_vs_performance(self):
        """Plot parameters vs performance."""
        if not self.swept_parameters:
            return
        
        n_params = len(self.swept_parameters)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, param in enumerate(self.swept_parameters):
            ax = axes[i]
            
            ax.scatter(
                self.successful_df[param], 
                self.successful_df['best_val_loss'],
                alpha=0.7
            )
            
            ax.set_xlabel(param)
            ax.set_ylabel('Best Validation Loss')
            ax.set_title(f'{param} vs Performance')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_params, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'parameter_vs_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parameter plots saved to: {save_path}")
    
    def plot_performance_distribution(self):
        """Plot performance distribution."""
        if len(self.successful_df) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Validation loss distribution
        ax1.hist(self.successful_df['best_val_loss'], bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Best Validation Loss')
        ax1.set_ylabel('Count')
        ax1.set_title('Validation Loss Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Training time vs performance
        if 'training_time' in self.successful_df.columns:
            ax2.scatter(
                self.successful_df['training_time'],
                self.successful_df['best_val_loss'],
                alpha=0.7
            )
            ax2.set_xlabel('Training Time (seconds)')
            ax2.set_ylabel('Best Validation Loss')
            ax2.set_title('Training Time vs Performance')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'performance_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance distribution saved to: {save_path}")
    
    def generate_recommendations(self) -> Dict:
        """Generate simple recommendations."""
        if len(self.successful_df) == 0:
            return {}
        
        # Best overall
        best_exp = self.successful_df.loc[self.successful_df['best_val_loss'].idxmin()]
        
        # Best parameter ranges (top 25%)
        top_25_percent = self.successful_df.nsmallest(
            max(1, int(len(self.successful_df) * 0.25)), 'best_val_loss'
        )
        
        recommendations = {
            'best_experiment': {
                'id': best_exp['experiment_id'],
                'val_loss': best_exp['best_val_loss'],
                'parameters': {param: best_exp[param] for param in self.swept_parameters if param in best_exp}
            },
            'good_parameter_ranges': {}
        }
        
        for param in self.swept_parameters:
            if param in top_25_percent.columns and top_25_percent[param].nunique() > 1:
                recommendations['good_parameter_ranges'][param] = {
                    'min': top_25_percent[param].min(),
                    'max': top_25_percent[param].max(),
                    'best_values': sorted(top_25_percent[param].unique())
                }
        
        return recommendations
    
    def save_summary_report(self):
        """Save simple summary report."""
        report_path = os.path.join(self.results_dir, "summary_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("PARAMETER SWEEP SUMMARY REPORT\n")
            f.write("="*40 + "\n\n")
            
            f.write(f"Total experiments: {len(self.results_df)}\n")
            f.write(f"Successful: {len(self.successful_df)}\n")
            f.write(f"Success rate: {len(self.successful_df)/len(self.results_df)*100:.1f}%\n\n")
            
            if self.swept_parameters:
                f.write("Swept parameters:\n")
                for param in self.swept_parameters:
                    values = sorted(self.results_df[param].unique())
                    f.write(f"  {param}: {values}\n")
                f.write("\n")
            
            if len(self.successful_df) > 0:
                best_exp = self.successful_df.loc[self.successful_df['best_val_loss'].idxmin()]
                f.write(f"Best experiment: {best_exp['experiment_id']}\n")
                f.write(f"Best val_loss: {best_exp['best_val_loss']:.6f}\n")
                f.write(f"Best val_mae: {best_exp['best_val_mae']:.6f}\n")
                
                f.write("\nBest parameters:\n")
                for param in self.swept_parameters:
                    if param in best_exp:
                        f.write(f"  {param}: {best_exp[param]}\n")
        
        print(f"Summary report saved to: {report_path}")
    
    def run_complete_analysis(self):
        """Run complete analysis."""
        print("Running sweep analysis...")
        
        self.print_parameter_summary()
        self.print_best_results()
        self.analyze_parameter_importance()
        
        print("\nGenerating plots...")
        self.plot_parameter_vs_performance()
        self.plot_performance_distribution()
        
        recommendations = self.generate_recommendations()
        self.save_summary_report()
        
        print(f"\nAnalysis complete! Results in: {self.plots_dir}")
        return recommendations


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze parameter sweep results')
    parser.add_argument('results_dir', type=str, help='Directory containing sweep results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    analyzer = SweepAnalyzer(args.results_dir)
    recommendations = analyzer.run_complete_analysis()
    
    # Print final recommendations
    if recommendations and 'best_experiment' in recommendations:
        print(f"\n{'='*50}")
        print("RECOMMENDATIONS")
        print(f"{'='*50}")
        
        best = recommendations['best_experiment']
        print(f"Best configuration: {best['id']}")
        print(f"Validation loss: {best['val_loss']:.6f}")
        print(f"Parameters:")
        for param, value in best['parameters'].items():
            print(f"  {param}: {value}")
        
        if 'good_parameter_ranges' in recommendations:
            print(f"\nGood parameter ranges:")
            for param, info in recommendations['good_parameter_ranges'].items():
                print(f"  {param}: {info['min']} to {info['max']}")


if __name__ == "__main__":
    main()