"""Simplified analysis script for optimized parameter sweep results."""

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


class SimplifiedSweepAnalyzer:
    """Analyze parameter sweep results with focus on model comparison."""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.results_df = None
        self.successful_df = None
        self.plots_dir = os.path.join(results_dir, "analysis")
        
        os.makedirs(self.plots_dir, exist_ok=True)
        self.load_results()
    
    def load_results(self):
        """Load results from CSV file."""
        results_path = os.path.join(self.results_dir, "results.csv")
        
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        self.results_df = pd.read_csv(results_path)
        self.successful_df = self.results_df[self.results_df['status'] == 'success'].copy()
        
        print(f"Loaded {len(self.results_df)} experiments")
        print(f"Successful: {len(self.successful_df)} ({len(self.successful_df)/len(self.results_df)*100:.1f}%)")
    
    def get_swept_parameters(self):
        """Find parameters that were varied in the sweep."""
        if len(self.successful_df) == 0:
            return []
        
        swept_params = []
        numeric_cols = self.successful_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['exp_id', 'training_time', 'best_epoch', 'total_epochs'] and \
               col.startswith(('best_', 'final_')) == False:
                if self.successful_df[col].nunique() > 1:
                    swept_params.append(col)
        
        # Add categorical parameters
        for col in ['model_type', 'use_attention_mask', 'use_attention_pooling']:
            if col in self.successful_df.columns and self.successful_df[col].nunique() > 1:
                swept_params.append(col)
        
        return swept_params
    
    def print_summary(self):
        """Print analysis summary."""
        print("\n" + "="*60)
        print("SWEEP ANALYSIS SUMMARY")
        print("="*60)
        print(f"Results directory: {os.path.basename(self.results_dir)}")
        
        if len(self.successful_df) == 0:
            print("No successful experiments to analyze.")
            return
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"  Total experiments: {len(self.results_df)}")
        print(f"  Successful: {len(self.successful_df)}")
        print(f"  Success rate: {len(self.successful_df)/len(self.results_df)*100:.1f}%")
        
        if 'training_time' in self.successful_df.columns:
            avg_time = self.successful_df['training_time'].mean()
            print(f"  Average training time: {avg_time:.1f}s")
        
        # Model type analysis
        if 'model_type' in self.successful_df.columns:
            print(f"\nModel Types:")
            model_counts = self.successful_df['model_type'].value_counts()
            for model_type, count in model_counts.items():
                best_loss = self.successful_df[self.successful_df['model_type'] == model_type]['best_val_loss'].min()
                print(f"  {model_type}: {count} experiments (best loss: {best_loss:.4f})")
        
        # Parameter analysis
        swept_params = self.get_swept_parameters()
        if swept_params:
            print(f"\nSwept Parameters: {', '.join(swept_params)}")
        
        print("="*60)
    
    def print_best_results(self, top_n: int = 5):
        """Print top results."""
        if len(self.successful_df) == 0:
            return
        
        best_df = self.successful_df.nsmallest(top_n, 'best_val_loss')
        
        print(f"\nTOP {top_n} RESULTS:")
        print("-" * 50)
        
        for i, (_, row) in enumerate(best_df.iterrows()):
            print(f"{i+1}. Exp {row['exp_id']}")
            print(f"   Loss: {row['best_val_loss']:.4f}, MAE: {row['best_val_mae']:.4f}")
            
            if 'model_type' in row:
                print(f"   Model: {row['model_type']}")
            
            # Show key parameters
            key_params = ['learning_rate', 'd_model', 'num_heads', 'cell_encoder_units', 
                         'event_encoder_units', 'use_attention_mask']
            params_str = []
            for param in key_params:
                if param in row and pd.notna(row[param]):
                    params_str.append(f"{param}={row[param]}")
            
            if params_str:
                print(f"   Params: {', '.join(params_str)}")
            print()
    
    def plot_model_comparison(self):
        """Plot model type comparison if available."""
        if 'model_type' not in self.successful_df.columns:
            return
        
        model_types = self.successful_df['model_type'].unique()
        if len(model_types) < 2:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance distribution
        ax1 = axes[0]
        for model_type in model_types:
            data = self.successful_df[self.successful_df['model_type'] == model_type]['best_val_loss']
            ax1.hist(data, alpha=0.7, label=f'{model_type} ({len(data)} exp)', bins=15)
        
        ax1.set_xlabel('Best Validation Loss')
        ax1.set_ylabel('Count')
        ax1.set_title('Performance Distribution by Model Type')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        ax2 = axes[1]
        model_data = []
        model_labels = []
        for model_type in model_types:
            data = self.successful_df[self.successful_df['model_type'] == model_type]['best_val_loss']
            model_data.append(data.values)
            model_labels.append(f'{model_type}\n(n={len(data)})')
        
        bp = ax2.boxplot(model_data, labels=model_labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('Best Validation Loss')
        ax2.set_title('Performance Box Plot Comparison')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model comparison plot saved to: {save_path}")
    
    def plot_parameter_effects(self):
        """Plot parameter vs performance."""
        swept_params = self.get_swept_parameters()
        if not swept_params:
            return
        
        # Focus on most important parameters
        important_params = []
        for param in ['learning_rate', 'd_model', 'num_heads', 'cell_dropout_rate', 'attention_hidden_units']:
            if param in swept_params:
                important_params.append(param)
        
        if not important_params:
            important_params = swept_params[:4]  # Take first 4
        
        n_params = len(important_params)
        if n_params == 0:
            return
        
        n_cols = min(2, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, param in enumerate(important_params):
            ax = axes[i] if n_params > 1 else axes[0]
            
            # Different plot types based on parameter type
            if self.successful_df[param].dtype in ['object', 'bool']:
                # Categorical parameter - box plot
                categories = self.successful_df[param].unique()
                data_by_cat = []
                labels = []
                for cat in categories:
                    cat_data = self.successful_df[self.successful_df[param] == cat]['best_val_loss']
                    data_by_cat.append(cat_data.values)
                    labels.append(f'{cat}\n(n={len(cat_data)})')
                
                bp = ax.boxplot(data_by_cat, labels=labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                ax.set_ylabel('Best Validation Loss')
            else:
                # Numerical parameter - scatter plot
                if 'model_type' in self.successful_df.columns:
                    for model_type in self.successful_df['model_type'].unique():
                        model_data = self.successful_df[self.successful_df['model_type'] == model_type]
                        ax.scatter(model_data[param], model_data['best_val_loss'], 
                                 alpha=0.7, label=model_type, s=50)
                    ax.legend()
                else:
                    ax.scatter(self.successful_df[param], self.successful_df['best_val_loss'],
                             alpha=0.7, s=50)
                ax.set_xlabel(param)
                ax.set_ylabel('Best Validation Loss')
            
            ax.set_title(f'{param} vs Performance')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_params, len(axes)):
            if n_params > 1:
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'parameter_effects.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parameter effects plot saved to: {save_path}")
    
    def plot_training_progress(self):
        """Plot training time vs performance."""
        if 'training_time' not in self.successful_df.columns:
            return
        
        plt.figure(figsize=(10, 6))
        
        if 'model_type' in self.successful_df.columns:
            for model_type in self.successful_df['model_type'].unique():
                model_data = self.successful_df[self.successful_df['model_type'] == model_type]
                plt.scatter(model_data['training_time'], model_data['best_val_loss'],
                          alpha=0.7, label=model_type, s=50)
            plt.legend()
        else:
            plt.scatter(self.successful_df['training_time'], self.successful_df['best_val_loss'],
                      alpha=0.7, s=50)
        
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Best Validation Loss')
        plt.title('Training Time vs Performance')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.plots_dir, 'training_efficiency.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training efficiency plot saved to: {save_path}")
    
    def save_summary_report(self):
        """Save text summary report."""
        report_path = os.path.join(self.results_dir, "analysis_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("PARAMETER SWEEP ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Total experiments: {len(self.results_df)}\n")
            f.write(f"Successful: {len(self.successful_df)}\n")
            f.write(f"Success rate: {len(self.successful_df)/len(self.results_df)*100:.1f}%\n\n")
            
            if len(self.successful_df) > 0:
                best_exp = self.successful_df.loc[self.successful_df['best_val_loss'].idxmin()]
                f.write("Best Results:\n")
                f.write(f"Experiment ID: {best_exp['exp_id']}\n")
                f.write(f"Best val_loss: {best_exp['best_val_loss']:.6f}\n")
                f.write(f"Best val_mae: {best_exp['best_val_mae']:.6f}\n")
                
                if 'model_type' in best_exp:
                    f.write(f"Model type: {best_exp['model_type']}\n")
                
                # Model comparison
                if 'model_type' in self.successful_df.columns:
                    f.write(f"\nModel Comparison:\n")
                    for model_type in self.successful_df['model_type'].unique():
                        model_results = self.successful_df[self.successful_df['model_type'] == model_type]
                        best_loss = model_results['best_val_loss'].min()
                        avg_loss = model_results['best_val_loss'].mean()
                        f.write(f"{model_type}: best={best_loss:.4f}, avg={avg_loss:.4f} ({len(model_results)} exp)\n")
        
        print(f"Summary report saved to: {report_path}")
    
    def run_analysis(self):
        """Run complete analysis."""
        print("Running simplified sweep analysis...")
        
        self.print_summary()
        self.print_best_results()
        
        print("\nGenerating plots...")
        self.plot_model_comparison()
        self.plot_parameter_effects()
        self.plot_training_progress()
        
        self.save_summary_report()
        
        print(f"\nAnalysis complete! Results in: {self.plots_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze optimized parameter sweep results')
    parser.add_argument('results_dir', type=str, help='Directory containing sweep results')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top results to show')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1
    
    try:
        analyzer = SimplifiedSweepAnalyzer(args.results_dir)
        analyzer.run_analysis()
        return 0
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
