"""Visualization utilities for model evaluation."""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import seaborn as sns

from config.base_config import BaseConfig


class Visualizer:
    """Handle visualization of training and evaluation results."""
    
    def __init__(self, config: BaseConfig):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Set matplotlib parameters for better plots
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        
    def plot_training_history(
        self, 
        history: Dict[str, np.ndarray], 
        save_path: Optional[str] = None
    ):
        """
        Plot training history metrics.
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, "training_metrics.png")
        
        # Determine number of subplots needed (exclude RMSE metrics)
        metrics = [key for key in history.keys() if not key.startswith('val_') and 'root_mean_squared_error' not in key]
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot training metric
            ax.plot(history[metric], label=f'Training {metric.upper()}', linewidth=2)
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax.plot(history[val_metric], label=f'Validation {metric.upper()}', linewidth=2)
            
            ax.set_title(f'Model {metric.upper()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.upper())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Find best epoch for validation metric
            if val_metric in history:
                best_epoch = np.argmin(history[val_metric]) if 'loss' in metric else np.argmax(history[val_metric])
                best_val = history[val_metric][best_epoch]
                ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7)
                ax.text(0.02, 0.98, f'Best: {best_val:.4f} (epoch {best_epoch + 1})', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
        plt.close()
    
    def plot_prediction_results(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metrics: Dict[str, float],
        save_path: Optional[str] = None,
        plot_type: str = '2d_hist'  # 新增参数
    ):
        """
        Plot predicted vs actual values as 2D histogram or scatter plot.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: Evaluation metrics
            save_path: Path to save the plot
            plot_type: Type of plot ('2d_hist', 'hexbin', or 'scatter')
        """
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, "prediction_results.png")
        
        plt.figure(figsize=(10, 8))
        
        # Determine plot range
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        
        # Add some padding
        range_padding = (max_val - min_val) * 0.05
        plot_min = min_val - range_padding
        plot_max = max_val + range_padding
        
        if plot_type == '2d_hist':
            # Create 2D histogram
            bins = min(50, int(np.sqrt(len(y_true))))  # Adaptive bin number
            hist, xedges, yedges = np.histogram2d(
                y_true, y_pred, 
                bins=bins, 
                range=[[plot_min, plot_max], [plot_min, plot_max]]
            )
            
            # Plot 2D histogram
            im = plt.imshow(
                hist.T, 
                origin='lower',
                extent=[plot_min, plot_max, plot_min, plot_max],
                cmap='Blues',
                aspect='equal',
                interpolation='bilinear'
            )
            
            # Add colorbar
            cbar = plt.colorbar(im, label='Number of Points')
            cbar.ax.tick_params(labelsize=9)
            
        elif plot_type == 'hexbin':
            # Create hexagonal binning plot
            hb = plt.hexbin(
                y_true, y_pred,
                gridsize=30,
                cmap='Blues',
                mincnt=1,
                extent=[plot_min, plot_max, plot_min, plot_max]
            )
            
            # Add colorbar
            cbar = plt.colorbar(hb, label='Number of Points')
            cbar.ax.tick_params(labelsize=9)
            
        else:  # Default to scatter plot
            # Create scatter plot with alpha for density visualization
            plt.scatter(y_true, y_pred, alpha=0.6, s=20, c='blue')
        
        # Perfect prediction line
        plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2, 
                label='Perfect Prediction', alpha=0.8)
        
        # Add metrics text box
        metrics_text = f"R² = {metrics['r_squared']:.4f}\n"
        metrics_text += f"RMSE = {metrics['rmse']:.4f}\n"
        metrics_text += f"MAE = {metrics['mae']:.4f}\n"
        metrics_text += f"Correlation = {metrics['correlation']:.4f}\n"
        metrics_text += f"N = {len(y_true):,} points"
        
        # Position text box in upper left
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                fontsize=10)
        
        # Add statistical information in lower right
        bias = np.mean(y_pred - y_true)
        stats_text = f"Bias = {bias:.4f}\n"
        stats_text += f"Std Error = {metrics['error_std']:.4f}"
        
        plt.text(0.95, 0.05, stats_text, transform=plt.gca().transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
                fontsize=9)
        
        # Formatting
        plt.xlabel('Actual Vertex Time', fontsize=12)
        plt.ylabel('Predicted Vertex Time', fontsize=12)
        plt.title(f'{self.config.model_name}: Predicted vs Actual Values (2D Histogram)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Equal aspect ratio and axis limits
        plt.axis('equal')
        plt.xlim(plot_min, plot_max)
        plt.ylim(plot_min, plot_max)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction results plot saved to: {save_path}")
        plt.close()
    
    def plot_prediction_results_comparison(
        self,
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Create a comparison plot with both scatter and 2D histogram views.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: Evaluation metrics
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, "prediction_results_comparison.png")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Determine plot range
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        range_padding = (max_val - min_val) * 0.05
        plot_min = min_val - range_padding
        plot_max = max_val + range_padding
        
        # Left plot: 2D Histogram
        bins = min(50, int(np.sqrt(len(y_true))))
        hist, xedges, yedges = np.histogram2d(
            y_true, y_pred, 
            bins=bins, 
            range=[[plot_min, plot_max], [plot_min, plot_max]]
        )
        
        im1 = ax1.imshow(
            hist.T, 
            origin='lower',
            extent=[plot_min, plot_max, plot_min, plot_max],
            cmap='Blues',
            aspect='equal',
            interpolation='bilinear'
        )
        
        # Perfect prediction line for histogram
        ax1.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2, 
                label='Perfect Prediction', alpha=0.8)
        
        ax1.set_xlabel('Actual Vertex Time', fontsize=11)
        ax1.set_ylabel('Predicted Vertex Time', fontsize=11)
        ax1.set_title('2D Histogram View', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(plot_min, plot_max)
        ax1.set_ylim(plot_min, plot_max)
        
        # Add colorbar for left plot
        cbar1 = plt.colorbar(im1, ax=ax1, label='Count')
        cbar1.ax.tick_params(labelsize=9)
        
        # Right plot: Hexbin
        hb = ax2.hexbin(
            y_true, y_pred,
            gridsize=35,
            cmap='Reds',
            mincnt=1,
            extent=[plot_min, plot_max, plot_min, plot_max]
        )
        
        # Perfect prediction line for hexbin
        ax2.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', linewidth=2, 
                label='Perfect Prediction', alpha=0.8)
        
        ax2.set_xlabel('Actual Vertex Time', fontsize=11)
        ax2.set_ylabel('Predicted Vertex Time', fontsize=11)
        ax2.set_title('Hexagonal Binning View', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(plot_min, plot_max)
        ax2.set_ylim(plot_min, plot_max)
        
        # Add colorbar for right plot
        cbar2 = plt.colorbar(hb, ax=ax2, label='Count')
        cbar2.ax.tick_params(labelsize=9)
        
        # Add metrics text box (shared)
        metrics_text = f"R² = {metrics['r_squared']:.4f}  |  RMSE = {metrics['rmse']:.4f}  |  MAE = {metrics['mae']:.4f}  |  N = {len(y_true):,}"
        
        fig.suptitle(f'{self.config.model_name}: Predicted vs Actual Values - Density Visualization', 
                    fontsize=14, y=0.95)
        
        # Add metrics as figure text
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.88)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction results comparison plot saved to: {save_path}")
        plt.close()
    
    def plot_histogram_comparison(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot histogram comparison of actual vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, "histogram_comparison.png")
        
        # Set matplotlib parameters to match the reference style
        import matplotlib as mpl
        mpl.rcParams['figure.dpi'] = 120
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.linewidth'] = 1.2
        mpl.rcParams['xtick.major.width'] = 1.0
        mpl.rcParams['ytick.major.width'] = 1.0
        mpl.rcParams['lines.linewidth'] = 1.5
        mpl.rcParams['lines.markersize'] = 6
        mpl.rcParams['legend.frameon'] = True
        mpl.rcParams['legend.framealpha'] = 0.9
        mpl.rcParams['legend.edgecolor'] = 'gray'
        mpl.rcParams['legend.fancybox'] = True
        mpl.rcParams['savefig.format'] = 'png'
        mpl.rcParams['savefig.bbox'] = 'tight'
        mpl.rcParams['savefig.pad_inches'] = 0.1
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate statistics
        true_mean = np.mean(y_true)
        true_std = np.std(y_true)
        pred_mean = np.mean(y_pred)
        pred_std = np.std(y_pred)
        
        # Create histogram with bin width = 10, limited to ±2000 range
        bins = np.arange(-2000, 2010, 10)
        
        # Plot both histograms with stepfilled style to match reference
        true_counts, bin_edges = np.histogram(y_true, bins=bins)
        pred_counts, _ = np.histogram(y_pred, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot actual values
        ax.hist(bin_centers, bins=bin_edges, weights=true_counts,
               histtype='stepfilled', color='#4682B4', edgecolor='blue',
               linewidth=2, alpha=0.5)
        
        # Plot predicted values  
        ax.hist(bin_centers, bins=bin_edges, weights=pred_counts,
               histtype='stepfilled', color='#D46A6A', edgecolor='red', 
               linewidth=2, alpha=0.5)
        
        # Set axis labels and title
        ax.set_xlabel("Time [ps]", fontsize=12)
        ax.set_ylabel("Counts", fontsize=12)
        ax.set_title("Reco. t0", fontsize=14)
        
        # Increase y-axis by 10% to accommodate legend
        ymax = max(np.max(true_counts), np.max(pred_counts)) * 1.1
        ax.set_ylim(0, ymax)
        ax.set_xlim(-2000, 2000)
        
        # Add custom legend with statistics (inline format)
        blue_patch = plt.Rectangle((0, 0), 1, 1, fc='#4682B4', ec='blue', alpha=0.5)
        red_patch = plt.Rectangle((0, 0), 1, 1, fc='#D46A6A', ec='red', alpha=0.5)
        
        legend_labels = [
            f"Events: N = {len(y_true):,}",
            f"Actual (μ = {true_mean:.2f}, σ = {true_std:.2f} ps)",
            f"Predicted (μ = {pred_mean:.2f}, σ = {pred_std:.2f} ps)"
        ]
        
        # Add handles for legend
        legend_handles = [plt.Rectangle((0,0),0,0,alpha=0.0), blue_patch, red_patch]
        
        ax.legend(legend_handles, legend_labels, 
                 loc='upper left', fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram comparison plot saved to: {save_path}")
        plt.close()
    
    def _fit_double_gaussian(
        self,
        bin_centers: np.ndarray,
        counts: np.ndarray,
        pileup_sigma: float,
        fix_pileup: bool,
        error_mean: float,
        error_std: float
    ) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
        """
        Fit double Gaussian to error distribution (matching C++ ROOT implementation).

        NOTE: Unlike single Gaussian, double Gaussian fits over the ENTIRE data range,
        not just the core region. This matches the C++ ROOT implementation.

        Args:
            bin_centers: Histogram bin centers (full range)
            counts: Histogram bin counts (full range)
            pileup_sigma: Background Gaussian width [ps]
            fix_pileup: Whether to fix background width
            error_mean: Error mean for initial guess
            error_std: Error std for initial guess

        Returns:
            Tuple of (fit_mean, fit_sigma_core, fit_params) or (None, None, None) if fit fails
        """
        try:
            from scipy.optimize import curve_fit

            # Define double Gaussian function
            # f(x) = a1*exp(-0.5*((x-mu)/sigma1)^2) + a2*exp(-0.5*((x-mu)/sigma2)^2)
            def double_gaussian_func(x, a1, a2, mu, sigma1, sigma2):
                return (a1 * np.exp(-0.5 * ((x - mu) / sigma1) ** 2) +
                       a2 * np.exp(-0.5 * ((x - mu) / sigma2) ** 2))

            # Define double Gaussian with fixed sigma2
            def double_gaussian_fixed_sigma2(x, a1, a2, mu, sigma1):
                return (a1 * np.exp(-0.5 * ((x - mu) / sigma1) ** 2) +
                       a2 * np.exp(-0.5 * ((x - mu) / pileup_sigma) ** 2))

            # Use ALL data for double Gaussian fitting (matching C++ ROOT implementation)
            # No range restriction - fit the entire histogram
            fit_x = bin_centers
            fit_y = counts

            # Debug: print fitting range and data statistics
            print(f"  Double Gaussian fitting range: [{np.min(fit_x):.1f}, {np.max(fit_x):.1f}] ps")
            print(f"  Number of bins: {len(fit_x)}, Total counts: {np.sum(fit_y):.0f}")
            print(f"  Peak bin value: {np.max(fit_y):.0f} at x={fit_x[np.argmax(fit_y)]:.1f} ps")

            if len(fit_x) <= 4:  # Need at least 5 points for reasonable fit
                return None, None, None

            # Calculate fit uncertainties (Poisson statistics)
            fit_error = np.sqrt(fit_y)
            fit_error[fit_error == 0] = 1

            # Initial parameter guesses (matching C++ implementation)
            max_count = np.max(fit_y)
            p0_a1 = 0.8 * max_count      # Core amplitude: 80% of peak
            p0_a2 = 0.01 * max_count     # Background amplitude: 1% of peak
            p0_mu = 0.0                  # Mean fixed at zero
            p0_sigma1 = 26.0             # Core width: ~26ps initial guess
            p0_sigma2 = pileup_sigma     # Background width: 175ps

            if fix_pileup:
                # Fit with fixed background width (matches C++ with fixbkg=true)
                initial_params = [p0_a1, p0_a2, p0_mu, p0_sigma1]

                # Parameter bounds: a1>0, a2>0, mu≈0 (very tight), sigma1>0.1
                # Note: mu must have range to satisfy scipy bounds requirement, use [-0.1, 0.1] for tight constraint
                bounds = ([0, 0, -0.1, 0.1], [np.inf, np.inf, 0.1, np.inf])

                popt, pcov = curve_fit(
                    double_gaussian_fixed_sigma2, fit_x, fit_y,
                    p0=initial_params, sigma=fit_error, absolute_sigma=True,
                    bounds=bounds
                )

                # Extract parameters
                fit_mean = popt[2]
                fit_sigma_core = abs(popt[3])

                # Reconstruct full parameter array for plotting
                fit_params = np.array([popt[0], popt[1], popt[2], popt[3], pileup_sigma])

                # Debug: print fitted parameters
                print(f"  Double Gaussian fit SUCCESS (fixed σ_bkg={pileup_sigma:.0f}ps):")
                print(f"    N_core={popt[0]:.1f}, N_bkg={popt[1]:.1f}, μ={popt[2]:.2f}, σ_core={popt[3]:.2f} ps")

            else:
                # Fit with free background width (matches C++ with fixbkg=false)
                initial_params = [p0_a1, p0_a2, p0_mu, p0_sigma1, p0_sigma2]

                # Parameter bounds: a1>0, a2>0, mu≈0 (very tight), sigma1>0.1, sigma2>0.1
                # Note: mu must have range to satisfy scipy bounds requirement, use [-0.1, 0.1] for tight constraint
                bounds = ([0, 0, -0.1, 0.1, 0.1], [np.inf, np.inf, 0.1, np.inf, np.inf])

                popt, pcov = curve_fit(
                    double_gaussian_func, fit_x, fit_y,
                    p0=initial_params, sigma=fit_error, absolute_sigma=True,
                    bounds=bounds
                )

                # Extract parameters
                fit_mean = popt[2]
                fit_sigma_core = abs(popt[3])
                fit_params = popt

                # Debug: print fitted parameters
                print(f"  Double Gaussian fit SUCCESS (free σ_bkg):")
                print(f"    N_core={popt[0]:.1f}, N_bkg={popt[1]:.1f}, μ={popt[2]:.2f}, σ_core={popt[3]:.2f}, σ_bkg={popt[4]:.2f} ps")

            return fit_mean, fit_sigma_core, fit_params

        except Exception as e:
            print(f"  WARNING: Double Gaussian fit FAILED: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        baseline_predictions: Optional[np.ndarray] = None
    ):
        """
        Plot distribution of prediction errors with optional baseline comparison.

        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
            baseline_predictions: Optional baseline method predictions for comparison
        """
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, "error_distribution.png")

        errors = y_pred - y_true
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        
        # Set matplotlib parameters to match the reference style
        import matplotlib as mpl
        mpl.rcParams['figure.dpi'] = 120
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.linewidth'] = 1.2
        mpl.rcParams['xtick.major.width'] = 1.0
        mpl.rcParams['ytick.major.width'] = 1.0
        mpl.rcParams['lines.linewidth'] = 1.5
        mpl.rcParams['lines.markersize'] = 6
        mpl.rcParams['legend.frameon'] = True
        mpl.rcParams['legend.framealpha'] = 0.9
        mpl.rcParams['legend.edgecolor'] = 'gray'
        mpl.rcParams['legend.fancybox'] = True
        mpl.rcParams['savefig.format'] = 'png'
        mpl.rcParams['savefig.bbox'] = 'tight'
        mpl.rcParams['savefig.pad_inches'] = 0.1
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram with bin width = 10, limited to ±2000 range
        bins = np.arange(-2000, 2010, 10)
        bin_width = 10
        
        # Plot ML model errors using stepfilled style to match reference
        ml_counts, bin_edges = np.histogram(errors, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ax.hist(bin_centers, bins=bin_edges, weights=ml_counts,
               histtype='stepfilled', color='#4682B4', edgecolor='blue',
               linewidth=2, alpha=0.5)
        
        # Calculate baseline errors and plot if provided
        baseline_errors = None
        baseline_counts = None
        if baseline_predictions is not None:
            baseline_errors = baseline_predictions.flatten() - y_true
            baseline_counts, _ = np.histogram(baseline_errors, bins=bins)
            baseline_mean = np.mean(baseline_errors)
            baseline_std = np.std(baseline_errors)
            
            ax.hist(bin_centers, bins=bin_edges, weights=baseline_counts,
                   histtype='stepfilled', color='#D46A6A', edgecolor='red', 
                   linewidth=2, alpha=0.5)
        
        # Get fitting parameters from config (with defaults for backward compatibility)
        fit_method = getattr(self.config, 'error_fit_method', 'single_gaussian')
        fit_range = getattr(self.config, 'error_fit_range', 120)
        pileup_sigma = getattr(self.config, 'pileup_sigma', 175.0)
        fix_pileup = getattr(self.config, 'fix_pileup_sigma', True)

        # Gaussian fitting function (single Gaussian)
        def gaussian_func(x, a, mu, sigma):
            return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        # Double Gaussian fitting function
        def double_gaussian_func(x, a1, a2, mu, sigma1, sigma2):
            return (a1 * np.exp(-0.5 * ((x - mu) / sigma1) ** 2) +
                   a2 * np.exp(-0.5 * ((x - mu) / sigma2) ** 2))

        # Fit ML model errors
        try:
            from scipy.optimize import curve_fit

            if fit_method == 'double_gaussian':
                # Use double Gaussian fitting (fits entire histogram range, not just core)
                ml_fit_mean, ml_fit_std, ml_fit_params = self._fit_double_gaussian(
                    bin_centers, ml_counts, pileup_sigma, fix_pileup,
                    error_mean, error_std
                )

                if ml_fit_params is not None:
                    # Plot fitted curve extended to ±2000ps display range
                    x_fit = np.linspace(-2000, 2000, 1000)
                    y_fit = double_gaussian_func(x_fit, *ml_fit_params)
                    ax.plot(x_fit, y_fit, 'b--', linewidth=2)
                else:
                    # Fallback to data statistics
                    ml_fit_mean, ml_fit_std = error_mean, error_std

            else:
                # Use single Gaussian fitting (default)
                mask = (bin_centers >= -fit_range) & (bin_centers <= fit_range)

                if np.sum(mask) > 3:
                    fit_x = bin_centers[mask]
                    fit_y = ml_counts[mask]
                    fit_error = np.sqrt(fit_y)
                    fit_error[fit_error == 0] = 1

                    # Initial parameters for fit (amplitude, mean, sigma)
                    p0 = [np.max(fit_y), error_mean, error_std]

                    popt, pcov = curve_fit(gaussian_func, fit_x, fit_y,
                                          p0=p0, sigma=fit_error, absolute_sigma=True)
                    ml_fit_mean, ml_fit_std = popt[1], abs(popt[2])

                    # Plot fitted curve extended to ±2000ps display range
                    x_fit = np.linspace(-2000, 2000, 1000)
                    y_fit = gaussian_func(x_fit, *popt)
                    ax.plot(x_fit, y_fit, 'b--', linewidth=2)
                else:
                    ml_fit_mean, ml_fit_std = error_mean, error_std

        except Exception as e:
            print(f"Warning: Gaussian fit failed: {e}")
            ml_fit_mean, ml_fit_std = error_mean, error_std
        
        # Fit baseline errors if provided
        baseline_fit_mean, baseline_fit_std = None, None
        if baseline_predictions is not None:
            try:
                if fit_method == 'double_gaussian':
                    # Use double Gaussian fitting for baseline (fits entire histogram range)
                    baseline_fit_mean, baseline_fit_std, baseline_fit_params = self._fit_double_gaussian(
                        bin_centers, baseline_counts, pileup_sigma, fix_pileup,
                        baseline_mean, baseline_std
                    )

                    if baseline_fit_params is not None:
                        # Plot fitted curve extended to ±2000ps display range
                        x_fit = np.linspace(-2000, 2000, 1000)
                        y_fit = double_gaussian_func(x_fit, *baseline_fit_params)
                        ax.plot(x_fit, y_fit, 'r--', linewidth=2)
                    else:
                        # Fallback to data statistics
                        baseline_fit_mean, baseline_fit_std = baseline_mean, baseline_std

                else:
                    # Use single Gaussian fitting (default)
                    mask = (bin_centers >= -fit_range) & (bin_centers <= fit_range)
                    if np.sum(mask) > 3:
                        fit_x = bin_centers[mask]
                        fit_y = baseline_counts[mask]
                        fit_error = np.sqrt(fit_y)
                        fit_error[fit_error == 0] = 1

                        p0 = [np.max(fit_y), baseline_mean, baseline_std]

                        popt, pcov = curve_fit(gaussian_func, fit_x, fit_y,
                                              p0=p0, sigma=fit_error, absolute_sigma=True)
                        baseline_fit_mean, baseline_fit_std = popt[1], abs(popt[2])

                        x_fit = np.linspace(-2000, 2000, 1000)
                        y_fit = gaussian_func(x_fit, *popt)
                        ax.plot(x_fit, y_fit, 'r--', linewidth=2)
                    else:
                        baseline_fit_mean, baseline_fit_std = baseline_mean, baseline_std

            except Exception as e:
                print(f"Warning: Baseline Gaussian fit failed: {e}")
                baseline_fit_mean, baseline_fit_std = baseline_mean, baseline_std
        
        # Set axis labels and title
        ax.set_xlabel("Time [ps]", fontsize=12)
        ax.set_ylabel("Counts", fontsize=12)
        ax.set_title("Reco. t0", fontsize=14)
        
        # Increase y-axis by 10% to accommodate legend
        if baseline_predictions is not None:
            ymax = max(np.max(ml_counts), np.max(baseline_counts)) * 1.1
        else:
            ymax = np.max(ml_counts) * 1.1
        ax.set_ylim(0, ymax)
        ax.set_xlim(-2000, 2000)
        
        # Add custom legend with fit parameters
        blue_patch = plt.Rectangle((0, 0), 1, 1, fc='#4682B4', ec='blue', alpha=0.5)
        blue_line = plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2)

        # Create fit method description for legend
        if fit_method == 'double_gaussian':
            fit_label = f"Double Gaussian fit (σ_core, σ_bkg={pileup_sigma:.0f}ps fixed)"
            sigma_label = "σ_core"  # Core Gaussian width
        else:
            fit_label = "Single Gaussian fit"
            sigma_label = "σ"  # Single Gaussian width

        legend_labels = [
            f"Events: N = {len(errors):,}",
            f"ML model",
            f"ML data: μ = {error_mean:.2f}, σ = {error_std:.2f} ps",
            f"ML fit ({fit_label}): μ = {ml_fit_mean:.2f}, {sigma_label} = {ml_fit_std:.2f} ps"
        ]

        legend_handles = [plt.Rectangle((0,0),0,0,alpha=0.0), blue_patch,
                         plt.Rectangle((0,0),0,0,alpha=0.0), blue_line]
        
        if baseline_predictions is not None:
            red_patch = plt.Rectangle((0, 0), 1, 1, fc='#D46A6A', ec='red', alpha=0.5)
            red_line = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2)

            legend_labels.extend([
                "Baseline method",
                f"Baseline data: μ = {baseline_mean:.2f}, σ = {baseline_std:.2f} ps",
                f"Baseline fit ({fit_label}): μ = {baseline_fit_mean:.2f}, {sigma_label} = {baseline_fit_std:.2f} ps"
            ])

            legend_handles.extend([red_patch,
                                  plt.Rectangle((0,0),0,0,alpha=0.0), red_line])
        
        ax.legend(legend_handles, legend_labels, 
                 loc='upper left', fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to: {save_path}")
        plt.close()
    
    def plot_residuals_vs_predicted(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot residuals vs predicted values to check for patterns.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, "residuals_vs_predicted.png")
        
        residuals = y_pred - y_true
        n_events = len(y_true)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_pred, residuals, alpha=0.6, s=20, label=f'Residuals (N={n_events:,} events)')
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero line')
        
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals (Predicted - Actual)')
        plt.title(f'{self.config.model_name}: Residuals vs Predicted Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Residuals plot saved to: {save_path}")
        plt.close()
    
    def plot_baseline_corrections_scatter(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        baseline_predictions: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot scatter plot of baseline corrections vs truth for baseline-guided models.
        
        Args:
            y_true: True vertex time values
            y_pred: ML model predictions (final predictions)
            baseline_predictions: Baseline method predictions
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, "baseline_corrections_scatter.png")
        
        # Calculate the corrections made by the ML model
        # Corrections = final_prediction - baseline_prediction
        corrections = y_pred - baseline_predictions.flatten()
        
        # Calculate statistics
        correction_mean = np.mean(corrections)
        correction_std = np.std(corrections)
        correction_rmse = np.sqrt(np.mean(corrections**2))
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with alpha for density visualization
        plt.scatter(y_true, corrections, alpha=0.6, s=15, c='darkgreen', 
                   label=f'ML Corrections (N={len(corrections):,})')
        
        # Add horizontal line at zero (no correction)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2, 
                   label='No Correction', alpha=0.8)
        
        # Add statistics text box
        stats_text = f"Correction Statistics:\n"
        stats_text += f"Mean: {correction_mean:.2f} ps\n"
        stats_text += f"Std: {correction_std:.2f} ps\n"
        stats_text += f"RMSE: {correction_rmse:.2f} ps\n"
        stats_text += f"Range: [{np.min(corrections):.1f}, {np.max(corrections):.1f}] ps"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9))
        
        # Calculate correlation between truth and corrections
        correlation = np.corrcoef(y_true, corrections)[0, 1]
        corr_text = f"Correlation with Truth: {correlation:.4f}"
        
        plt.text(0.98, 0.02, corr_text, transform=plt.gca().transAxes,
                verticalalignment='bottom', horizontalalignment='right', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # Formatting
        plt.xlabel('Truth Vertex Time [ps]', fontsize=12)
        plt.ylabel('ML Model Corrections to Baseline [ps]', fontsize=12)
        plt.title(f'{self.config.model_name}: ML Model Corrections vs Truth Vertex Time', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Baseline corrections scatter plot saved to: {save_path}")
        plt.close()
    
    def create_comprehensive_evaluation_plots(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metrics: Dict[str, float],
        training_history: Optional[Dict[str, np.ndarray]] = None,
        plot_prediction_type: str = '2d_hist',
        baseline_predictions: Optional[np.ndarray] = None
    ):
        """
        Create all evaluation plots with configurable prediction plot type.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: Evaluation metrics
            training_history: Optional training history for plotting
            plot_prediction_type: Type of prediction plot ('2d_hist', 'hexbin', 'scatter', 'comparison')
            baseline_predictions: Optional baseline method predictions for comparison
        """
        # Ensure plots directory exists
        os.makedirs(self.config.plots_dir, exist_ok=True)
        
        print("Creating evaluation plots...")
        
        # Plot training history if provided
        if training_history is not None:
            self.plot_training_history(training_history)
        
        # Plot prediction results based on type
        if plot_prediction_type == 'comparison':
            self.plot_prediction_results_comparison(y_true, y_pred, metrics)
        else:
            self.plot_prediction_results(y_true, y_pred, metrics, plot_type=plot_prediction_type)
        
        # Plot histogram comparison
        self.plot_histogram_comparison(y_true, y_pred)
        
        # Plot error distribution with baseline comparison if available
        self.plot_error_distribution(y_true, y_pred, baseline_predictions=baseline_predictions)
        
        # Plot residuals
        self.plot_residuals_vs_predicted(y_true, y_pred)
        
        # Plot baseline corrections scatter plot if this is a baseline-guided model
        if baseline_predictions is not None:
            self.plot_baseline_corrections_scatter(y_true, y_pred, baseline_predictions)
        
        print(f"All evaluation plots saved to: {self.config.plots_dir}")
    
    def plot_feature_importance(
        self, 
        feature_names: list, 
        importance_scores: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, "feature_importance.png")
        
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_scores[sorted_indices]
        
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
        plt.barh(range(len(sorted_features)), sorted_scores)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Importance Score')
        plt.title(f'{self.config.model_name}: Feature Importance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
        plt.close()
