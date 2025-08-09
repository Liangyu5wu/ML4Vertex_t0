"""Data preprocessing and normalization utilities."""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from config.base_config import BaseConfig


class DataProcessor:
    """Handle data preprocessing, normalization, and dataset creation."""
    
    def __init__(self, config: BaseConfig):
        """
        Initialize data processor.
        
        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config
        
        # Energy bin edges for calibration: [1-1.5, 1.5-2, 2-3, 3-4, 4-5, 5-10, >10]
        self.energy_bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, float('inf')]
        
    def get_energy_bin_index(self, energy: float) -> int:
        """Get energy bin index for calibration parameter lookup."""
        if energy < 1.0:
            return 0  # Use first bin for energies < 1 GeV
        
        for i in range(len(self.energy_bins) - 1):
            if self.energy_bins[i] <= energy < self.energy_bins[i + 1]:
                return i
        
        return len(self.energy_bins) - 2  # Last bin for energies >= 10 GeV
    
    def apply_time_calibration(self, cell_sequences: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Apply detector time calibration to cell sequences.
        
        Args:
            cell_sequences: Original cell sequences
            
        Returns:
            Cell sequences with calibrated time
        """
        if not self.config.use_detector_params:
            return cell_sequences
        
        # Load calibration data
        calibration_data = self.config.load_calibration_data()
        
        # Get indices of required features
        try:
            barrel_idx = self.config.cell_features.index('Cell_Barrel')
            layer_idx = self.config.cell_features.index('Cell_layer')
            energy_idx = self.config.cell_features.index('Cell_e')
            time_idx = self.config.cell_features.index('Cell_time_TOF_corrected')
        except ValueError as e:
            raise ValueError(f"Required feature not found for time calibration: {e}")
        
        # Create parameter lookup using loaded data
        param_lookup = {
            (1, 1): calibration_data['EMB1_params'],  # Barrel, Layer 1
            (1, 2): calibration_data['EMB2_params'],  # Barrel, Layer 2
            (1, 3): calibration_data['EMB3_params'],  # Barrel, Layer 3
            (0, 1): calibration_data['EME1_params'],  # Endcap, Layer 1
            (0, 2): calibration_data['EME2_params'],  # Endcap, Layer 2
            (0, 3): calibration_data['EME3_params'],  # Endcap, Layer 3
        }
        
        calibrated_sequences = []
        
        for sequence in cell_sequences:
            calibrated_sequence = []
            
            for cell in sequence:
                calibrated_cell = cell.copy()
                
                # Get cell properties
                barrel = int(cell[barrel_idx])
                layer = int(cell[layer_idx])
                energy = cell[energy_idx]
                time_tof = cell[time_idx]
                
                # Get calibration parameters
                detector_params = param_lookup.get((barrel, layer), [0.0] * 7)
                
                # Get energy bin index and calibration value
                energy_bin_idx = self.get_energy_bin_index(energy)
                calibration_value = detector_params[energy_bin_idx]
                
                # Apply calibration: corrected_time = tof_corrected_time - calibration_value
                calibrated_time = time_tof - calibration_value
                calibrated_cell[time_idx] = calibrated_time
                
                calibrated_sequence.append(calibrated_cell)
            
            calibrated_sequences.append(calibrated_sequence)
        
        return calibrated_sequences
    
    def plot_calibration_validation(
        self, 
        original_sequences: List[List[List[float]]], 
        calibrated_sequences: List[List[List[float]]]
    ):
        """
        Plot calibration validation histograms for specified detector region.
        
        Args:
            original_sequences: Original cell sequences before calibration
            calibrated_sequences: Cell sequences after calibration
        """
        if not self.config.calibration_validation or not self.config.use_detector_params:
            return
        
        # Get feature indices
        try:
            barrel_idx = self.config.cell_features.index('Cell_Barrel')
            layer_idx = self.config.cell_features.index('Cell_layer')
            energy_idx = self.config.cell_features.index('Cell_e')
            time_idx = self.config.cell_features.index('Cell_time_TOF_corrected')
        except ValueError as e:
            print(f"Cannot create calibration validation plot: {e}")
            return
        
        # Energy bin labels
        energy_labels = ['1-1.5 GeV', '1.5-2 GeV', '2-3 GeV', '3-4 GeV', 
                        '4-5 GeV', '5-10 GeV', '>10 GeV']
        
        # Collect data for specified detector region
        original_data = [[] for _ in range(7)]  # 7 energy bins
        calibrated_data = [[] for _ in range(7)]
        
        for seq_idx, (orig_seq, cal_seq) in enumerate(zip(original_sequences, calibrated_sequences)):
            for cell_idx, (orig_cell, cal_cell) in enumerate(zip(orig_seq, cal_seq)):
                barrel = int(orig_cell[barrel_idx])
                layer = int(orig_cell[layer_idx])
                energy = orig_cell[energy_idx]
                
                # Check if cell matches validation criteria
                if (barrel == self.config.validation_detector_type and 
                    layer == self.config.validation_layer):
                    
                    bin_idx = self.get_energy_bin_index(energy)
                    original_data[bin_idx].append(orig_cell[time_idx])
                    calibrated_data[bin_idx].append(cal_cell[time_idx])
        
        # Create plots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        detector_name = f"{'Barrel' if self.config.validation_detector_type == 1 else 'Endcap'} Layer {self.config.validation_layer}"
        fig.suptitle(f'Time Calibration Validation - {detector_name}', fontsize=14)
        
        for i in range(7):
            ax = axes[i]
            
            orig_times = np.array(original_data[i])
            cal_times = np.array(calibrated_data[i])
            
            if len(orig_times) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{energy_labels[i]}\nN=0')
                continue
            
            # Calculate statistics
            orig_mean, orig_std = np.mean(orig_times), np.std(orig_times)
            cal_mean, cal_std = np.mean(cal_times), np.std(cal_times)
            
            # Plot histograms
            bins = np.linspace(min(np.min(orig_times), np.min(cal_times)),
                              max(np.max(orig_times), np.max(cal_times)), 30)
            
            ax.hist(orig_times, bins=bins, alpha=0.6, color='blue', 
                   label=f'Before: μ={orig_mean:.2f}, σ={orig_std:.2f}')
            ax.hist(cal_times, bins=bins, alpha=0.6, color='red',
                   label=f'After: μ={cal_mean:.2f}, σ={cal_std:.2f}')
            
            ax.set_title(f'{energy_labels[i]}\nN={len(orig_times)}')
            ax.set_xlabel('Time [ns]')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        fig.delaxes(axes[7])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.model_dir, "calibration_validation.png")
        os.makedirs(self.config.model_dir, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Calibration validation plot saved to: {plot_path}")
        print(f"Validation region: {detector_name}")
    
    def gaussian_func(self, x, a, mu, sigma):
        """Gaussian function for fitting."""
        return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    def calculate_traditional_t0(
        self, 
        cell_sequences: List[List[List[float]]], 
        vertex_times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate traditional (non-ML) t0 for each event.
        
        Args:
            cell_sequences: Calibrated cell sequences
            vertex_times: True vertex times
            
        Returns:
            Tuple of (traditional_t0, t0_errors)
        """
        # Load calibration data for sigma values
        calibration_data = self.config.load_calibration_data()
        
        # Sigma lookup tables using loaded data
        sigma_lookup = {
            (1, 1): calibration_data['EMB1_sigma'],  # Barrel, Layer 1
            (1, 2): calibration_data['EMB2_sigma'],  # Barrel, Layer 2
            (1, 3): calibration_data['EMB3_sigma'],  # Barrel, Layer 3
            (0, 1): calibration_data['EME1_sigma'],  # Endcap, Layer 1
            (0, 2): calibration_data['EME2_sigma'],  # Endcap, Layer 2
            (0, 3): calibration_data['EME3_sigma'],  # Endcap, Layer 3
        }
        
        # Get feature indices
        try:
            barrel_idx = self.config.cell_features.index('Cell_Barrel')
            layer_idx = self.config.cell_features.index('Cell_layer')
            energy_idx = self.config.cell_features.index('Cell_e')
            time_idx = self.config.cell_features.index('Cell_time_TOF_corrected')
        except ValueError as e:
            raise ValueError(f"Required feature not found for traditional t0 calculation: {e}")
        
        traditional_t0 = []
        
        for sequence in cell_sequences:
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for cell in sequence:
                barrel = int(cell[barrel_idx])
                layer = int(cell[layer_idx])
                energy = cell[energy_idx]
                time = cell[time_idx]
                
                # Get sigma for this cell
                sigma_params = sigma_lookup.get((barrel, layer), [1000.0] * 7)
                energy_bin_idx = self.get_energy_bin_index(energy)
                sigma = sigma_params[energy_bin_idx]
                
                # Weight = 1/sigma^2
                weight = 1.0 / (sigma * sigma)
                
                weighted_sum += weight * time
                weight_sum += weight
            
            if weight_sum > 0:
                t0 = weighted_sum / weight_sum
            else:
                t0 = 0.0
            
            traditional_t0.append(t0)
        
        traditional_t0 = np.array(traditional_t0)
        t0_errors = traditional_t0 - vertex_times
        
        return traditional_t0, t0_errors
    
    def create_baseline_check_plots(
        self, 
        cell_sequences: List[List[List[float]]], 
        vertex_times: np.ndarray
    ):
        """
        Create baseline check plots using traditional t0 calculation.
        
        Args:
            cell_sequences: Calibrated cell sequences
            vertex_times: True vertex times
        """
        print("Creating baseline check plots...")
        
        # Calculate traditional t0
        traditional_t0, t0_errors = self.calculate_traditional_t0(cell_sequences, vertex_times)
        
        # Create baseline_check directory
        baseline_dir = os.path.join(self.config.model_dir, "baseline_check")
        os.makedirs(baseline_dir, exist_ok=True)
        
        # Plot 1: Traditional t0 distribution
        self._plot_traditional_t0_distribution(traditional_t0, baseline_dir)
        
        # Plot 2: t0 error distribution
        self._plot_t0_error_distribution(t0_errors, baseline_dir)
        
        # Plot 3: Traditional t0 vs true t0 2D histogram
        self._plot_t0_vs_true_2d(traditional_t0, vertex_times, baseline_dir)
        
        print(f"Baseline check plots saved to: {baseline_dir}")
    
    def _plot_traditional_t0_distribution(self, traditional_t0: np.ndarray, save_dir: str):
        """Plot traditional t0 distribution with Gaussian fit."""
        plt.figure(figsize=(10, 6))
        
        # Create histogram with bin width = 10, limited to ±2000 range
        bins = np.arange(-2000, 2010, 10)  # -2000 to +2000 with bin width 10
        
        counts, bin_edges, _ = plt.hist(traditional_t0, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        
        # Calculate basic statistics
        mean_all = np.mean(traditional_t0)
        std_all = np.std(traditional_t0)
        
        # Gaussian fit on restricted range
        fit_range = self.config.gaussian_fit_range
        mask = (traditional_t0 >= -fit_range) & (traditional_t0 <= fit_range)
        if np.sum(mask) > 10:  # Need enough points for fitting
            fit_data = traditional_t0[mask]
            try:
                # Initial guess for Gaussian parameters
                hist_fit, bin_centers = np.histogram(fit_data, bins=50)
                bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2
                
                # Fit Gaussian
                initial_guess = [np.max(hist_fit), np.mean(fit_data), np.std(fit_data)]
                popt, _ = curve_fit(self.gaussian_func, bin_centers, hist_fit, p0=initial_guess)
                
                fit_mean, fit_std = popt[1], abs(popt[2])
                
                # Plot fitted Gaussian
                x_fit = np.linspace(-fit_range, fit_range, 200)
                y_fit = self.gaussian_func(x_fit, *popt)
                # Scale to match histogram
                scale_factor = 10 * len(traditional_t0) / len(fit_data)
                plt.plot(x_fit, y_fit * scale_factor, 'r-', linewidth=2, 
                        label=f'Gaussian fit (±{fit_range}): μ={fit_mean:.2f}, σ={fit_std:.2f}')
                
            except Exception:
                fit_mean, fit_std = np.mean(fit_data), np.std(fit_data)
        else:
            fit_mean, fit_std = mean_all, std_all
        
        plt.xlabel('Traditional t0 [ns]')
        plt.ylabel('Count')
        plt.title('Traditional t0 Distribution')
        plt.legend([f'All data: μ={mean_all:.2f}, σ={std_all:.2f}, N={len(traditional_t0)}',
                   f'Fit range ±{fit_range}: μ={fit_mean:.2f}, σ={fit_std:.2f}'])
        plt.grid(True, alpha=0.3)
        plt.xlim(-2000, 2000)  # Limit x-axis to ±2000
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'traditional_t0_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_t0_error_distribution(self, t0_errors: np.ndarray, save_dir: str):
        """Plot t0 error distribution with Gaussian fit."""
        plt.figure(figsize=(10, 6))
        
        # Create histogram with bin width = 10, limited to ±2000 range
        bins = np.arange(-2000, 2010, 10)  # -2000 to +2000 with bin width 10
        
        counts, bin_edges, _ = plt.hist(t0_errors, bins=bins, alpha=0.7, color='green', edgecolor='black')
        
        # Calculate basic statistics
        mean_all = np.mean(t0_errors)
        std_all = np.std(t0_errors)
        
        # Gaussian fit on restricted range
        fit_range = self.config.gaussian_fit_range
        mask = (t0_errors >= -fit_range) & (t0_errors <= fit_range)
        if np.sum(mask) > 10:
            fit_data = t0_errors[mask]
            try:
                hist_fit, bin_centers = np.histogram(fit_data, bins=50)
                bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2
                
                initial_guess = [np.max(hist_fit), np.mean(fit_data), np.std(fit_data)]
                popt, _ = curve_fit(self.gaussian_func, bin_centers, hist_fit, p0=initial_guess)
                
                fit_mean, fit_std = popt[1], abs(popt[2])
                
                # Plot fitted Gaussian
                x_fit = np.linspace(-fit_range, fit_range, 200)
                y_fit = self.gaussian_func(x_fit, *popt)
                scale_factor = 10 * len(t0_errors) / len(fit_data)
                plt.plot(x_fit, y_fit * scale_factor, 'r-', linewidth=2,
                        label=f'Gaussian fit (±{fit_range}): μ={fit_mean:.2f}, σ={fit_std:.2f}')
                
            except Exception:
                fit_mean, fit_std = np.mean(fit_data), np.std(fit_data)
        else:
            fit_mean, fit_std = mean_all, std_all
        
        plt.xlabel('Traditional t0 - True t0 [ns]')
        plt.ylabel('Count')
        plt.title('Traditional t0 Error Distribution')
        plt.legend([f'All data: μ={mean_all:.2f}, σ={std_all:.2f}, N={len(t0_errors)}',
                   f'Fit range ±{fit_range}: μ={fit_mean:.2f}, σ={fit_std:.2f}'])
        plt.grid(True, alpha=0.3)
        plt.xlim(-2000, 2000)  # Limit x-axis to ±2000
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 't0_error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_t0_vs_true_2d(self, traditional_t0: np.ndarray, vertex_times: np.ndarray, save_dir: str):
        """Plot traditional t0 vs true t0 as 2D histogram."""
        plt.figure(figsize=(10, 8))
        
        # Fixed plot range to ±2000
        plot_min, plot_max = -2000, 2000
        
        # Create 2D histogram
        bins = 80  # 400 bins across 4000 range gives 50 bins per 1000 units
        hist, xedges, yedges = np.histogram2d(
            vertex_times, traditional_t0,
            bins=bins,
            range=[[plot_min, plot_max], [plot_min, plot_max]]
        )
        
        im = plt.imshow(
            hist.T,
            origin='lower',
            extent=[plot_min, plot_max, plot_min, plot_max],
            cmap='Blues',
            aspect='equal',
            interpolation='bilinear'
        )
        
        # Perfect prediction line
        plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2,
                label='Perfect Prediction', alpha=0.8)
        
        # Calculate metrics
        correlation = np.corrcoef(vertex_times, traditional_t0)[0, 1]
        rmse = np.sqrt(np.mean((traditional_t0 - vertex_times) ** 2))
        mae = np.mean(np.abs(traditional_t0 - vertex_times))
        
        plt.xlabel('True Vertex Time [ns]')
        plt.ylabel('Traditional t0 [ns]')
        plt.title('Traditional t0 vs True t0 (2D Histogram)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, label='Count')
        cbar.ax.tick_params(labelsize=9)
        
        # Add metrics text
        metrics_text = f"Correlation = {correlation:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nN = {len(traditional_t0):,}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                fontsize=10)
        
        plt.xlim(plot_min, plot_max)  # Limit both axes to ±2000
        plt.ylim(plot_min, plot_max)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'traditional_t0_vs_true_2d.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    
    
    def split_data(
        self, 
        cell_sequences: List[List[List[float]]], 
        vertex_features: np.ndarray, 
        vertex_times: np.ndarray
    ) -> Tuple[Tuple[List, List, List], Tuple[np.ndarray, np.ndarray, np.ndarray], 
               Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            cell_sequences: List of cell sequences
            vertex_features: Array of vertex features
            vertex_times: Array of vertex times
            
        Returns:
            Tuple of ((train_cells, val_cells, test_cells), 
                     (train_vertex, val_vertex, test_vertex),
                     (train_times, val_times, test_times))
        """
        indices = np.arange(len(vertex_times))
        train_indices, temp_indices = train_test_split(
            indices, test_size=self.config.test_size, random_state=self.config.random_state
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=self.config.val_split, random_state=self.config.random_state
        )
        
        train_cells = [cell_sequences[i] for i in train_indices]
        val_cells = [cell_sequences[i] for i in val_indices]
        test_cells = [cell_sequences[i] for i in test_indices]
        
        train_vertex = vertex_features[train_indices]
        val_vertex = vertex_features[val_indices]
        test_vertex = vertex_features[test_indices]
        
        train_times = vertex_times[train_indices]
        val_times = vertex_times[val_indices]
        test_times = vertex_times[test_indices]
        
        print(f"Data split sizes:")
        print(f"Training set: {len(train_cells)} events")
        print(f"Validation set: {len(val_cells)} events")
        print(f"Test set: {len(test_cells)} events")
        
        return ((train_cells, val_cells, test_cells),
                (train_vertex, val_vertex, test_vertex),
                (train_times, val_times, test_times))
    
    def normalize_features(
        self,
        train_cells: List[List[List[float]]],
        val_cells: List[List[List[float]]],
        test_cells: List[List[List[float]]],
        train_vertex: np.ndarray,
        val_vertex: np.ndarray,
        test_vertex: np.ndarray,
        train_times: np.ndarray,
        val_times: np.ndarray,
        test_times: np.ndarray
    ) -> Tuple[Tuple[List, List, List], Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, Any]]:
        """
        Apply time calibration and normalize cell and vertex features.
        
        Args:
            train_cells, val_cells, test_cells: Cell sequences for each split
            train_vertex, val_vertex, test_vertex: Vertex features for each split
            
        Returns:
            Tuple of normalized data and normalization parameters
        """
        # Apply time calibration first
        original_cells_copy = [[[cell for cell in seq] for seq in split] 
                              for split in [train_cells, val_cells, test_cells]]
        
        train_cells = self.apply_time_calibration(train_cells)
        val_cells = self.apply_time_calibration(val_cells)
        test_cells = self.apply_time_calibration(test_cells)
        
        # Create calibration validation plot if enabled
        if self.config.calibration_validation and self.config.use_detector_params:
            print("Creating calibration validation plot...")
            # Use training data for validation plot
            self.plot_calibration_validation(original_cells_copy[0], train_cells)
        
        # Create baseline check plots if detector params are used
        if self.config.use_detector_params:
            print("Creating baseline check plots...")
            # Use training data for baseline check
            self.create_baseline_check_plots(train_cells, train_times)
        
        
        # Normalize cell features
        train_cells_norm, val_cells_norm, test_cells_norm, cell_norm_params = \
            self._normalize_cell_features(train_cells, val_cells, test_cells)
        
        # Normalize vertex features
        train_vertex_norm, val_vertex_norm, test_vertex_norm, vertex_norm_params = \
            self._normalize_vertex_features(train_vertex, val_vertex, test_vertex)
        
        norm_params = {
            'cell_means': cell_norm_params['means'],
            'cell_stds': cell_norm_params['stds'],
            'vertex_means': vertex_norm_params['means'],
            'vertex_stds': vertex_norm_params['stds']
        }
        
        return ((train_cells_norm, val_cells_norm, test_cells_norm),
                (train_vertex_norm, val_vertex_norm, test_vertex_norm),
                norm_params)
    
    def _normalize_cell_features(
        self,
        train_cells: List[List[List[float]]],
        val_cells: List[List[List[float]]],
        test_cells: List[List[List[float]]]
    ) -> Tuple[List, List, List, Dict[str, List]]:
        """Normalize cell features based on training data statistics."""
        # Collect all training feature values
        all_train_values = [[] for _ in range(len(self.config.cell_features))]
        
        for sequence in train_cells:
            for cell in sequence:
                for feat_idx in range(len(self.config.cell_features)):
                    if feat_idx < len(cell):
                        all_train_values[feat_idx].append(cell[feat_idx])
        
        # Compute normalization parameters
        cell_means = []
        cell_stds = []
        skip_indices = self._get_skip_indices()
        
        for feat_idx in range(len(self.config.cell_features)):
            if feat_idx in skip_indices:
                cell_means.append(0.0)
                cell_stds.append(1.0)
            else:
                values = np.array(all_train_values[feat_idx])
                mean_val = np.mean(values)
                std_val = np.std(values)
                cell_means.append(mean_val)
                cell_stds.append(std_val if std_val > 0 else 1.0)
        
        # Apply normalization
        train_cells_norm = self._apply_cell_normalization(train_cells, cell_means, cell_stds)
        val_cells_norm = self._apply_cell_normalization(val_cells, cell_means, cell_stds)
        test_cells_norm = self._apply_cell_normalization(test_cells, cell_means, cell_stds)
        
        return (train_cells_norm, val_cells_norm, test_cells_norm,
                {'means': cell_means, 'stds': cell_stds})
    
    def _normalize_vertex_features(
        self,
        train_vertex: np.ndarray,
        val_vertex: np.ndarray,
        test_vertex: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Normalize vertex features based on training data statistics."""
        vertex_means = np.mean(train_vertex, axis=0)
        vertex_stds = np.std(train_vertex, axis=0)
        vertex_stds = np.where(vertex_stds > 0, vertex_stds, 1)
        
        train_vertex_norm = (train_vertex - vertex_means) / vertex_stds
        val_vertex_norm = (val_vertex - vertex_means) / vertex_stds
        test_vertex_norm = (test_vertex - vertex_means) / vertex_stds
        
        return (train_vertex_norm, val_vertex_norm, test_vertex_norm,
                {'means': vertex_means, 'stds': vertex_stds})
    
    def _get_skip_indices(self) -> List[int]:
        """Get indices of features to skip normalization for."""
        skip_indices = []
        for feature in self.config.skip_normalization:
            if feature in self.config.cell_features:
                skip_indices.append(self.config.cell_features.index(feature))
        return skip_indices
    
    def _apply_cell_normalization(
        self,
        cell_sequences: List[List[List[float]]],
        means: List[float],
        stds: List[float]
    ) -> List[List[List[float]]]:
        """Apply normalization to cell sequences."""
        normalized_sequences = []
        for sequence in cell_sequences:
            normalized_sequence = []
            for cell in sequence:
                normalized_cell = []
                for feat_idx, value in enumerate(cell):
                    if feat_idx < len(means):
                        if stds[feat_idx] != 1.0:
                            normalized_value = (value - means[feat_idx]) / stds[feat_idx]
                        else:
                            normalized_value = value
                    else:
                        normalized_value = value
                    normalized_cell.append(normalized_value)
                normalized_sequence.append(normalized_cell)
            normalized_sequences.append(normalized_sequence)
        return normalized_sequences
    
    def create_padded_dataset(
        self,
        cell_sequences: List[List[List[float]]],
        vertex_features: np.ndarray,
        vertex_times: np.ndarray,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """
        Create padded TensorFlow dataset from sequences.
        
        Args:
            cell_sequences: Variable-length cell sequences (with calibrated time)
            vertex_features: Vertex feature arrays
            vertex_times: Target vertex times
            shuffle: Whether to shuffle the dataset
            
        Returns:
            Batched and prefetched TensorFlow dataset
        """
        # Find maximum sequence length
        max_seq_len = max(len(seq) for seq in cell_sequences)
        
        # Feature dimension is just the original cell features (no extra detector params)
        feature_dim = len(self.config.cell_features)
        
        # Pad all sequences to max length
        padded_cells = np.zeros((len(cell_sequences), max_seq_len, feature_dim))
        for i, seq in enumerate(cell_sequences):
            seq_len = len(seq)
            padded_cells[i, :seq_len, :] = seq
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            {'cell_sequence': padded_cells, 'vertex_features': vertex_features},
            vertex_times
        ))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        return dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
    
    def create_prediction_batches(
        self,
        cell_sequences: List[List[List[float]]],
        vertex_features: np.ndarray,
        vertex_times: np.ndarray
    ):
        """
        Create batches for prediction from variable-length sequences.
        
        Args:
            cell_sequences: Variable-length cell sequences (with calibrated time)
            vertex_features: Vertex feature arrays
            vertex_times: Target vertex times
            
        Yields:
            Batches of data for prediction
        """
        # Sort by sequence length for more efficient batching
        lengths = [len(seq) for seq in cell_sequences]
        indices = np.argsort(lengths)
        
        # Feature dimension is just the original cell features
        feature_dim = len(self.config.cell_features)
        
        for i in range(0, len(indices), self.config.batch_size):
            batch_indices = indices[i:i+self.config.batch_size]
            
            # Find max length in this batch
            batch_lengths = [lengths[idx] for idx in batch_indices]
            max_length = max(batch_lengths)
            
            # Pad sequences in this batch to max_length
            batch_cells = np.zeros((len(batch_indices), max_length, feature_dim))
            batch_vertex = np.zeros((len(batch_indices), len(vertex_features[0])))
            batch_times = np.zeros(len(batch_indices))
            
            for j, idx in enumerate(batch_indices):
                seq = cell_sequences[idx]
                seq_len = len(seq)
                
                # Fill in the actual sequence
                for k in range(seq_len):
                    batch_cells[j, k, :] = seq[k]
                # Padding is automatically 0 from np.zeros
                
                batch_vertex[j] = vertex_features[idx]
                batch_times[j] = vertex_times[idx]
            
            # Return batch inputs
            yield {'cell_sequence': batch_cells, 'vertex_features': batch_vertex}, batch_times
