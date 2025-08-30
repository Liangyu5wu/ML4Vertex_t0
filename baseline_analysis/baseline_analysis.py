#!/usr/bin/env python3
"""
Baseline Time Reconstruction Analysis Tool

This script performs detailed analysis of baseline time reconstruction methods,
focusing on understanding why certain events have poor reconstruction performance.
It analyzes the worst-performing events and compares feature distributions.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import argparse
from datetime import datetime
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict, Any
import shutil
import logging


class BaselineAnalysisConfig:
    """Configuration class for baseline analysis."""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration from YAML file."""
        if config_file is None:
            config_file = Path(__file__).parent / "baseline_analysis_config.yaml"
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Load all configuration parameters
        for key, value in config_data.items():
            setattr(self, key, value)
        
        # Load calibration data
        self.calibration_data = self._load_calibration_data()
        
        # Energy bins for calibration
        self.energy_bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, float('inf')]
    
    def _load_calibration_data(self) -> Dict[str, List[float]]:
        """Load calibration data from external file."""
        # Try current directory first, then parent directory
        calibration_path = Path("calibration_data") / self.calibration_data_file
        if not calibration_path.exists():
            calibration_path = Path("../calibration_data") / self.calibration_data_file
        
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration data file not found. Tried paths:\n"
                                  f"  - calibration_data/{self.calibration_data_file}\n"
                                  f"  - ../calibration_data/{self.calibration_data_file}\n"
                                  f"Please ensure calibration data is available.")
        
        print(f"Loading calibration data from: {calibration_path}")
        
        calibration_data = {}
        with open(calibration_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if ':' in line:
                        key, values_str = line.split(':', 1)
                        key = key.strip()
                        values = [float(x.strip()) for x in values_str.split(',')]
                        calibration_data[key] = values
        
        return calibration_data


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_file = output_dir / "analysis_log.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_and_filter_data(config: BaselineAnalysisConfig, logger: logging.Logger) -> Tuple[List, np.ndarray, List]:
    """Load and filter data from HDF5 files."""
    logger.info(f"Loading data from {config.data_dir}")
    logger.info(f"Number of files: {config.num_files}")
    logger.info(f"Track matching: {config.use_track_features}, Jet matching: {config.use_jet_features}")
    
    all_cell_sequences = []
    all_vertex_times = []
    all_raw_cell_data = []  # Store raw cell data for feature analysis
    
    total_events = 0
    valid_events = 0
    total_cells_before = 0
    total_cells_after = 0
    
    for i in range(config.num_files):
        file_path = os.path.join(config.data_dir, f"output_{i:03d}.h5")
        
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} not found, skipping...")
            continue
            
        logger.info(f"Processing {file_path}...")
        
        with h5py.File(file_path, 'r') as f:
            vertex_data = f['HSvertex'][:]
            cells_data = f['cells'][:]
            
            total_events += len(vertex_data)
            
            for event_idx in range(len(vertex_data)):
                vertex_time = vertex_data[event_idx]['HSvertex_time']
                event_cells = cells_data[event_idx]
                
                if len(event_cells) == 0:
                    continue
                
                total_cells_before += len(event_cells)
                
                # Apply filtering based on configuration
                filtered_cells = apply_cell_filtering(event_cells, config)
                
                if len(filtered_cells) < config.min_cells:
                    continue
                
                # Additional layer filtering
                layer_filtered_cells = []
                for cell in filtered_cells:
                    if cell['Cell_layer'] in [1, 2, 3]:
                        layer_filtered_cells.append(cell)
                
                if len(layer_filtered_cells) < config.min_cells:
                    continue
                
                total_cells_after += len(layer_filtered_cells)
                valid_events += 1
                
                # Store raw cell data for feature analysis
                all_raw_cell_data.append(layer_filtered_cells)
                
                # Convert to list format for processing
                cell_sequence = []
                for cell in layer_filtered_cells:
                    cell_features = [
                        cell['Cell_time_TOF_corrected'],
                        cell['Cell_e'],
                        cell['Cell_Barrel'],
                        cell['Cell_layer']
                    ]
                    cell_sequence.append(cell_features)
                
                all_cell_sequences.append(cell_sequence)
                all_vertex_times.append(vertex_time)
    
    logger.info(f"Data loading summary:")
    logger.info(f"  Total events: {total_events}")
    logger.info(f"  Valid events: {valid_events}")
    logger.info(f"  Total cells before filtering: {total_cells_before}")
    logger.info(f"  Total cells after filtering: {total_cells_after}")
    if total_cells_before > 0:
        retention_rate = (total_cells_after / total_cells_before) * 100
        logger.info(f"  Cell retention rate: {retention_rate:.1f}%")
    
    return all_cell_sequences, np.array(all_vertex_times), all_raw_cell_data


def apply_cell_filtering(event_cells, config: BaselineAnalysisConfig):
    """Apply cell filtering based on configuration."""
    mask = np.ones(len(event_cells), dtype=bool)
    
    # Apply valid cell filter
    if config.require_valid_cells:
        valid_mask = event_cells['valid'] == True
        mask = mask & valid_mask
    
    # Apply track or jet matching filter
    if config.use_track_features and config.use_cell_track_matching:
        track_matching_mask = event_cells['matched_track_HS'] == 1
        mask = mask & track_matching_mask
    elif config.use_jet_features and config.use_cell_jet_matching:
        jet_matching_mask = event_cells['cell_jet_matched'] == True
        mask = mask & jet_matching_mask
    
    return event_cells[mask]


def get_energy_bin_index(energy: float, energy_bins: List[float]) -> int:
    """Get energy bin index for calibration parameter lookup."""
    if energy < 1.0:
        return 0
    
    for i in range(len(energy_bins) - 1):
        if energy_bins[i] <= energy < energy_bins[i + 1]:
            return i
    
    return len(energy_bins) - 2


def apply_time_calibration(cell_sequences: List, config: BaselineAnalysisConfig, logger: logging.Logger) -> List:
    """Apply detector time calibration to cell sequences."""
    logger.info("Applying time calibration...")
    
    # Parameter lookup - using 1-based layer indexing
    param_lookup = {
        (1, 1): config.calibration_data['EMB1_params'],
        (1, 2): config.calibration_data['EMB2_params'],
        (1, 3): config.calibration_data['EMB3_params'],
        (0, 1): config.calibration_data['EME1_params'],
        (0, 2): config.calibration_data['EME2_params'],
        (0, 3): config.calibration_data['EME3_params'],
    }
    
    calibrated_sequences = []
    
    for sequence in cell_sequences:
        calibrated_sequence = []
        
        for cell in sequence:
            calibrated_cell = cell.copy()
            time_tof = cell[0]
            energy = cell[1]
            barrel = int(cell[2])
            layer = int(cell[3])
            
            detector_params = param_lookup.get((barrel, layer), [0.0] * 7)
            energy_bin_idx = get_energy_bin_index(energy, config.energy_bins)
            calibration_value = detector_params[energy_bin_idx]
            
            calibrated_time = time_tof - calibration_value
            calibrated_cell[0] = calibrated_time
            
            calibrated_sequence.append(calibrated_cell)
        
        calibrated_sequences.append(calibrated_sequence)
    
    return calibrated_sequences


def calculate_baseline_t0(cell_sequences: List, vertex_times: np.ndarray, 
                         config: BaselineAnalysisConfig, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate baseline t0 for each event using weighted average."""
    logger.info("Calculating baseline t0...")
    
    # Sigma lookup tables
    sigma_lookup = {
        (1, 1): config.calibration_data['EMB1_sigma'],
        (1, 2): config.calibration_data['EMB2_sigma'],
        (1, 3): config.calibration_data['EMB3_sigma'],
        (0, 1): config.calibration_data['EME1_sigma'],
        (0, 2): config.calibration_data['EME2_sigma'],
        (0, 3): config.calibration_data['EME3_sigma'],
    }
    
    # Parameter lookup for original times
    param_lookup = {
        (1, 1): config.calibration_data['EMB1_params'],
        (1, 2): config.calibration_data['EMB2_params'],
        (1, 3): config.calibration_data['EMB3_params'],
        (0, 1): config.calibration_data['EME1_params'],
        (0, 2): config.calibration_data['EME2_params'],
        (0, 3): config.calibration_data['EME3_params'],
    }
    
    baseline_t0 = []
    
    for event_idx, sequence in enumerate(cell_sequences):
        weighted_sum = 0.0
        weight_sum = 0.0
        
        calibrated_cell_times = []
        original_cell_times = []
        
        for cell in sequence:
            calibrated_time = cell[0]
            energy = cell[1]
            barrel = int(cell[2])
            layer = int(cell[3])
            
            calibrated_cell_times.append(calibrated_time)
            
            # Calculate original time
            detector_params = param_lookup.get((barrel, layer), [0.0] * 7)
            energy_bin_idx = get_energy_bin_index(energy, config.energy_bins)
            calibration_value = detector_params[energy_bin_idx]
            original_time = calibrated_time + calibration_value
            original_cell_times.append(original_time)
            
            # Get sigma for this cell
            sigma_params = sigma_lookup.get((barrel, layer), [1000.0] * 7)
            sigma = sigma_params[energy_bin_idx]
            
            # Weight = 1/sigma^2
            weight = 1.0 / (sigma * sigma)
            
            weighted_sum += weight * calibrated_time
            weight_sum += weight
        
        if weight_sum > 0:
            t0 = weighted_sum / weight_sum
        else:
            t0 = 0.0
        
        baseline_t0.append(t0)
        
        # Print debug info for first few events
        if event_idx < 10:
            logger.info(f"Event {event_idx}:")
            logger.info(f"  Truth vertex time: {vertex_times[event_idx]:.4f} ps")
            logger.info(f"  Number of filtered cells: {len(calibrated_cell_times)}")
            
            original_times_str = ", ".join([f'{t:.1f}' for t in original_cell_times])
            logger.info(f"  Original cell times: {original_times_str} ps")
            
            calibrated_times_str = ", ".join([f'{t:.1f}' for t in calibrated_cell_times])
            logger.info(f"  Calibrated cell times: {calibrated_times_str} ps")
            
            logger.info(f"  Reconstructed vertex time: {t0:.4f} ps")
            logger.info(f"  Error (reco - truth): {t0 - vertex_times[event_idx]:.4f} ps")
    
    baseline_t0 = np.array(baseline_t0)
    t0_errors = baseline_t0 - vertex_times
    
    logger.info(f"Baseline t0 calculation completed for {len(baseline_t0)} events")
    
    return baseline_t0, t0_errors


def analyze_worst_events(baseline_t0: np.ndarray, vertex_times: np.ndarray, 
                        raw_cell_data: List, config: BaselineAnalysisConfig, 
                        logger: logging.Logger):
    """Analyze and log details of worst-performing events."""
    t0_errors = baseline_t0 - vertex_times
    error_abs = np.abs(t0_errors)
    
    # Get indices of worst events
    worst_indices = np.argsort(error_abs)[-config.top_worst_events:][::-1]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TOP {config.top_worst_events} WORST RECONSTRUCTION EVENTS")
    logger.info(f"{'='*60}")
    
    for i, event_idx in enumerate(worst_indices):
        logger.info(f"\nWorst Event #{i+1} (Event Index: {event_idx}):")
        logger.info(f"  Truth time: {vertex_times[event_idx]:.4f} ps")
        logger.info(f"  Reconstructed time: {baseline_t0[event_idx]:.4f} ps")
        logger.info(f"  Error: {t0_errors[event_idx]:.4f} ps")
        logger.info(f"  Absolute error: {error_abs[event_idx]:.4f} ps")
        
        # Analyze cell properties for this event
        event_cells = raw_cell_data[event_idx]
        logger.info(f"  Number of cells: {len(event_cells)}")
        
        # Cell energy statistics
        cell_energies = [cell['Cell_e'] for cell in event_cells]
        logger.info(f"  Cell energies: mean={np.mean(cell_energies):.2f}, "
                   f"std={np.std(cell_energies):.2f}, "
                   f"min={np.min(cell_energies):.2f}, "
                   f"max={np.max(cell_energies):.2f} GeV")
        
        # Cell time statistics (before calibration)
        cell_times = [cell['Cell_time_TOF_corrected'] for cell in event_cells]
        logger.info(f"  Cell times: mean={np.mean(cell_times):.2f}, "
                   f"std={np.std(cell_times):.2f}, "
                   f"min={np.min(cell_times):.2f}, "
                   f"max={np.max(cell_times):.2f} ps")
        
        # Layer distribution
        layers = [cell['Cell_layer'] for cell in event_cells]
        layer_counts = {layer: layers.count(layer) for layer in set(layers)}
        logger.info(f"  Layer distribution: {layer_counts}")
        
        # Barrel/Endcap distribution
        barrels = [cell['Cell_Barrel'] for cell in event_cells]
        barrel_counts = {('Barrel' if b else 'Endcap'): barrels.count(b) for b in set(barrels)}
        logger.info(f"  Detector region: {barrel_counts}")


def gaussian_func(x, a, mu, sigma):
    """Gaussian function for fitting."""
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def create_baseline_plots(baseline_t0: np.ndarray, vertex_times: np.ndarray, 
                         t0_errors: np.ndarray, config: BaselineAnalysisConfig, 
                         output_dir: Path, logger: logging.Logger):
    """Create standard baseline check plots."""
    logger.info("Creating baseline check plots...")
    
    baseline_plots_dir = output_dir / "baseline_plots"
    baseline_plots_dir.mkdir(exist_ok=True)
    
    plot_range = config.plot_x_range
    bins = np.linspace(plot_range[0], plot_range[1], config.plot_bins)
    
    # Plot 1: t0 error distribution
    plt.figure(figsize=(10, 6))
    counts, bin_edges, _ = plt.hist(t0_errors, bins=bins, alpha=0.7, color='green', edgecolor='black')
    
    # Gaussian fit on restricted range
    fit_range = config.gaussian_fit_range
    mask = (t0_errors >= -fit_range) & (t0_errors <= fit_range)
    
    if np.sum(mask) > 10:
        fit_data = t0_errors[mask]
        try:
            fit_bins = np.linspace(-fit_range, fit_range, 50)
            hist_fit, fit_bin_edges = np.histogram(fit_data, bins=fit_bins)
            bin_centers = (fit_bin_edges[:-1] + fit_bin_edges[1:]) / 2
            
            nonzero_mask = hist_fit > 0
            if np.sum(nonzero_mask) > 3:
                initial_guess = [np.max(hist_fit), np.mean(fit_data), np.std(fit_data)]
                popt, _ = curve_fit(gaussian_func, bin_centers[nonzero_mask], hist_fit[nonzero_mask], p0=initial_guess)
                
                fit_mean, fit_std = popt[1], abs(popt[2])
                x_fit = np.linspace(-fit_range, fit_range, 200)
                y_fit = gaussian_func(x_fit, *popt)
                plt.plot(x_fit, y_fit, 'r-', linewidth=2,
                        label=f'Gaussian fit (±{fit_range}): μ={fit_mean:.2f}, σ={fit_std:.2f}')
        except Exception:
            pass
    
    mean_error = np.mean(t0_errors)
    std_error = np.std(t0_errors)
    plt.xlabel('Baseline t0 - True t0 [ps]')
    plt.ylabel('Count')
    plt.title('Baseline t0 Error Distribution')
    plt.legend([f'All data: μ={mean_error:.2f}, σ={std_error:.2f}, N={len(t0_errors)}'])
    plt.grid(True, alpha=0.3)
    plt.xlim(plot_range[0], plot_range[1])
    
    plt.tight_layout()
    plt.savefig(baseline_plots_dir / 't0_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Baseline t0 distribution
    plt.figure(figsize=(10, 6))
    plt.hist(baseline_t0, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    
    mean_t0 = np.mean(baseline_t0)
    std_t0 = np.std(baseline_t0)
    plt.xlabel('Baseline t0 [ps]')
    plt.ylabel('Count')
    plt.title('Baseline t0 Distribution')
    plt.text(0.05, 0.95, f'μ={mean_t0:.2f}, σ={std_t0:.2f}, N={len(baseline_t0)}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    plt.grid(True, alpha=0.3)
    plt.xlim(plot_range[0], plot_range[1])
    
    plt.tight_layout()
    plt.savefig(baseline_plots_dir / 'traditional_t0_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: 2D histogram - baseline t0 vs true t0
    plt.figure(figsize=(10, 8))
    
    plot_range_2d = [-1000, 1000]
    hist, xedges, yedges = np.histogram2d(
        vertex_times, baseline_t0,
        bins=80,
        range=[plot_range_2d, plot_range_2d]
    )
    
    im = plt.imshow(
        hist.T,
        origin='lower',
        extent=[plot_range_2d[0], plot_range_2d[1], plot_range_2d[0], plot_range_2d[1]],
        cmap='Blues',
        aspect='equal',
        interpolation='bilinear'
    )
    
    # Perfect prediction line
    plt.plot(plot_range_2d, plot_range_2d, 'r--', linewidth=2,
            label='Perfect Prediction', alpha=0.8)
    
    # Calculate metrics
    correlation = np.corrcoef(vertex_times, baseline_t0)[0, 1]
    rmse = np.sqrt(np.mean((baseline_t0 - vertex_times) ** 2))
    mae = np.mean(np.abs(baseline_t0 - vertex_times))
    
    plt.xlabel('True Vertex Time [ps]')
    plt.ylabel('Baseline t0 [ps]')
    plt.title('Baseline t0 vs True t0')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, label='Count')
    
    # Add metrics text
    metrics_text = f"Correlation = {correlation:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nN = {len(baseline_t0):,}"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
            fontsize=10)
    
    plt.xlim(plot_range_2d[0], plot_range_2d[1])
    plt.ylim(plot_range_2d[0], plot_range_2d[1])
    
    plt.tight_layout()
    plt.savefig(baseline_plots_dir / 'traditional_t0_vs_true_2d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Baseline plots saved to {baseline_plots_dir}")


def create_feature_comparison_plots(raw_cell_data: List, t0_errors: np.ndarray, 
                                  config: BaselineAnalysisConfig, output_dir: Path, 
                                  logger: logging.Logger):
    """Create feature distribution comparison plots between best and worst events."""
    logger.info("Creating feature comparison plots...")
    
    feature_plots_dir = output_dir / "feature_comparison"
    feature_plots_dir.mkdir(exist_ok=True)
    
    error_abs = np.abs(t0_errors)
    sample_size = config.feature_comparison_sample_size
    
    # Get indices for best and worst events
    best_indices = np.argsort(error_abs)[:sample_size]
    worst_indices = np.argsort(error_abs)[-sample_size:]
    
    logger.info(f"Comparing {sample_size} best vs {sample_size} worst events")
    
    # Collect features for best and worst events
    best_features = {feature: [] for feature in config.comparison_features}
    worst_features = {feature: [] for feature in config.comparison_features}
    
    # Collect features from raw cell data
    for idx in best_indices:
        event_cells = raw_cell_data[idx]
        for cell in event_cells:
            for feature in config.comparison_features:
                if feature in cell.dtype.names:
                    best_features[feature].append(cell[feature])
    
    for idx in worst_indices:
        event_cells = raw_cell_data[idx]
        for cell in event_cells:
            for feature in config.comparison_features:
                if feature in cell.dtype.names:
                    worst_features[feature].append(cell[feature])
    
    # Create comparison plots for each feature
    for feature in config.comparison_features:
        if len(best_features[feature]) == 0 or len(worst_features[feature]) == 0:
            continue
            
        plt.figure(figsize=(12, 6))
        
        best_data = np.array(best_features[feature])
        worst_data = np.array(worst_features[feature])
        
        # Determine appropriate bins
        all_data = np.concatenate([best_data, worst_data])
        bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 50)
        
        # Plot histograms
        plt.hist(best_data, bins=bins, alpha=0.7, color='green', 
                label=f'Best events (N={len(best_data)})')
        plt.hist(worst_data, bins=bins, alpha=0.7, color='red', 
                label=f'Worst events (N={len(worst_data)})')
        
        # Calculate statistics
        best_mean, best_std = np.mean(best_data), np.std(best_data)
        worst_mean, worst_std = np.mean(worst_data), np.std(worst_data)
        
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.title(f'Feature Comparison: {feature}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Best: μ={best_mean:.3f}, σ={best_std:.3f}\nWorst: μ={worst_mean:.3f}, σ={worst_std:.3f}"
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                fontsize=9)
        
        plt.tight_layout()
        plt.savefig(feature_plots_dir / f'{feature}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create reco-truth comparison plots
    best_errors = t0_errors[best_indices]
    worst_errors = t0_errors[worst_indices]
    
    plt.figure(figsize=(12, 6))
    bins = np.linspace(-2000, 2000, 100)
    plt.hist(best_errors, bins=bins, alpha=0.7, color='green', 
            label=f'Best events (N={len(best_errors)})')
    plt.hist(worst_errors, bins=bins, alpha=0.7, color='red', 
            label=f'Worst events (N={len(worst_errors)})')
    
    plt.xlabel('Reco - Truth [ps]')
    plt.ylabel('Count')
    plt.title('Error Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    best_err_mean, best_err_std = np.mean(best_errors), np.std(best_errors)
    worst_err_mean, worst_err_std = np.mean(worst_errors), np.std(worst_errors)
    
    stats_text = f"Best: μ={best_err_mean:.2f}, σ={best_err_std:.2f}\nWorst: μ={worst_err_mean:.2f}, σ={worst_err_std:.2f}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(feature_plots_dir / 'error_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature comparison plots saved to {feature_plots_dir}")


def create_additional_analysis_plots(raw_cell_data: List, baseline_t0: np.ndarray, 
                                   vertex_times: np.ndarray, t0_errors: np.ndarray,
                                   config: BaselineAnalysisConfig, output_dir: Path, 
                                   logger: logging.Logger):
    """Create additional analysis plots for understanding reconstruction failures."""
    logger.info("Creating additional analysis plots...")
    
    additional_plots_dir = output_dir / "additional_analysis"
    additional_plots_dir.mkdir(exist_ok=True)
    
    # Plot 1: Error vs number of cells
    cell_counts = [len(event_cells) for event_cells in raw_cell_data]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(cell_counts, np.abs(t0_errors), alpha=0.5, s=1)
    plt.xlabel('Number of Cells per Event')
    plt.ylabel('|Reconstruction Error| [ps]')
    plt.title('Reconstruction Error vs Number of Cells')
    plt.grid(True, alpha=0.3)
    
    # Add correlation
    correlation = np.corrcoef(cell_counts, np.abs(t0_errors))[0, 1]
    plt.text(0.05, 0.95, f'Correlation = {correlation:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(additional_plots_dir / 'error_vs_ncells.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Error vs cell energy spread
    energy_spreads = []
    for event_cells in raw_cell_data:
        energies = [cell['Cell_e'] for cell in event_cells]
        energy_spreads.append(np.std(energies))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(energy_spreads, np.abs(t0_errors), alpha=0.5, s=1)
    plt.xlabel('Cell Energy Spread (std) [GeV]')
    plt.ylabel('|Reconstruction Error| [ps]')
    plt.title('Reconstruction Error vs Cell Energy Spread')
    plt.grid(True, alpha=0.3)
    
    correlation = np.corrcoef(energy_spreads, np.abs(t0_errors))[0, 1]
    plt.text(0.05, 0.95, f'Correlation = {correlation:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(additional_plots_dir / 'error_vs_energy_spread.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Error vs cell time spread
    time_spreads = []
    for event_cells in raw_cell_data:
        times = [cell['Cell_time_TOF_corrected'] for cell in event_cells]
        time_spreads.append(np.std(times))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(time_spreads, np.abs(t0_errors), alpha=0.5, s=1)
    plt.xlabel('Cell Time Spread (std) [ps]')
    plt.ylabel('|Reconstruction Error| [ps]')
    plt.title('Reconstruction Error vs Cell Time Spread')
    plt.grid(True, alpha=0.3)
    
    correlation = np.corrcoef(time_spreads, np.abs(t0_errors))[0, 1]
    plt.text(0.05, 0.95, f'Correlation = {correlation:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(additional_plots_dir / 'error_vs_time_spread.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Layer composition analysis
    layer_compositions = []
    for event_cells in raw_cell_data:
        layers = [cell['Cell_layer'] for cell in event_cells]
        layer_counts = {1: 0, 2: 0, 3: 0}
        for layer in layers:
            if layer in layer_counts:
                layer_counts[layer] += 1
        total = sum(layer_counts.values())
        if total > 0:
            layer_frac_1 = layer_counts[1] / total
            layer_compositions.append(layer_frac_1)
        else:
            layer_compositions.append(0.0)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(layer_compositions, np.abs(t0_errors), alpha=0.5, s=1)
    plt.xlabel('Fraction of Layer 1 Cells')
    plt.ylabel('|Reconstruction Error| [ps]')
    plt.title('Reconstruction Error vs Layer 1 Cell Fraction')
    plt.grid(True, alpha=0.3)
    
    correlation = np.corrcoef(layer_compositions, np.abs(t0_errors))[0, 1]
    plt.text(0.05, 0.95, f'Correlation = {correlation:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(additional_plots_dir / 'error_vs_layer1_fraction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Additional analysis plots saved to {additional_plots_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Baseline Time Reconstruction Analysis Tool')
    parser.add_argument('--config', type=str, 
                       default='baseline_analysis_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--top-events', type=int, default=None,
                       help='Number of worst events to analyze in detail')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for feature comparison plots')
    
    args = parser.parse_args()
    
    # Load configuration
    config = BaselineAnalysisConfig(args.config)
    
    # Override config with command line arguments
    if args.top_events is not None:
        config.top_worst_events = args.top_events
    if args.sample_size is not None:
        config.feature_comparison_sample_size = args.sample_size
    
    # Create output directory
    if config.create_timestamp_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.output_base_dir) / f"baseline_analysis_{timestamp}"
    else:
        output_dir = Path(config.output_base_dir) / "baseline_analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy configuration file to output directory
    shutil.copy(args.config, output_dir / "config_used.yaml")
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info(f"{'='*60}")
    logger.info("BASELINE TIME RECONSTRUCTION ANALYSIS")
    logger.info(f"{'='*60}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Matching type: {'Track' if config.use_track_features else 'Jet' if config.use_jet_features else 'None'}")
    
    try:
        # Load and filter data
        cell_sequences, vertex_times, raw_cell_data = load_and_filter_data(config, logger)
        
        if len(cell_sequences) == 0:
            logger.error("No valid events found. Exiting.")
            return
        
        # Apply time calibration
        calibrated_sequences = apply_time_calibration(cell_sequences, config, logger)
        
        # Calculate baseline t0
        baseline_t0, t0_errors = calculate_baseline_t0(calibrated_sequences, vertex_times, config, logger)
        
        # Analyze worst events
        analyze_worst_events(baseline_t0, vertex_times, raw_cell_data, config, logger)
        
        # Create baseline plots
        create_baseline_plots(baseline_t0, vertex_times, t0_errors, config, output_dir, logger)
        
        # Create feature comparison plots
        create_feature_comparison_plots(raw_cell_data, t0_errors, config, output_dir, logger)
        
        # Create additional analysis plots
        create_additional_analysis_plots(raw_cell_data, baseline_t0, vertex_times, 
                                       t0_errors, config, output_dir, logger)
        
        # Calculate and log summary statistics
        correlation = np.corrcoef(vertex_times, baseline_t0)[0, 1]
        rmse = np.sqrt(np.mean((baseline_t0 - vertex_times) ** 2))
        mae = np.mean(np.abs(baseline_t0 - vertex_times))
        mean_error = np.mean(t0_errors)
        std_error = np.std(t0_errors)
        
        logger.info(f"\n{'='*60}")
        logger.info("ANALYSIS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Events processed: {len(baseline_t0)}")
        logger.info(f"Correlation: {correlation:.4f}")
        logger.info(f"RMSE: {rmse:.4f} ps")
        logger.info(f"MAE: {mae:.4f} ps")
        logger.info(f"Mean error: {mean_error:.4f} ps")
        logger.info(f"Error std: {std_error:.4f} ps")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()