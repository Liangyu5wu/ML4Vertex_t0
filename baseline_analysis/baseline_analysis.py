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

# Add the src directory to the path to import DataLoader and BaseConfig
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))
from data.data_loader import DataLoader
from config.base_config import BaseConfig


class BaselineAnalysisConfig(BaseConfig):
    """Configuration class for baseline analysis, inheriting from BaseConfig."""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration from YAML file."""
        if config_file is None:
            config_file = Path(__file__).parent / "baseline_analysis_config.yaml"
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Initialize with defaults from BaseConfig
        super().__init__()
        
        # Override with values from YAML file
        for key, value in config_data.items():
            setattr(self, key, value)
        
        # Run post_init to setup feature lists
        self.__post_init__()
        
        # Load calibration data
        self.calibration_data = self.load_calibration_data()
        
        # Energy bins for calibration  
        self.energy_bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, float('inf')]
    
    def load_calibration_data(self) -> Dict[str, List[float]]:
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
    """Load and filter data from HDF5 files using optimized filtering approach."""
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
                
                # Apply optimized cell filtering
                filtered_cells = apply_optimized_cell_filtering(event_cells, config)
                
                if len(filtered_cells) < config.min_cells:
                    continue
                
                # Apply baseline method filter if enabled
                if config.use_baseline_method_filter:
                    baseline_error = calculate_baseline_t0_error(filtered_cells, vertex_time, config)
                    if baseline_error > config.baseline_method_threshold:
                        continue
                
                total_cells_after += len(filtered_cells)
                valid_events += 1
                
                # Store filtered cell data for feature analysis
                all_raw_cell_data.append(filtered_cells)
                
                # Convert to list format for processing
                cell_sequence = []
                for cell in filtered_cells:
                    cell_features = [
                        cell['Cell_time_TOF_corrected'],
                        cell['Cell_e'], 
                        cell['Cell_Barrel'],
                        cell['Cell_layer']
                    ]
                    cell_sequence.append(cell_features)
                
                all_cell_sequences.append(cell_sequence)
                all_vertex_times.append(vertex_time)
    
    logger.info(f"Data loading and filtering summary:")
    logger.info(f"  Total events: {total_events}")
    logger.info(f"  Valid events after filtering: {valid_events}")
    logger.info(f"  Total cells before filtering: {total_cells_before}")
    logger.info(f"  Total cells after filtering: {total_cells_after}")
    if total_events > 0:
        event_retention_rate = (valid_events / total_events) * 100
        logger.info(f"  Event retention rate: {event_retention_rate:.1f}%")
    if total_cells_before > 0:
        cell_retention_rate = (total_cells_after / total_cells_before) * 100
        logger.info(f"  Cell retention rate: {cell_retention_rate:.1f}%")
    
    logger.info(f"Applied filters:")
    logger.info(f"  - Valid cells: {config.require_valid_cells}")
    if config.use_track_features:
        logger.info(f"  - Track matching: {config.use_cell_track_matching}")
    if config.use_jet_features:
        logger.info(f"  - Jet matching: {config.use_cell_jet_matching}")
    logger.info(f"  - Time quality cut: {config.use_time_quality_cut}")
    if config.use_time_quality_cut:
        logger.info(f"    └─ Formula: |cell_time| <= sqrt({config.vertex_time_sigma}^2 + sigma_cell^2) * {config.time_quality_n_sigma}")
    logger.info(f"  - Layer filtering: layers 1, 2, 3 only")
    logger.info(f"  - Baseline method filter: {config.use_baseline_method_filter}")
    if config.use_baseline_method_filter:
        logger.info(f"    └─ Threshold: ±{config.baseline_method_threshold:.1f} ps")
    
    return all_cell_sequences, np.array(all_vertex_times), all_raw_cell_data


def apply_optimized_cell_filtering(event_cells: np.ndarray, config: BaselineAnalysisConfig) -> np.ndarray:
    """Apply optimized cell filtering using the validated logic from DataLoader."""
    mask = np.ones(len(event_cells), dtype=bool)
    
    # Apply valid cell filter
    if config.require_valid_cells:
        valid_mask = event_cells['valid'] == True
        mask = mask & valid_mask
    
    # Apply cell-track matching filter
    if config.use_cell_track_matching:
        track_matching_mask = event_cells['matched_track_HS'] == 1
        mask = mask & track_matching_mask
    
    # Apply cell-jet matching filter
    if config.use_cell_jet_matching:
        if 'cell_jet_matched' in event_cells.dtype.names:
            jet_matching_mask = event_cells['cell_jet_matched'] == True
            mask = mask & jet_matching_mask
    
    # Apply layer filtering - only keep cells with layers 1, 2, 3
    if 'Cell_layer' in event_cells.dtype.names:
        layer_mask = np.isin(event_cells['Cell_layer'], [1, 2, 3])
        mask = mask & layer_mask
    
    # Apply additional custom filters
    if config.additional_cell_filters:
        for filter_key, filter_value in config.additional_cell_filters.items():
            if filter_key in event_cells.dtype.names:
                additional_mask = event_cells[filter_key] == filter_value
                mask = mask & additional_mask
    
    # Apply basic filtering first
    filtered_cells = event_cells[mask]
    
    # Apply time quality cut
    if config.use_time_quality_cut:
        filtered_cells = apply_time_quality_cut(filtered_cells, config)
    
    return filtered_cells


def apply_time_quality_cut(event_cells: np.ndarray, config: BaselineAnalysisConfig) -> np.ndarray:
    """Apply time quality cut based on statistical uncertainty using correct formula."""
    if not config.use_time_quality_cut or len(event_cells) == 0:
        return event_cells
    
    # Sigma lookup tables using pre-loaded calibration data
    sigma_lookup = {
        (1, 1): config.calibration_data['EMB1_sigma'],  # Barrel, Layer 1
        (1, 2): config.calibration_data['EMB2_sigma'],  # Barrel, Layer 2
        (1, 3): config.calibration_data['EMB3_sigma'],  # Barrel, Layer 3
        (0, 1): config.calibration_data['EME1_sigma'],  # Endcap, Layer 1
        (0, 2): config.calibration_data['EME2_sigma'],  # Endcap, Layer 2
        (0, 3): config.calibration_data['EME3_sigma'],  # Endcap, Layer 3
    }
    
    # Energy bins for calibration
    energy_bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, float('inf')]
    
    mask = np.ones(len(event_cells), dtype=bool)
    
    for i, cell in enumerate(event_cells):
        try:
            barrel = int(cell['Cell_Barrel'])
            layer = int(cell['Cell_layer'])
            energy = cell['Cell_e']
            cell_time = cell['Cell_time_TOF_corrected']
            
            # Skip cells with invalid layer
            if layer not in [1, 2, 3]:
                mask[i] = False
                continue
            
            # Get sigma for this cell
            sigma_params = sigma_lookup.get((barrel, layer), [1000.0] * 7)
            energy_bin_idx = get_energy_bin_index(energy, energy_bins)
            
            # Add bounds checking for array access
            if energy_bin_idx >= len(sigma_params):
                energy_bin_idx = len(sigma_params) - 1
            elif energy_bin_idx < 0:
                energy_bin_idx = 0
            
            sigma_cell = sigma_params[energy_bin_idx]
            
            # Calculate total uncertainty: σ_total = √(σ_vertex² + σ_cell²)
            sigma_total = np.sqrt(config.vertex_time_sigma**2 + sigma_cell**2)
            
            # Apply n-sigma cut: |cell_time| < n_sigma × σ_total
            cut_threshold = config.time_quality_n_sigma * sigma_total
            
            if abs(cell_time) > cut_threshold:
                mask[i] = False
                
        except (KeyError, ValueError, IndexError):
            # Skip cells with missing or invalid data
            mask[i] = False
    
    return event_cells[mask]


def calculate_baseline_t0_error(event_cells: np.ndarray, true_vertex_time: float, config: BaselineAnalysisConfig) -> float:
    """Calculate baseline t0 error for baseline method filtering."""
    if len(event_cells) == 0:
        return float('inf')
    
    # Energy bins for calibration
    energy_bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, float('inf')]
    
    # Parameter and sigma lookup tables
    param_lookup = {
        (1, 1): config.calibration_data['EMB1_params'],  # Barrel, Layer 1
        (1, 2): config.calibration_data['EMB2_params'],  # Barrel, Layer 2
        (1, 3): config.calibration_data['EMB3_params'],  # Barrel, Layer 3
        (0, 1): config.calibration_data['EME1_params'],  # Endcap, Layer 1
        (0, 2): config.calibration_data['EME2_params'],  # Endcap, Layer 2
        (0, 3): config.calibration_data['EME3_params'],  # Endcap, Layer 3
    }
    
    sigma_lookup = {
        (1, 1): config.calibration_data['EMB1_sigma'],  # Barrel, Layer 1
        (1, 2): config.calibration_data['EMB2_sigma'],  # Barrel, Layer 2
        (1, 3): config.calibration_data['EMB3_sigma'],  # Barrel, Layer 3
        (0, 1): config.calibration_data['EME1_sigma'],  # Endcap, Layer 1
        (0, 2): config.calibration_data['EME2_sigma'],  # Endcap, Layer 2
        (0, 3): config.calibration_data['EME3_sigma'],  # Endcap, Layer 3
    }
    
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for cell in event_cells:
        try:
            barrel = int(cell['Cell_Barrel'])
            layer = int(cell['Cell_layer'])
            energy = cell['Cell_e']
            time_tof = cell['Cell_time_TOF_corrected']
            
            # Skip cells with invalid layer
            if layer not in [1, 2, 3]:
                continue
            
            # Get calibration parameters and sigma
            detector_params = param_lookup.get((barrel, layer), [0.0] * 7)
            sigma_params = sigma_lookup.get((barrel, layer), [1000.0] * 7)
            
            energy_bin_idx = get_energy_bin_index(energy, energy_bins)
            
            # Add bounds checking
            if energy_bin_idx >= len(detector_params):
                energy_bin_idx = len(detector_params) - 1
            elif energy_bin_idx < 0:
                energy_bin_idx = 0
            
            calibration_value = detector_params[energy_bin_idx]
            sigma = sigma_params[energy_bin_idx]
            
            # Apply calibration
            calibrated_time = time_tof - calibration_value
            
            # Weight = 1/sigma^2
            weight = 1.0 / (sigma * sigma)
            
            weighted_sum += weight * calibrated_time
            weight_sum += weight
            
        except (KeyError, ValueError, IndexError):
            continue
    
    if weight_sum > 0:
        baseline_t0 = weighted_sum / weight_sum
        return abs(baseline_t0 - true_vertex_time)
    else:
        return float('inf')


def get_energy_bin_index(energy: float, energy_bins: List[float]) -> int:
    """Get energy bin index for calibration parameter lookup."""
    if energy < 1.0:
        return 0
    
    for i in range(len(energy_bins) - 1):
        if energy_bins[i] <= energy < energy_bins[i + 1]:
            return i
    
    return len(energy_bins) - 2


# Removed apply_time_calibration function - calibration is now handled within DataLoader's baseline calculation


def calculate_baseline_t0(raw_cell_data: List, vertex_times: np.ndarray, 
                         config: BaselineAnalysisConfig, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate baseline t0 for each event using optimized implementation."""
    logger.info("Calculating baseline t0...")
    
    # Energy bins for calibration
    energy_bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, float('inf')]
    
    # Parameter and sigma lookup tables
    param_lookup = {
        (1, 1): config.calibration_data['EMB1_params'],  # Barrel, Layer 1
        (1, 2): config.calibration_data['EMB2_params'],  # Barrel, Layer 2
        (1, 3): config.calibration_data['EMB3_params'],  # Barrel, Layer 3
        (0, 1): config.calibration_data['EME1_params'],  # Endcap, Layer 1
        (0, 2): config.calibration_data['EME2_params'],  # Endcap, Layer 2
        (0, 3): config.calibration_data['EME3_params'],  # Endcap, Layer 3
    }
    
    sigma_lookup = {
        (1, 1): config.calibration_data['EMB1_sigma'],  # Barrel, Layer 1
        (1, 2): config.calibration_data['EMB2_sigma'],  # Barrel, Layer 2
        (1, 3): config.calibration_data['EMB3_sigma'],  # Barrel, Layer 3
        (0, 1): config.calibration_data['EME1_sigma'],  # Endcap, Layer 1
        (0, 2): config.calibration_data['EME2_sigma'],  # Endcap, Layer 2
        (0, 3): config.calibration_data['EME3_sigma'],  # Endcap, Layer 3
    }
    
    baseline_t0 = []
    
    for event_idx, event_cells in enumerate(raw_cell_data):
        weighted_sum = 0.0
        weight_sum = 0.0
        
        original_cell_times = []
        calibrated_cell_times = []
        
        for cell in event_cells:
            try:
                barrel = int(cell['Cell_Barrel'])
                layer = int(cell['Cell_layer'])
                energy = cell['Cell_e']
                time_tof = cell['Cell_time_TOF_corrected']
                
                original_cell_times.append(time_tof)
                
                # Skip cells with invalid layer
                if layer not in [1, 2, 3]:
                    continue
                
                # Get calibration parameters and sigma
                detector_params = param_lookup.get((barrel, layer), [0.0] * 7)
                sigma_params = sigma_lookup.get((barrel, layer), [1000.0] * 7)
                
                energy_bin_idx = get_energy_bin_index(energy, energy_bins)
                
                # Add bounds checking
                if energy_bin_idx >= len(detector_params):
                    energy_bin_idx = len(detector_params) - 1
                elif energy_bin_idx < 0:
                    energy_bin_idx = 0
                
                calibration_value = detector_params[energy_bin_idx]
                sigma = sigma_params[energy_bin_idx]
                
                # Apply calibration
                calibrated_time = time_tof - calibration_value
                calibrated_cell_times.append(calibrated_time)
                
                # Weight = 1/sigma^2
                weight = 1.0 / (sigma * sigma)
                
                weighted_sum += weight * calibrated_time
                weight_sum += weight
                
            except (KeyError, ValueError, IndexError):
                continue
        
        if weight_sum > 0:
            t0 = weighted_sum / weight_sum
        else:
            t0 = 0.0
        
        baseline_t0.append(t0)
        
        # Print debug info for first few events
        if event_idx < 10:
            logger.info(f"Event {event_idx}:")
            logger.info(f"  Truth vertex time: {vertex_times[event_idx]:.4f} ps")
            logger.info(f"  Number of filtered cells: {len(original_cell_times)}")
            
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
        
        # Calculate calibrated times for this event
        param_lookup = {
            (1, 1): config.calibration_data['EMB1_params'],
            (1, 2): config.calibration_data['EMB2_params'], 
            (1, 3): config.calibration_data['EMB3_params'],
            (0, 1): config.calibration_data['EME1_params'],
            (0, 2): config.calibration_data['EME2_params'],
            (0, 3): config.calibration_data['EME3_params'],
        }
        
        original_cell_times = []
        calibrated_cell_times = []
        
        for cell in event_cells:
            time_tof = cell['Cell_time_TOF_corrected']
            energy = cell['Cell_e']
            barrel = int(cell['Cell_Barrel'])
            layer = int(cell['Cell_layer'])
            
            original_cell_times.append(time_tof)
            
            # Apply calibration
            detector_params = param_lookup.get((barrel, layer), [0.0] * 7)
            energy_bin_idx = get_energy_bin_index(energy, config.energy_bins)
            calibration_value = detector_params[energy_bin_idx]
            calibrated_time = time_tof - calibration_value
            calibrated_cell_times.append(calibrated_time)
        
        # Print all cell times before and after calibration
        original_times_str = ", ".join([f'{t:.1f}' for t in original_cell_times])
        logger.info(f"  Original cell times (before calibration): {original_times_str} ps")
        
        calibrated_times_str = ", ".join([f'{t:.1f}' for t in calibrated_cell_times])  
        logger.info(f"  Calibrated cell times (after calibration): {calibrated_times_str} ps")
        
        # Cell time statistics
        logger.info(f"  Cell time statistics - Original: mean={np.mean(original_cell_times):.2f}, "
                   f"std={np.std(original_cell_times):.2f} ps")
        logger.info(f"  Cell time statistics - Calibrated: mean={np.mean(calibrated_cell_times):.2f}, "
                   f"std={np.std(calibrated_cell_times):.2f} ps")
        
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
            # Estimate Gaussian parameters from data
            fit_mean = np.mean(fit_data)
            fit_std = np.std(fit_data)
            
            # Use the same bins as the original histogram for amplitude calculation
            fit_bins_count = 50  # Number of bins for fitting
            fit_bin_width = (2 * fit_range) / fit_bins_count
            
            # Calculate proper amplitude: total counts in fit range * bin_width / (std * sqrt(2*pi))
            fit_amplitude = len(fit_data) * fit_bin_width / (fit_std * np.sqrt(2 * np.pi))
            
            # Fit Gaussian to data using scipy.optimize.curve_fit on the actual data distribution
            from scipy.stats import norm
            x_fit = np.linspace(-fit_range, fit_range, 200)
            
            # Create a proper normalized Gaussian fit
            def gaussian_pdf_scaled(x, amplitude, mean, std):
                return amplitude * np.exp(-0.5 * ((x - mean) / std) ** 2)
            
            # Initial guess based on data statistics
            initial_guess = [fit_amplitude, fit_mean, fit_std]
            
            # Use the original histogram data for fitting  
            hist_values, bin_edges = np.histogram(fit_data, bins=np.linspace(-fit_range, fit_range, fit_bins_count))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Only fit non-zero bins
            nonzero_mask = hist_values > 0
            if np.sum(nonzero_mask) >= 3:
                popt, _ = curve_fit(gaussian_pdf_scaled, bin_centers[nonzero_mask], 
                                  hist_values[nonzero_mask], p0=initial_guess, maxfev=2000)
                
                fit_amplitude, fit_mean, fit_std = popt
                fit_std = abs(fit_std)
                
                y_fit = gaussian_pdf_scaled(x_fit, fit_amplitude, fit_mean, fit_std)
                plt.plot(x_fit, y_fit, 'r-', linewidth=2,
                        label=f'Gaussian fit (±{fit_range}): μ={fit_mean:.2f}, σ={fit_std:.2f}')
            else:
                # Fallback: simple analytical Gaussian
                y_fit = fit_amplitude * np.exp(-0.5 * ((x_fit - fit_mean) / fit_std) ** 2)
                plt.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7,
                        label=f'Analytical fit (±{fit_range}): μ={fit_mean:.2f}, σ={fit_std:.2f}')
        except Exception as e:
            # Fallback: simple analytical Gaussian based on data statistics
            fit_mean = np.mean(fit_data)
            fit_std = np.std(fit_data)
            fit_bins_count = 50
            fit_bin_width = (2 * fit_range) / fit_bins_count
            fit_amplitude = len(fit_data) * fit_bin_width / (fit_std * np.sqrt(2 * np.pi))
            
            x_fit = np.linspace(-fit_range, fit_range, 200)
            y_fit = fit_amplitude * np.exp(-0.5 * ((x_fit - fit_mean) / fit_std) ** 2)
            plt.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7,
                    label=f'Analytical fit (±{fit_range}): μ={fit_mean:.2f}, σ={fit_std:.2f}')
    
    mean_error = np.mean(t0_errors)
    std_error = np.std(t0_errors)
    plt.xlabel('Baseline t0 - True t0 [ps]')
    plt.ylabel('Count')
    plt.title('Baseline t0 Error Distribution')
    
    # Add data statistics as text
    plt.text(0.02, 0.98, f'All data: μ={mean_error:.2f}, σ={std_error:.2f}, N={len(t0_errors)}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
             fontsize=10)
    
    # Show legend only if there are fit lines
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(loc='upper right')
    
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


def create_feature_comparison_plots(raw_cell_data: List, baseline_t0: np.ndarray, t0_errors: np.ndarray, 
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
    
    # Create reconstruction time distribution comparison plots
    # Get the actual reconstruction times for best and worst events
    best_reco_times = baseline_t0[best_indices] 
    worst_reco_times = baseline_t0[worst_indices]
    
    plt.figure(figsize=(12, 6))
    bins = np.linspace(-2000, 2000, 100)
    plt.hist(best_reco_times, bins=bins, alpha=0.7, color='green', 
            label=f'Best events (N={len(best_reco_times)})')
    plt.hist(worst_reco_times, bins=bins, alpha=0.7, color='red', 
            label=f'Worst events (N={len(worst_reco_times)})')
    
    plt.xlabel('Reconstructed Time [ps]')
    plt.ylabel('Count')
    plt.title('Reconstruction Time Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    best_reco_mean, best_reco_std = np.mean(best_reco_times), np.std(best_reco_times)
    worst_reco_mean, worst_reco_std = np.mean(worst_reco_times), np.std(worst_reco_times)
    
    stats_text = f"Best: μ={best_reco_mean:.2f}, σ={best_reco_std:.2f}\nWorst: μ={worst_reco_mean:.2f}, σ={worst_reco_std:.2f}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(feature_plots_dir / 'reconstruction_time_comparison.png', dpi=300, bbox_inches='tight')
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
    plt.scatter(cell_counts, t0_errors, alpha=0.5, s=1)
    plt.xlabel('Number of Cells per Event')
    plt.ylabel('Reconstruction Error [ps]')
    plt.title('Reconstruction Error vs Number of Cells')
    plt.grid(True, alpha=0.3)
    
    # Add data size info
    plt.text(0.05, 0.95, f'N = {len(cell_counts):,} events', 
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
    plt.scatter(energy_spreads, t0_errors, alpha=0.5, s=1)
    plt.xlabel('Cell Energy Spread (std) [GeV]')
    plt.ylabel('Reconstruction Error [ps]')
    plt.title('Reconstruction Error vs Cell Energy Spread')
    plt.grid(True, alpha=0.3)
    
    plt.text(0.05, 0.95, f'N = {len(energy_spreads):,} events', 
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
    plt.scatter(time_spreads, t0_errors, alpha=0.5, s=1)
    plt.xlabel('Cell Time Spread (std) [ps]')
    plt.ylabel('Reconstruction Error [ps]')
    plt.title('Reconstruction Error vs Cell Time Spread')
    plt.grid(True, alpha=0.3)
    
    plt.text(0.05, 0.95, f'N = {len(time_spreads):,} events', 
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
    plt.scatter(layer_compositions, t0_errors, alpha=0.5, s=1)
    plt.xlabel('Fraction of Layer 1 Cells')
    plt.ylabel('Reconstruction Error [ps]')
    plt.title('Reconstruction Error vs Layer 1 Cell Fraction')
    plt.grid(True, alpha=0.3)
    
    plt.text(0.05, 0.95, f'N = {len(layer_compositions):,} events', 
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
        
        # Calculate baseline t0 using filtered raw cell data
        baseline_t0, t0_errors = calculate_baseline_t0(raw_cell_data, vertex_times, config, logger)
        
        # Note: Baseline method filtering is now handled during data loading in load_and_filter_data()
        logger.info(f"Baseline method filtering: {'enabled' if config.use_baseline_method_filter else 'disabled'}")
        if config.use_baseline_method_filter:
            logger.info(f"  Threshold: ±{config.baseline_method_threshold:.1f} ps (applied during data loading)")
        
        # Analyze worst events
        analyze_worst_events(baseline_t0, vertex_times, raw_cell_data, config, logger)
        
        # Create baseline plots
        create_baseline_plots(baseline_t0, vertex_times, t0_errors, config, output_dir, logger)
        
        # Create feature comparison plots
        create_feature_comparison_plots(raw_cell_data, baseline_t0, t0_errors, config, output_dir, logger)
        
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