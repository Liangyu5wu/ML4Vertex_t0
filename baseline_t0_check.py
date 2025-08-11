#!/usr/bin/env python3
"""
Enhanced baseline t0 reconstruction check with parallel cell-track and cell-jet matching analysis.
This script performs both cell-track matching and cell-jet matching based t0 reconstruction
using different calibration data sets and saves results in parallel subdirectories.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict


class SimpleConfig:
    """Simplified configuration for baseline t0 reconstruction."""
    
    def __init__(self, calibration_file: str = "HStrackmatching_calibration.txt"):
        # Data parameters
        self.data_dir = "/fs/ddn/sdf/group/atlas/d/liangyu/jetML/datasets/h5/selected_h5_with_jets/"
        self.num_files = 5
        self.min_cells = 1
        
        # Cell filtering parameters
        self.use_cell_track_matching = True
        self.require_valid_cells = True
        
        # Energy bins for calibration: [1-1.5, 1.5-2, 2-3, 3-4, 4-5, 5-10, >10]
        self.energy_bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, float('inf')]
        
        # Gaussian fit range for plots
        self.gaussian_fit_range = 120
        
        # Calibration data file
        self.calibration_file = calibration_file
        self.calibration_data = self.load_calibration_data()
        
    def load_calibration_data(self) -> Dict[str, List[float]]:
        """Load calibration data from external file."""
        calibration_path = Path("calibration_data") / self.calibration_file
        
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration data file not found: {calibration_path}")
        
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
        
        print(f"Loaded calibration data from: {calibration_path}")
        return calibration_data


def load_and_filter_data(config: SimpleConfig, matching_type: str = "track") -> Tuple[List, np.ndarray]:
    """
    Load and filter data from HDF5 files.
    
    Args:
        config: Configuration object
        matching_type: "track" for cell-track matching, "jet" for cell-jet matching
        
    Returns:
        Tuple of (filtered_cell_sequences, vertex_times)
    """
    print(f"Loading data from {config.data_dir}")
    print(f"Number of files: {config.num_files}")
    print(f"Matching type: {matching_type}")
    
    if matching_type == "track":
        print(f"Cell filtering - Track matching: {config.use_cell_track_matching}, Valid cells: {config.require_valid_cells}")
    else:
        print(f"Cell filtering - Jet matching: True, Valid cells: {config.require_valid_cells}")
    
    all_cell_sequences = []
    all_vertex_times = []
    
    total_events = 0
    valid_events = 0
    total_cells_before = 0
    total_cells_after = 0
    
    for i in range(config.num_files):
        file_path = os.path.join(config.data_dir, f"output_{i:03d}.h5")
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping...")
            continue
            
        print(f"Processing {file_path}...")
        
        with h5py.File(file_path, 'r') as f:
            vertex_data = f['HSvertex'][:]
            cells_data = f['cells'][:]
            
            total_events += len(vertex_data)
            
            for event_idx in range(len(vertex_data)):
                # Get vertex time (target)
                vertex_time = vertex_data[event_idx]['HSvertex_time']
                
                # Get cells for this event
                event_cells = cells_data[event_idx]
                
                if len(event_cells) == 0:
                    continue
                
                total_cells_before += len(event_cells)
                
                # Apply filtering based on matching type
                if matching_type == "track":
                    filtered_cells = apply_cell_track_filtering(event_cells, config)
                else:  # jet matching
                    filtered_cells = apply_cell_jet_filtering(event_cells, config)
                
                if len(filtered_cells) < config.min_cells:
                    continue
                
                # Additional layer filtering - only keep cells with layers 1, 2, 3
                layer_filtered_cells = []
                for cell in filtered_cells:
                    if cell['Cell_layer'] in [1, 2, 3]:
                        layer_filtered_cells.append(cell)
                
                # Skip events with no valid layer cells
                if len(layer_filtered_cells) < config.min_cells:
                    continue
                
                total_cells_after += len(layer_filtered_cells)
                valid_events += 1
                
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
    
    print(f"\nData loading summary:")
    print(f"  Total events: {total_events}")
    print(f"  Valid events: {valid_events}")
    print(f"  Total cells before filtering: {total_cells_before}")
    print(f"  Total cells after filtering: {total_cells_after}")
    if total_cells_before > 0:
        retention_rate = (total_cells_after / total_cells_before) * 100
        print(f"  Cell retention rate: {retention_rate:.1f}%")
    
    return all_cell_sequences, np.array(all_vertex_times)


def apply_cell_track_filtering(event_cells, config: SimpleConfig):
    """Apply cell-track matching filtering based on configuration."""
    mask = np.ones(len(event_cells), dtype=bool)
    
    # Apply valid cell filter
    if config.require_valid_cells:
        valid_mask = event_cells['valid'] == True
        mask = mask & valid_mask
    
    # Apply cell-track matching filter
    if config.use_cell_track_matching:
        track_matching_mask = event_cells['matched_track_HS'] == 1
        mask = mask & track_matching_mask
    
    return event_cells[mask]


def apply_cell_jet_filtering(event_cells, config: SimpleConfig):
    """Apply cell-jet matching filtering based on configuration."""
    mask = np.ones(len(event_cells), dtype=bool)
    
    # Apply valid cell filter
    if config.require_valid_cells:
        valid_mask = event_cells['valid'] == True
        mask = mask & valid_mask
    
    # Apply cell-jet matching filter
    jet_matching_mask = event_cells['cell_jet_matched'] == True
    mask = mask & jet_matching_mask
    
    return event_cells[mask]


def get_energy_bin_index(energy: float, energy_bins: List[float]) -> int:
    """Get energy bin index for calibration parameter lookup."""
    if energy < 1.0:
        return 0  # Use first bin for energies < 1 GeV
    
    for i in range(len(energy_bins) - 1):
        if energy_bins[i] <= energy < energy_bins[i + 1]:
            return i
    
    return len(energy_bins) - 2  # Last bin for energies >= 10 GeV


def apply_time_calibration(cell_sequences: List, config: SimpleConfig) -> List:
    """
    Apply detector time calibration to cell sequences.
    
    Args:
        cell_sequences: Original cell sequences (already layer-filtered)
        config: Configuration with calibration data
        
    Returns:
        Cell sequences with calibrated time
    """
    print("Applying time calibration...")
    
    # Parameter lookup - using 1-based layer indexing
    param_lookup = {
        (1, 1): config.calibration_data['EMB1_params'],  # Barrel, Layer 1
        (1, 2): config.calibration_data['EMB2_params'],  # Barrel, Layer 2
        (1, 3): config.calibration_data['EMB3_params'],  # Barrel, Layer 3
        (0, 1): config.calibration_data['EME1_params'],  # Endcap, Layer 1
        (0, 2): config.calibration_data['EME2_params'],  # Endcap, Layer 2
        (0, 3): config.calibration_data['EME3_params'],  # Endcap, Layer 3
    }
    
    calibrated_sequences = []
    
    for sequence in cell_sequences:
        calibrated_sequence = []
        
        for cell in sequence:
            calibrated_cell = cell.copy()
            
            # Extract cell properties
            time_tof = cell[0]  # Cell_time_TOF_corrected
            energy = cell[1]    # Cell_e
            barrel = int(cell[2])  # Cell_Barrel
            layer = int(cell[3])   # Cell_layer (already 1, 2, 3)
            
            # Get calibration parameters - use layer directly
            detector_params = param_lookup.get((barrel, layer), [0.0] * 7)
            
            # Get energy bin index and calibration value
            energy_bin_idx = get_energy_bin_index(energy, config.energy_bins)
            calibration_value = detector_params[energy_bin_idx]
            
            # Apply calibration: corrected_time = tof_corrected_time - calibration_value
            calibrated_time = time_tof - calibration_value
            calibrated_cell[0] = calibrated_time
            
            calibrated_sequence.append(calibrated_cell)
        
        calibrated_sequences.append(calibrated_sequence)
    
    return calibrated_sequences


def calculate_traditional_t0(cell_sequences: List, vertex_times: np.ndarray, config: SimpleConfig, 
                           matching_type: str = "track") -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate traditional (non-ML) t0 for each event using weighted average.
    
    Args:
        cell_sequences: Calibrated cell sequences (already layer-filtered)
        vertex_times: True vertex times
        config: Configuration with calibration data
        matching_type: "track" or "jet" for output labeling
        
    Returns:
        Tuple of (traditional_t0, t0_errors)
    """
    print(f"Calculating traditional t0 for {matching_type} matching...")
    
    # Sigma lookup tables - using 1-based layer indexing
    sigma_lookup = {
        (1, 1): config.calibration_data['EMB1_sigma'],  # Barrel, Layer 1
        (1, 2): config.calibration_data['EMB2_sigma'],  # Barrel, Layer 2
        (1, 3): config.calibration_data['EMB3_sigma'],  # Barrel, Layer 3
        (0, 1): config.calibration_data['EME1_sigma'],  # Endcap, Layer 1
        (0, 2): config.calibration_data['EME2_sigma'],  # Endcap, Layer 2
        (0, 3): config.calibration_data['EME3_sigma'],  # Endcap, Layer 3
    }
    
    # Parameter lookup for getting original times - using 1-based layer indexing
    param_lookup = {
        (1, 1): config.calibration_data['EMB1_params'],  # Barrel, Layer 1
        (1, 2): config.calibration_data['EMB2_params'],  # Barrel, Layer 2
        (1, 3): config.calibration_data['EMB3_params'],  # Barrel, Layer 3
        (0, 1): config.calibration_data['EME1_params'],  # Endcap, Layer 1
        (0, 2): config.calibration_data['EME2_params'],  # Endcap, Layer 2
        (0, 3): config.calibration_data['EME3_params'],  # Endcap, Layer 3
    }
    
    traditional_t0 = []
    
    for event_idx, sequence in enumerate(cell_sequences):
        weighted_sum = 0.0
        weight_sum = 0.0
        
        # Collect cell times for debugging output
        calibrated_cell_times = []
        original_cell_times = []
        
        for cell in sequence:
            calibrated_time = cell[0]  # Already calibrated time
            energy = cell[1]           # Cell_e
            barrel = int(cell[2])      # Cell_Barrel
            layer = int(cell[3])       # Cell_layer (already 1, 2, 3)
            
            calibrated_cell_times.append(calibrated_time)
            
            # Calculate original time (before calibration) - use layer directly
            detector_params = param_lookup.get((barrel, layer), [0.0] * 7)
            energy_bin_idx = get_energy_bin_index(energy, config.energy_bins)
            calibration_value = detector_params[energy_bin_idx]
            original_time = calibrated_time + calibration_value  # Add back the calibration
            original_cell_times.append(original_time)
            
            # Get sigma for this cell - use layer directly
            sigma_params = sigma_lookup.get((barrel, layer), [1000.0] * 7)
            energy_bin_idx = get_energy_bin_index(energy, config.energy_bins)
            sigma = sigma_params[energy_bin_idx]
            
            # Weight = 1/sigma^2
            weight = 1.0 / (sigma * sigma)
            
            weighted_sum += weight * calibrated_time
            weight_sum += weight
        
        if weight_sum > 0:
            t0 = weighted_sum / weight_sum
        else:
            t0 = 0.0
        
        traditional_t0.append(t0)
        
        # Print debug info for first n events
        if event_idx < 10:  # Print first 10 events
            print(f"\n{matching_type.capitalize()} matching - Event {event_idx}:")
            print(f"  Truth vertex time: {vertex_times[event_idx]:.4f} ns")
            print(f"  Number of filtered cells: {len(calibrated_cell_times)}")
            
            # Print all original cell times (before calibration)
            original_times_str = ", ".join([f'{t:.1f}' for t in original_cell_times])
            print(f"  Original cell times (before calibration): {original_times_str} ns")
            
            # Print all calibrated cell times (after calibration)
            calibrated_times_str = ", ".join([f'{t:.1f}' for t in calibrated_cell_times])
            print(f"  Calibrated cell times (after calibration): {calibrated_times_str} ns")
            
            print(f"  Reconstructed vertex time: {t0:.4f} ns")
            print(f"  Error (reco - truth): {t0 - vertex_times[event_idx]:.4f} ns")
    
    traditional_t0 = np.array(traditional_t0)
    t0_errors = traditional_t0 - vertex_times
    
    print(f"\nTraditional t0 calculation completed for {len(traditional_t0)} events using {matching_type} matching")
    
    return traditional_t0, t0_errors


def gaussian_func(x, a, mu, sigma):
    """Gaussian function for fitting."""
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def plot_t0_distribution(traditional_t0: np.ndarray, config: SimpleConfig, save_path: str, matching_type: str):
    """Plot traditional t0 distribution with Gaussian fit."""
    plt.figure(figsize=(10, 6))
    
    # Create histogram with bin width = 10, limited to ±2000 range
    bins = np.arange(-2000, 2010, 10)
    
    counts, bin_edges, _ = plt.hist(traditional_t0, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    
    # Calculate basic statistics
    mean_all = np.mean(traditional_t0)
    std_all = np.std(traditional_t0)
    
    # Gaussian fit on restricted range
    fit_range = config.gaussian_fit_range
    mask = (traditional_t0 >= -fit_range) & (traditional_t0 <= fit_range)
    
    if np.sum(mask) > 10:  # Need enough points for fitting
        fit_data = traditional_t0[mask]
        try:
            # Initial guess for Gaussian parameters
            hist_fit, bin_centers = np.histogram(fit_data, bins=50)
            bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2
            
            # Fit Gaussian
            initial_guess = [np.max(hist_fit), np.mean(fit_data), np.std(fit_data)]
            popt, _ = curve_fit(gaussian_func, bin_centers, hist_fit, p0=initial_guess)
            
            fit_mean, fit_std = popt[1], abs(popt[2])
            
            # Plot fitted Gaussian
            x_fit = np.linspace(-fit_range, fit_range, 200)
            y_fit = gaussian_func(x_fit, *popt)
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
    plt.title(f'Traditional t0 Distribution ({matching_type.capitalize()} Matching)')
    plt.legend([f'All data: μ={mean_all:.2f}, σ={std_all:.2f}, N={len(traditional_t0)}',
               f'Fit range ±{fit_range}: μ={fit_mean:.2f}, σ={fit_std:.2f}'])
    plt.grid(True, alpha=0.3)
    plt.xlim(-2000, 2000)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Traditional t0 distribution plot saved to: {save_path}")


def plot_error_distribution(t0_errors: np.ndarray, config: SimpleConfig, save_path: str, matching_type: str):
    """Plot t0 error distribution with Gaussian fit."""
    plt.figure(figsize=(10, 6))
    
    # Create histogram with bin width = 10, limited to ±2000 range
    bins = np.arange(-2000, 2010, 10)
    
    counts, bin_edges, _ = plt.hist(t0_errors, bins=bins, alpha=0.7, color='green', edgecolor='black')
    
    # Calculate basic statistics
    mean_all = np.mean(t0_errors)
    std_all = np.std(t0_errors)
    
    # Gaussian fit on restricted range
    fit_range = config.gaussian_fit_range
    mask = (t0_errors >= -fit_range) & (t0_errors <= fit_range)
    
    if np.sum(mask) > 10:
        fit_data = t0_errors[mask]
        try:
            hist_fit, bin_centers = np.histogram(fit_data, bins=50)
            bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2
            
            initial_guess = [np.max(hist_fit), np.mean(fit_data), np.std(fit_data)]
            popt, _ = curve_fit(gaussian_func, bin_centers, hist_fit, p0=initial_guess)
            
            fit_mean, fit_std = popt[1], abs(popt[2])
            
            # Plot fitted Gaussian
            x_fit = np.linspace(-fit_range, fit_range, 200)
            y_fit = gaussian_func(x_fit, *popt)
            scale_factor = 10 * len(t0_errors) / len(fit_data)
            plt.plot(x_fit, y_fit * scale_factor, 'r-', linewidth=2,
                    label=f'Gaussian fit (±{fit_range}): μ={fit_mean:.2f}, σ={fit_std:.2f}')
            
        except Exception:
            fit_mean, fit_std = np.mean(fit_data), np.std(fit_data)
    else:
        fit_mean, fit_std = mean_all, std_all
    
    plt.xlabel('Traditional t0 - True t0 [ns]')
    plt.ylabel('Count')
    plt.title(f'Traditional t0 Error Distribution ({matching_type.capitalize()} Matching)')
    plt.legend([f'All data: μ={mean_all:.2f}, σ={std_all:.2f}, N={len(t0_errors)}',
               f'Fit range ±{fit_range}: μ={fit_mean:.2f}, σ={fit_std:.2f}'])
    plt.grid(True, alpha=0.3)
    plt.xlim(-2000, 2000)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Traditional t0 error distribution plot saved to: {save_path}")


def plot_true_vertex_time_distribution(vertex_times: np.ndarray, save_path: str, matching_type: str):
    """Plot true vertex time distribution with statistics."""
    plt.figure(figsize=(10, 6))
    
    # Create histogram with bin width = 10, limited to ±2000 range
    bins = np.arange(-2000, 2010, 10)
    
    counts, bin_edges, _ = plt.hist(vertex_times, bins=bins, alpha=0.7, color='purple', edgecolor='black')
    
    # Calculate statistics
    mean_val = np.mean(vertex_times)
    std_val = np.std(vertex_times)
    
    # Add vertical lines for mean and ±1σ
    plt.axvline(x=mean_val, color='red', linestyle='-', linewidth=2, 
               label=f'Mean: {mean_val:.2f} ns')
    plt.axvline(x=mean_val + std_val, color='red', linestyle=':', linewidth=2, 
               label=f'+1σ: {std_val:.2f} ns')
    plt.axvline(x=mean_val - std_val, color='red', linestyle=':', linewidth=2, 
               label=f'-1σ')
    
    plt.xlabel('True Vertex Time [ns]')
    plt.ylabel('Count')
    plt.title(f'True Vertex Time Distribution ({matching_type.capitalize()} Matching)')
    plt.legend([f'Data: μ={mean_val:.2f}, σ={std_val:.2f}, N={len(vertex_times)}'])
    plt.grid(True, alpha=0.3)
    plt.xlim(-2000, 2000)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"True vertex time distribution plot saved to: {save_path}")


def plot_2d_histogram(traditional_t0: np.ndarray, vertex_times: np.ndarray, save_path: str, matching_type: str):
    """Plot traditional t0 vs true t0 as 2D histogram."""
    plt.figure(figsize=(10, 8))
    
    # Fixed plot range to ±2000
    plot_min, plot_max = -2000, 2000
    
    # Create 2D histogram
    bins = 80
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
    plt.title(f'Traditional t0 vs True t0 ({matching_type.capitalize()} Matching)')
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
    
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Traditional t0 vs true 2D histogram saved to: {save_path}")


def run_analysis(config: SimpleConfig, output_dir: Path, matching_type: str) -> Dict[str, float]:
    """
    Run complete analysis for either track or jet matching.
    
    Args:
        config: Configuration object
        output_dir: Output directory for this analysis
        matching_type: "track" or "jet"
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"{matching_type.upper()} MATCHING ANALYSIS")
    print(f"{'='*60}")
    print(f"Using calibration file: {config.calibration_file}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load and filter data
    cell_sequences, vertex_times = load_and_filter_data(config, matching_type)
    
    if len(cell_sequences) == 0:
        print(f"No valid events found for {matching_type} matching. Skipping.")
        return {}
    
    # Apply time calibration
    calibrated_sequences = apply_time_calibration(cell_sequences, config)
    
    # Calculate traditional t0
    traditional_t0, t0_errors = calculate_traditional_t0(calibrated_sequences, vertex_times, config, matching_type)
    
    # Generate plots
    print(f"\nGenerating plots for {matching_type} matching...")
    
    # Plot 0: True vertex time distribution
    plot_true_vertex_time_distribution(
        vertex_times, 
        output_dir / 'true_vertex_time_distribution.png', 
        matching_type
    )
    
    # Plot 1: Traditional t0 distribution
    plot_t0_distribution(
        traditional_t0, 
        config, 
        output_dir / 'traditional_t0_distribution.png', 
        matching_type
    )
    
    # Plot 2: t0 error distribution
    plot_error_distribution(
        t0_errors, 
        config, 
        output_dir / 't0_error_distribution.png', 
        matching_type
    )
    
    # Plot 3: Traditional t0 vs true t0 2D histogram
    plot_2d_histogram(
        traditional_t0, 
        vertex_times, 
        output_dir / 'traditional_t0_vs_true_2d.png', 
        matching_type
    )
    
    # Calculate summary statistics
    correlation = np.corrcoef(vertex_times, traditional_t0)[0, 1]
    rmse = np.sqrt(np.mean((traditional_t0 - vertex_times) ** 2))
    mae = np.mean(np.abs(traditional_t0 - vertex_times))
    mean_error = np.mean(t0_errors)
    std_error = np.std(t0_errors)
    
    results = {
        'num_events': len(traditional_t0),
        'correlation': correlation,
        'rmse': rmse,
        'mae': mae,
        'mean_error': mean_error,
        'std_error': std_error,
        'mean_t0': np.mean(traditional_t0),
        'std_t0': np.std(traditional_t0)
    }
    
    print(f"\n{matching_type.capitalize()} matching analysis completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Events processed: {results['num_events']}")
    print(f"Correlation: {results['correlation']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    
    return results


def create_comparison_summary(track_results: Dict, jet_results: Dict, output_dir: Path):
    """Create a comparison summary of track vs jet matching results."""
    summary_path = output_dir / 'comparison_summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write("BASELINE T0 RECONSTRUCTION COMPARISON SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("Analysis Overview:\n")
        f.write("- Cell-Track Matching: Uses cells matched to hard-scatter tracks\n")
        f.write("- Cell-Jet Matching: Uses cells matched to jets\n")
        f.write("- Different calibration parameters used for each method\n\n")
        
        if track_results:
            f.write("CELL-TRACK MATCHING RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"  Events processed: {track_results['num_events']}\n")
            f.write(f"  Correlation: {track_results['correlation']:.4f}\n")
            f.write(f"  RMSE: {track_results['rmse']:.4f} ns\n")
            f.write(f"  MAE: {track_results['mae']:.4f} ns\n")
            f.write(f"  Mean error: {track_results['mean_error']:.4f} ns\n")
            f.write(f"  Error std: {track_results['std_error']:.4f} ns\n")
            f.write(f"  Mean t0: {track_results['mean_t0']:.4f} ns\n")
            f.write(f"  t0 std: {track_results['std_t0']:.4f} ns\n\n")
        else:
            f.write("CELL-TRACK MATCHING RESULTS: No valid events\n\n")
        
        if jet_results:
            f.write("CELL-JET MATCHING RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"  Events processed: {jet_results['num_events']}\n")
            f.write(f"  Correlation: {jet_results['correlation']:.4f}\n")
            f.write(f"  RMSE: {jet_results['rmse']:.4f} ns\n")
            f.write(f"  MAE: {jet_results['mae']:.4f} ns\n")
            f.write(f"  Mean error: {jet_results['mean_error']:.4f} ns\n")
            f.write(f"  Error std: {jet_results['std_error']:.4f} ns\n")
            f.write(f"  Mean t0: {jet_results['mean_t0']:.4f} ns\n")
            f.write(f"  t0 std: {jet_results['std_t0']:.4f} ns\n\n")
        else:
            f.write("CELL-JET MATCHING RESULTS: No valid events\n\n")
        
        if track_results and jet_results:
            f.write("COMPARISON:\n")
            f.write("-" * 30 + "\n")
            corr_diff = jet_results['correlation'] - track_results['correlation']
            rmse_diff = jet_results['rmse'] - track_results['rmse']
            mae_diff = jet_results['mae'] - track_results['mae']
            
            f.write(f"  Correlation difference (jet - track): {corr_diff:+.4f}\n")
            f.write(f"  RMSE difference (jet - track): {rmse_diff:+.4f} ns\n")
            f.write(f"  MAE difference (jet - track): {mae_diff:+.4f} ns\n")
            
            if corr_diff > 0:
                f.write("  → Jet matching shows better correlation\n")
            else:
                f.write("  → Track matching shows better correlation\n")
                
            if rmse_diff < 0:
                f.write("  → Jet matching shows better RMSE\n")
            else:
                f.write("  → Track matching shows better RMSE\n")
    
    print(f"Comparison summary saved to: {summary_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Enhanced baseline t0 reconstruction check with parallel track/jet analysis')
    parser.add_argument('--data-dir', type=str, default='/fs/ddn/sdf/group/atlas/d/liangyu/jetML/datasets/h5/selected_h5_with_jets/',
                       help='Directory containing HDF5 data files')
    parser.add_argument('--num-files', type=int, default=5,
                       help='Number of files to process')
    parser.add_argument('--output-dir', type=str, default='baseline_check_output',
                       help='Output directory for plots')
    parser.add_argument('--no-track-matching', action='store_true',
                       help='Disable track matching filter')
    parser.add_argument('--no-valid-cells', action='store_true',
                       help='Disable valid cells filter')
    parser.add_argument('--min-cells', type=int, default=1,
                       help='Minimum number of cells per event')
    parser.add_argument('--skip-track', action='store_true',
                       help='Skip cell-track matching analysis')
    parser.add_argument('--skip-jet', action='store_true',
                       help='Skip cell-jet matching analysis')
    
    args = parser.parse_args()
    
    # Create main output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("ENHANCED BASELINE T0 RECONSTRUCTION CHECK")
    print("="*60)
    print("Performing parallel analysis of cell-track and cell-jet matching")
    
    results = {}
    
    # Run cell-track matching analysis
    if not args.skip_track:
        track_config = SimpleConfig("HStrackmatching_calibration.txt")
        track_config.data_dir = args.data_dir
        track_config.num_files = args.num_files
        track_config.min_cells = args.min_cells
        track_config.use_cell_track_matching = not args.no_track_matching
        track_config.require_valid_cells = not args.no_valid_cells
        
        track_output_dir = output_dir / "cell_track_results"
        results['track'] = run_analysis(track_config, track_output_dir, "track")
    else:
        results['track'] = {}
        print("\nSkipping cell-track matching analysis")
    
    # Run cell-jet matching analysis
    if not args.skip_jet:
        jet_config = SimpleConfig("cell_jet_calibration.txt")
        jet_config.data_dir = args.data_dir
        jet_config.num_files = args.num_files
        jet_config.min_cells = args.min_cells
        jet_config.require_valid_cells = not args.no_valid_cells
        
        jet_output_dir = output_dir / "cell_jet_results"
        results['jet'] = run_analysis(jet_config, jet_output_dir, "jet")
    else:
        results['jet'] = {}
        print("\nSkipping cell-jet matching analysis")
    
    # Create comparison summary
    if results['track'] and results['jet']:
        create_comparison_summary(results['track'], results['jet'], output_dir)
    
    print("\n" + "="*60)
    print("ENHANCED BASELINE CHECK COMPLETED!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    
    if results['track']:
        print(f"Cell-track results in: {output_dir / 'cell_track_results'}")
    if results['jet']:
        print(f"Cell-jet results in: {output_dir / 'cell_jet_results'}")
    
    if results['track'] and results['jet']:
        print(f"Comparison summary: {output_dir / 'comparison_summary.txt'}")


if __name__ == "__main__":
    main()
