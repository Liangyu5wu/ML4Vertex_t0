#!/usr/bin/env python3
"""
Standalone script to reproduce non-ML vertex t0 reconstruction method.
This script replicates the baseline_check functionality for validation.
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
    
    def __init__(self):
        # Data parameters
        self.data_dir = "../selected_h5/"
        self.num_files = 5
        self.min_cells = 1
        
        # Cell filtering parameters
        self.use_cell_track_matching = True
        self.require_valid_cells = True
        
        # Energy bins for calibration: [1-1.5, 1.5-2, 2-3, 3-4, 4-5, 5-10, >10]
        self.energy_bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, float('inf')]
        
        # Gaussian fit range for plots
        self.gaussian_fit_range = 120
        
        # Calibration data (from HStrackmatching_calibration.txt)
        self.calibration_data = {
            # Barrel Layer 1 (EMB1)
            'EMB1_params': [48.5266, 37.56, 28.9393, 23.1505, 18.5468, 13.0141, 8.03724],
            'EMB1_sigma': [416.994, 293.206, 208.321, 148.768, 117.756, 106.804, 57.6545],
            
            # Barrel Layer 2 (EMB2)
            'EMB2_params': [46.2244, 41.5079, 38.5544, 36.9812, 31.2718, 29.7469, 19.331],
            'EMB2_sigma': [2001.56, 1423.38, 1010.24, 720.392, 551.854, 357.594, 144.162],
            
            # Barrel Layer 3 (EMB3)
            'EMB3_params': [104.325, 106.119, 71.1017, 75.151, 51.2334, 48.2088, 46.6502],
            'EMB3_sigma': [1215.53, 880.826, 680.742, 468.689, 372.184, 279.134, 162.288],
            
            # Endcap Layer 1 (EME1)
            'EME1_params': [125.348, 102.888, 86.7558, 59.7355, 55.3299, 41.3032, 23.646],
            'EME1_sigma': [855.662, 589.529, 435.052, 314.788, 252.453, 185.536, 76.5333],
            
            # Endcap Layer 2 (EME2)
            'EME2_params': [272.149, 224.475, 173.443, 135.829, 113.05, 83.8009, 37.1829],
            'EME2_sigma': [1708.6, 1243.34, 881.465, 627.823, 486.99, 311.032, 106.533],
            
            # Endcap Layer 3 (EME3)
            'EME3_params': [189.356, 140.293, 111.232, 86.8784, 69.0834, 60.5034, 38.5008],
            'EME3_sigma': [1137.06, 803.044, 602.152, 403.393, 318.327, 210.827, 99.697]
        }


def load_and_filter_data(config: SimpleConfig) -> Tuple[List, np.ndarray]:
    """
    Load and filter data from HDF5 files.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (filtered_cell_sequences, vertex_times)
    """
    print(f"Loading data from {config.data_dir}")
    print(f"Number of files: {config.num_files}")
    print(f"Cell filtering - Track matching: {config.use_cell_track_matching}, Valid cells: {config.require_valid_cells}")
    
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
                
                # Apply filtering
                filtered_cells = apply_cell_filtering(event_cells, config)
                
                if len(filtered_cells) < config.min_cells:
                    continue
                
                total_cells_after += len(filtered_cells)
                valid_events += 1
                
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
    
    print(f"\nData loading summary:")
    print(f"  Total events: {total_events}")
    print(f"  Valid events: {valid_events}")
    print(f"  Total cells before filtering: {total_cells_before}")
    print(f"  Total cells after filtering: {total_cells_after}")
    if total_cells_before > 0:
        retention_rate = (total_cells_after / total_cells_before) * 100
        print(f"  Cell retention rate: {retention_rate:.1f}%")
    
    return all_cell_sequences, np.array(all_vertex_times)


def apply_cell_filtering(event_cells, config: SimpleConfig):
    """Apply cell filtering based on configuration."""
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
        cell_sequences: Original cell sequences
        config: Configuration with calibration data
        
    Returns:
        Cell sequences with calibrated time
    """
    print("Applying time calibration...")
    
    # Parameter lookup
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
            layer = int(cell[3])   # Cell_layer
            
            # Get calibration parameters
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


def calculate_traditional_t0(cell_sequences: List, vertex_times: np.ndarray, config: SimpleConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate traditional (non-ML) t0 for each event using weighted average.
    
    Args:
        cell_sequences: Calibrated cell sequences
        vertex_times: True vertex times
        config: Configuration with calibration data
        
    Returns:
        Tuple of (traditional_t0, t0_errors)
    """
    print("Calculating traditional t0...")
    
    # Sigma lookup tables
    sigma_lookup = {
        (1, 1): config.calibration_data['EMB1_sigma'],  # Barrel, Layer 1
        (1, 2): config.calibration_data['EMB2_sigma'],  # Barrel, Layer 2
        (1, 3): config.calibration_data['EMB3_sigma'],  # Barrel, Layer 3
        (0, 1): config.calibration_data['EME1_sigma'],  # Endcap, Layer 1
        (0, 2): config.calibration_data['EME2_sigma'],  # Endcap, Layer 2
        (0, 3): config.calibration_data['EME3_sigma'],  # Endcap, Layer 3
    }
    
    traditional_t0 = []
    
    for sequence in cell_sequences:
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for cell in sequence:
            time = cell[0]         # Calibrated time
            energy = cell[1]       # Cell_e
            barrel = int(cell[2])  # Cell_Barrel
            layer = int(cell[3])   # Cell_layer
            
            # Get sigma for this cell
            sigma_params = sigma_lookup.get((barrel, layer), [1000.0] * 7)
            energy_bin_idx = get_energy_bin_index(energy, config.energy_bins)
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
    
    print(f"Traditional t0 calculation completed for {len(traditional_t0)} events")
    
    return traditional_t0, t0_errors


def gaussian_func(x, a, mu, sigma):
    """Gaussian function for fitting."""
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def plot_t0_distribution(traditional_t0: np.ndarray, config: SimpleConfig, save_path: str):
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
    plt.title('Traditional t0 Distribution')
    plt.legend([f'All data: μ={mean_all:.2f}, σ={std_all:.2f}, N={len(traditional_t0)}',
               f'Fit range ±{fit_range}: μ={fit_mean:.2f}, σ={fit_std:.2f}'])
    plt.grid(True, alpha=0.3)
    plt.xlim(-2000, 2000)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Traditional t0 distribution plot saved to: {save_path}")


def plot_error_distribution(t0_errors: np.ndarray, config: SimpleConfig, save_path: str):
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
    plt.title('Traditional t0 Error Distribution')
    plt.legend([f'All data: μ={mean_all:.2f}, σ={std_all:.2f}, N={len(t0_errors)}',
               f'Fit range ±{fit_range}: μ={fit_mean:.2f}, σ={fit_std:.2f}'])
    plt.grid(True, alpha=0.3)
    plt.xlim(-2000, 2000)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Traditional t0 error distribution plot saved to: {save_path}")


def plot_true_vertex_time_distribution(vertex_times: np.ndarray, save_path: str):
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
    plt.title('True Vertex Time Distribution')
    plt.legend([f'Data: μ={mean_val:.2f}, σ={std_val:.2f}, N={len(vertex_times)}'])
    plt.grid(True, alpha=0.3)
    plt.xlim(-2000, 2000)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"True vertex time distribution plot saved to: {save_path}")


def plot_2d_histogram(traditional_t0: np.ndarray, vertex_times: np.ndarray, save_path: str):
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
    
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Traditional t0 vs true 2D histogram saved to: {save_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Baseline t0 reconstruction check')
    parser.add_argument('--data-dir', type=str, default='../selected_h5/',
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
    
    args = parser.parse_args()
    
    # Create configuration
    config = SimpleConfig()
    config.data_dir = args.data_dir
    config.num_files = args.num_files
    config.min_cells = args.min_cells
    config.use_cell_track_matching = not args.no_track_matching
    config.require_valid_cells = not args.no_valid_cells
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("BASELINE T0 RECONSTRUCTION CHECK")
    print("="*60)
    
    # Load and filter data
    cell_sequences, vertex_times = load_and_filter_data(config)
    
    if len(cell_sequences) == 0:
        print("No valid events found. Exiting.")
        return
    
    # Apply time calibration
    calibrated_sequences = apply_time_calibration(cell_sequences, config)
    
    # Calculate traditional t0
    traditional_t0, t0_errors = calculate_traditional_t0(calibrated_sequences, vertex_times, config)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot 0: True vertex time distribution
    plot_true_vertex_time_distribution(vertex_times, output_dir / 'true_vertex_time_distribution.png')
    
    # Plot 1: Traditional t0 distribution
    plot_t0_distribution(traditional_t0, config, output_dir / 'traditional_t0_distribution.png')
    
    # Plot 2: t0 error distribution
    plot_error_distribution(t0_errors, config, output_dir / 't0_error_distribution.png')
    
    # Plot 3: Traditional t0 vs true t0 2D histogram
    plot_2d_histogram(traditional_t0, vertex_times, output_dir / 'traditional_t0_vs_true_2d.png')
    
    print("\n" + "="*60)
    print("BASELINE CHECK COMPLETED!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Processed {len(traditional_t0)} events from {config.num_files} files")


if __name__ == "__main__":
    main()
