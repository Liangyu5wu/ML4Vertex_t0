"""Data loading utilities for HDF5 files with configurable cell filtering."""

import os
import h5py
import numpy as np
from typing import List, Tuple, Optional
from config.base_config import BaseConfig


class DataLoader:
    """Load and preprocess data from HDF5 files with configurable cell filtering."""
    
    def __init__(self, config: BaseConfig):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object containing data parameters
        """
        self.config = config
        
    def get_file_paths(self) -> List[str]:
        """Get list of HDF5 file paths."""
        return [
            os.path.join(self.config.data_dir, f"output_{i:03d}.h5") 
            for i in range(self.config.num_files)
        ]
    
    def apply_time_quality_cut(self, event_cells: np.ndarray) -> np.ndarray:
        """Apply time quality cut based on configuration."""
        if not self.config.use_time_quality_cut or len(event_cells) == 0:
            return event_cells
        
        return self._apply_time_quality_cut_unified(event_cells, apply_calibration=self.config.use_detector_params)
    
    def _apply_time_quality_cut_unified(self, event_cells: np.ndarray, apply_calibration: bool = True) -> np.ndarray:
        """Unified time quality cut implementation."""
        try:
            calibration_data = self.config.load_calibration_data()
            use_full_uncertainty = True
        except Exception:
            # Fallback to vertex uncertainty only if calibration data unavailable
            print("Warning: Cannot load calibration data. Using vertex uncertainty only.")
            use_full_uncertainty = False
        
        if not use_full_uncertainty:
            # Simple fallback: vertex uncertainty only
            mask = np.ones(len(event_cells), dtype=bool)
            cut_threshold = self.config.time_quality_n_sigma * self.config.vertex_time_sigma
            for i, cell in enumerate(event_cells):
                try:
                    cell_time = cell['Cell_time_TOF_corrected']
                    if abs(cell_time) > cut_threshold:
                        mask[i] = False
                except (KeyError, ValueError):
                    mask[i] = False
            return event_cells[mask]
        
        # Full uncertainty calculation
        sigma_lookup = {
            (1, 1): calibration_data['EMB1_sigma'], (1, 2): calibration_data['EMB2_sigma'], (1, 3): calibration_data['EMB3_sigma'],
            (0, 1): calibration_data['EME1_sigma'], (0, 2): calibration_data['EME2_sigma'], (0, 3): calibration_data['EME3_sigma'],
        }
        
        if apply_calibration:
            param_lookup = {
                (1, 1): calibration_data['EMB1_params'], (1, 2): calibration_data['EMB2_params'], (1, 3): calibration_data['EMB3_params'],
                (0, 1): calibration_data['EME1_params'], (0, 2): calibration_data['EME2_params'], (0, 3): calibration_data['EME3_params'],
            }
        
        energy_bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, float('inf')]
        
        def get_energy_bin_index(energy: float) -> int:
            if energy < 1.0: return 0
            for i in range(len(energy_bins) - 1):
                if energy_bins[i] <= energy < energy_bins[i + 1]: return i
            return len(energy_bins) - 2
        
        mask = np.ones(len(event_cells), dtype=bool)
        
        for i, cell in enumerate(event_cells):
            try:
                barrel, layer, energy = int(cell['Cell_Barrel']), int(cell['Cell_layer']), cell['Cell_e']
                cell_time = cell['Cell_time_TOF_corrected']
                
                if layer not in [1, 2, 3]:
                    mask[i] = False
                    continue
                
                energy_bin_idx = max(0, min(get_energy_bin_index(energy), len(sigma_lookup.get((barrel, layer), [1000.0] * 7)) - 1))
                sigma_cell = sigma_lookup.get((barrel, layer), [1000.0] * 7)[energy_bin_idx]
                
                # Apply detector calibration if enabled
                if apply_calibration:
                    calibration_value = param_lookup.get((barrel, layer), [0.0] * 7)[energy_bin_idx]
                    cell_time = cell_time - calibration_value
                
                # Apply cut with full uncertainty
                sigma_total = np.sqrt(self.config.vertex_time_sigma**2 + sigma_cell**2)
                cut_threshold = self.config.time_quality_n_sigma * sigma_total
                
                if abs(cell_time) > cut_threshold:
                    mask[i] = False
                    
            except (KeyError, ValueError, IndexError):
                mask[i] = False
        
        return event_cells[mask]
    
    def apply_cell_filtering(self, event_cells: np.ndarray) -> np.ndarray:
        """
        Apply configurable cell filtering based on configuration.
        
        Args:
            event_cells: Array of cells for a single event
            
        Returns:
            Filtered array of cells
        """
        # Start with all cells
        mask = np.ones(len(event_cells), dtype=bool)
        
        # Apply valid cell filter
        if self.config.require_valid_cells:
            valid_mask = event_cells['valid'] == True
            mask = mask & valid_mask
        
        # Apply cell-track matching filter
        if self.config.use_cell_track_matching:
            track_matching_mask = event_cells['matched_track_HS'] == 1
            mask = mask & track_matching_mask
        
        # NEW: Apply cell-jet matching filter
        if self.config.use_cell_jet_matching:
            # Check if jet matching field exists in data
            if 'cell_jet_matched' in event_cells.dtype.names:
                jet_matching_mask = event_cells['cell_jet_matched'] == True
                mask = mask & jet_matching_mask
            else:
                print("Warning: cell_jet_matched field not found in data. Jet matching filter skipped.")
        
        # Apply layer filtering - only keep cells with layers 1, 2, 3
        # This ensures consistency with baseline t0 calculation requirements
        if 'Cell_layer' in event_cells.dtype.names:
            layer_mask = np.isin(event_cells['Cell_layer'], [1, 2, 3])
            mask = mask & layer_mask
        else:
            print("Warning: Cell_layer not found in cell data. Layer filtering skipped.")
        
        # Apply additional custom filters
        if self.config.additional_cell_filters:
            for filter_key, filter_value in self.config.additional_cell_filters.items():
                if filter_key in event_cells.dtype.names:
                    # Support >= filtering for energy thresholds
                    if filter_key == 'Cell_e' and isinstance(filter_value, (int, float)):
                        additional_mask = event_cells[filter_key] >= filter_value
                    else:
                        additional_mask = event_cells[filter_key] == filter_value
                    mask = mask & additional_mask
                else:
                    print(f"Warning: Filter key '{filter_key}' not found in cell data. Skipping this filter.")
        
        # Apply time quality cut before final filtering
        filtered_cells = event_cells[mask]
        if self.config.use_time_quality_cut:
            filtered_cells = self.apply_time_quality_cut(filtered_cells)
        
        return filtered_cells
    
    def calculate_baseline_t0_error(self, event_cells: np.ndarray, true_vertex_time: float) -> float:
        """
        Calculate baseline (non-ML) t0 error for event-level filtering.
        
        Args:
            event_cells: Array of cells for a single event (after basic filtering)
            true_vertex_time: True vertex time for this event
            
        Returns:
            Absolute error in ps between baseline t0 and true vertex time
        """
        if len(event_cells) == 0:
            return float('inf')  # Invalid event
        
        try:
            # Load calibration data for sigma and parameter values
            calibration_data = self.config.load_calibration_data()
        except Exception:
            return 0.0  # Skip filtering if calibration data unavailable
        
        # Energy bins for calibration: [1-1.5, 1.5-2, 2-3, 3-4, 4-5, 5-10, >10]
        energy_bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, float('inf')]
        
        def get_energy_bin_index(energy: float) -> int:
            """Get energy bin index for calibration parameter lookup."""
            if energy < 1.0:
                return 0
            for i in range(len(energy_bins) - 1):
                if energy_bins[i] <= energy < energy_bins[i + 1]:
                    return i
            return len(energy_bins) - 2
        
        # Parameter and sigma lookup tables
        param_lookup = {
            (1, 1): calibration_data['EMB1_params'],  # Barrel, Layer 1
            (1, 2): calibration_data['EMB2_params'],  # Barrel, Layer 2
            (1, 3): calibration_data['EMB3_params'],  # Barrel, Layer 3
            (0, 1): calibration_data['EME1_params'],  # Endcap, Layer 1
            (0, 2): calibration_data['EME2_params'],  # Endcap, Layer 2
            (0, 3): calibration_data['EME3_params'],  # Endcap, Layer 3
        }
        
        sigma_lookup = {
            (1, 1): calibration_data['EMB1_sigma'],  # Barrel, Layer 1
            (1, 2): calibration_data['EMB2_sigma'],  # Barrel, Layer 2
            (1, 3): calibration_data['EMB3_sigma'],  # Barrel, Layer 3
            (0, 1): calibration_data['EME1_sigma'],  # Endcap, Layer 1
            (0, 2): calibration_data['EME2_sigma'],  # Endcap, Layer 2
            (0, 3): calibration_data['EME3_sigma'],  # Endcap, Layer 3
        }
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for cell in event_cells:
            try:
                # Extract cell properties
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
                
                energy_bin_idx = get_energy_bin_index(energy)
                
                # Add bounds checking
                if energy_bin_idx >= len(detector_params):
                    energy_bin_idx = len(detector_params) - 1
                elif energy_bin_idx < 0:
                    energy_bin_idx = 0
                
                calibration_value = detector_params[energy_bin_idx]
                sigma = sigma_params[energy_bin_idx]
                
                # Apply calibration: corrected_time = tof_corrected_time - calibration_value
                calibrated_time = time_tof - calibration_value
                
                # Weight = 1/sigma^2
                weight = 1.0 / (sigma * sigma)
                
                weighted_sum += weight * calibrated_time
                weight_sum += weight
                
            except (KeyError, ValueError, IndexError):
                # Skip cells with missing or invalid data
                continue
        
        if weight_sum > 0:
            baseline_t0 = weighted_sum / weight_sum
            # Calculate absolute error (all data already in ps)
            error_ps = abs(baseline_t0 - true_vertex_time)
            return error_ps
        else:
            return float('inf')  # Invalid calculation
    
    def calculate_baseline_t0_prediction(self, event_cells: np.ndarray) -> float:
        """
        Calculate baseline (non-ML) t0 prediction for residual learning.
        
        Args:
            event_cells: Array of cells for a single event (after basic filtering)
            
        Returns:
            Baseline t0 prediction in ps (or 0.0 if calculation fails)
        """
        if len(event_cells) == 0:
            return 0.0  # Return 0 for empty events
        
        try:
            # Load calibration data for sigma and parameter values
            calibration_data = self.config.load_calibration_data()
        except Exception:
            return 0.0  # Return 0 if calibration data unavailable
        
        # Energy bins for calibration: [1-1.5, 1.5-2, 2-3, 3-4, 4-5, 5-10, >10]
        energy_bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, float('inf')]
        
        def get_energy_bin_index(energy: float) -> int:
            """Get energy bin index for calibration parameter lookup."""
            if energy < 1.0:
                return 0
            for i in range(len(energy_bins) - 1):
                if energy_bins[i] <= energy < energy_bins[i + 1]:
                    return i
            return len(energy_bins) - 2
        
        # Parameter and sigma lookup tables
        param_lookup = {
            (1, 1): calibration_data['EMB1_params'],  # Barrel, Layer 1
            (1, 2): calibration_data['EMB2_params'],  # Barrel, Layer 2
            (1, 3): calibration_data['EMB3_params'],  # Barrel, Layer 3
            (0, 1): calibration_data['EME1_params'],  # Endcap, Layer 1
            (0, 2): calibration_data['EME2_params'],  # Endcap, Layer 2
            (0, 3): calibration_data['EME3_params'],  # Endcap, Layer 3
        }
        
        sigma_lookup = {
            (1, 1): calibration_data['EMB1_sigma'],  # Barrel, Layer 1
            (1, 2): calibration_data['EMB2_sigma'],  # Barrel, Layer 2
            (1, 3): calibration_data['EMB3_sigma'],  # Barrel, Layer 3
            (0, 1): calibration_data['EME1_sigma'],  # Endcap, Layer 1
            (0, 2): calibration_data['EME2_sigma'],  # Endcap, Layer 2
            (0, 3): calibration_data['EME3_sigma'],  # Endcap, Layer 3
        }
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for cell in event_cells:
            try:
                # Extract cell properties
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
                
                energy_bin_idx = get_energy_bin_index(energy)
                
                # Add bounds checking
                if energy_bin_idx >= len(detector_params):
                    energy_bin_idx = len(detector_params) - 1
                elif energy_bin_idx < 0:
                    energy_bin_idx = 0
                
                calibration_value = detector_params[energy_bin_idx]
                sigma = sigma_params[energy_bin_idx]
                
                # Apply calibration: corrected_time = tof_corrected_time - calibration_value
                calibrated_time = time_tof - calibration_value
                
                # Weight = 1/sigma^2
                weight = 1.0 / (sigma * sigma)
                
                weighted_sum += weight * calibrated_time
                weight_sum += weight
                
            except (KeyError, ValueError, IndexError):
                # Skip cells with missing or invalid data
                continue
        
        if weight_sum > 0:
            baseline_t0 = weighted_sum / weight_sum
            return baseline_t0
        else:
            return 0.0  # Return 0 for invalid calculation
    
    def apply_baseline_method_filter(
        self, 
        event_cells: np.ndarray, 
        true_vertex_time: float
    ) -> bool:
        """
        Apply baseline method performance filter.
        
        Args:
            event_cells: Array of cells for a single event (after basic filtering)
            true_vertex_time: True vertex time for this event
            
        Returns:
            True if event passes baseline method filter, False otherwise
        """
        if not self.config.use_baseline_method_filter:
            return True
        
        error_ps = self.calculate_baseline_t0_error(event_cells, true_vertex_time)
        return error_ps <= self.config.baseline_method_threshold
    
    def get_filtering_statistics(self, event_cells: np.ndarray) -> dict:
        """
        Get statistics about cell filtering for debugging/monitoring.
        
        Args:
            event_cells: Original array of cells
            
        Returns:
            Dictionary with filtering statistics
        """
        stats = {
            'total_cells': len(event_cells),
            'valid_cells': 0,
            'track_matched_cells': 0,
            'jet_matched_cells': 0,  # NEW
            'time_quality_passed_cells': 0,  # NEW
            'final_filtered_cells': 0
        }
        
        if len(event_cells) == 0:
            return stats
        
        # Count valid cells
        if 'valid' in event_cells.dtype.names:
            stats['valid_cells'] = np.sum(event_cells['valid'] == True)
        
        # Count track-matched cells
        if 'matched_track_HS' in event_cells.dtype.names:
            stats['track_matched_cells'] = np.sum(event_cells['matched_track_HS'] == 1)
        
        # NEW: Count jet-matched cells
        if 'cell_jet_matched' in event_cells.dtype.names:
            stats['jet_matched_cells'] = np.sum(event_cells['cell_jet_matched'] == True)
        
        # NEW: Count cells passing time quality cut (if enabled)
        if self.config.use_time_quality_cut:
            temp_filtered = self.apply_cell_filtering(event_cells)
            # Remove time quality cut temporarily to count cells before this filter
            original_time_cut = self.config.use_time_quality_cut
            self.config.use_time_quality_cut = False
            cells_before_time_cut = self.apply_cell_filtering(event_cells)
            self.config.use_time_quality_cut = original_time_cut
            
            time_cut_filtered = self.apply_time_quality_cut(cells_before_time_cut)
            stats['time_quality_passed_cells'] = len(time_cut_filtered)
        
        # Count cells after all filtering
        filtered_cells = self.apply_cell_filtering(event_cells)
        stats['final_filtered_cells'] = len(filtered_cells)
        
        return stats
    
    def load_data_from_files(
        self, 
        file_paths: Optional[List[str]] = None,
        print_filtering_stats: bool = True
    ) -> Tuple[List[List[List[float]]], np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from HDF5 files with configurable cell filtering.
        
        For backward compatibility, this method loads only cell data.
        Use load_data_with_jets_and_tracks_from_files for multi-input models.
        """
        result = self.load_data_with_jets_and_tracks_from_files(file_paths, print_filtering_stats)
        # Return only the first 4 elements for backward compatibility
        return result[:4]
    
    def load_data_with_jets_and_tracks_from_files(
        self, 
        file_paths: Optional[List[str]] = None,
        print_filtering_stats: bool = True
    ) -> Tuple[List[List[List[float]]], np.ndarray, np.ndarray, np.ndarray, 
               Optional[List[List[List[float]]]], Optional[List[List[List[float]]]]]:
        """
        Load data from HDF5 files with jets and tracks support.
        
        Args:
            file_paths: List of file paths to load. If None, uses default paths.
            print_filtering_stats: Whether to print cell filtering statistics
            
        Returns:
            Tuple of (cell_sequences, vertex_features, vertex_times, sequence_lengths, jet_sequences, track_sequences)
        """
        if file_paths is None:
            file_paths = self.get_file_paths()
            
        all_cell_sequences = []
        all_vertex_features = []
        all_vertex_times = []
        all_jet_sequences = [] if hasattr(self.config, 'use_event_jets') and self.config.use_event_jets else None
        all_track_sequences = [] if hasattr(self.config, 'use_event_tracks') and self.config.use_event_tracks else None
        sequence_lengths = []
        
        # Print configuration
        print(f"Data loading configuration:")
        print(f"  Cell features used: {self.config.cell_features}")
        print(f"  Use event jets: {getattr(self.config, 'use_event_jets', False)}")
        print(f"  Use event tracks: {getattr(self.config, 'use_event_tracks', False)}")
        if hasattr(self.config, 'use_event_jets') and self.config.use_event_jets:
            print(f"  Max jets per event: {self.config.max_jets}, Min jets: {self.config.min_jets}")
        if hasattr(self.config, 'use_event_tracks') and self.config.use_event_tracks:
            print(f"  Max tracks per event: {self.config.max_tracks}, Min tracks: {self.config.min_tracks}")
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping...")
                continue
                
            print(f"Processing {file_path}...")
            try:
                result = self._process_file_with_jets_tracks(file_path)
                if all_jet_sequences is not None and all_track_sequences is not None:
                    cell_seq, vertex_feat, vertex_time, seq_len, jet_seq, track_seq = result
                    all_jet_sequences.extend(jet_seq)
                    all_track_sequences.extend(track_seq)
                else:
                    cell_seq, vertex_feat, vertex_time, seq_len, _, _ = result
                    
                all_cell_sequences.extend(cell_seq)
                all_vertex_features.extend(vertex_feat)
                all_vertex_times.extend(vertex_time)
                sequence_lengths.extend(seq_len)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        sequence_lengths = np.array(sequence_lengths)
        print(f"Processed {len(all_vertex_times)} valid events")
        
        return (all_cell_sequences, np.array(all_vertex_features), 
                np.array(all_vertex_times), sequence_lengths,
                all_jet_sequences, all_track_sequences)
    
    def load_data_with_baselines_from_files(
        self, 
        file_paths: Optional[List[str]] = None,
        print_filtering_stats: bool = True
    ) -> Tuple[List[List[List[float]]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from HDF5 files with baseline predictions for residual learning.
        
        Args:
            file_paths: List of file paths to load. If None, uses default paths.
            print_filtering_stats: Whether to print cell filtering statistics
            
        Returns:
            Tuple of (cell_sequences, vertex_features, vertex_times, sequence_lengths, baseline_predictions)
        """
        # Load regular data first
        cell_sequences, vertex_features, vertex_times, sequence_lengths = self.load_data_from_files(
            file_paths, print_filtering_stats
        )
        
        print("Computing baseline predictions for residual learning...")
        baseline_predictions = []
        
        # Process each event to compute baseline predictions
        for i, cell_seq in enumerate(cell_sequences):
            # Convert cell sequence back to structured array format for baseline calculation
            # We need to reconstruct the event_cells format from the processed sequences
            if len(cell_seq) == 0:
                baseline_pred = 0.0
            else:
                # Create structured array from cell sequence
                # Note: This requires matching the cell sequence format to the expected structure
                event_cells = self._reconstruct_event_cells_from_sequence(cell_seq)
                baseline_pred = self.calculate_baseline_t0_prediction(event_cells)
            
            baseline_predictions.append(baseline_pred)
            
            if (i + 1) % 10000 == 0:
                print(f"Computed baseline predictions for {i + 1}/{len(cell_sequences)} events")
        
        baseline_predictions = np.array(baseline_predictions)
        print(f"Baseline predictions computed. Mean: {np.mean(baseline_predictions):.2f}, "
              f"Std: {np.std(baseline_predictions):.2f}")
        
        return (cell_sequences, vertex_features, vertex_times, sequence_lengths, baseline_predictions)
    
    def _reconstruct_event_cells_from_sequence(self, cell_seq: List[List[float]]) -> np.ndarray:
        """
        Reconstruct structured event_cells array from processed cell sequence.
        
        Args:
            cell_seq: Processed cell sequence (list of feature vectors)
            
        Returns:
            Structured array compatible with baseline calculation
        """
        if len(cell_seq) == 0:
            # Return empty structured array with correct dtype
            return np.array([], dtype=[
                ('Cell_Barrel', 'i4'),
                ('Cell_layer', 'i4'),
                ('Cell_e', 'f8'),
                ('Cell_time_TOF_corrected', 'f8')
            ])
        
        # Map feature indices based on config.cell_features
        feature_names = self.config.cell_features
        
        # Find indices of required features for baseline calculation
        barrel_idx = feature_names.index('Cell_Barrel') if 'Cell_Barrel' in feature_names else -1
        layer_idx = feature_names.index('Cell_layer') if 'Cell_layer' in feature_names else -1
        energy_idx = feature_names.index('Cell_e') if 'Cell_e' in feature_names else -1
        time_idx = feature_names.index('Cell_time_TOF_corrected') if 'Cell_time_TOF_corrected' in feature_names else -1
        
        # Create structured array
        num_cells = len(cell_seq)
        event_cells = np.zeros(num_cells, dtype=[
            ('Cell_Barrel', 'i4'),
            ('Cell_layer', 'i4'),
            ('Cell_e', 'f8'),
            ('Cell_time_TOF_corrected', 'f8')
        ])
        
        for i, cell_features in enumerate(cell_seq):
            # Extract required features (with defaults if not available)
            event_cells[i]['Cell_Barrel'] = int(cell_features[barrel_idx]) if barrel_idx >= 0 else 1
            event_cells[i]['Cell_layer'] = int(cell_features[layer_idx]) if layer_idx >= 0 else 1
            event_cells[i]['Cell_e'] = float(cell_features[energy_idx]) if energy_idx >= 0 else 1.0
            event_cells[i]['Cell_time_TOF_corrected'] = float(cell_features[time_idx]) if time_idx >= 0 else 0.0
        
        return event_cells
    
    def _process_file(self, file_path: str) -> Tuple[List, List, List, List, dict]:
        """Process a single HDF5 file with detailed statistics."""
        cell_sequences = []
        vertex_features = []
        vertex_times = []
        sequence_lengths = []
        
        # File-level statistics
        file_stats = {
            'total_events': 0,
            'events_with_cells': 0,
            'events_after_min_cells_filter': 0,
            'events_after_baseline_filter': 0,  # NEW
            'total_cells_before_filtering': 0,
            'total_cells_after_filtering': 0
        }
        
        with h5py.File(file_path, 'r') as f:
            vertex_data = f['HSvertex'][:]
            cells_data = f['cells'][:]
            
            file_stats['total_events'] = len(vertex_data)
            
            for i in range(len(vertex_data)):
                # Extract vertex features for global context
                if self.config.use_spatial_features:
                    vertex_reco = [
                        vertex_data[i]['HSvertex_reco_x'],
                        vertex_data[i]['HSvertex_reco_y'],
                        vertex_data[i]['HSvertex_reco_z']
                    ]
                else:
                    vertex_reco = [0.0, 0.0, 0.0]
                
                # Process cells for this event
                event_cells = cells_data[i]
                
                if len(event_cells) > 0:
                    file_stats['events_with_cells'] += 1
                    file_stats['total_cells_before_filtering'] += len(event_cells)
                
                # Apply configurable cell filtering
                valid_cells = self.apply_cell_filtering(event_cells)
                
                if len(valid_cells) > 0:
                    file_stats['total_cells_after_filtering'] += len(valid_cells)
                
                # Skip events with too few cells
                if len(valid_cells) < self.config.min_cells:
                    continue
                
                file_stats['events_after_min_cells_filter'] += 1
                
                # Apply baseline method filter if enabled
                vertex_time = vertex_data[i]['HSvertex_time']
                if not self.apply_baseline_method_filter(valid_cells, vertex_time):
                    continue
                
                file_stats['events_after_baseline_filter'] += 1
                
                # Process cells for this event
                sequence = self._process_event_cells(valid_cells)
                if sequence is None:
                    continue
                
                cell_sequences.append(sequence)
                vertex_features.append(vertex_reco)
                vertex_times.append(vertex_time)
                sequence_lengths.append(len(sequence))
        
        return cell_sequences, vertex_features, vertex_times, sequence_lengths, file_stats
    
    def _process_file_with_jets_tracks(self, file_path: str) -> Tuple[List, List, List, List, List, List]:
        """Process a single HDF5 file with jets and tracks support."""
        cell_sequences = []
        vertex_features = []
        vertex_times = []
        sequence_lengths = []
        jet_sequences = []
        track_sequences = []
        
        with h5py.File(file_path, 'r') as f:
            vertex_data = f['HSvertex'][:]
            cells_data = f['cells'][:]
            
            # Load jets and tracks data if enabled
            jets_data = f.get('jets', None) if hasattr(self.config, 'use_event_jets') and self.config.use_event_jets else None
            tracks_data = f.get('tracks', None) if hasattr(self.config, 'use_event_tracks') and self.config.use_event_tracks else None
            
            for i in range(len(vertex_data)):
                # Extract vertex features (not using spatial features for new models)
                vertex_reco = [0.0, 0.0, 0.0]
                
                # Process cells for this event
                event_cells = cells_data[i]
                valid_cells = self.apply_cell_filtering(event_cells)
                
                # Skip events with too few cells
                if len(valid_cells) < self.config.min_cells:
                    continue
                
                # Apply baseline method filter if enabled
                vertex_time = vertex_data[i]['HSvertex_time']
                if not self.apply_baseline_method_filter(valid_cells, vertex_time):
                    continue
                
                # Process cells
                sequence = self._process_event_cells(valid_cells)
                if sequence is None:
                    continue
                
                # Process jets if enabled
                jet_sequence = []
                if jets_data is not None:
                    event_jets = jets_data[i]
                    jet_sequence = self._process_event_jets(event_jets)
                
                # Process tracks if enabled
                track_sequence = []
                if tracks_data is not None:
                    event_tracks = tracks_data[i]
                    track_sequence = self._process_event_tracks(event_tracks)
                
                cell_sequences.append(sequence)
                vertex_features.append(vertex_reco)
                vertex_times.append(vertex_time)
                sequence_lengths.append(len(sequence))
                jet_sequences.append(jet_sequence)
                track_sequences.append(track_sequence)
        
        return cell_sequences, vertex_features, vertex_times, sequence_lengths, jet_sequences, track_sequences
    
    def _process_event_jets(self, event_jets: np.ndarray) -> List[List[float]]:
        """Process jets for a single event."""
        # Filter jets: AntiKt4EMTopoJets_selected=1 and valid=true
        valid_mask = (event_jets['valid'] == True) & (event_jets['AntiKt4EMTopoJets_selected'] == 1)
        valid_jets = event_jets[valid_mask]
        
        # Sort by pt (descending)
        sorted_indices = np.argsort(-valid_jets['AntiKt4EMTopoJets_pt'])
        sorted_jets = valid_jets[sorted_indices]
        
        # Take up to max_jets
        n_jets_to_use = min(len(sorted_jets), self.config.max_jets)
        
        # Create jet sequence
        jet_sequence = []
        for jet_idx in range(n_jets_to_use):
            jet = sorted_jets[jet_idx]
            jet_features = [
                jet['AntiKt4EMTopoJets_pt'],
                jet['AntiKt4EMTopoJets_eta'],
                jet['AntiKt4EMTopoJets_phi'],
                jet['AntiKt4EMTopoJets_width']
            ]
            jet_sequence.append(jet_features)
        
        # Pad to max_jets if necessary
        while len(jet_sequence) < self.config.max_jets:
            padding_values = [
                self.config.jet_padding_values['pt'],
                self.config.jet_padding_values['eta'],
                self.config.jet_padding_values['phi'],
                self.config.jet_padding_values['width']
            ]
            jet_sequence.append(padding_values)
        
        return jet_sequence
    
    def _process_event_tracks(self, event_tracks: np.ndarray) -> List[List[float]]:
        """Process tracks for a single event."""
        # Filter tracks: valid=true and Track_isGoodFromHS_old_files=1
        valid_mask = (event_tracks['valid'] == True) & (event_tracks['Track_isGoodFromHS_old_files'] == 1)
        valid_tracks = event_tracks[valid_mask]
        
        # Sort by pt (descending) to get top N tracks
        sorted_indices = np.argsort(-valid_tracks['Track_pt'])
        sorted_tracks = valid_tracks[sorted_indices]
        
        # Take up to max_tracks
        n_tracks_to_use = min(len(sorted_tracks), self.config.max_tracks)
        
        # Create track sequence
        track_sequence = []
        for track_idx in range(n_tracks_to_use):
            track = sorted_tracks[track_idx]
            track_features = [
                track['Track_pt'],
                track['Track_eta'],
                track['Track_phi'],
                track['Track_d0'],
                track['Track_z0']
            ]
            track_sequence.append(track_features)
        
        # Pad to max_tracks if necessary
        while len(track_sequence) < self.config.max_tracks:
            padding_values = [
                self.config.track_padding_values['pt'],
                self.config.track_padding_values['eta'],
                self.config.track_padding_values['phi'],
                self.config.track_padding_values['d0'],
                self.config.track_padding_values['z0']
            ]
            track_sequence.append(padding_values)
        
        return track_sequence
    
    def _process_event_cells(self, valid_cells: np.ndarray) -> Optional[List[List[float]]]:
        """Process cells for a single event."""
        # Sort cells by selection feature
        if self.config.cell_selection_feature in ['Cell_e', 'Cell_significance', 'matched_track_pt']:
            sorted_indices = np.argsort(-valid_cells[self.config.cell_selection_feature])
        else:
            sorted_indices = np.argsort(-valid_cells[self.config.cell_selection_feature])
        
        # Take up to max_cells
        n_cells_to_use = min(len(valid_cells), self.config.max_cells)
        sorted_cells = valid_cells[sorted_indices[:n_cells_to_use]]
        
        # Create sequence of cell features
        sequence = []
        for cell_idx in range(n_cells_to_use):
            cell_features_values = []
            for feature in self.config.cell_features:
                if feature in sorted_cells.dtype.names:
                    cell_features_values.append(sorted_cells[feature][cell_idx])
                else:
                    # NEW: Better error handling for missing features
                    if self.config.use_jet_features and feature in self.config.jet_features:
                        print(f"Warning: Jet feature '{feature}' not found in cell data. Using 0.0 as default.")
                        print(f"Available fields: {list(sorted_cells.dtype.names)}")
                    else:
                        print(f"Warning: Feature '{feature}' not found in cell data. Using 0.0 as default.")
                    cell_features_values.append(0.0)
            sequence.append(cell_features_values)
        
        return sequence
    
    def _print_filtering_statistics(self, stats: dict):
        """Print detailed cell filtering statistics."""
        print("\n" + "="*60)
        print("CELL FILTERING STATISTICS")
        print("="*60)
        
        print(f"Events:")
        print(f"  Total events processed: {stats['total_events']}")
        print(f"  Events with cells: {stats['events_with_cells']}")
        print(f"  Events after min_cells filter: {stats['events_after_min_cells_filter']}")
        
        # NEW: Show baseline filter statistics
        if self.config.use_baseline_method_filter:
            print(f"  Events after baseline filter: {stats['events_after_baseline_filter']}")
            final_events = stats['events_after_baseline_filter']
        else:
            final_events = stats['events_after_min_cells_filter']
        
        if stats['events_with_cells'] > 0:
            event_retention_rate = (final_events / stats['events_with_cells']) * 100
            print(f"  Final event retention rate: {event_retention_rate:.1f}%")
            
            # Show baseline filter impact if enabled
            if self.config.use_baseline_method_filter and stats['events_after_min_cells_filter'] > 0:
                baseline_retention = (stats['events_after_baseline_filter'] / stats['events_after_min_cells_filter']) * 100
                baseline_removed = stats['events_after_min_cells_filter'] - stats['events_after_baseline_filter']
                print(f"  Events removed by baseline filter: {baseline_removed}")
                print(f"  Baseline filter retention rate: {baseline_retention:.1f}%")
        
        print(f"\nCells:")
        print(f"  Total cells before filtering: {stats['total_cells_before_filtering']}")
        print(f"  Total cells after filtering: {stats['total_cells_after_filtering']}")
        
        if stats['total_cells_before_filtering'] > 0:
            cell_retention_rate = (stats['total_cells_after_filtering'] / stats['total_cells_before_filtering']) * 100
            cells_removed = stats['total_cells_before_filtering'] - stats['total_cells_after_filtering']
            print(f"  Cells removed: {cells_removed}")
            print(f"  Cell retention rate: {cell_retention_rate:.1f}%")
            
            if final_events > 0:
                avg_cells_per_event = stats['total_cells_after_filtering'] / final_events
                print(f"  Average cells per event (after filtering): {avg_cells_per_event:.1f}")
        
        print(f"\nFiltering Configuration:")
        print(f"  {self.config.get_cell_filtering_description()}")
        
        # NEW: Add time quality cut information
        if self.config.use_time_quality_cut:
            print(f"\nTime Quality Cut:")
            print(f"  σ_vertex: {self.config.vertex_time_sigma} ps")
            print(f"  Cut threshold: {self.config.time_quality_n_sigma}σ_total")
        
        # NEW: Add baseline method filter information
        if self.config.use_baseline_method_filter:
            print(f"\nBaseline Method Filter:")
            print(f"  Error threshold: ±{self.config.baseline_method_threshold} ps")
            print(f"  Only events with baseline method error ≤ {self.config.baseline_method_threshold} ps are included")
        
        # NEW: Add jet feature information
        if self.config.use_jet_features:
            print(f"\nJet Features:")
            print(f"  Enabled jet features: {self.config.jet_features}")
        
        print("="*60)
    
    def _print_sequence_statistics(self, sequence_lengths: np.ndarray):
        """Print statistics about sequence lengths."""
        if len(sequence_lengths) == 0:
            print("No valid sequences found!")
            return
            
        print(f"\nSequence length statistics:")
        print(f"  Mean: {np.mean(sequence_lengths):.2f}")
        print(f"  Std: {np.std(sequence_lengths):.2f}")
        print(f"  Min: {np.min(sequence_lengths)}")
        print(f"  Max: {np.max(sequence_lengths)}")
        print(f"  Median: {np.median(sequence_lengths):.2f}")
        
        # Show distribution of sequence lengths
        unique_lengths, counts = np.unique(sequence_lengths, return_counts=True)
        print(f"  Most common lengths:")
        for length, count in sorted(zip(unique_lengths, counts), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / len(sequence_lengths)) * 100
            print(f"    Length {length}: {count} events ({percentage:.1f}%)")
    
    def check_jet_features_availability(self, file_path: str) -> dict:
        """
        Check availability of jet features in a specific file.
        
        Args:
            file_path: Path to HDF5 file to check
            
        Returns:
            Dictionary with jet feature availability information
        """
        availability = {
            'file_path': file_path,
            'file_exists': False,
            'has_cells_data': False,
            'available_jet_fields': [],
            'missing_jet_fields': [],
            'jet_features_ready': False
        }
        
        if not os.path.exists(file_path):
            return availability
        
        availability['file_exists'] = True
        
        try:
            with h5py.File(file_path, 'r') as f:
                if 'cells' in f:
                    availability['has_cells_data'] = True
                    cells_data = f['cells']
                    
                    # Check first event's cell structure
                    if len(cells_data) > 0:
                        first_event_cells = cells_data[0]
                        if len(first_event_cells) > 0:
                            available_fields = list(first_event_cells.dtype.names)
                            
                            # Check for jet-related fields
                            expected_jet_fields = [
                                'cell_jet_matched', 'matched_jet_pt', 'matched_jet_eta',
                                'matched_jet_phi', 'matched_jet_width', 'matched_jet_deltaR'
                            ]
                            
                            for field in expected_jet_fields:
                                if field in available_fields:
                                    availability['available_jet_fields'].append(field)
                                else:
                                    availability['missing_jet_fields'].append(field)
                            
                            # Check if all required jet features are available
                            if self.config.use_jet_features:
                                required_jet_features = self.config.jet_features + ['cell_jet_matched']
                                availability['jet_features_ready'] = all(
                                    field in available_fields for field in required_jet_features
                                )
                            else:
                                availability['jet_features_ready'] = True
                                
        except Exception as e:
            print(f"Error checking jet features in {file_path}: {e}")
        
        return availability
    
    def validate_jet_features_in_dataset(self) -> bool:
        """
        Validate that jet features are available in the dataset.
        
        Returns:
            True if jet features are available or not needed, False otherwise
        """
        if not self.config.use_jet_features and not self.config.use_cell_jet_matching:
            return True
        
        print("Validating jet features availability in dataset...")
        
        file_paths = self.get_file_paths()[:3]  # Check first 3 files
        
        all_ready = True
        for file_path in file_paths:
            if os.path.exists(file_path):
                availability = self.check_jet_features_availability(file_path)
                
                if not availability['jet_features_ready']:
                    print(f"Warning: Jet features not ready in {file_path}")
                    print(f"  Missing fields: {availability['missing_jet_fields']}")
                    all_ready = False
                else:
                    print(f"✓ Jet features available in {os.path.basename(file_path)}")
        
        if not all_ready:
            print("Error: Some files are missing required jet features.")
            print("Please ensure your H5 files contain the enhanced cell-jet matching data.")
            return False
        
        print("✓ Jet features validation passed")
        return True
