"""Data loader for multi-input models with jets and tracks support."""

import os
import h5py
import numpy as np
from typing import List, Tuple, Optional
from config.base_config import BaseConfig


class MultiInputDataLoader:
    """Load and preprocess data for multi-input models with jets and tracks."""
    
    def __init__(self, config: BaseConfig):
        self.config = config
        
    def get_file_paths(self) -> List[str]:
        return [
            os.path.join(self.config.data_dir, f"output_{i:03d}.h5") 
            for i in range(self.config.num_files)
        ]
    
    def apply_cell_filtering(self, event_cells: np.ndarray) -> np.ndarray:
        """Apply cell filtering based on configuration."""
        mask = np.ones(len(event_cells), dtype=bool)
        
        if self.config.require_valid_cells:
            valid_mask = event_cells['valid'] == True
            mask = mask & valid_mask
        
        # Apply layer filtering
        if 'Cell_layer' in event_cells.dtype.names:
            layer_mask = np.isin(event_cells['Cell_layer'], [1, 2, 3])
            mask = mask & layer_mask
        
        # Apply additional custom filters
        if self.config.additional_cell_filters:
            for filter_key, filter_value in self.config.additional_cell_filters.items():
                if filter_key in event_cells.dtype.names:
                    if filter_key == 'Cell_e' and isinstance(filter_value, (int, float)):
                        additional_mask = event_cells[filter_key] >= filter_value
                    else:
                        additional_mask = event_cells[filter_key] == filter_value
                    mask = mask & additional_mask
        
        filtered_cells = event_cells[mask]
        
        # Apply time quality cut if enabled
        if self.config.use_time_quality_cut:
            try:
                filtered_cells = self.apply_time_quality_cut(filtered_cells)
            except Exception:
                pass  # Skip if calibration data unavailable
        
        return filtered_cells
    
    def apply_time_quality_cut(self, event_cells: np.ndarray) -> np.ndarray:
        """Apply time quality cut using calibration data."""
        if not self.config.use_time_quality_cut or len(event_cells) == 0:
            return event_cells
        
        calibration_data = self.config.load_calibration_data()
        
        # Sigma lookup tables
        sigma_lookup = {
            (1, 1): calibration_data['EMB1_sigma'],
            (1, 2): calibration_data['EMB2_sigma'],
            (1, 3): calibration_data['EMB3_sigma'],
            (0, 1): calibration_data['EME1_sigma'],
            (0, 2): calibration_data['EME2_sigma'],
            (0, 3): calibration_data['EME3_sigma'],
        }
        
        energy_bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, float('inf')]
        
        def get_energy_bin_index(energy: float) -> int:
            if energy < 1.0:
                return 0
            for i in range(len(energy_bins) - 1):
                if energy_bins[i] <= energy < energy_bins[i + 1]:
                    return i
            return len(energy_bins) - 2
        
        mask = np.ones(len(event_cells), dtype=bool)
        
        for i, cell in enumerate(event_cells):
            try:
                barrel = int(cell['Cell_Barrel'])
                layer = int(cell['Cell_layer'])
                energy = cell['Cell_e']
                cell_time = cell['Cell_time_TOF_corrected']
                
                if layer not in [1, 2, 3]:
                    mask[i] = False
                    continue
                
                sigma_params = sigma_lookup.get((barrel, layer), [1000.0] * 7)
                energy_bin_idx = get_energy_bin_index(energy)
                
                if energy_bin_idx >= len(sigma_params):
                    energy_bin_idx = len(sigma_params) - 1
                elif energy_bin_idx < 0:
                    energy_bin_idx = 0
                
                sigma_cell = sigma_params[energy_bin_idx]
                sigma_total = np.sqrt(self.config.vertex_time_sigma**2 + sigma_cell**2)
                cut_threshold = self.config.time_quality_n_sigma * sigma_total
                
                if abs(cell_time) > cut_threshold:
                    mask[i] = False
                    
            except (KeyError, ValueError, IndexError):
                mask[i] = False
        
        return event_cells[mask]
    
    def load_data_from_files(self, file_paths: Optional[List[str]] = None) -> Tuple:
        """Load multi-input data with jets and tracks."""
        if file_paths is None:
            file_paths = self.get_file_paths()
        
        all_cell_sequences = []
        all_vertex_features = []
        all_vertex_times = []
        all_jet_sequences = []
        all_track_sequences = []
        sequence_lengths = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
                
            with h5py.File(file_path, 'r') as f:
                vertex_data = f['HSvertex'][:]
                cells_data = f['cells'][:]
                jets_data = f['jets'][:]
                tracks_data = f['tracks'][:]
                
                for i in range(len(vertex_data)):
                    # Process cells
                    event_cells = cells_data[i]
                    valid_cells = self.apply_cell_filtering(event_cells)
                    
                    if len(valid_cells) < self.config.min_cells:
                        continue
                    
                    # Process cells sequence
                    cell_sequence = self._process_event_cells(valid_cells)
                    if cell_sequence is None:
                        continue
                    
                    # Process jets
                    jet_sequence = self._process_event_jets(jets_data[i])
                    
                    # Process tracks  
                    track_sequence = self._process_event_tracks(tracks_data[i])
                    
                    # Vertex features based on spatial features configuration
                    if self.config.use_spatial_features:
                        vertex_reco = [
                            vertex_data[i]['HSvertex_reco_x'],
                            vertex_data[i]['HSvertex_reco_y'],
                            vertex_data[i]['HSvertex_reco_z']
                        ]
                    else:
                        vertex_reco = [0.0, 0.0, 0.0]
                    vertex_time = vertex_data[i]['HSvertex_time']
                    
                    all_cell_sequences.append(cell_sequence)
                    all_vertex_features.append(vertex_reco)
                    all_vertex_times.append(vertex_time)
                    all_jet_sequences.append(jet_sequence)
                    all_track_sequences.append(track_sequence)
                    sequence_lengths.append(len(cell_sequence))
        
        return (all_cell_sequences, np.array(all_vertex_features), 
                np.array(all_vertex_times), sequence_lengths,
                all_jet_sequences, all_track_sequences)
    
    def _process_event_cells(self, valid_cells: np.ndarray) -> Optional[List[List[float]]]:
        """Process cells for a single event."""
        if self.config.cell_selection_feature in ['Cell_e', 'Cell_significance']:
            sorted_indices = np.argsort(-valid_cells[self.config.cell_selection_feature])
        else:
            sorted_indices = np.argsort(-valid_cells[self.config.cell_selection_feature])
        
        n_cells_to_use = min(len(valid_cells), self.config.max_cells)
        sorted_cells = valid_cells[sorted_indices[:n_cells_to_use]]
        
        sequence = []
        for cell_idx in range(n_cells_to_use):
            cell_features_values = []
            for feature in self.config.cell_features:
                if feature in sorted_cells.dtype.names:
                    cell_features_values.append(sorted_cells[feature][cell_idx])
                else:
                    cell_features_values.append(0.0)
            sequence.append(cell_features_values)
        
        return sequence
    
    def _process_event_jets(self, event_jets: np.ndarray) -> List[List[float]]:
        """Process jets for a single event."""
        valid_mask = (event_jets['valid'] == True) & (event_jets['AntiKt4EMTopoJets_selected'] == 1)
        valid_jets = event_jets[valid_mask]
        
        sorted_indices = np.argsort(-valid_jets['AntiKt4EMTopoJets_pt'])
        sorted_jets = valid_jets[sorted_indices]
        
        n_jets_to_use = min(len(sorted_jets), self.config.max_jets)
        
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
        
        # Pad to max_jets
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
        valid_mask = (event_tracks['valid'] == True) & (event_tracks['Track_isGoodFromHS_old_files'] == 1)
        valid_tracks = event_tracks[valid_mask]
        
        sorted_indices = np.argsort(-valid_tracks['Track_pt'])
        sorted_tracks = valid_tracks[sorted_indices]
        
        n_tracks_to_use = min(len(sorted_tracks), self.config.max_tracks)
        
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
        
        # Pad to max_tracks
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
    
    def calculate_baseline_method_predictions(self, cell_sequences: List, vertex_times: np.ndarray) -> np.ndarray:
        """
        Calculate baseline method predictions using sigma-weighted averaging.
        
        Args:
            cell_sequences: List of cell sequences for each event
            vertex_times: True vertex times (for logging only)
            
        Returns:
            Array of baseline method predictions
        """
        print("Calculating baseline method predictions for multi-input model...")
        
        # Load calibration data 
        calibration_file = getattr(self.config, 'calibration_data_file', 'multi_input_calibration.txt')
        calibration_data = self._load_calibration_data(calibration_file)
        
        # Energy bins for calibration: [1-1.5, 1.5-2, 2-3, 3-4, 4-5, 5-10, >10]
        energy_bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, float('inf')]
        
        # Sigma lookup tables - using 1-based layer indexing
        sigma_lookup = {
            (1, 1): calibration_data['EMB1_sigma'],  # Barrel, Layer 1
            (1, 2): calibration_data['EMB2_sigma'],  # Barrel, Layer 2
            (1, 3): calibration_data['EMB3_sigma'],  # Barrel, Layer 3
            (0, 1): calibration_data['EME1_sigma'],  # Endcap, Layer 1
            (0, 2): calibration_data['EME2_sigma'],  # Endcap, Layer 2
            (0, 3): calibration_data['EME3_sigma'],  # Endcap, Layer 3
        }
        
        # Parameter lookup for calibration - using 1-based layer indexing
        param_lookup = {
            (1, 1): calibration_data['EMB1_params'],  # Barrel, Layer 1
            (1, 2): calibration_data['EMB2_params'],  # Barrel, Layer 2
            (1, 3): calibration_data['EMB3_params'],  # Barrel, Layer 3
            (0, 1): calibration_data['EME1_params'],  # Endcap, Layer 1
            (0, 2): calibration_data['EME2_params'],  # Endcap, Layer 2
            (0, 3): calibration_data['EME3_params'],  # Endcap, Layer 3
        }
        
        baseline_predictions = []
        
        for event_idx, sequence in enumerate(cell_sequences):
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for cell_features in sequence:
                # Extract cell properties - assuming standard order
                # [eta, phi, barrel, layer, time_TOF_corrected, energy, significance]
                if len(cell_features) >= 6:
                    time_tof = cell_features[4]  # Cell_time_TOF_corrected
                    energy = cell_features[5]    # Cell_e
                    barrel = int(cell_features[2])  # Cell_Barrel
                    layer = int(cell_features[3])   # Cell_layer
                else:
                    continue  # Skip malformed cells
                
                # Apply calibration: corrected_time = tof_corrected_time - calibration_value
                detector_params = param_lookup.get((barrel, layer), [0.0] * 7)
                energy_bin_idx = self._get_energy_bin_index(energy, energy_bins)
                calibration_value = detector_params[energy_bin_idx]
                calibrated_time = time_tof - calibration_value
                
                # Get sigma for this cell
                sigma_params = sigma_lookup.get((barrel, layer), [1000.0] * 7)
                sigma = sigma_params[energy_bin_idx]
                
                # Weight = 1/sigma^2
                weight = 1.0 / (sigma * sigma)
                
                weighted_sum += weight * calibrated_time
                weight_sum += weight
            
            if weight_sum > 0:
                baseline_t0 = weighted_sum / weight_sum
            else:
                baseline_t0 = 0.0
            
            baseline_predictions.append(baseline_t0)
        
        baseline_predictions = np.array(baseline_predictions)
        print(f"Baseline method predictions calculated for {len(baseline_predictions)} events")
        
        return baseline_predictions
    
    def _load_calibration_data(self, calibration_file: str) -> dict:
        """Load calibration data from external file."""
        from pathlib import Path
        calibration_path = Path("calibration_data") / calibration_file
        
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
    
    def _get_energy_bin_index(self, energy: float, energy_bins: list) -> int:
        """Get energy bin index for calibration parameter lookup."""
        if energy < 1.0:
            return 0  # Use first bin for energies < 1 GeV
        
        for i in range(len(energy_bins) - 1):
            if energy_bins[i] <= energy < energy_bins[i + 1]:
                return i
        
        return len(energy_bins) - 2  # Last bin for energies >= 10 GeV