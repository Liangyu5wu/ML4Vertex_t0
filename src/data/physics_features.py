"""Physics-informed feature engineering for vertex time prediction."""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class PhysicsFeatureEngineer:
    """Add physics-informed features to cell data."""
    
    def __init__(self, config):
        """
        Initialize physics feature engineer.
        
        Args:
            config: Configuration object with calibration data
        """
        self.config = config
        
        # Energy bins for sigma lookup: [1-1.5, 1.5-2, 2-3, 3-4, 4-5, 5-10, >10]
        self.energy_bins = getattr(config, 'sigma_energy_bins', [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0])
        
        # Load calibration data if available
        self.calibration_data = None
        if hasattr(config, 'calibration_data_file') and config.use_detector_params:
            try:
                self.calibration_data = self._load_calibration_data()
            except Exception as e:
                logger.warning(f"Could not load calibration data: {e}")
    
    def _load_calibration_data(self) -> Dict[str, List[float]]:
        """Load calibration data from external file."""
        from pathlib import Path
        
        calibration_path = Path("calibration_data") / self.config.calibration_data_file
        
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
        
        return calibration_data
    
    def _get_energy_bin_index(self, energy: float) -> int:
        """Get energy bin index for sigma lookup."""
        if energy < self.energy_bins[0]:
            return 0  # Use first bin for energies < 1 GeV
        
        for i in range(len(self.energy_bins) - 1):
            if self.energy_bins[i] <= energy < self.energy_bins[i + 1]:
                return i
        
        return len(self.energy_bins) - 2  # Last bin for energies >= 10 GeV
    
    def _get_cell_sigma(self, barrel: int, layer: int, energy: float) -> float:
        """
        Get expected time measurement uncertainty (sigma) for a cell.
        
        Args:
            barrel: 1 for barrel, 0 for endcap
            layer: Layer number (1, 2, 3)
            energy: Cell energy in GeV
            
        Returns:
            Expected time sigma in ps
        """
        if self.calibration_data is None:
            # Fallback: use energy-dependent approximation
            return 50.0 + 100.0 / max(energy, 0.1)  # Rough approximation
        
        # Sigma lookup tables
        sigma_lookup = {
            (1, 1): self.calibration_data.get('EMB1_sigma', [100.0] * 7),
            (1, 2): self.calibration_data.get('EMB2_sigma', [100.0] * 7),
            (1, 3): self.calibration_data.get('EMB3_sigma', [100.0] * 7),
            (0, 1): self.calibration_data.get('EME1_sigma', [100.0] * 7),
            (0, 2): self.calibration_data.get('EME2_sigma', [100.0] * 7),
            (0, 3): self.calibration_data.get('EME3_sigma', [100.0] * 7),
        }
        
        detector_sigmas = sigma_lookup.get((barrel, layer), [100.0] * 7)
        energy_bin_idx = self._get_energy_bin_index(energy)
        
        return detector_sigmas[min(energy_bin_idx, len(detector_sigmas) - 1)]
    
    def add_physics_features(self, cell_sequences: List[List[List[float]]], feature_names: List[str]) -> Tuple[List[List[List[float]]], List[str]]:
        """
        Add physics-informed features to cell sequences.
        
        Args:
            cell_sequences: Original cell sequences
            feature_names: Original feature names
            
        Returns:
            Tuple of (enhanced_sequences, enhanced_feature_names)
        """
        # Find indices of required features
        try:
            energy_idx = feature_names.index('Cell_e')
            barrel_idx = feature_names.index('Cell_Barrel') 
            layer_idx = feature_names.index('Cell_layer')
            time_idx = feature_names.index('Cell_time_TOF_corrected')
        except ValueError as e:
            logger.error(f"Required feature not found: {e}")
            return cell_sequences, feature_names
        
        enhanced_sequences = []
        
        for sequence in cell_sequences:
            enhanced_sequence = []
            
            for cell in sequence:
                enhanced_cell = cell.copy()
                
                # Extract cell properties
                energy = cell[energy_idx]
                barrel = int(cell[barrel_idx])
                layer = int(cell[layer_idx])
                time = cell[time_idx]
                
                # Add physics-informed features
                
                # 1. Expected measurement uncertainty (sigma)
                sigma = self._get_cell_sigma(barrel, layer, energy)
                enhanced_cell.append(sigma)
                
                # 2. Measurement weight (1/sigma^2) - traditional weighting
                weight = 1.0 / (sigma * sigma) if sigma > 0 else 0.0
                enhanced_cell.append(weight)
                
                # 3. Energy-normalized time (helps with different energy scales)
                energy_norm_time = time / max(energy, 0.1)
                enhanced_cell.append(energy_norm_time)
                
                # 4. Log energy (physics-motivated feature)
                log_energy = np.log(max(energy, 0.01))
                enhanced_cell.append(log_energy)
                
                # 5. Time significance (time/sigma ratio)
                time_significance = time / max(sigma, 1.0)
                enhanced_cell.append(time_significance)
                
                # 6. Quality indicator (higher energy and lower sigma = better)
                quality = energy / max(sigma, 1.0)
                enhanced_cell.append(quality)
                
                enhanced_sequence.append(enhanced_cell)
            
            enhanced_sequences.append(enhanced_sequence)
        
        # Update feature names
        enhanced_feature_names = feature_names + [
            'cell_sigma',           # Expected time uncertainty
            'cell_weight',          # Traditional 1/sigma^2 weight
            'energy_norm_time',     # Time normalized by energy
            'log_energy',           # Log of cell energy
            'time_significance',    # Time/sigma ratio
            'quality_indicator'     # Energy/sigma quality metric
        ]
        
        logger.info(f"Added {len(enhanced_feature_names) - len(feature_names)} physics-informed features")
        
        return enhanced_sequences, enhanced_feature_names
    
    def compute_event_level_features(self, cell_sequences: List[List[List[float]]], feature_names: List[str]) -> np.ndarray:
        """
        Compute event-level physics features that can be used as auxiliary inputs.
        
        Args:
            cell_sequences: Cell sequences with features
            feature_names: Feature names
            
        Returns:
            Event-level feature array of shape (num_events, num_event_features)
        """
        try:
            energy_idx = feature_names.index('Cell_e')
            weight_idx = feature_names.index('cell_weight') if 'cell_weight' in feature_names else None
            time_idx = feature_names.index('Cell_time_TOF_corrected')
        except ValueError:
            # Return empty features if required indices not found
            return np.zeros((len(cell_sequences), 0))
        
        event_features = []
        
        for sequence in cell_sequences:
            if len(sequence) == 0:
                # Handle empty sequences
                event_features.append([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            
            # Extract features for this event
            energies = np.array([cell[energy_idx] for cell in sequence])
            times = np.array([cell[time_idx] for cell in sequence])
            
            if weight_idx is not None:
                weights = np.array([cell[weight_idx] for cell in sequence])
            else:
                weights = np.ones(len(sequence))
            
            # Compute event-level features
            total_energy = np.sum(energies)
            num_cells = len(sequence)
            avg_energy = total_energy / num_cells
            energy_spread = np.std(energies)
            
            # Weighted average time (similar to traditional method)
            if np.sum(weights) > 0:
                weighted_avg_time = np.sum(times * weights) / np.sum(weights)
            else:
                weighted_avg_time = np.mean(times)
            
            event_features.append([
                total_energy,
                num_cells,
                avg_energy,
                energy_spread,
                weighted_avg_time
            ])
        
        return np.array(event_features)