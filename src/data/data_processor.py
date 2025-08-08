"""Data preprocessing and normalization utilities."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
        
        # Get indices of required features
        try:
            barrel_idx = self.config.cell_features.index('Cell_Barrel')
            layer_idx = self.config.cell_features.index('Cell_layer')
            energy_idx = self.config.cell_features.index('Cell_e')
            time_idx = self.config.cell_features.index('Cell_time_TOF_corrected')
        except ValueError as e:
            raise ValueError(f"Required feature not found for time calibration: {e}")
        
        # Create parameter lookup
        param_lookup = {
            (1, 1): self.config.emb1_params,  # Barrel, Layer 1
            (1, 2): self.config.emb2_params,  # Barrel, Layer 2
            (1, 3): self.config.emb3_params,  # Barrel, Layer 3
            (0, 1): self.config.eme1_params,  # Endcap, Layer 1
            (0, 2): self.config.eme2_params,  # Endcap, Layer 2
            (0, 3): self.config.eme3_params,  # Endcap, Layer 3
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
        test_vertex: np.ndarray
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
