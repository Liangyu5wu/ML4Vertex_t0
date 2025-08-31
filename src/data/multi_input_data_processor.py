"""Data processor for multi-input models with jets and tracks."""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .data_processor import DataProcessor


class MultiInputDataProcessor(DataProcessor):
    """Data processor for multi-input models with jets and tracks support."""
    
    def normalize_features(self, train_cells, val_cells, test_cells,
                          train_vertex, val_vertex, test_vertex,
                          train_jets, val_jets, test_jets,
                          train_tracks, val_tracks, test_tracks,
                          train_times, val_times, test_times):
        """Normalize features for multi-input data."""
        
        # Normalize cell features using parent class method
        (train_cells_norm, val_cells_norm, test_cells_norm), \
        (train_vertex_norm, val_vertex_norm, test_vertex_norm), \
        norm_params = super().normalize_features(
            train_cells, val_cells, test_cells,
            train_vertex, val_vertex, test_vertex, 
            train_times, val_times, test_times
        )
        
        # Normalize jet features
        train_jets_flat = np.array(train_jets).reshape(-1, len(train_jets[0][0]))
        val_jets_flat = np.array(val_jets).reshape(-1, len(val_jets[0][0]))
        test_jets_flat = np.array(test_jets).reshape(-1, len(test_jets[0][0]))
        
        jet_scaler = StandardScaler()
        train_jets_norm_flat = jet_scaler.fit_transform(train_jets_flat)
        val_jets_norm_flat = jet_scaler.transform(val_jets_flat)
        test_jets_norm_flat = jet_scaler.transform(test_jets_flat)
        
        train_jets_norm = train_jets_norm_flat.reshape(len(train_jets), len(train_jets[0]), -1).tolist()
        val_jets_norm = val_jets_norm_flat.reshape(len(val_jets), len(val_jets[0]), -1).tolist()
        test_jets_norm = test_jets_norm_flat.reshape(len(test_jets), len(test_jets[0]), -1).tolist()
        
        # Normalize track features  
        train_tracks_flat = np.array(train_tracks).reshape(-1, len(train_tracks[0][0]))
        val_tracks_flat = np.array(val_tracks).reshape(-1, len(val_tracks[0][0]))
        test_tracks_flat = np.array(test_tracks).reshape(-1, len(test_tracks[0][0]))
        
        track_scaler = StandardScaler()
        train_tracks_norm_flat = track_scaler.fit_transform(train_tracks_flat)
        val_tracks_norm_flat = track_scaler.transform(val_tracks_flat)
        test_tracks_norm_flat = track_scaler.transform(test_tracks_flat)
        
        train_tracks_norm = train_tracks_norm_flat.reshape(len(train_tracks), len(train_tracks[0]), -1).tolist()
        val_tracks_norm = val_tracks_norm_flat.reshape(len(val_tracks), len(val_tracks[0]), -1).tolist()
        test_tracks_norm = test_tracks_norm_flat.reshape(len(test_tracks), len(test_tracks[0]), -1).tolist()
        
        # Add scalers to norm_params
        norm_params['jet_scaler'] = jet_scaler
        norm_params['track_scaler'] = track_scaler
        
        return (
            (train_cells_norm, val_cells_norm, test_cells_norm),
            (train_vertex_norm, val_vertex_norm, test_vertex_norm),
            (train_jets_norm, val_jets_norm, test_jets_norm),
            (train_tracks_norm, val_tracks_norm, test_tracks_norm),
            norm_params
        )
    
    def split_data(self, cell_sequences, vertex_features, vertex_times, 
                   jet_sequences, track_sequences):
        """Split data into train/val/test sets."""
        
        # Use parent class method for basic splitting
        (train_cells, val_cells, test_cells), \
        (train_vertex, val_vertex, test_vertex), \
        (train_times, val_times, test_times) = super().split_data(
            cell_sequences, vertex_features, vertex_times
        )
        
        # Generate same indices for jets and tracks
        indices = np.arange(len(vertex_times))
        train_indices, temp_indices = train_test_split(
            indices, test_size=self.config.test_size, random_state=self.config.random_state
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=self.config.val_split, random_state=self.config.random_state
        )
        
        # Split jets and tracks using same indices
        train_jets = [jet_sequences[i] for i in train_indices]
        val_jets = [jet_sequences[i] for i in val_indices]
        test_jets = [jet_sequences[i] for i in test_indices]
        
        train_tracks = [track_sequences[i] for i in train_indices]
        val_tracks = [track_sequences[i] for i in val_indices]
        test_tracks = [track_sequences[i] for i in test_indices]
        
        return (
            (train_cells, val_cells, test_cells),
            (train_vertex, val_vertex, test_vertex), 
            (train_jets, val_jets, test_jets),
            (train_tracks, val_tracks, test_tracks),
            (train_times, val_times, test_times)
        )
    
    def _pad_sequences_with_masks(self, cell_sequences):
        """
        Pad cell sequences and create attention masks.
        
        Args:
            cell_sequences: Variable-length cell sequences
            
        Returns:
            Tuple of (padded_sequences, attention_masks)
        """
        if not cell_sequences:
            return np.array([]), np.array([])
        
        # Find maximum sequence length
        max_seq_len = max(len(seq) for seq in cell_sequences)
        
        # Feature dimension
        feature_dim = len(self.config.cell_features)
        
        # Apply smart padding using parent class method
        padded_cells = self.apply_smart_padding(cell_sequences, max_seq_len, feature_dim)
        
        # Create attention mask using parent class method
        attention_masks = self.create_attention_mask(cell_sequences, max_seq_len)
        
        return padded_cells, attention_masks
    
    def create_multi_input_dataset(self, cell_sequences, vertex_features, 
                                  jet_sequences, track_sequences, vertex_times, 
                                  shuffle=True):
        """Create dataset for multi-input models."""
        
        # Pad cell sequences and create masks
        padded_cells, masks = self._pad_sequences_with_masks(cell_sequences)
        
        # Convert to tensors
        cells_tensor = tf.constant(padded_cells, dtype=tf.float32)
        vertex_tensor = tf.constant(vertex_features, dtype=tf.float32)
        jets_tensor = tf.constant(jet_sequences, dtype=tf.float32)
        tracks_tensor = tf.constant(track_sequences, dtype=tf.float32)
        masks_tensor = tf.constant(masks, dtype=tf.bool)
        times_tensor = tf.constant(vertex_times, dtype=tf.float32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'cell_inputs': cells_tensor,
                'vertex_inputs': vertex_tensor,
                'jet_inputs': jets_tensor,
                'track_inputs': tracks_tensor,
                'attention_mask': masks_tensor
            },
            times_tensor
        ))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, seed=42)
        
        return dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
    
    def create_multi_input_prediction_batches(self, cell_sequences, vertex_features,
                                            jet_sequences, track_sequences, vertex_times):
        """Create prediction batches for multi-input models."""
        
        batch_size = self.config.batch_size
        num_samples = len(cell_sequences)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            
            # Get batch data
            batch_cells = cell_sequences[start_idx:end_idx]
            batch_vertex = vertex_features[start_idx:end_idx]
            batch_jets = jet_sequences[start_idx:end_idx]
            batch_tracks = track_sequences[start_idx:end_idx]
            batch_times = vertex_times[start_idx:end_idx]
            
            # Pad and create tensors
            padded_cells, masks = self._pad_sequences_with_masks(batch_cells)
            
            batch_data = {
                'cell_inputs': tf.constant(padded_cells, dtype=tf.float32),
                'vertex_inputs': tf.constant(batch_vertex, dtype=tf.float32), 
                'jet_inputs': tf.constant(batch_jets, dtype=tf.float32),
                'track_inputs': tf.constant(batch_tracks, dtype=tf.float32),
                'attention_mask': tf.constant(masks, dtype=tf.bool)
            }
            
            yield batch_data, batch_times