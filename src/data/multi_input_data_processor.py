"""Data processor for multi-input models with jets and tracks."""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .data_processor import DataProcessor


class MultiInputDataProcessor(DataProcessor):
    """Data processor for multi-input models with jets and tracks support."""

    def _normalize_variable_length_features(self, train_seqs, val_seqs, test_seqs):
        """
        Normalize variable-length sequences (jets or tracks).
        Only real data is used for computing statistics, padding values are excluded.

        Args:
            train_seqs: List of variable-length sequences for training
            val_seqs: List of variable-length sequences for validation
            test_seqs: List of variable-length sequences for testing

        Returns:
            Tuple of (train_norm, val_norm, test_norm, scaler)
        """
        # Collect all real features from training data
        all_train_features = []
        for event_seq in train_seqs:
            for item in event_seq:
                all_train_features.append(item)

        # Convert to array and fit scaler
        train_array = np.array(all_train_features)
        scaler = StandardScaler()
        scaler.fit(train_array)

        # Normalize each split while preserving variable-length structure
        def normalize_split(sequences):
            normalized_seqs = []
            for event_seq in sequences:
                normalized_event = []
                for item in event_seq:
                    normalized_item = scaler.transform([item])[0].tolist()
                    normalized_event.append(normalized_item)
                normalized_seqs.append(normalized_event)
            return normalized_seqs

        train_norm = normalize_split(train_seqs)
        val_norm = normalize_split(val_seqs)
        test_norm = normalize_split(test_seqs)

        return train_norm, val_norm, test_norm, scaler

    def normalize_features(self, train_cells, val_cells, test_cells,
                          train_vertex, val_vertex, test_vertex,
                          train_jets, val_jets, test_jets,
                          train_tracks, val_tracks, test_tracks,
                          train_times, val_times, test_times):
        """
        Normalize features for multi-input data.

        Jets and tracks are variable-length sequences at this point (no padding yet).
        Normalization is computed only from real data, excluding any padding values.
        """

        # Normalize cell features using parent class method
        (train_cells_norm, val_cells_norm, test_cells_norm), \
        (train_vertex_norm, val_vertex_norm, test_vertex_norm), \
        norm_params = super().normalize_features(
            train_cells, val_cells, test_cells,
            train_vertex, val_vertex, test_vertex,
            train_times, val_times, test_times
        )

        # Normalize jet features (only real data, no padding)
        train_jets_norm, val_jets_norm, test_jets_norm, jet_scaler = \
            self._normalize_variable_length_features(train_jets, val_jets, test_jets)

        # Normalize track features (only real data, no padding)
        train_tracks_norm, val_tracks_norm, test_tracks_norm, track_scaler = \
            self._normalize_variable_length_features(train_tracks, val_tracks, test_tracks)

        # Store scalers as instance variables for padding methods
        self.jet_scaler = jet_scaler
        self.track_scaler = track_scaler

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

    def _pad_jet_sequences(self, jet_sequences):
        """
        Pad variable-length jet sequences to max_jets.

        Padding values are taken from config and transformed to normalized space.

        Args:
            jet_sequences: List of variable-length jet sequences (already normalized)

        Returns:
            Padded numpy array of shape (num_events, max_jets, num_jet_features)
        """
        num_events = len(jet_sequences)
        num_jet_features = len(self.config.jet_features)

        # Create padding vector in normalized space using configured values
        padding_vector = np.zeros(num_jet_features)
        if hasattr(self, 'jet_scaler') and self.jet_scaler is not None:
            for i, feature_name in enumerate(self.config.jet_features):
                original_padding = self.config.jet_padding_values[feature_name]
                # Transform original padding value to normalized space
                padding_vector[i] = (original_padding - self.jet_scaler.mean_[i]) / self.jet_scaler.scale_[i]

        # Initialize with padding values
        padded = np.tile(padding_vector, (num_events, self.config.max_jets, 1))

        # Fill in real data
        for i, event_jets in enumerate(jet_sequences):
            num_jets = len(event_jets)
            if num_jets > 0:
                padded[i, :num_jets, :] = event_jets

        return padded

    def _pad_track_sequences(self, track_sequences):
        """
        Pad variable-length track sequences to max_tracks.

        Padding values are taken from config and transformed to normalized space.

        Args:
            track_sequences: List of variable-length track sequences (already normalized)

        Returns:
            Padded numpy array of shape (num_events, max_tracks, num_track_features)
        """
        num_events = len(track_sequences)
        num_track_features = len(self.config.track_features)

        # Create padding vector in normalized space using configured values
        padding_vector = np.zeros(num_track_features)
        if hasattr(self, 'track_scaler') and self.track_scaler is not None:
            for i, feature_name in enumerate(self.config.track_features):
                original_padding = self.config.track_padding_values[feature_name]
                # Transform original padding value to normalized space
                padding_vector[i] = (original_padding - self.track_scaler.mean_[i]) / self.track_scaler.scale_[i]

        # Initialize with padding values
        padded = np.tile(padding_vector, (num_events, self.config.max_tracks, 1))

        # Fill in real data
        for i, event_tracks in enumerate(track_sequences):
            num_tracks = len(event_tracks)
            if num_tracks > 0:
                padded[i, :num_tracks, :] = event_tracks

        return padded

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

        # Pad jet sequences (after normalization)
        padded_jets = self._pad_jet_sequences(jet_sequences)

        # Pad track sequences (after normalization)
        padded_tracks = self._pad_track_sequences(track_sequences)

        # Convert to tensors
        cells_tensor = tf.constant(padded_cells, dtype=tf.float32)
        vertex_tensor = tf.constant(vertex_features, dtype=tf.float32)
        jets_tensor = tf.constant(padded_jets, dtype=tf.float32)
        tracks_tensor = tf.constant(padded_tracks, dtype=tf.float32)
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
            padded_jets = self._pad_jet_sequences(batch_jets)
            padded_tracks = self._pad_track_sequences(batch_tracks)

            batch_data = {
                'cell_inputs': tf.constant(padded_cells, dtype=tf.float32),
                'vertex_inputs': tf.constant(batch_vertex, dtype=tf.float32),
                'jet_inputs': tf.constant(padded_jets, dtype=tf.float32),
                'track_inputs': tf.constant(padded_tracks, dtype=tf.float32),
                'attention_mask': tf.constant(masks, dtype=tf.bool)
            }

            yield batch_data, batch_times