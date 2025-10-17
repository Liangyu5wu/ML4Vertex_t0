"""Data processor for HGTD multi-input models with LAr cells, jets, LAr tracks, and HGTD tracks."""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .multi_input_data_processor import MultiInputDataProcessor


class HGTDMultiInputDataProcessor(MultiInputDataProcessor):
    """Data processor for HGTD multi-input models with LAr cells, jets, LAr tracks, and HGTD tracks support."""

    def normalize_features(self, train_cells, val_cells, test_cells,
                          train_vertex, val_vertex, test_vertex,
                          train_jets, val_jets, test_jets,
                          train_tracks, val_tracks, test_tracks,
                          train_hgtd_tracks, val_hgtd_tracks, test_hgtd_tracks,
                          train_times, val_times, test_times):
        """
        Normalize features for HGTD multi-input data.

        Jets, tracks, and HGTD tracks are variable-length sequences at this point (no padding yet).
        Normalization is computed only from real data, excluding any padding values.
        """

        # Normalize cell features, vertex features, jets, and tracks using parent class method
        (train_cells_norm, val_cells_norm, test_cells_norm), \
        (train_vertex_norm, val_vertex_norm, test_vertex_norm), \
        (train_jets_norm, val_jets_norm, test_jets_norm), \
        (train_tracks_norm, val_tracks_norm, test_tracks_norm), \
        norm_params = super().normalize_features(
            train_cells, val_cells, test_cells,
            train_vertex, val_vertex, test_vertex,
            train_jets, val_jets, test_jets,
            train_tracks, val_tracks, test_tracks,
            train_times, val_times, test_times
        )

        # Normalize HGTD track features (only real data, no padding)
        train_hgtd_tracks_norm, val_hgtd_tracks_norm, test_hgtd_tracks_norm, hgtd_track_scaler = \
            self._normalize_variable_length_features(train_hgtd_tracks, val_hgtd_tracks, test_hgtd_tracks)

        # Store HGTD track scaler as instance variable for padding methods
        self.hgtd_track_scaler = hgtd_track_scaler

        # Add HGTD track scaler to norm_params
        norm_params['hgtd_track_scaler'] = hgtd_track_scaler

        return (
            (train_cells_norm, val_cells_norm, test_cells_norm),
            (train_vertex_norm, val_vertex_norm, test_vertex_norm),
            (train_jets_norm, val_jets_norm, test_jets_norm),
            (train_tracks_norm, val_tracks_norm, test_tracks_norm),
            (train_hgtd_tracks_norm, val_hgtd_tracks_norm, test_hgtd_tracks_norm),
            norm_params
        )

    def _pad_hgtd_track_sequences(self, hgtd_track_sequences):
        """
        Pad variable-length HGTD track sequences to max_hgtd_tracks.

        Padding values are taken from config and transformed to normalized space.

        Args:
            hgtd_track_sequences: List of variable-length HGTD track sequences (already normalized)

        Returns:
            Padded numpy array of shape (num_events, max_hgtd_tracks, num_hgtd_track_features)
        """
        num_events = len(hgtd_track_sequences)
        num_hgtd_track_features = len(self.config.hgtd_track_features)

        # Create padding vector in normalized space using configured values
        padding_vector = np.zeros(num_hgtd_track_features)
        if hasattr(self, 'hgtd_track_scaler') and self.hgtd_track_scaler is not None:
            for i, feature_name in enumerate(self.config.hgtd_track_features):
                original_padding = self.config.hgtd_track_padding_values[feature_name]
                # Transform original padding value to normalized space
                padding_vector[i] = (original_padding - self.hgtd_track_scaler.mean_[i]) / self.hgtd_track_scaler.scale_[i]

        # Initialize with padding values
        padded = np.tile(padding_vector, (num_events, self.config.max_hgtd_tracks, 1))

        # Fill in real data
        for i, event_hgtd_tracks in enumerate(hgtd_track_sequences):
            num_hgtd_tracks = len(event_hgtd_tracks)
            if num_hgtd_tracks > 0:
                padded[i, :num_hgtd_tracks, :] = event_hgtd_tracks

        return padded

    def split_data(self, cell_sequences, vertex_features, vertex_times,
                   jet_sequences, track_sequences, hgtd_track_sequences):
        """Split data into train/val/test sets."""

        # Generate indices for splitting
        indices = np.arange(len(vertex_times))
        train_indices, temp_indices = train_test_split(
            indices, test_size=self.config.test_size, random_state=self.config.random_state
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=self.config.val_split, random_state=self.config.random_state
        )

        # Split cells
        train_cells = [cell_sequences[i] for i in train_indices]
        val_cells = [cell_sequences[i] for i in val_indices]
        test_cells = [cell_sequences[i] for i in test_indices]

        # Split vertex features
        train_vertex = vertex_features[train_indices]
        val_vertex = vertex_features[val_indices]
        test_vertex = vertex_features[test_indices]

        # Split jets
        train_jets = [jet_sequences[i] for i in train_indices]
        val_jets = [jet_sequences[i] for i in val_indices]
        test_jets = [jet_sequences[i] for i in test_indices]

        # Split LAr tracks
        train_tracks = [track_sequences[i] for i in train_indices]
        val_tracks = [track_sequences[i] for i in val_indices]
        test_tracks = [track_sequences[i] for i in test_indices]

        # Split HGTD tracks
        train_hgtd_tracks = [hgtd_track_sequences[i] for i in train_indices]
        val_hgtd_tracks = [hgtd_track_sequences[i] for i in val_indices]
        test_hgtd_tracks = [hgtd_track_sequences[i] for i in test_indices]

        # Split times
        train_times = vertex_times[train_indices]
        val_times = vertex_times[val_indices]
        test_times = vertex_times[test_indices]

        return (
            (train_cells, val_cells, test_cells),
            (train_vertex, val_vertex, test_vertex),
            (train_jets, val_jets, test_jets),
            (train_tracks, val_tracks, test_tracks),
            (train_hgtd_tracks, val_hgtd_tracks, test_hgtd_tracks),
            (train_times, val_times, test_times)
        )

    def create_hgtd_multi_input_dataset(self, cell_sequences, vertex_features,
                                        jet_sequences, track_sequences, hgtd_track_sequences,
                                        vertex_times, shuffle=True):
        """Create dataset for HGTD multi-input models."""

        # Pad cell sequences and create masks
        padded_cells, masks = self._pad_sequences_with_masks(cell_sequences)

        # Pad jet sequences (after normalization)
        padded_jets = self._pad_jet_sequences(jet_sequences)

        # Pad LAr track sequences (after normalization)
        padded_tracks = self._pad_track_sequences(track_sequences)

        # Pad HGTD track sequences (after normalization)
        padded_hgtd_tracks = self._pad_hgtd_track_sequences(hgtd_track_sequences)

        # Convert to tensors
        cells_tensor = tf.constant(padded_cells, dtype=tf.float32)
        vertex_tensor = tf.constant(vertex_features, dtype=tf.float32)
        jets_tensor = tf.constant(padded_jets, dtype=tf.float32)
        tracks_tensor = tf.constant(padded_tracks, dtype=tf.float32)
        hgtd_tracks_tensor = tf.constant(padded_hgtd_tracks, dtype=tf.float32)
        masks_tensor = tf.constant(masks, dtype=tf.bool)
        times_tensor = tf.constant(vertex_times, dtype=tf.float32)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'cell_inputs': cells_tensor,
                'vertex_inputs': vertex_tensor,
                'jet_inputs': jets_tensor,
                'track_inputs': tracks_tensor,
                'hgtd_track_inputs': hgtd_tracks_tensor,
                'attention_mask': masks_tensor
            },
            times_tensor
        ))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, seed=42)

        return dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

    def create_hgtd_multi_input_prediction_batches(self, cell_sequences, vertex_features,
                                                   jet_sequences, track_sequences, hgtd_track_sequences,
                                                   vertex_times):
        """Create prediction batches for HGTD multi-input models."""

        batch_size = self.config.batch_size
        num_samples = len(cell_sequences)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            # Get batch data
            batch_cells = cell_sequences[start_idx:end_idx]
            batch_vertex = vertex_features[start_idx:end_idx]
            batch_jets = jet_sequences[start_idx:end_idx]
            batch_tracks = track_sequences[start_idx:end_idx]
            batch_hgtd_tracks = hgtd_track_sequences[start_idx:end_idx]
            batch_times = vertex_times[start_idx:end_idx]

            # Pad and create tensors
            padded_cells, masks = self._pad_sequences_with_masks(batch_cells)
            padded_jets = self._pad_jet_sequences(batch_jets)
            padded_tracks = self._pad_track_sequences(batch_tracks)
            padded_hgtd_tracks = self._pad_hgtd_track_sequences(batch_hgtd_tracks)

            batch_data = {
                'cell_inputs': tf.constant(padded_cells, dtype=tf.float32),
                'vertex_inputs': tf.constant(batch_vertex, dtype=tf.float32),
                'jet_inputs': tf.constant(padded_jets, dtype=tf.float32),
                'track_inputs': tf.constant(padded_tracks, dtype=tf.float32),
                'hgtd_track_inputs': tf.constant(padded_hgtd_tracks, dtype=tf.float32),
                'attention_mask': tf.constant(masks, dtype=tf.bool)
            }

            yield batch_data, batch_times
