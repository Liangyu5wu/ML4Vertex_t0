"""Data processor for HGTD-only models using only HGTD tracks (no LAr data)."""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .data_processor import DataProcessor


class HGTDOnlyDataProcessor(DataProcessor):
    """Data processor for HGTD-only models with HGTD tracks support only."""

    def split_data(self, hgtd_track_sequences, vertex_features, vertex_times):
        """Split data into train/val/test sets."""

        # Generate indices for splitting
        indices = np.arange(len(vertex_times))
        train_indices, temp_indices = train_test_split(
            indices, test_size=self.config.test_size, random_state=self.config.random_state
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=self.config.val_split, random_state=self.config.random_state
        )

        # Split HGTD tracks
        train_hgtd_tracks = [hgtd_track_sequences[i] for i in train_indices]
        val_hgtd_tracks = [hgtd_track_sequences[i] for i in val_indices]
        test_hgtd_tracks = [hgtd_track_sequences[i] for i in test_indices]

        # Split vertex features
        train_vertex = vertex_features[train_indices]
        val_vertex = vertex_features[val_indices]
        test_vertex = vertex_features[test_indices]

        # Split times
        train_times = vertex_times[train_indices]
        val_times = vertex_times[val_indices]
        test_times = vertex_times[test_indices]

        return (
            (train_hgtd_tracks, val_hgtd_tracks, test_hgtd_tracks),
            (train_vertex, val_vertex, test_vertex),
            (train_times, val_times, test_times)
        )

    def _normalize_variable_length_features(self, train_sequences, val_sequences, test_sequences):
        """
        Normalize variable-length feature sequences.

        Returns normalized sequences and the scaler used.
        """
        # Collect all real features from training data (no padding)
        all_train_features = []
        for event_sequence in train_sequences:
            for features in event_sequence:
                all_train_features.append(features)

        if len(all_train_features) == 0:
            raise ValueError("No training features found for normalization")

        all_train_features = np.array(all_train_features)

        # Fit scaler on training data only
        scaler = StandardScaler()
        scaler.fit(all_train_features)

        # Normalize all sequences
        def normalize_sequences(sequences):
            normalized = []
            for event_sequence in sequences:
                if len(event_sequence) > 0:
                    event_array = np.array(event_sequence)
                    event_normalized = scaler.transform(event_array)
                    normalized.append(event_normalized.tolist())
                else:
                    normalized.append([])
            return normalized

        train_normalized = normalize_sequences(train_sequences)
        val_normalized = normalize_sequences(val_sequences)
        test_normalized = normalize_sequences(test_sequences)

        return train_normalized, val_normalized, test_normalized, scaler

    def normalize_features(self, train_hgtd_tracks, val_hgtd_tracks, test_hgtd_tracks,
                          train_vertex, val_vertex, test_vertex,
                          train_times, val_times, test_times):
        """
        Normalize features for HGTD-only data.

        HGTD tracks are variable-length sequences at this point (no padding yet).
        Normalization is computed only from real data, excluding any padding values.
        """

        # Normalize HGTD track features (only real data, no padding)
        train_hgtd_tracks_norm, val_hgtd_tracks_norm, test_hgtd_tracks_norm, hgtd_track_scaler = \
            self._normalize_variable_length_features(train_hgtd_tracks, val_hgtd_tracks, test_hgtd_tracks)

        # Store HGTD track scaler as instance variable for padding methods
        self.hgtd_track_scaler = hgtd_track_scaler

        # Normalize vertex features
        vertex_scaler = StandardScaler()
        train_vertex_norm = vertex_scaler.fit_transform(train_vertex)
        val_vertex_norm = vertex_scaler.transform(val_vertex)
        test_vertex_norm = vertex_scaler.transform(test_vertex)

        # Store normalization parameters
        norm_params = {
            'hgtd_track_scaler': hgtd_track_scaler,
            'vertex_scaler': vertex_scaler
        }

        return (
            (train_hgtd_tracks_norm, val_hgtd_tracks_norm, test_hgtd_tracks_norm),
            (train_vertex_norm, val_vertex_norm, test_vertex_norm),
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

    def create_hgtd_only_dataset(self, hgtd_track_sequences, vertex_features, vertex_times, shuffle=True):
        """Create dataset for HGTD-only models."""

        # Pad HGTD track sequences (after normalization)
        padded_hgtd_tracks = self._pad_hgtd_track_sequences(hgtd_track_sequences)

        # Convert to tensors
        hgtd_tracks_tensor = tf.constant(padded_hgtd_tracks, dtype=tf.float32)
        vertex_tensor = tf.constant(vertex_features, dtype=tf.float32)
        times_tensor = tf.constant(vertex_times, dtype=tf.float32)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'hgtd_track_inputs': hgtd_tracks_tensor,
                'vertex_inputs': vertex_tensor
            },
            times_tensor
        ))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, seed=42)

        return dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

    def create_hgtd_only_prediction_batches(self, hgtd_track_sequences, vertex_features, vertex_times):
        """Create prediction batches for HGTD-only models."""

        batch_size = self.config.batch_size
        num_samples = len(hgtd_track_sequences)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            # Get batch data
            batch_hgtd_tracks = hgtd_track_sequences[start_idx:end_idx]
            batch_vertex = vertex_features[start_idx:end_idx]
            batch_times = vertex_times[start_idx:end_idx]

            # Pad and create tensors
            padded_hgtd_tracks = self._pad_hgtd_track_sequences(batch_hgtd_tracks)

            batch_data = {
                'hgtd_track_inputs': tf.constant(padded_hgtd_tracks, dtype=tf.float32),
                'vertex_inputs': tf.constant(batch_vertex, dtype=tf.float32)
            }

            yield batch_data, batch_times
