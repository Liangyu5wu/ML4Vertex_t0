"""HGTD-only DNN model using only HGTD tracks for vertex time prediction (no LAr data)."""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, Any

from ..common.model_utils import (
    root_mean_squared_error, get_standard_metrics, get_common_custom_objects,
    load_model_with_fallback, get_model_summary_string, count_model_parameters
)
from config.dnn_config import DNNConfig


class HGTDOnlyDNNModel:
    """HGTD-only DNN model for vertex time prediction using only HGTD tracks."""

    def __init__(self, config: DNNConfig):
        self.config = config
        self.model = None

    def _get_loss_function(self):
        if self.config.loss_function == 'huber':
            return tf.keras.losses.Huber(delta=self.config.huber_delta)
        elif self.config.loss_function == 'mse':
            return 'mse'
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")

    def build_model(self, hgtd_track_feature_dim: int, vertex_dim: int) -> tf.keras.Model:
        """
        Build HGTD-only DNN model with HGTD tracks and vertex features.

        Architecture:
            HGTD Tracks → Track Encoder MLP → Global Average Pooling ┐
            Vertex Features → Dense Processing ─────────────────────┤→ Concat → Event MLP → Vertex Time
        """

        # Input layers
        hgtd_track_inputs = layers.Input(shape=(self.config.max_hgtd_tracks, hgtd_track_feature_dim), name='hgtd_track_inputs')
        vertex_inputs = layers.Input(shape=(vertex_dim,), name='vertex_inputs')

        # HGTD track processing - configurable encoder
        hgtd_track_x = hgtd_track_inputs
        for i, (units, dropout_rate) in enumerate(zip(self.config.hgtd_track_encoder_units, self.config.hgtd_track_dropout_rates)):
            hgtd_track_x = layers.Dense(units, activation=self.config.hgtd_track_activation, name=f'hgtd_track_encoder_{i}')(hgtd_track_x)

            if self.config.hgtd_track_use_batch_norm:
                hgtd_track_x = layers.BatchNormalization(name=f'hgtd_track_bn_{i}')(hgtd_track_x)

            hgtd_track_x = layers.Dropout(dropout_rate, name=f'hgtd_track_dropout_{i}')(hgtd_track_x)

        # Global average pooling for HGTD tracks
        hgtd_track_representation = layers.GlobalAveragePooling1D(name='hgtd_track_global_pool')(hgtd_track_x)

        # Vertex feature processing
        vertex_x = layers.Dense(32, activation='relu', name='vertex_dense')(vertex_inputs)

        # Feature combination
        combined = layers.Concatenate(name='combine_features')([hgtd_track_representation, vertex_x])

        # Event-level processing
        x = combined
        for i, (units, dropout_rate) in enumerate(zip(
            self.config.event_encoder_units,
            self.config.event_dropout_rates
        )):
            x = layers.Dense(units, activation='relu', name=f'event_encoder_{i}')(x)

            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'event_bn_{i}')(x)

            x = layers.Dropout(dropout_rate, name=f'event_dropout_{i}')(x)

        # Output layer
        output = layers.Dense(1, name='vertex_time')(x)

        # Create model
        model = models.Model(
            inputs=[hgtd_track_inputs, vertex_inputs],
            outputs=output,
            name='hgtd_only_dnn'
        )

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self._get_loss_function(),
            metrics=get_standard_metrics()
        )

        self.model = model
        return model

    def get_model(self) -> tf.keras.Model:
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        return self.model

    def save_model(self, filepath: str = None):
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        if filepath is None:
            filepath = self.config.model_path

        # Save in H5 format
        self.model.save(filepath, save_format='h5')
        print(f"HGTD-only DNN model saved to: {filepath}")

    @staticmethod
    def load_model(filepath: str) -> tf.keras.Model:
        custom_objects = get_common_custom_objects()
        return load_model_with_fallback(filepath, custom_objects)

    def get_model_summary(self) -> str:
        if self.model is None:
            return "Model not built yet"
        return get_model_summary_string(self.model)

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        return count_model_parameters(self.model)
