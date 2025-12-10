"""HGTD Multi-input DNN model with LAr cells, jets, LAr tracks, and HGTD tracks for vertex time prediction."""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, Any

from ..common.model_utils import (
    root_mean_squared_error, get_standard_metrics, get_common_custom_objects,
    load_model_with_fallback, get_model_summary_string, count_model_parameters
)
from .multi_input_dnn_model import MaskedAttentionPooling
from config.dnn_config import DNNConfig


class HGTDMultiInputDNNModel:
    """HGTD Multi-input DNN model for vertex time prediction with LAr cells, jets, LAr tracks, and HGTD tracks."""

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

    def build_model(self, feature_dim: int, vertex_dim: int,
                    jet_feature_dim: int, track_feature_dim: int,
                    hgtd_track_feature_dim: int) -> tf.keras.Model:
        """Build HGTD multi-input DNN model with LAr cells, jets, LAr tracks, and HGTD tracks."""

        # Input layers
        cell_inputs = layers.Input(shape=(None, feature_dim), name='cell_inputs')
        vertex_inputs = layers.Input(shape=(vertex_dim,), name='vertex_inputs')
        jet_inputs = layers.Input(shape=(self.config.max_jets, jet_feature_dim), name='jet_inputs')
        track_inputs = layers.Input(shape=(self.config.max_tracks, track_feature_dim), name='track_inputs')
        hgtd_track_inputs = layers.Input(shape=(self.config.max_hgtd_tracks, hgtd_track_feature_dim), name='hgtd_track_inputs')
        mask_inputs = layers.Input(shape=(None,), dtype=tf.bool, name='attention_mask')

        # Cell processing (same as multi_input_dnn)
        x = cell_inputs
        for i, units in enumerate(self.config.cell_encoder_units):
            x = layers.Dense(units, activation=self.config.cell_activation, name=f'cell_encoder_{i}')(x)
            x = layers.Dropout(self.config.cell_dropout_rate, name=f'cell_dropout_{i}')(x)

        # Cell attention pooling
        if self.config.use_attention_pooling:
            cell_representation = MaskedAttentionPooling(
                hidden_units=self.config.attention_hidden_units,
                name='cell_attention_pooling'
            )(x, mask=mask_inputs)
        else:
            # Masked global average pooling
            mask_expanded = tf.expand_dims(tf.cast(mask_inputs, tf.float32), axis=-1)
            masked_sum = tf.reduce_sum(x * mask_expanded, axis=1)
            mask_count = tf.reduce_sum(mask_expanded, axis=1)
            mask_count = tf.maximum(mask_count, 1.0)
            cell_representation = masked_sum / mask_count

        # Jet processing (same as multi_input_dnn)
        jet_x = jet_inputs
        for i, (units, dropout_rate) in enumerate(zip(self.config.jet_encoder_units, self.config.jet_dropout_rates)):
            jet_x = layers.Dense(units, activation=self.config.jet_activation, name=f'jet_encoder_{i}')(jet_x)

            if self.config.jet_use_batch_norm:
                jet_x = layers.BatchNormalization(name=f'jet_bn_{i}')(jet_x)

            jet_x = layers.Dropout(dropout_rate, name=f'jet_dropout_{i}')(jet_x)
        jet_representation = layers.GlobalAveragePooling1D(name='jet_global_pool')(jet_x)

        # LAr track processing (same as multi_input_dnn)
        track_x = track_inputs
        for i, (units, dropout_rate) in enumerate(zip(self.config.track_encoder_units, self.config.track_dropout_rates)):
            track_x = layers.Dense(units, activation=self.config.track_activation, name=f'track_encoder_{i}')(track_x)

            if self.config.track_use_batch_norm:
                track_x = layers.BatchNormalization(name=f'track_bn_{i}')(track_x)

            track_x = layers.Dropout(dropout_rate, name=f'track_dropout_{i}')(track_x)
        track_representation = layers.GlobalAveragePooling1D(name='track_global_pool')(track_x)

        # HGTD track processing (NEW - configurable architecture)
        hgtd_track_x = hgtd_track_inputs
        for i, (units, dropout_rate) in enumerate(zip(self.config.hgtd_track_encoder_units, self.config.hgtd_track_dropout_rates)):
            hgtd_track_x = layers.Dense(units, activation=self.config.hgtd_track_activation, name=f'hgtd_track_encoder_{i}')(hgtd_track_x)

            if self.config.hgtd_track_use_batch_norm:
                hgtd_track_x = layers.BatchNormalization(name=f'hgtd_track_bn_{i}')(hgtd_track_x)

            hgtd_track_x = layers.Dropout(dropout_rate, name=f'hgtd_track_dropout_{i}')(hgtd_track_x)
        hgtd_track_representation = layers.GlobalAveragePooling1D(name='hgtd_track_global_pool')(hgtd_track_x)

        # Feature combination (including vertex features and HGTD tracks)
        combined = layers.Concatenate(name='combine_all_features')([
            cell_representation, vertex_inputs, jet_representation, track_representation, hgtd_track_representation
        ])

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
            inputs=[cell_inputs, vertex_inputs, jet_inputs, track_inputs, hgtd_track_inputs, mask_inputs],
            outputs=output,
            name='hgtd_multi_input_dnn'
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

        # Save in H5 format with custom objects
        self.model.save(filepath, save_format='h5')
        print(f"HGTD Multi-input DNN model saved to: {filepath}")

    @staticmethod
    def load_model(filepath: str) -> tf.keras.Model:
        custom_objects = get_common_custom_objects()
        custom_objects['MaskedAttentionPooling'] = MaskedAttentionPooling

        return load_model_with_fallback(filepath, custom_objects)

    def get_model_summary(self) -> str:
        if self.model is None:
            return "Model not built yet"
        return get_model_summary_string(self.model)

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        return count_model_parameters(self.model)
