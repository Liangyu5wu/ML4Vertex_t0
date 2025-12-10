"""Unified layer implementations for all models."""

import tensorflow as tf
from tensorflow.keras import layers


class MaskedAttentionPooling(layers.Layer):
    """Attention-based pooling with mask support for DNN models."""

    def __init__(self, hidden_units: int = 32, **kwargs):
        """Initialize masked attention pooling layer."""
        super(MaskedAttentionPooling, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.attention_net = tf.keras.Sequential([
            layers.Dense(hidden_units, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, inputs, mask=None):
        """
        Apply masked attention pooling.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, features)
            mask: Attention mask of shape (batch_size, seq_len)
                 True for valid positions, False for padding positions

        Returns:
            Pooled tensor of shape (batch_size, features)
        """
        # Calculate attention scores
        attention_scores = self.attention_net(inputs)  # (batch_size, seq_len, 1)

        if mask is not None:
            # Convert mask to float and expand dimensions
            mask = tf.cast(mask, tf.float32)  # (batch_size, seq_len)
            mask = tf.expand_dims(mask, axis=-1)  # (batch_size, seq_len, 1)

            # Apply mask by adding large negative values to padded positions
            attention_scores += (1.0 - mask) * -1e9

        # Calculate attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # (batch_size, seq_len, 1)

        # Apply attention weights
        weighted_inputs = inputs * attention_weights  # (batch_size, seq_len, features)
        output = tf.reduce_sum(weighted_inputs, axis=1)  # (batch_size, features)

        return output

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({'hidden_units': self.hidden_units})
        return config


class MaskedGlobalAveragePooling1D(layers.Layer):
    """Global average pooling with attention mask support for Transformer models."""

    def __init__(self, **kwargs):
        """Initialize masked global average pooling layer."""
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        """
        Apply masked global average pooling.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, features)
            mask: Attention mask of shape (batch_size, seq_len)
                 True for valid positions, False for padding positions

        Returns:
            Pooled tensor of shape (batch_size, features)
        """
        if mask is None:
            # Fallback to standard global average pooling
            return tf.reduce_mean(inputs, axis=1)

        # Convert mask to float and expand dimensions for broadcasting
        mask = tf.cast(mask, tf.float32)  # (batch_size, seq_len)
        mask = tf.expand_dims(mask, axis=-1)  # (batch_size, seq_len, 1)

        # Apply mask to inputs
        masked_inputs = inputs * mask  # (batch_size, seq_len, features)

        # Calculate sum and count of valid positions
        masked_sum = tf.reduce_sum(masked_inputs, axis=1)  # (batch_size, features)
        mask_count = tf.reduce_sum(mask, axis=1)  # (batch_size, 1)

        # Avoid division by zero
        mask_count = tf.maximum(mask_count, 1.0)

        # Calculate masked average
        masked_average = masked_sum / mask_count  # (batch_size, features)

        return masked_average

    def get_config(self):
        """Get layer configuration for serialization."""
        return super().get_config()


class FeatureEncoder(layers.Layer):
    """Configurable feature encoder for jets, tracks, and HGTD tracks.

    This unified encoder replaces the duplicated encoder blocks across multiple models.
    """

    def __init__(
        self,
        encoder_units: list,
        dropout_rates: list = None,
        activation: str = 'relu',
        use_batch_norm: bool = False,
        name: str = 'feature_encoder',
        **kwargs
    ):
        """
        Initialize feature encoder.

        Args:
            encoder_units: List of hidden units for each dense layer
            dropout_rates: List of dropout rates (one per layer, or None for no dropout)
            activation: Activation function to use
            use_batch_norm: Whether to use batch normalization
            name: Layer name
        """
        super(FeatureEncoder, self).__init__(name=name, **kwargs)
        self.encoder_units = encoder_units
        self.dropout_rates = dropout_rates if dropout_rates else [0.0] * len(encoder_units)
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        # Validate inputs
        if len(self.dropout_rates) != len(self.encoder_units):
            raise ValueError("dropout_rates must have the same length as encoder_units")

        # Build layers
        self.dense_layers = []
        self.dropout_layers = []
        self.batch_norm_layers = []

        for i, (units, dropout_rate) in enumerate(zip(self.encoder_units, self.dropout_rates)):
            self.dense_layers.append(
                layers.Dense(units, activation=self.activation, name=f'{name}_dense_{i}')
            )
            if use_batch_norm:
                self.batch_norm_layers.append(
                    layers.BatchNormalization(name=f'{name}_bn_{i}')
                )
            if dropout_rate > 0:
                self.dropout_layers.append(
                    layers.Dropout(dropout_rate, name=f'{name}_dropout_{i}')
                )
            else:
                self.dropout_layers.append(None)

        # Global pooling layer
        self.global_pool = layers.GlobalAveragePooling1D(name=f'{name}_global_pool')

    def call(self, inputs, training=None):
        """
        Apply encoder to input features.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, features)
            training: Training mode flag

        Returns:
            Encoded tensor of shape (batch_size, encoder_units[-1])
        """
        x = inputs

        # Apply dense layers with optional batch norm and dropout
        for i, (dense, dropout, batch_norm) in enumerate(
            zip(self.dense_layers, self.dropout_layers,
                self.batch_norm_layers if self.use_batch_norm else [None] * len(self.dense_layers))
        ):
            x = dense(x)
            if self.use_batch_norm and batch_norm is not None:
                x = batch_norm(x, training=training)
            if dropout is not None:
                x = dropout(x, training=training)

        # Global average pooling
        x = self.global_pool(x)

        return x

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'encoder_units': self.encoder_units,
            'dropout_rates': self.dropout_rates,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm
        })
        return config


# Export all layers for easy importing
__all__ = [
    'MaskedAttentionPooling',
    'MaskedGlobalAveragePooling1D',
    'FeatureEncoder'
]
