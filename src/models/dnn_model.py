"""Two-stage DNN model implementation for vertex time prediction with attention pooling."""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, Any

from .model_utils import (
    root_mean_squared_error, get_standard_metrics, get_common_custom_objects,
    load_model_with_fallback, get_model_summary_string, count_model_parameters,
    create_dummy_vertex_connection
)
from config.dnn_config import DNNConfig


class MaskedAttentionPooling(layers.Layer):
    """Attention-based pooling with mask support."""
    
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


class DNNModel:
    """Two-stage DNN model for vertex time prediction."""
    
    def __init__(self, config: DNNConfig):
        """
        Initialize DNN model.
        
        Args:
            config: DNN configuration object
        """
        self.config = config
        self.model = None
        
    def _get_loss_function(self):
        """Get loss function based on configuration."""
        if self.config.loss_function == 'huber':
            return tf.keras.losses.Huber(delta=self.config.huber_delta)
        elif self.config.loss_function == 'mse':
            return 'mse'
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
        
    def build_model(self, feature_dim: int, vertex_dim: int) -> tf.keras.Model:
        """
        Build the DNN model architecture (backward compatibility).
        
        Args:
            feature_dim: Dimension of cell features
            vertex_dim: Dimension of vertex features
            
        Returns:
            Compiled Keras model
        """
        return self._build_model_internal(feature_dim, vertex_dim, use_mask=False)
    
    def build_model_with_mask(self, feature_dim: int, vertex_dim: int) -> tf.keras.Model:
        """
        Build the DNN model architecture with attention mask support.
        
        Args:
            feature_dim: Dimension of cell features
            vertex_dim: Dimension of vertex features
            
        Returns:
            Compiled Keras model with mask support
        """
        return self._build_model_internal(feature_dim, vertex_dim, use_mask=True)
    
    def _build_model_internal(self, feature_dim: int, vertex_dim: int, use_mask: bool = False) -> tf.keras.Model:
        """
        Internal method to build the DNN model architecture.
        
        Args:
            feature_dim: Dimension of cell features
            vertex_dim: Dimension of vertex features
            use_mask: Whether to use attention mask support
            
        Returns:
            Compiled Keras model
        """
        # Validate configuration
        self.config.validate_config()
        
        # Cell sequence input (flattened to 2D for DNN processing)
        cell_inputs = layers.Input(shape=(None, feature_dim), name='cell_sequence')
        
        # Vertex features input
        vertex_inputs = layers.Input(shape=(vertex_dim,), name='vertex_features')
        
        # Attention mask input (only for masked version)
        if use_mask:
            mask_inputs = layers.Input(shape=(None,), dtype=tf.bool, name='attention_mask')
            current_mask = mask_inputs
        else:
            mask_inputs = None
            current_mask = None
        
        # Stage 1: Cell-level processing
        x = cell_inputs
        for i, units in enumerate(self.config.cell_encoder_units):
            x = layers.Dense(
                units, 
                activation=self.config.cell_activation,
                name=f'cell_encoder_{i}'
            )(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'cell_bn_{i}')(x)
                
            x = layers.Dropout(self.config.cell_dropout_rate, name=f'cell_dropout_{i}')(x)
        
        # Stage 2: Attention pooling with physics features
        use_physics = getattr(self.config, 'use_physics_informed_features', False)
        if use_physics and feature_dim > 10:
            # Use physics weights (cell_weight is 5th from end in physics features)
            physics_weights = cell_inputs[:, :, -5]
            # Normalize weights within each sequence to avoid gradient issues
            physics_weights = tf.nn.softmax(physics_weights, axis=1)
            weights = tf.expand_dims(physics_weights, axis=-1)
            if current_mask is not None:
                mask_expanded = tf.expand_dims(tf.cast(current_mask, tf.float32), axis=-1)
                weights = weights * mask_expanded
            # Normalize weights
            weight_sum = tf.reduce_sum(weights, axis=1, keepdims=True)
            weight_sum = tf.maximum(weight_sum, 1e-8)
            weights = weights / weight_sum
            # Apply weighted pooling
            cell_representation = tf.reduce_sum(x * weights, axis=1)
        elif self.config.use_attention_pooling:
            cell_representation = MaskedAttentionPooling(
                hidden_units=self.config.attention_hidden_units,
                name='attention_pooling'
            )(x, mask=current_mask)
        else:
            # Fallback to average pooling
            if use_mask and current_mask is not None:
                # Masked average pooling
                mask_expanded = tf.expand_dims(tf.cast(current_mask, tf.float32), axis=-1)
                masked_sum = tf.reduce_sum(x * mask_expanded, axis=1)
                mask_count = tf.reduce_sum(mask_expanded, axis=1)
                mask_count = tf.maximum(mask_count, 1.0)
                cell_representation = masked_sum / mask_count
            else:
                cell_representation = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Stage 3: Vertex feature processing and combination
        if self.config.use_spatial_features:
            # Process vertex features
            vertex_processed = vertex_inputs
            combined = layers.Concatenate(name='combine_features')([cell_representation, vertex_processed])
        else:
            # Create simplified dummy connection for unused vertex features
            vertex_zeros = create_dummy_vertex_connection(vertex_inputs, cell_representation.shape[-1])
            combined = layers.Add(name='add_dummy_vertex')([cell_representation, vertex_zeros])
        
        # Stage 4: Event-level processing
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
        if use_mask:
            model = models.Model(inputs=[cell_inputs, vertex_inputs, mask_inputs], outputs=output)
        else:
            model = models.Model(inputs=[cell_inputs, vertex_inputs], outputs=output)
        
        # Compile model with configurable loss function
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self._get_loss_function(),
            metrics=get_standard_metrics()
        )
        
        self.model = model
        return model
    
    def get_model(self) -> tf.keras.Model:
        """Get the built model."""
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        return self.model
    
    def save_model(self, filepath: str = None):
        """
        Save the model to file.
        
        Args:
            filepath: Path to save the model. If None, uses config path.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")
            
        if filepath is None:
            filepath = self.config.model_path
            
        self.model.save(filepath, save_format='h5')
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> tf.keras.Model:
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded Keras model
        """
        # Combine common custom objects with DNN-specific ones
        custom_objects = get_common_custom_objects()
        custom_objects.update({
            'MaskedAttentionPooling': MaskedAttentionPooling
        })
        
        return load_model_with_fallback(filepath, custom_objects)
    
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        return get_model_summary_string(self.model)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        return count_model_parameters(self.model)
