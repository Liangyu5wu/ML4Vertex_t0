"""Improved Transformer model with physics-informed features and weighted attention."""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, Any
import numpy as np

from .transformer_layers import PositionalEncoding, TransformerBlock
from .model_utils import (
    root_mean_squared_error, get_loss_function, get_standard_metrics,
    get_common_custom_objects, load_model_with_fallback, get_model_summary_string,
    count_model_parameters, create_dummy_vertex_connection
)
from config.transformer_config import TransformerConfig


class PhysicsInformedAttention(layers.Layer):
    """Attention layer that incorporates physics-based weights."""
    
    def __init__(self, **kwargs):
        super(PhysicsInformedAttention, self).__init__(**kwargs)
    
    def call(self, inputs, physics_weights=None, mask=None):
        """
        Apply physics-informed attention.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, features)
            physics_weights: Physics-based weights of shape (batch_size, seq_len)
            mask: Attention mask of shape (batch_size, seq_len)
        
        Returns:
            Weighted tensor of shape (batch_size, features)
        """
        if physics_weights is not None:
            # Use physics weights as attention weights
            attention_weights = physics_weights
            
            # Expand dimensions for broadcasting
            attention_weights = tf.expand_dims(attention_weights, axis=-1)
            
            # Apply mask if provided
            if mask is not None:
                mask_expanded = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
                attention_weights = attention_weights * mask_expanded
            
            # Normalize weights
            weight_sum = tf.reduce_sum(attention_weights, axis=1, keepdims=True)
            weight_sum = tf.maximum(weight_sum, 1e-8)  # Avoid division by zero
            attention_weights = attention_weights / weight_sum
            
            # Apply weighted pooling
            weighted_inputs = inputs * attention_weights
            output = tf.reduce_sum(weighted_inputs, axis=1)
            
            return output
        else:
            # Fallback to standard global average pooling
            if mask is not None:
                mask_expanded = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
                masked_sum = tf.reduce_sum(inputs * mask_expanded, axis=1)
                mask_count = tf.reduce_sum(mask_expanded, axis=1)
                mask_count = tf.maximum(mask_count, 1.0)
                return masked_sum / mask_count
            else:
                return tf.reduce_mean(inputs, axis=1)
    
    def get_config(self):
        return super().get_config()


class ImprovedTransformerModel:
    """Improved Transformer model with physics-informed features."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize improved transformer model.
        
        Args:
            config: Transformer configuration object
        """
        self.config = config
        self.model = None
        
        # Check if physics-informed features are enabled
        self.use_physics_features = getattr(config, 'use_physics_informed_features', False)
        
    def build_model_with_mask(self, feature_dim: int, vertex_dim: int, event_feature_dim: int = 0) -> tf.keras.Model:
        """
        Build the improved transformer model with physics-informed features.
        
        Args:
            feature_dim: Dimension of cell features (including physics features)
            vertex_dim: Dimension of vertex features  
            event_feature_dim: Dimension of event-level features
            
        Returns:
            Compiled Keras model with mask support
        """
        # Validate configuration
        self.config.validate_config()
        
        # Cell sequence input
        cell_inputs = layers.Input(shape=(None, feature_dim), name='cell_sequence')
        
        # Vertex features input
        vertex_inputs = layers.Input(shape=(vertex_dim,), name='vertex_features')
        
        # Attention mask input
        mask_inputs = layers.Input(shape=(None,), dtype=tf.bool, name='attention_mask')
        
        # Event-level features input (optional)
        if event_feature_dim > 0:
            event_inputs = layers.Input(shape=(event_feature_dim,), name='event_features')
        else:
            event_inputs = None
        
        # Extract physics weights if available
        if self.use_physics_features and feature_dim > 10:  # Assuming physics features are added
            # Assume 'cell_weight' is at index -5 (6th from end) in the enhanced features
            weight_index = -5
            physics_weights = cell_inputs[:, :, weight_index]
        else:
            physics_weights = None
        
        # Project cell features to d_model dimensions
        x = layers.Dense(self.config.d_model, name='cell_projection')(cell_inputs)
        
        # Add positional encoding
        x = PositionalEncoding(self.config.max_position, self.config.d_model, name='pos_encoding')(x)
        
        # Apply dropout to input
        x = layers.Dropout(self.config.dropout_rate, name='input_dropout')(x)
        
        # Stack transformer blocks
        for i in range(self.config.num_transformer_blocks):
            x = TransformerBlock(
                self.config.d_model, 
                self.config.num_heads, 
                self.config.dff, 
                rate=self.config.dropout_rate,
                name=f'transformer_block_{i}'
            )(x, mask=mask_inputs)
        
        # Apply physics-informed attention pooling
        if self.use_physics_features:
            cell_representation = PhysicsInformedAttention(name='physics_attention')(
                x, physics_weights=physics_weights, mask=mask_inputs
            )
        else:
            # Standard masked global average pooling
            mask_expanded = tf.expand_dims(tf.cast(mask_inputs, tf.float32), axis=-1)
            masked_sum = tf.reduce_sum(x * mask_expanded, axis=1)
            mask_count = tf.reduce_sum(mask_expanded, axis=1)
            mask_count = tf.maximum(mask_count, 1.0)
            cell_representation = masked_sum / mask_count
        
        # Process vertex features if needed
        if self.config.use_spatial_features:
            vertex_dense = layers.Dense(
                self.config.vertex_dense_units, 
                activation='relu',
                name='vertex_processing'
            )(vertex_inputs)
            combined_features = [cell_representation, vertex_dense]
        else:
            vertex_zeros = create_dummy_vertex_connection(vertex_inputs, self.config.d_model)
            combined_features = [layers.Add(name='add_dummy_vertex')([cell_representation, vertex_zeros])]
        
        # Add event-level features if provided
        if event_inputs is not None:
            event_processed = layers.Dense(
                self.config.d_model // 4, 
                activation='relu',
                name='event_processing'
            )(event_inputs)
            combined_features.append(event_processed)
        
        # Combine all features
        if len(combined_features) > 1:
            combined = layers.Concatenate(name='combine_all_features')(combined_features)
        else:
            combined = combined_features[0]
        
        # Final prediction layers with residual connections
        x = combined
        for i, (units, dropout_rate) in enumerate(zip(
            self.config.final_dense_units, 
            self.config.final_dropout_rates
        )):
            # Residual connection for layers of same dimension
            residual = x
            
            x = layers.Dense(units, activation='relu', name=f'final_dense_{i}')(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'final_bn_{i}')(x)
                
            x = layers.Dropout(dropout_rate, name=f'final_dropout_{i}')(x)
            
            # Add residual connection if dimensions match
            if residual.shape[-1] == units:
                x = layers.Add(name=f'residual_{i}')([x, residual])
        
        # Output layer
        output = layers.Dense(1, name='vertex_time')(x)
        
        # Create model
        inputs = [cell_inputs, vertex_inputs, mask_inputs]
        if event_inputs is not None:
            inputs.append(event_inputs)
            
        model = models.Model(inputs=inputs, outputs=output)
        
        # Compile with improved settings
        loss_function = get_loss_function(self.config.loss_function, getattr(self.config, 'huber_delta', 1.0))
        
        # Use different optimizers based on configuration
        if hasattr(self.config, 'optimizer') and self.config.optimizer == 'adamw':
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.config.learning_rate,
                weight_decay=1e-5
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate,
                clipnorm=1.0  # Gradient clipping
            )
        
        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=get_standard_metrics()
        )
        
        self.model = model
        return model
    
    def get_model(self) -> tf.keras.Model:
        """Get the built model."""
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model_with_mask() first.")
        return self.model
    
    def save_model(self, filepath: str = None):
        """Save the model to file."""
        if self.model is None:
            raise ValueError("Model has not been built yet.")
            
        if filepath is None:
            filepath = self.config.model_path
            
        self.model.save(filepath, save_format='h5')
        print(f"Improved model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> tf.keras.Model:
        """Load a saved improved model."""
        custom_objects = get_common_custom_objects()
        custom_objects.update({
            'PositionalEncoding': PositionalEncoding,
            'TransformerBlock': TransformerBlock,
            'PhysicsInformedAttention': PhysicsInformedAttention
        })
        
        return load_model_with_fallback(filepath, custom_objects)
    
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        return get_model_summary_string(self.model)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        return count_model_parameters(self.model)