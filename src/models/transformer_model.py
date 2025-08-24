"""Transformer model implementation for vertex time prediction with mask support."""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, Any

from .transformer_layers import PositionalEncoding, MultiHeadSelfAttention, TransformerBlock
from .model_utils import (
    root_mean_squared_error, get_loss_function, get_standard_metrics,
    get_common_custom_objects, load_model_with_fallback, get_model_summary_string,
    count_model_parameters, create_dummy_vertex_connection
)
from config.transformer_config import TransformerConfig


class MaskedGlobalAveragePooling1D(layers.Layer):
    """Global average pooling with attention mask support."""
    
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


class TransformerModel:
    """Transformer model for vertex time prediction with mask support."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize transformer model.
        
        Args:
            config: Transformer configuration object
        """
        self.config = config
        self.model = None
        
    def build_model(self, feature_dim: int, vertex_dim: int) -> tf.keras.Model:
        """
        Build the transformer model architecture (backward compatibility).
        
        Args:
            feature_dim: Dimension of cell features (original cell features only)
            vertex_dim: Dimension of vertex features
            
        Returns:
            Compiled Keras model
        """
        return self._build_model_internal(feature_dim, vertex_dim, use_mask=False)
    
    def build_model_with_mask(self, feature_dim: int, vertex_dim: int) -> tf.keras.Model:
        """
        Build the transformer model architecture with attention mask support.
        
        Args:
            feature_dim: Dimension of cell features (original cell features only)
            vertex_dim: Dimension of vertex features
            
        Returns:
            Compiled Keras model with mask support
        """
        return self._build_model_internal(feature_dim, vertex_dim, use_mask=True)
    
    def _build_model_internal(self, feature_dim: int, vertex_dim: int, use_mask: bool = False) -> tf.keras.Model:
        """
        Internal method to build the transformer model architecture.
        
        Args:
            feature_dim: Dimension of cell features
            vertex_dim: Dimension of vertex features
            use_mask: Whether to use attention mask support
            
        Returns:
            Compiled Keras model
        """
        # Validate configuration
        self.config.validate_config()
        
        # Cell sequence input (variable length, will be padded during batching)
        cell_inputs = layers.Input(shape=(None, feature_dim), name='cell_sequence')
        
        # Vertex features input (always present for interface consistency)
        vertex_inputs = layers.Input(shape=(vertex_dim,), name='vertex_features')
        
        # Attention mask input (only for masked version)
        if use_mask:
            mask_inputs = layers.Input(shape=(None,), dtype=tf.bool, name='attention_mask')
            current_mask = mask_inputs
        else:
            mask_inputs = None
            current_mask = None
        
        # Project cell features to d_model dimensions
        x = layers.Dense(self.config.d_model)(cell_inputs)
        
        # Add positional encoding
        x = PositionalEncoding(self.config.max_position, self.config.d_model)(x)
        
        # Stack transformer blocks
        for i in range(self.config.num_transformer_blocks):
            x = TransformerBlock(
                self.config.d_model, 
                self.config.num_heads, 
                self.config.dff, 
                rate=self.config.dropout_rate
            )(x, mask=current_mask)
        
        # Global average pooling (with or without mask)
        if use_mask:
            cell_representation = MaskedGlobalAveragePooling1D()(x, mask=current_mask)
        else:
            cell_representation = layers.GlobalAveragePooling1D()(x)
        
        # Conditionally process vertex features
        if self.config.use_spatial_features:
            # Use vertex features normally
            vertex_dense = layers.Dense(
                self.config.vertex_dense_units, 
                activation='relu'
            )(vertex_inputs)
            # Combine cell and vertex representations
            combined = layers.Concatenate()([cell_representation, vertex_dense])
        else:
            # Create simplified dummy connection for unused vertex features
            vertex_zeros = create_dummy_vertex_connection(vertex_inputs, self.config.d_model)
            combined = layers.Add()([cell_representation, vertex_zeros])
        
        # Final prediction layers
        x = combined
        for i, (units, dropout_rate) in enumerate(zip(
            self.config.final_dense_units, 
            self.config.final_dropout_rates
        )):
            x = layers.Dense(units, activation='relu')(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization()(x)
                
            x = layers.Dropout(dropout_rate)(x)
        
        # Output layer
        output = layers.Dense(1, name='vertex_time')(x)
        
        # Create model (inputs depend on whether mask is used)
        if use_mask:
            model = models.Model(inputs=[cell_inputs, vertex_inputs, mask_inputs], outputs=output)
        else:
            model = models.Model(inputs=[cell_inputs, vertex_inputs], outputs=output)

        loss_function = get_loss_function(self.config.loss_function, self.config.huber_delta)
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
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
            
        # Use save_weights and save architecture separately for better compatibility
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
        # Combine common custom objects with transformer-specific ones
        custom_objects = get_common_custom_objects()
        custom_objects.update({
            'PositionalEncoding': PositionalEncoding,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'TransformerBlock': TransformerBlock,
            'MaskedGlobalAveragePooling1D': MaskedGlobalAveragePooling1D
        })
        
        return load_model_with_fallback(filepath, custom_objects)
    
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        return get_model_summary_string(self.model)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        return count_model_parameters(self.model)
