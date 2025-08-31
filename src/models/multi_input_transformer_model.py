"""Multi-input Transformer model with jets and tracks for vertex time prediction."""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, Any

from .transformer_layers import PositionalEncoding, MultiHeadSelfAttention, TransformerBlock
from .model_utils import (
    root_mean_squared_error, get_loss_function, get_standard_metrics,
    get_common_custom_objects, load_model_with_fallback, get_model_summary_string,
    count_model_parameters
)
from config.transformer_config import TransformerConfig


class MaskedGlobalAveragePooling1D(layers.Layer):
    """Global average pooling with attention mask support."""
    
    def __init__(self, **kwargs):
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)
    
    def call(self, inputs, mask=None):
        if mask is None:
            return tf.reduce_mean(inputs, axis=1)
        
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        
        masked_inputs = inputs * mask
        masked_sum = tf.reduce_sum(masked_inputs, axis=1)
        mask_count = tf.reduce_sum(mask, axis=1)
        mask_count = tf.maximum(mask_count, 1.0)
        
        return masked_sum / mask_count
    
    def get_config(self):
        return super().get_config()


class MultiInputTransformerModel:
    """Multi-input Transformer model for vertex time prediction with jets and tracks."""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.model = None
        
    def build_model(self, feature_dim: int, vertex_dim: int,
                    jet_feature_dim: int, track_feature_dim: int) -> tf.keras.Model:
        """Build multi-input Transformer model with jets and tracks."""
        
        # Validate configuration
        self.config.validate_config()
        
        # Input layers
        cell_inputs = layers.Input(shape=(None, feature_dim), name='cell_sequence')
        vertex_inputs = layers.Input(shape=(vertex_dim,), name='vertex_features')
        jet_inputs = layers.Input(shape=(self.config.max_jets, jet_feature_dim), name='jet_inputs')
        track_inputs = layers.Input(shape=(self.config.max_tracks, track_feature_dim), name='track_inputs')
        mask_inputs = layers.Input(shape=(None,), dtype=tf.bool, name='attention_mask')
        
        # Cell processing with transformer
        x = layers.Dense(self.config.d_model, activation='linear', name='input_projection')(cell_inputs)
        x = PositionalEncoding(max_position=self.config.max_position_encoding, d_model=self.config.d_model)(x)
        
        # Apply transformer blocks
        for i in range(self.config.num_layers):
            x = TransformerBlock(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                dff=self.config.dff,
                rate=self.config.dropout_rate,
                name=f'transformer_block_{i}'
            )(x, mask=mask_inputs)
        
        # Global pooling with mask support
        cell_representation = MaskedGlobalAveragePooling1D(name='masked_global_pool')(x, mask=mask_inputs)
        
        # Jet processing with configurable architecture
        jet_x = jet_inputs
        for i, (units, dropout_rate) in enumerate(zip(self.config.jet_encoder_units, self.config.jet_dropout_rates)):
            jet_x = layers.Dense(units, activation=self.config.jet_activation, name=f'jet_encoder_{i}')(jet_x)
            
            if self.config.jet_use_batch_norm:
                jet_x = layers.BatchNormalization(name=f'jet_bn_{i}')(jet_x)
                
            jet_x = layers.Dropout(dropout_rate, name=f'jet_dropout_{i}')(jet_x)
        jet_representation = layers.GlobalAveragePooling1D(name='jet_global_pool')(jet_x)
        
        # Track processing with configurable architecture
        track_x = track_inputs
        for i, (units, dropout_rate) in enumerate(zip(self.config.track_encoder_units, self.config.track_dropout_rates)):
            track_x = layers.Dense(units, activation=self.config.track_activation, name=f'track_encoder_{i}')(track_x)
            
            if self.config.track_use_batch_norm:
                track_x = layers.BatchNormalization(name=f'track_bn_{i}')(track_x)
                
            track_x = layers.Dropout(dropout_rate, name=f'track_dropout_{i}')(track_x)
        track_representation = layers.GlobalAveragePooling1D(name='track_global_pool')(track_x)
        
        # Feature combination (including vertex features)
        combined = layers.Concatenate(name='combine_all_features')([
            cell_representation, vertex_inputs, jet_representation, track_representation
        ])
        
        # Final prediction layers
        x = combined
        for i, (units, dropout_rate) in enumerate(zip(
            self.config.final_dense_units, 
            self.config.final_dropout_rates
        )):
            x = layers.Dense(units, activation='relu', name=f'final_dense_{i}')(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'final_bn_{i}')(x)
                
            x = layers.Dropout(dropout_rate, name=f'final_dropout_{i}')(x)
        
        # Output layer
        output = layers.Dense(1, name='vertex_time')(x)
        
        # Create model
        model = models.Model(
            inputs=[cell_inputs, vertex_inputs, jet_inputs, track_inputs, mask_inputs], 
            outputs=output,
            name='multi_input_transformer'
        )

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
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        return self.model
    
    def save_model(self, filepath: str = None):
        if self.model is None:
            raise ValueError("Model has not been built yet.")
        
        if filepath is None:
            filepath = self.config.model_path
        
        self.model.save(filepath, save_format='h5')
        print(f"Multi-input Transformer model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> tf.keras.Model:
        custom_objects = get_common_custom_objects()
        custom_objects['MaskedGlobalAveragePooling1D'] = MaskedGlobalAveragePooling1D
        custom_objects['PositionalEncoding'] = PositionalEncoding
        custom_objects['MultiHeadSelfAttention'] = MultiHeadSelfAttention
        custom_objects['TransformerBlock'] = TransformerBlock
        
        return load_model_with_fallback(filepath, custom_objects)
    
    def get_model_summary(self) -> str:
        if self.model is None:
            return "Model not built yet"
        return get_model_summary_string(self.model)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        return count_model_parameters(self.model)