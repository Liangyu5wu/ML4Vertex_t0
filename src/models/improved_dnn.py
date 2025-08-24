"""Improved DNN model with physics-informed features."""

import tensorflow as tf
from tensorflow.keras import layers, models

from .dnn_model import DNNModel, MaskedAttentionPooling
from .model_utils import get_loss_function, get_standard_metrics, get_common_custom_objects, load_model_with_fallback
from config.dnn_config import DNNConfig


class ImprovedDNNModel(DNNModel):
    """Improved DNN model with physics-informed features."""
    
    def __init__(self, config: DNNConfig):
        super().__init__(config)
        self.use_physics_features = getattr(config, 'use_physics_informed_features', False)
    
    def build_model_with_mask(self, feature_dim: int, vertex_dim: int, event_feature_dim: int = 0) -> tf.keras.Model:
        """Build improved DNN with physics features."""
        self.config.validate_config()
        
        # Inputs
        cell_inputs = layers.Input(shape=(None, feature_dim), name='cell_sequence')
        vertex_inputs = layers.Input(shape=(vertex_dim,), name='vertex_features')
        mask_inputs = layers.Input(shape=(None,), dtype=tf.bool, name='attention_mask')
        
        if event_feature_dim > 0:
            event_inputs = layers.Input(shape=(event_feature_dim,), name='event_features')
        else:
            event_inputs = None
        
        # Extract physics weights if available
        if self.use_physics_features and feature_dim > 10:
            physics_weights = cell_inputs[:, :, -5]  # cell_weight is 5th from end
        else:
            physics_weights = None
        
        # Stage 1: Cell-level processing
        x = cell_inputs
        for i, units in enumerate(self.config.cell_encoder_units):
            x = layers.Dense(units, activation='relu', name=f'cell_{i}')(x)
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'cell_bn_{i}')(x)
            x = layers.Dropout(self.config.cell_dropout_rate)(x)
        
        # Stage 2: Physics-informed pooling
        if self.use_physics_features and physics_weights is not None:
            # Use physics weights directly
            weights = tf.expand_dims(physics_weights, axis=-1)
            if mask_inputs is not None:
                mask_expanded = tf.expand_dims(tf.cast(mask_inputs, tf.float32), axis=-1)
                weights = weights * mask_expanded
            
            # Normalize weights
            weight_sum = tf.reduce_sum(weights, axis=1, keepdims=True)
            weight_sum = tf.maximum(weight_sum, 1e-8)
            weights = weights / weight_sum
            
            # Apply weighted pooling
            cell_representation = tf.reduce_sum(x * weights, axis=1)
        else:
            # Standard attention pooling
            cell_representation = MaskedAttentionPooling(
                self.config.attention_hidden_units
            )(x, mask=mask_inputs)
        
        # Stage 3: Combine features
        combined_features = [cell_representation]
        
        if self.config.use_spatial_features:
            vertex_processed = layers.Dense(32, activation='relu')(vertex_inputs)
            combined_features.append(vertex_processed)
        
        if event_inputs is not None:
            event_processed = layers.Dense(32, activation='relu')(event_inputs)
            combined_features.append(event_processed)
        
        if len(combined_features) > 1:
            combined = layers.Concatenate()(combined_features)
        else:
            combined = combined_features[0]
        
        # Stage 4: Event-level processing with residuals
        x = combined
        for i, (units, dropout) in enumerate(zip(
            self.config.event_encoder_units,
            self.config.event_dropout_rates
        )):
            residual = x
            x = layers.Dense(units, activation='relu')(x)
            if self.config.use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)
            
            # Residual connection if same dimension
            if residual.shape[-1] == units:
                x = layers.Add()([x, residual])
        
        # Output
        output = layers.Dense(1, name='vertex_time')(x)
        
        # Create model
        inputs = [cell_inputs, vertex_inputs, mask_inputs]
        if event_inputs is not None:
            inputs.append(event_inputs)
        
        model = models.Model(inputs=inputs, outputs=output)
        
        # Compile
        loss_function = get_loss_function(self.config.loss_function, 
                                        getattr(self.config, 'huber_delta', 1.0))
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=1.0
        )
        
        model.compile(optimizer=optimizer, loss=loss_function, metrics=get_standard_metrics())
        self.model = model
        return model
    
    @staticmethod
    def load_model(filepath: str) -> tf.keras.Model:
        """Load a saved improved DNN model."""
        custom_objects = get_common_custom_objects()
        custom_objects.update({'MaskedAttentionPooling': MaskedAttentionPooling})
        return load_model_with_fallback(filepath, custom_objects)