"""Baseline-guided DNN model for vertex time prediction with residual learning."""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, Any
import numpy as np

from .model_utils import (
    root_mean_squared_error, get_standard_metrics, get_common_custom_objects,
    load_model_with_fallback, get_model_summary_string, count_model_parameters
)
from config.dnn_config import DNNConfig


class BaselineGuidedDNN:
    """Baseline-guided DNN model using residual learning approach."""
    
    def __init__(self, config: DNNConfig):
        """
        Initialize baseline-guided DNN model.
        
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
        Build the baseline-guided DNN model architecture.
        
        Args:
            feature_dim: Dimension of cell features
            vertex_dim: Dimension of vertex features
            
        Returns:
            Compiled Keras model
        """
        # Validate configuration
        self.config.validate_config()
        
        # Cell features input
        cell_inputs = layers.Input(shape=(None, feature_dim), name='cell_sequence')
        
        # Vertex features input
        vertex_inputs = layers.Input(shape=(vertex_dim,), name='vertex_features')
        
        # Baseline prediction input (single value per event)
        baseline_inputs = layers.Input(shape=(1,), name='baseline_prediction')
        
        # Process cell features through simple dense layers
        x = cell_inputs
        
        # Apply dense layers to each cell
        for i, units in enumerate([128, 64, 32]):
            x = layers.Dense(units, activation='relu', name=f'cell_dense_{i}')(x)
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'cell_bn_{i}')(x)
            x = layers.Dropout(0.1, name=f'cell_dropout_{i}')(x)
        
        # Simple average pooling (no attention needed)
        cell_representation = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Combine all features
        if self.config.use_spatial_features:
            combined = layers.Concatenate(name='combine_features')([
                cell_representation, 
                vertex_inputs, 
                baseline_inputs
            ])
        else:
            # Create zero vertex connection for consistency
            # Use a dense layer with zero weights instead of Lambda to avoid tf scoping issues
            vertex_zeros = layers.Dense(
                1, 
                use_bias=False,
                kernel_initializer='zeros',
                trainable=False,
                name='vertex_zeros'
            )(vertex_inputs)
            combined = layers.Concatenate(name='combine_features')([
                cell_representation,
                vertex_zeros, 
                baseline_inputs
            ])
        
        # Event-level processing (3-4 layers as requested)
        x = combined
        event_units = [128, 64, 32, 16]
        dropout_rates = [0.2, 0.2, 0.1, 0.1]
        
        for i, (units, dropout_rate) in enumerate(zip(event_units, dropout_rates)):
            x = layers.Dense(units, activation='relu', name=f'event_dense_{i}')(x)
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'event_bn_{i}')(x)
            x = layers.Dropout(dropout_rate, name=f'event_dropout_{i}')(x)
        
        # Output residual correction
        # This layer outputs a residual value (can be positive, negative, or zero)
        residual = layers.Dense(1, name='residual_output')(x)
        
        # Final prediction: baseline + residual
        # The model optimizes: |final_prediction - true_vertex_time|^2
        # where final_prediction = baseline_prediction + model_residual
        # If residual ≈ 0, performance equals baseline method
        # If residual ≠ 0, model learns improvements over baseline
        final_output = layers.Add(name='final_prediction')([baseline_inputs, residual])
        
        # Create model
        model = models.Model(
            inputs=[cell_inputs, vertex_inputs, baseline_inputs], 
            outputs=final_output,
            name='baseline_guided_dnn'
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
        """Get the built model."""
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        return self.model
    
    def predict(self, cell_sequences, vertex_features, baseline_predictions):
        """
        Make predictions using the model.
        
        Args:
            cell_sequences: Cell sequence data
            vertex_features: Vertex feature data
            baseline_predictions: Baseline prediction values
            
        Returns:
            Final time predictions (baseline + residual)
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")
        
        return self.model.predict([cell_sequences, vertex_features, baseline_predictions])
    
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
        custom_objects = get_common_custom_objects()
        return load_model_with_fallback(filepath, custom_objects)
    
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        return get_model_summary_string(self.model)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        return count_model_parameters(self.model)