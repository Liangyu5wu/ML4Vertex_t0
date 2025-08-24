"""Shared utilities for model implementations."""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import models
from typing import Dict, Any


def root_mean_squared_error(y_true, y_pred):
    """Custom RMSE metric for both Transformer and DNN models."""
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def get_loss_function(loss_name: str, delta: float = 1.0):
    """
    Get loss function based on configuration.
    
    Args:
        loss_name: Name of the loss function ('mse' or 'huber')
        delta: Delta parameter for Huber loss
        
    Returns:
        Loss function for compilation
    """
    if loss_name == 'mse':
        return 'mse'
    elif loss_name == 'huber':
        return tf.keras.losses.Huber(delta=delta)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def get_standard_metrics():
    """Get standard metrics used by both model types."""
    return ['mae', root_mean_squared_error, tf.keras.metrics.MeanSquaredError(name='mse_metric')]


def get_common_custom_objects():
    """Get common custom objects for model loading."""
    return {
        'root_mean_squared_error': root_mean_squared_error,
        'mse': tf.keras.losses.MeanSquaredError(),
        'mae': tf.keras.metrics.MeanAbsoluteError(),
        'mse_metric': tf.keras.metrics.MeanSquaredError(name='mse_metric'),
        'Huber': tf.keras.losses.Huber
    }


def load_model_with_fallback(filepath: str, custom_objects: Dict[str, Any] = None) -> tf.keras.Model:
    """
    Load model with fallback error handling for both .h5 and .keras formats.
    
    Args:
        filepath: Path to the saved model
        custom_objects: Custom objects dictionary for loading
        
    Returns:
        Loaded Keras model
    """
    if custom_objects is None:
        custom_objects = get_common_custom_objects()
    
    try:
        model = models.load_model(filepath, custom_objects=custom_objects)
        print(f"Model loaded successfully with compilation intact.")
        return model
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        
        # Try alternative loading method for .h5 files
        if filepath.endswith('.h5'):
            print("Attempting alternative loading method for .h5 file...")
            try:
                model = tf.keras.models.load_model(filepath, custom_objects=custom_objects, compile=False)
                print("Model architecture loaded successfully. Re-compiling...")
                
                # Re-compile the model with standard configuration
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                model.compile(
                    optimizer=optimizer,
                    loss='mse',
                    metrics=get_standard_metrics()
                )
                print("Model re-compiled successfully.")
                return model
            except Exception as e2:
                print(f"Alternative loading method also failed: {e2}")
                raise e2
        else:
            raise e


def get_model_summary_string(model: tf.keras.Model) -> str:
    """Get model summary as string."""
    if model is None:
        raise ValueError("Model has not been built yet.")
        
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)


def count_model_parameters(model: tf.keras.Model) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with parameter counts
    """
    if model is None:
        raise ValueError("Model has not been built yet.")
        
    total_params = model.count_params()
    trainable_params = sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def create_dummy_vertex_connection(vertex_inputs, target_dim: int):
    """
    Create a simplified dummy connection for unused vertex features.
    
    Args:
        vertex_inputs: Vertex input tensor
        target_dim: Target dimension to match
        
    Returns:
        Zero tensor with correct dimensions
    """
    from tensorflow.keras import layers
    
    # Simple approach: create zeros with the right shape
    vertex_processed = layers.Dense(target_dim, use_bias=False)(vertex_inputs)
    vertex_zeros = layers.Lambda(lambda x: x * 0, name='dummy_vertex_zeros')(vertex_processed)
    return vertex_zeros