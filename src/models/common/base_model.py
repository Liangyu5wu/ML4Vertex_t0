"""Base model class for all vertex time prediction models."""

import tensorflow as tf
from typing import Dict, Any
from abc import ABC, abstractmethod

from .model_utils import get_model_summary_string, get_standard_metrics


class BaseVertexModel(ABC):
    """Abstract base class for all vertex time prediction models.

    This class provides common functionality shared across all model types:
    - Loss function configuration
    - Model getter methods
    - Model summary utilities

    Subclasses must implement the build_model methods for their specific architectures.
    """

    def __init__(self, config):
        """
        Initialize base model.

        Args:
            config: Configuration object (DNNConfig, TransformerConfig, etc.)
        """
        self.config = config
        self.model = None

    def _get_loss_function(self):
        """
        Get loss function based on configuration.

        Returns:
            Loss function (Huber or MSE)
        """
        if self.config.loss_function == 'huber':
            return tf.keras.losses.Huber(delta=self.config.huber_delta)
        elif self.config.loss_function == 'mse':
            return 'mse'
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")

    def get_model(self) -> tf.keras.Model:
        """
        Get the built model.

        Returns:
            Compiled Keras model

        Raises:
            ValueError: If model has not been built yet
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        return self.model

    def get_model_summary(self) -> str:
        """
        Get a string representation of the model summary.

        Returns:
            Model summary as string

        Raises:
            ValueError: If model has not been built yet
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        return get_model_summary_string(self.model)

    @abstractmethod
    def build_model(self, *args, **kwargs) -> tf.keras.Model:
        """
        Build the model architecture.

        This method must be implemented by subclasses to define their specific architecture.

        Returns:
            Compiled Keras model
        """
        pass

    def compile_model(self, model: tf.keras.Model, metrics: list = None) -> tf.keras.Model:
        """
        Compile the model with loss function and metrics.

        Args:
            model: Keras model to compile
            metrics: List of metrics (optional, will use standard metrics if None)

        Returns:
            Compiled Keras model
        """
        loss_fn = self._get_loss_function()

        # Use provided metrics or import standard ones
        if metrics is None:
            from .model_utils import get_standard_metrics
            metrics = get_standard_metrics()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=loss_fn,
            metrics=metrics
        )

        return model


# Export for easy importing
__all__ = ['BaseVertexModel']
