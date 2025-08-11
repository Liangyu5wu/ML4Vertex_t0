"""Training utilities for vertex time prediction models."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from typing import Dict, Any, Tuple, Optional

from config.base_config import BaseConfig
from src.models.transformer_model import TransformerModel


class Trainer:
    """Handle model training with callbacks and monitoring."""
    
    def __init__(self, config: BaseConfig, model: TransformerModel):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            model: Model to train
        """
        self.config = config
        self.model = model
        self.history = None
        
    def get_callbacks(self) -> list:
        """
        Create training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=self.config.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                save_format='h5'  # Explicitly use h5 format for better compatibility
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=getattr(self.config, 'lr_reduction_factor', 0.5),
                patience=self.config.lr_patience,
                min_lr=self.config.min_lr,
                verbose=1
            )
        ]
        
        return callbacks_list
    
    def train(
        self, 
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Ensure model directory exists
        self.config.create_directories()
        
        # Save configuration
        self.config.save_config()
        
        print(f"Starting training for {self.config.epochs} epochs...")
        print(f"Model will be saved to: {self.config.model_path}")
        
        # Get model summary
        keras_model = self.model.get_model()
        print("\nModel Summary:")
        print(self.model.get_model_summary())
        
        # Print parameter count
        param_count = self.model.count_parameters()
        print(f"\nModel Parameters:")
        print(f"  Total: {param_count['total']:,}")
        print(f"  Trainable: {param_count['trainable']:,}")
        print(f"  Non-trainable: {param_count['non_trainable']:,}")
        
        # Train the model
        self.history = keras_model.fit(
            train_dataset,
            epochs=self.config.epochs,
            validation_data=val_dataset,
            callbacks=self.get_callbacks(),
            verbose=verbose
        )
        
        print(f"Training completed. Best model saved to: {self.config.model_path}")
        
        return self.history
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training results.
        
        Returns:
            Dictionary with training summary
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet.")
        
        history_dict = self.history.history
        
        # Find best epoch
        best_epoch = np.argmin(history_dict['val_loss'])
        
        summary = {
            'total_epochs': len(history_dict['loss']),
            'best_epoch': best_epoch + 1,  # 1-indexed
            'best_val_loss': history_dict['val_loss'][best_epoch],
            'best_val_mae': history_dict['val_mae'][best_epoch],
            'final_train_loss': history_dict['loss'][-1],
            'final_train_mae': history_dict['mae'][-1],
            'final_val_loss': history_dict['val_loss'][-1],
            'final_val_mae': history_dict['val_mae'][-1]
        }
        
        # Add RMSE if available
        if 'val_root_mean_squared_error' in history_dict:
            summary['best_val_rmse'] = history_dict['val_root_mean_squared_error'][best_epoch]
            summary['final_val_rmse'] = history_dict['val_root_mean_squared_error'][-1]
        
        return summary
    
    def save_training_history(self, filepath: Optional[str] = None):
        """
        Save training history to file.
        
        Args:
            filepath: Path to save history. If None, saves to model directory.
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet.")
        
        if filepath is None:
            filepath = os.path.join(self.config.model_dir, "training_history.npz")
        
        # Convert history to numpy arrays and save
        history_dict = {key: np.array(values) for key, values in self.history.history.items()}
        np.savez(filepath, **history_dict)
        
        print(f"Training history saved to: {filepath}")
    
    @staticmethod
    def load_training_history(filepath: str) -> Dict[str, np.ndarray]:
        """
        Load training history from file.
        
        Args:
            filepath: Path to history file
            
        Returns:
            Dictionary with training history
        """
        loaded = np.load(filepath)
        return {key: loaded[key] for key in loaded.files}
    
    def resume_training(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        additional_epochs: int,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Resume training from a saved model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            additional_epochs: Number of additional epochs to train
            verbose: Verbosity level
            
        Returns:
            Training history for additional epochs
        """
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"No saved model found at {self.config.model_path}")
        
        print(f"Loading model from {self.config.model_path}")
        keras_model = TransformerModel.load_model(self.config.model_path)
        
        # Update the model in our TransformerModel wrapper
        self.model.model = keras_model
        
        print(f"Resuming training for {additional_epochs} additional epochs...")
        
        # Continue training
        additional_history = keras_model.fit(
            train_dataset,
            epochs=additional_epochs,
            validation_data=val_dataset,
            callbacks=self.get_callbacks(),
            verbose=verbose
        )
        
        return additional_history
    
    def validate_training_data(
        self, 
        train_dataset: tf.data.Dataset, 
        val_dataset: tf.data.Dataset
    ) -> Dict[str, Any]:
        """
        Validate training datasets and return statistics.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        print("Validating training datasets...")
        
        # Count batches and samples
        train_batches = 0
        train_samples = 0
        for batch in train_dataset:
            train_batches += 1
            train_samples += batch[0]['cell_sequence'].shape[0]
        
        val_batches = 0
        val_samples = 0
        for batch in val_dataset:
            val_batches += 1
            val_samples += batch[0]['cell_sequence'].shape[0]
        
        # Get sample batch to check shapes
        sample_batch = next(iter(train_dataset))
        cell_shape = sample_batch[0]['cell_sequence'].shape
        vertex_shape = sample_batch[0]['vertex_features'].shape
        target_shape = sample_batch[1].shape
        
        stats = {
            'train_batches': train_batches,
            'train_samples': train_samples,
            'val_batches': val_batches,
            'val_samples': val_samples,
            'cell_sequence_shape': cell_shape,
            'vertex_features_shape': vertex_shape,
            'target_shape': target_shape
        }
        
        print(f"Training dataset: {train_batches} batches, {train_samples} samples")
        print(f"Validation dataset: {val_batches} batches, {val_samples} samples")
        print(f"Cell sequence shape: {cell_shape}")
        print(f"Vertex features shape: {vertex_shape}")
        print(f"Target shape: {target_shape}")
        
        return stats
