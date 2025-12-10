"""Training utilities for vertex time prediction models."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from typing import Dict, Any, Tuple, Optional, Union

from config.base_config import BaseConfig
from src.models.transformer import TransformerModel
from src.models.dnn import DNNModel


class Trainer:
    """Handle model training with callbacks and monitoring."""
    
    def __init__(self, config: BaseConfig, model: Union[TransformerModel, DNNModel]):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            model: Model to train (either TransformerModel or DNNModel)
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
            )
        ]
        
        # ModelCheckpoint with version compatibility handling
        try:
            checkpoint = callbacks.ModelCheckpoint(
                filepath=self.config.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                save_format='h5'
            )
        except TypeError:
            # Fallback for versions that don't support save_format
            checkpoint = callbacks.ModelCheckpoint(
                filepath=self.config.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        
        callbacks_list.append(checkpoint)
        
        callbacks_list.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=getattr(self.config, 'lr_reduction_factor', 0.5),
                patience=self.config.lr_patience,
                min_lr=self.config.min_lr,
                verbose=1
            )
        )
        
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
        
        # Load model using the appropriate class method
        if isinstance(self.model, TransformerModel):
            keras_model = TransformerModel.load_model(self.config.model_path)
        else:
            from src.models.dnn_model import DNNModel
            keras_model = DNNModel.load_model(self.config.model_path)
        
        # Update the model in our model wrapper
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
        
        # Get sample batch to detect dataset format
        sample_batch = next(iter(train_dataset))
        input_keys = list(sample_batch[0].keys())
        
        # Detect dataset format
        if 'hgtd_track_inputs' in input_keys:
            # HGTD-only dataset format
            primary_key = 'hgtd_track_inputs'
            vertex_key = 'vertex_inputs'
            is_multi_input = False
            is_hgtd_only = True
            print("Detected HGTD-only dataset format")
        elif 'cell_inputs' in input_keys:
            # Multi-input dataset format
            primary_key = 'cell_inputs'
            vertex_key = 'vertex_inputs'
            is_multi_input = True
            is_hgtd_only = False
            print("Detected multi-input dataset format")
        else:
            # Regular dataset format
            primary_key = 'cell_sequence'
            vertex_key = 'vertex_features'
            is_multi_input = False
            is_hgtd_only = False
            print("Detected regular dataset format")

        # Count batches and samples
        train_batches = 0
        train_samples = 0
        for batch in train_dataset:
            train_batches += 1
            train_samples += batch[0][primary_key].shape[0]

        val_batches = 0
        val_samples = 0
        for batch in val_dataset:
            val_batches += 1
            val_samples += batch[0][primary_key].shape[0]

        # Get shapes from sample batch
        primary_shape = sample_batch[0][primary_key].shape
        vertex_shape = sample_batch[0][vertex_key].shape
        target_shape = sample_batch[1].shape

        stats = {
            'train_batches': train_batches,
            'train_samples': train_samples,
            'val_batches': val_batches,
            'val_samples': val_samples,
            'primary_input_shape': primary_shape,
            'vertex_features_shape': vertex_shape,
            'target_shape': target_shape,
            'is_multi_input': is_multi_input,
            'is_hgtd_only': is_hgtd_only
        }

        print(f"Training dataset: {train_batches} batches, {train_samples} samples")
        print(f"Validation dataset: {val_batches} batches, {val_samples} samples")
        if is_hgtd_only:
            print(f"HGTD track sequence shape: {primary_shape}")
        else:
            print(f"Cell sequence shape: {primary_shape}")
        print(f"Vertex features shape: {vertex_shape}")
        
        if is_multi_input:
            jet_shape = sample_batch[0]['jet_inputs'].shape
            track_shape = sample_batch[0]['track_inputs'].shape
            mask_shape = sample_batch[0]['attention_mask'].shape
            print(f"Jet inputs shape: {jet_shape}")
            print(f"Track inputs shape: {track_shape}")
            print(f"Attention mask shape: {mask_shape}")
            stats.update({
                'jet_inputs_shape': jet_shape,
                'track_inputs_shape': track_shape,
                'attention_mask_shape': mask_shape
            })
            
        print(f"Target shape: {target_shape}")
        
        return stats
