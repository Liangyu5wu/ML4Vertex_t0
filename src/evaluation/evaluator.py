"""Model evaluation utilities."""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, List, Optional

from config.base_config import BaseConfig
from src.models.transformer_model import TransformerModel
from src.data.data_processor import DataProcessor


class Evaluator:
    """Handle model evaluation and metrics computation."""
    
    def __init__(self, config: BaseConfig):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
    def evaluate_model(
        self,
        model: tf.keras.Model,
        test_dataset: tf.data.Dataset,
        verbose: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Args:
            model: Trained model
            test_dataset: Test dataset
            verbose: Verbosity level
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating model on test dataset...")
        
        # Evaluate using Keras evaluate method
        test_results = model.evaluate(test_dataset, verbose=verbose)
        
        # Get metric names
        metric_names = model.metrics_names
        
        # Create results dictionary
        keras_results = dict(zip(metric_names, test_results))
        
        print(f"Test Results:")
        for metric, value in keras_results.items():
            print(f"  {metric}: {value:.6f}")
        
        return keras_results
    
    def predict_and_evaluate(
        self,
        model: tf.keras.Model,
        cell_sequences: List[List[List[float]]],
        vertex_features: np.ndarray,
        vertex_times: np.ndarray,
        data_processor: DataProcessor
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Make predictions and compute detailed evaluation metrics.
        
        Args:
            model: Trained model
            cell_sequences: Test cell sequences
            vertex_features: Test vertex features
            vertex_times: True vertex times
            data_processor: Data processor for batch creation
            
        Returns:
            Tuple of (predictions, metrics_dict)
        """
        print("Making predictions on test data...")
        
        # Make predictions in batches
        y_pred_list = []
        y_true_list = []
        
        for batch_data, batch_true in data_processor.create_prediction_batches(
            cell_sequences, vertex_features, vertex_times
        ):
            batch_pred = model.predict(batch_data, verbose=0)
            y_pred_list.extend(batch_pred.flatten())
            y_true_list.extend(batch_true.flatten())
        
        y_pred = np.array(y_pred_list)
        y_true = np.array(y_true_list)
        
        print(f"Predictions completed. Test set size: {len(y_true)}")
        
        # Compute detailed metrics
        metrics = self.compute_metrics(y_true, y_pred)
        
        return y_pred, metrics
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with computed metrics
        """
        # Basic regression metrics
        mse = np.mean((y_pred - y_true) ** 2)
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Correlation coefficient
        correlation_matrix = np.corrcoef(y_true, y_pred)
        correlation = correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0
        
        # Error statistics
        errors = y_pred - y_true
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        error_median = np.median(errors)
        
        # Percentile-based metrics
        abs_errors = np.abs(errors)
        mae_50 = np.percentile(abs_errors, 50)  # Median absolute error
        mae_90 = np.percentile(abs_errors, 90)  # 90th percentile absolute error
        mae_95 = np.percentile(abs_errors, 95)  # 95th percentile absolute error
        
        # Maximum and minimum errors
        max_error = np.max(abs_errors)
        min_error = np.min(abs_errors)
        
        # Relative metrics (if no zero values)
        if np.all(y_true != 0):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
            relative_rmse = rmse / np.mean(np.abs(y_true))
        else:
            mape = float('inf')
            relative_rmse = float('inf')
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r_squared': r_squared,
            'correlation': correlation,
            'error_mean': error_mean,
            'error_std': error_std,
            'error_median': error_median,
            'mae_50': mae_50,
            'mae_90': mae_90,
            'mae_95': mae_95,
            'max_error': max_error,
            'min_error': min_error,
            'mape': mape,
            'relative_rmse': relative_rmse
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary of computed metrics
        """
        print("\n" + "="*50)
        print("EVALUATION METRICS SUMMARY")
        print("="*50)
        
        print(f"Basic Metrics:")
        print(f"  MSE:           {metrics['mse']:.6f}")
        print(f"  MAE:           {metrics['mae']:.6f}")
        print(f"  RMSE:          {metrics['rmse']:.6f}")
        print(f"  R²:            {metrics['r_squared']:.6f}")
        print(f"  Correlation:   {metrics['correlation']:.6f}")
        
        print(f"\nError Statistics:")
        print(f"  Mean Error:    {metrics['error_mean']:.6f}")
        print(f"  Error Std:     {metrics['error_std']:.6f}")
        print(f"  Median Error:  {metrics['error_median']:.6f}")
        
        print(f"\nPercentile Metrics:")
        print(f"  MAE (50th):    {metrics['mae_50']:.6f}")
        print(f"  MAE (90th):    {metrics['mae_90']:.6f}")
        print(f"  MAE (95th):    {metrics['mae_95']:.6f}")
        
        print(f"\nExtreme Values:")
        print(f"  Max Error:     {metrics['max_error']:.6f}")
        print(f"  Min Error:     {metrics['min_error']:.6f}")
        
        if metrics['mape'] != float('inf'):
            print(f"\nRelative Metrics:")
            print(f"  MAPE:          {metrics['mape']:.2f}%")
            print(f"  Relative RMSE: {metrics['relative_rmse']:.6f}")
        
        print("="*50)
    
    def print_sample_predictions(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        n_samples: int = 20
    ):
        """
        Print sample predictions for inspection.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            n_samples: Number of samples to print
        """
        n_samples = min(n_samples, len(y_true))
        
        print(f"\nFirst {n_samples} predictions:")
        print(f"{'Index':<8} {'True Value':<15} {'Predicted':<15} {'Error':<15} {'Abs Error':<15}")
        print("-" * 75)
        
        for i in range(n_samples):
            true_val = y_true[i]
            pred_val = y_pred[i]
            error = pred_val - true_val
            abs_error = abs(error)
            print(f"{i:<8} {true_val:<15.6f} {pred_val:<15.6f} {error:<15.6f} {abs_error:<15.6f}")
    
    def save_predictions(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        filepath: Optional[str] = None
    ):
        """
        Save predictions to file.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            filepath: Path to save predictions. If None, saves to model directory.
        """
        if filepath is None:
            filepath = f"{self.config.model_dir}/predictions.npz"
        
        np.savez(
            filepath,
            y_true=y_true,
            y_pred=y_pred,
            errors=y_pred - y_true
        )
        
        print(f"Predictions saved to: {filepath}")
    
    def load_predictions(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load predictions from file.
        
        Args:
            filepath: Path to predictions file
            
        Returns:
            Tuple of (y_true, y_pred)
        """
        data = np.load(filepath)
        return data['y_true'], data['y_pred']
