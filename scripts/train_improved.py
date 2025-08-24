#!/usr/bin/env python3
"""
Training script for improved transformer model with physics-informed features.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.base_config import load_config
from src.data.data_loader import DataLoader
from src.data.improved_processor import ImprovedDataProcessor
from src.models.improved_transformer import ImprovedTransformerModel
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.evaluation.visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train improved transformer model')
    parser.add_argument('--config-file', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--learning-rate', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--verbose', type=int, default=1, help='Training verbosity')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Apply command line overrides
    if args.epochs:
        config.epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.batch_size = args.batch_size
    
    print("="*60)
    print("IMPROVED TRANSFORMER MODEL TRAINING")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Physics features: {getattr(config, 'use_physics_informed_features', False)}")
    print(f"Data: {config.data_dir} ({config.num_files} files)")
    print(f"Training: {config.epochs} epochs, LR={config.learning_rate}, BS={config.batch_size}")
    print()
    
    # Create directories
    config.create_directories()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    data_loader = DataLoader(config)
    cell_sequences, vertex_times, vertex_features, feature_names = data_loader.load_and_preprocess_data()
    
    print(f"Loaded {len(cell_sequences)} events")
    print(f"Original features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    
    # Step 2: Enhanced data processing with physics features
    print("\nStep 2: Processing data with physics features...")
    data_processor = ImprovedDataProcessor(config)
    
    train_data, val_data, test_data = data_processor.create_datasets_with_physics(
        cell_sequences, vertex_times, vertex_features, feature_names
    )
    
    # Determine feature dimensions
    feature_dim = train_data['cell_sequences'].shape[-1]
    vertex_dim = train_data['vertex_features'].shape[-1]
    event_feature_dim = train_data['event_features'].shape[-1] if train_data['event_features'] is not None else 0
    
    print(f"Enhanced feature dimension: {feature_dim}")
    print(f"Event feature dimension: {event_feature_dim}")
    
    # Step 3: Create TensorFlow datasets
    print("\nStep 3: Creating TensorFlow datasets...")
    train_dataset, val_dataset, test_dataset = data_processor.create_tensorflow_datasets(
        train_data, val_data, test_data
    )
    
    # Step 4: Build improved model
    print("\nStep 4: Building improved transformer model...")
    model_builder = ImprovedTransformerModel(config)
    model = model_builder.build_model_with_mask(feature_dim, vertex_dim, event_feature_dim)
    
    print("Model architecture:")
    print(model_builder.get_model_summary())
    
    param_counts = model_builder.count_parameters()
    print(f"Model parameters: {param_counts}")
    
    # Step 5: Train model
    print("\nStep 5: Training model...")
    trainer = Trainer(config, model_builder)
    history = trainer.train(train_dataset, val_dataset, verbose=args.verbose)
    
    print(f"Training completed! Best model saved to: {config.model_path}")
    
    # Step 6: Evaluate model
    print("\nStep 6: Evaluating model...")
    evaluator = Evaluator(config)
    
    # Load best model for evaluation
    best_model = ImprovedTransformerModel.load_model(config.model_path)
    
    # Evaluate on test set
    test_predictions = evaluator.predict_with_model(best_model, test_dataset)
    test_metrics = evaluator.calculate_metrics(test_data['vertex_times'], test_predictions)
    
    print("Test set performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Step 7: Create evaluation plots
    print("\nStep 7: Creating evaluation plots...")
    visualizer = Visualizer(config)
    
    # Convert training history to numpy arrays
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = np.array(values)
    
    # Create comprehensive plots
    visualizer.create_comprehensive_evaluation_plots(
        test_data['vertex_times'], 
        test_predictions, 
        test_metrics,
        training_history=history_dict,
        plot_prediction_type='comparison'
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Model saved: {config.model_path}")
    print(f"Plots saved: {config.plots_dir}")
    print(f"Config saved: {config.config_path}")
    print()
    
    # Print final performance comparison
    print("PERFORMANCE SUMMARY:")
    print("-" * 30)
    for metric, value in test_metrics.items():
        print(f"{metric:>15}: {value:>8.4f}")
    
    return test_metrics


if __name__ == "__main__":
    import numpy as np
    main()