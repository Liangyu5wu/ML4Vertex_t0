#!/usr/bin/env python3
"""Unified training script for physics-informed models (Transformer and DNN)."""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.base_config import load_config
from src.data.data_loader import DataLoader
from src.data.physics_features import PhysicsFeatureEngineer
from src.data.data_processor import DataProcessor
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.evaluation.visualizer import Visualizer
import numpy as np
import tensorflow as tf


def create_physics_datasets(config, cell_sequences, vertex_times, vertex_features, feature_names):
    """Create datasets with physics features."""
    # Add physics features if enabled
    if getattr(config, 'use_physics_informed_features', False):
        print("Adding physics-informed features...")
        physics_engineer = PhysicsFeatureEngineer(config)
        cell_sequences, feature_names = physics_engineer.add_physics_features(
            cell_sequences, feature_names
        )
        event_features = physics_engineer.compute_event_level_features(
            cell_sequences, feature_names
        )
    else:
        event_features = np.empty((len(cell_sequences), 0))
    
    # Create datasets using existing processor
    processor = DataProcessor(config)
    
    # Apply normalization
    for sequence in cell_sequences:
        for cell in sequence:
            # Simple normalization for time features (if not skipped)
            if 'Cell_time_TOF_corrected' not in config.skip_normalization:
                time_idx = feature_names.index('Cell_time_TOF_corrected')
                cell[time_idx] = cell[time_idx] / 200.0  # Simple scaling
    
    # Pad sequences and create masks
    max_seq_len = min(max(len(seq) for seq in cell_sequences), config.max_cells)
    feature_dim = len(cell_sequences[0][0]) if cell_sequences[0] else len(feature_names)
    
    padded_sequences = processor.apply_smart_padding(cell_sequences, max_seq_len, feature_dim)
    attention_masks = processor.create_attention_mask(cell_sequences, max_seq_len)
    
    # Split data
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(padded_sequences))
    train_val_idx, test_idx = train_test_split(indices, test_size=config.test_size, random_state=config.random_state)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=config.val_split, random_state=config.random_state)
    
    def create_data_dict(indices):
        return {
            'cell_sequences': padded_sequences[indices],
            'vertex_times': vertex_times[indices],
            'vertex_features': vertex_features[indices],
            'attention_masks': attention_masks[indices],
            'event_features': event_features[indices] if event_features.shape[1] > 0 else None
        }
    
    return create_data_dict(train_idx), create_data_dict(val_idx), create_data_dict(test_idx), feature_dim


def create_tensorflow_datasets(train_data, val_data, test_data, batch_size):
    """Create TensorFlow datasets."""
    def make_dataset(data_dict, is_training=False):
        inputs = (data_dict['cell_sequences'], data_dict['vertex_features'], data_dict['attention_masks'])
        if data_dict['event_features'] is not None:
            inputs = inputs + (data_dict['event_features'],)
        
        dataset = tf.data.Dataset.from_tensor_slices((inputs, data_dict['vertex_times']))
        if is_training:
            dataset = dataset.shuffle(10000)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return (make_dataset(train_data, True), 
            make_dataset(val_data), 
            make_dataset(test_data))


def main():
    parser = argparse.ArgumentParser(description='Train physics-informed ML models')
    parser.add_argument('--config-file', type=str, required=True, help='Configuration file path')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config_file)
    if args.epochs: config.epochs = args.epochs
    if args.learning_rate: config.learning_rate = args.learning_rate
    if args.batch_size: config.batch_size = args.batch_size
    
    print(f"Training {config.model_name} ({config.model_architecture})")
    print(f"Physics features: {getattr(config, 'use_physics_informed_features', False)}")
    
    # Load data
    print("Loading data...")
    data_loader = DataLoader(config)
    cell_sequences, vertex_times, vertex_features, feature_names = data_loader.load_and_preprocess_data()
    
    # Create datasets with physics features
    print("Processing data...")
    train_data, val_data, test_data, feature_dim = create_physics_datasets(
        config, cell_sequences, vertex_times, vertex_features, feature_names
    )
    
    vertex_dim = train_data['vertex_features'].shape[-1]
    event_feature_dim = train_data['event_features'].shape[-1] if train_data['event_features'] is not None else 0
    
    print(f"Feature dimensions: cells={feature_dim}, vertex={vertex_dim}, event={event_feature_dim}")
    
    # Create TF datasets
    train_dataset, val_dataset, test_dataset = create_tensorflow_datasets(
        train_data, val_data, test_data, config.batch_size
    )
    
    # Build model
    print("Building model...")
    if config.model_architecture == "two_stage_dnn":
        from src.models.improved_dnn import ImprovedDNNModel
        model_builder = ImprovedDNNModel(config)
    else:
        from src.models.improved_transformer import ImprovedTransformerModel
        model_builder = ImprovedTransformerModel(config)
    
    model = model_builder.build_model_with_mask(feature_dim, vertex_dim, event_feature_dim)
    print(f"Model parameters: {model.count_params():,}")
    
    # Train
    print("Training...")
    config.create_directories()
    trainer = Trainer(config, model_builder)
    history = trainer.train(train_dataset, val_dataset)
    
    # Evaluate
    print("Evaluating...")
    evaluator = Evaluator(config)
    if config.model_architecture == "two_stage_dnn":
        from src.models.improved_dnn import ImprovedDNNModel
        best_model = ImprovedDNNModel.load_model(config.model_path)
    else:
        from src.models.improved_transformer import ImprovedTransformerModel
        best_model = ImprovedTransformerModel.load_model(config.model_path)
    
    predictions = evaluator.predict_with_model(best_model, test_dataset)
    metrics = evaluator.calculate_metrics(test_data['vertex_times'], predictions)
    
    print("Results:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Create plots
    visualizer = Visualizer(config)
    history_dict = {k: np.array(v) for k, v in history.history.items()}
    visualizer.create_comprehensive_evaluation_plots(
        test_data['vertex_times'], predictions, metrics, history_dict
    )
    
    print(f"Training completed! Model saved to: {config.model_path}")
    return metrics


if __name__ == "__main__":
    main()