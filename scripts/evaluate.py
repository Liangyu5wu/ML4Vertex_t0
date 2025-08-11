"""Main evaluation script for vertex time prediction models."""

# python scripts/evaluate.py --model-dir models/transformer_simple_experiment --load-data

import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.transformer_config import TransformerConfig
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.models.transformer_model import TransformerModel
from src.evaluation.evaluator import Evaluator
from src.evaluation.visualizer import Visualizer
from src.training.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate vertex time prediction model')
    
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing saved model and config')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing HDF5 data files (overrides config)')
    parser.add_argument('--load-data', action='store_true',
                       help='Load and process data (otherwise assumes data exists)')
    parser.add_argument('--create-plots', action='store_true', default=True,
                       help='Create evaluation plots')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0, 1, 2)')
    
    return parser.parse_args()


def load_config_and_model(model_dir):
    """Load configuration and model from directory."""
    try:
        # Load configuration
        config = TransformerConfig.load_config(model_dir)
        print(f"Loaded configuration from: {model_dir}")
        
        # Load model - try both .h5 and .keras formats for backward compatibility
        model_h5_path = os.path.join(model_dir, "model.h5")
        model_keras_path = os.path.join(model_dir, "model.keras")
        
        model_path = None
        if os.path.exists(model_h5_path):
            model_path = model_h5_path
        elif os.path.exists(model_keras_path):
            model_path = model_keras_path
        else:
            raise FileNotFoundError(f"No model file found in {model_dir}. Expected model.h5 or model.keras")
        
        keras_model = TransformerModel.load_model(model_path)
        print(f"Loaded model from: {model_path}")
        
        return config, keras_model
        
    except Exception as e:
        print(f"Error loading model or config: {e}")
        raise


def load_or_reuse_data(config, data_dir_override=None, load_data=False):
    """Load data or try to reuse existing processed data."""
    if data_dir_override:
        config.data_dir = data_dir_override
    
    if load_data:
        print("Loading and processing data...")
        
        # Load raw data
        data_loader = DataLoader(config)
        cell_sequences, vertex_features, vertex_times, sequence_lengths = \
            data_loader.load_data_from_files()
        
        # Process data
        data_processor = DataProcessor(config)
        
        # Split data (using same random state as training for consistency)
        (train_cells, val_cells, test_cells), \
        (train_vertex, val_vertex, test_vertex), \
        (train_times, val_times, test_times) = data_processor.split_data(
            cell_sequences, vertex_features, vertex_times
        )
        
        # Normalize features
        (train_cells_norm, val_cells_norm, test_cells_norm), \
        (train_vertex_norm, val_vertex_norm, test_vertex_norm), \
        norm_params = data_processor.normalize_features(
            train_cells, val_cells, test_cells,
            train_vertex, val_vertex, test_vertex,
            train_times, val_times, test_times
        )
        
        return (test_cells_norm, test_vertex_norm, test_times, data_processor)
    
    else:
        # This is a placeholder - in a real implementation, you might save/load 
        # processed data to avoid reprocessing
        print("Warning: --load-data not specified. You must provide processed data.")
        print("For now, will load and process data anyway...")
        return load_or_reuse_data(config, data_dir_override, load_data=True)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("="*60)
    print("VERTEX TIME PREDICTION MODEL EVALUATION")
    print("="*60)
    print(f"Model directory: {args.model_dir}")
    if args.data_dir:
        print(f"Data directory override: {args.data_dir}")
    print("="*60)
    
    try:
        # Load configuration and model
        print("\n1. Loading model and configuration...")
        config, keras_model = load_config_and_model(args.model_dir)
        
        # Update config with model directory for saving results
        config.models_base_dir = os.path.dirname(args.model_dir)
        config.model_name = os.path.basename(args.model_dir)
        
        # Load or process data
        print("\n2. Loading evaluation data...")
        test_cells_norm, test_vertex_norm, test_times, data_processor = \
            load_or_reuse_data(config, args.data_dir, args.load_data)
        
        print(f"Test data loaded: {len(test_times)} samples")
        
        # Initialize evaluator
        print("\n3. Evaluating model...")
        evaluator = Evaluator(config)
        
        # Create test dataset for Keras evaluation
        test_dataset = data_processor.create_padded_dataset(
            test_cells_norm, test_vertex_norm, test_times, shuffle=False
        )
        
        # Evaluate using Keras
        keras_metrics = evaluator.evaluate_model(keras_model, test_dataset, args.verbose)
        
        # Make predictions and compute detailed metrics
        print("\n4. Computing detailed metrics...")
        y_pred, detailed_metrics = evaluator.predict_and_evaluate(
            keras_model, test_cells_norm, test_vertex_norm, test_times, data_processor
        )
        
        # Print comprehensive metrics
        evaluator.print_metrics(detailed_metrics)
        
        # Print sample predictions
        evaluator.print_sample_predictions(test_times, y_pred, n_samples=20)
        
        # Save predictions
        evaluator.save_predictions(test_times, y_pred)
        
        # Create visualizations
        if args.create_plots:
            print("\n5. Creating evaluation plots...")
            visualizer = Visualizer(config)
            
            # Load training history if available
            history_path = os.path.join(args.model_dir, "training_history.npz")
            training_history = None
            if os.path.exists(history_path):
                training_history = Trainer.load_training_history(history_path)
                print(f"Loaded training history from: {history_path}")
            
            # Create all plots
            visualizer.create_comprehensive_evaluation_plots(
                test_times, y_pred, detailed_metrics, training_history
            )
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {config.model_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
