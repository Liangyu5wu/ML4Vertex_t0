"""Main evaluation script for vertex time prediction models with DNN support."""

# python scripts/evaluate.py --model-dir models/transformer_simple_experiment --load-data

import os
import sys
import argparse
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.transformer_config import TransformerConfig
from config.dnn_config import DNNConfig
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.data.multi_input_data_loader import MultiInputDataLoader
from src.data.multi_input_data_processor import MultiInputDataProcessor
from src.models.transformer_model import TransformerModel
from src.models.dnn_model import DNNModel
from src.models.baseline_guided_model import BaselineGuidedDNN
from src.models.multi_input_dnn_model import MultiInputDNNModel
from src.models.multi_input_transformer_model import MultiInputTransformerModel
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
        # Try to load configuration - check for DNN vs Transformer
        config_path = os.path.join(model_dir, "config.yaml")
        if os.path.exists(config_path):
            # Check config type by reading YAML content
            import yaml
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            if yaml_data.get('model_architecture') == 'baseline_guided_dnn':
                config = DNNConfig.load_config(model_dir)
                print(f"Loaded baseline-guided DNN configuration from: {model_dir}")
                is_dnn_model = False
                is_baseline_guided = True
                is_multi_input = False
            elif yaml_data.get('model_architecture') == 'multi_input_dnn':
                config = DNNConfig.load_config(model_dir)
                print(f"Loaded multi-input DNN configuration from: {model_dir}")
                is_dnn_model = True
                is_baseline_guided = False
                is_multi_input = True
            elif yaml_data.get('model_architecture') == 'multi_input_transformer':
                config = TransformerConfig.load_config(model_dir)
                print(f"Loaded multi-input Transformer configuration from: {model_dir}")
                is_dnn_model = False
                is_baseline_guided = False
                is_multi_input = True
            elif yaml_data.get('model_architecture') == 'two_stage_dnn' or 'cell_encoder_units' in yaml_data:
                config = DNNConfig.load_config(model_dir)
                print(f"Loaded DNN configuration from: {model_dir}")
                is_dnn_model = True
                is_baseline_guided = False
                is_multi_input = False
            else:
                config = TransformerConfig.load_config(model_dir)
                print(f"Loaded Transformer configuration from: {model_dir}")
                is_dnn_model = False
                is_baseline_guided = False
                is_multi_input = False
        else:
            # Fallback to JSON config (assume Transformer for compatibility)
            config = TransformerConfig.load_config(model_dir)
            print(f"Loaded configuration from: {model_dir}")
            is_dnn_model = False
            is_baseline_guided = False
            is_multi_input = False
        
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
        
        # Load model with appropriate custom objects
        if is_baseline_guided:
            keras_model = BaselineGuidedDNN.load_model(model_path)
        elif is_multi_input and is_dnn_model:
            keras_model = MultiInputDNNModel.load_model(model_path)
        elif is_multi_input:
            keras_model = MultiInputTransformerModel.load_model(model_path)
        elif is_dnn_model:
            keras_model = DNNModel.load_model(model_path)
        else:
            keras_model = TransformerModel.load_model(model_path)
        print(f"Loaded model from: {model_path}")
        
        # Display model type information
        is_dnn_model = is_baseline_guided or (hasattr(config, 'model_architecture') and 'dnn' in getattr(config, 'model_architecture', ''))
        print_model_info(keras_model, is_dnn_model, is_multi_input)
        
        return config, keras_model, is_baseline_guided, is_multi_input
        
    except Exception as e:
        print(f"Error loading model or config: {e}")
        raise


def print_model_info(model, is_dnn_model, is_multi_input=False):
    """Print information about the loaded model."""
    # Detect model type
    if hasattr(model, 'input'):
        if isinstance(model.input, list):
            num_inputs = len(model.input)
            input_names = [inp.name for inp in model.input]
        else:
            num_inputs = 1
            input_names = [model.input.name]
    else:
        num_inputs = 2  # Fallback
        input_names = ["unknown"]
    
    model_type = "DNN" if is_dnn_model else "Transformer"
    
    if is_multi_input:
        model_subtype = "Multi-input (jets+tracks)"
    elif num_inputs == 3:
        model_subtype = "Mask-enabled"
    else:
        model_subtype = "Traditional"
    
    print(f"\nModel Information:")
    print(f"  Type: {model_type} ({model_subtype})")
    print(f"  Inputs: {num_inputs} ({', '.join(input_names)})")


def load_or_reuse_data(config, data_dir_override=None, load_data=False, is_baseline_guided=False, is_multi_input=False):
    """Load data or try to reuse existing processed data."""
    if data_dir_override:
        config.data_dir = data_dir_override
    
    if load_data:
        print("Loading and processing data...")
        
        # Load raw data
        if is_multi_input:
            data_loader = MultiInputDataLoader(config)
            cell_sequences, vertex_features, vertex_times, sequence_lengths, jet_sequences, track_sequences = \
                data_loader.load_data_from_files()
            baseline_predictions = None
        else:
            data_loader = DataLoader(config)
            if is_baseline_guided:
                cell_sequences, vertex_features, vertex_times, sequence_lengths, baseline_predictions = \
                    data_loader.load_data_with_baselines_from_files()
                jet_sequences = track_sequences = None
            else:
                cell_sequences, vertex_features, vertex_times, sequence_lengths = \
                    data_loader.load_data_from_files()
                baseline_predictions = None
                jet_sequences = track_sequences = None
        
        # Process data
        if is_multi_input:
            from src.data.multi_input_data_processor import MultiInputDataProcessor
            data_processor = MultiInputDataProcessor(config)
            
            # Split multi-input data
            (train_cells, val_cells, test_cells), \
            (train_vertex, val_vertex, test_vertex), \
            (train_jets, val_jets, test_jets), \
            (train_tracks, val_tracks, test_tracks), \
            (train_times, val_times, test_times) = data_processor.split_data(
                cell_sequences, vertex_features, vertex_times, jet_sequences, track_sequences
            )
            
            # Normalize multi-input features
            (train_cells_norm, val_cells_norm, test_cells_norm), \
            (train_vertex_norm, val_vertex_norm, test_vertex_norm), \
            (train_jets_norm, val_jets_norm, test_jets_norm), \
            (train_tracks_norm, val_tracks_norm, test_tracks_norm), \
            norm_params = data_processor.normalize_features(
                train_cells, val_cells, test_cells,
                train_vertex, val_vertex, test_vertex,
                train_jets, val_jets, test_jets,
                train_tracks, val_tracks, test_tracks,
                train_times, val_times, test_times
            )
            
            return (test_cells_norm, test_vertex_norm, test_times, data_processor, test_jets_norm, test_tracks_norm)
        else:
            data_processor = DataProcessor(config)
            
            # Split data (using same random state as training for consistency)
            # Generate the same indices that split_data uses
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(vertex_times))
            train_indices, temp_indices = train_test_split(
                indices, test_size=config.test_size, random_state=config.random_state
            )
            val_indices, test_indices = train_test_split(
                temp_indices, test_size=config.val_split, random_state=config.random_state
            )
            
            # Split all data using the same indices
            (train_cells, val_cells, test_cells), \
            (train_vertex, val_vertex, test_vertex), \
            (train_times, val_times, test_times) = data_processor.split_data(
                cell_sequences, vertex_features, vertex_times
            )
            
            # Split baseline predictions using the same indices if needed
            if is_baseline_guided:
                train_baselines = baseline_predictions[train_indices]
                val_baselines = baseline_predictions[val_indices]
                test_baselines = baseline_predictions[test_indices]
            else:
                train_baselines = val_baselines = test_baselines = None
            
            # Normalize features
            (train_cells_norm, val_cells_norm, test_cells_norm), \
            (train_vertex_norm, val_vertex_norm, test_vertex_norm), \
            norm_params = data_processor.normalize_features(
                train_cells, val_cells, test_cells,
                train_vertex, val_vertex, test_vertex,
                train_times, val_times, test_times
            )
        
        if is_baseline_guided:
            return (test_cells_norm, test_vertex_norm, test_times, data_processor, test_baselines)
        else:
            return (test_cells_norm, test_vertex_norm, test_times, data_processor)
    
    else:
        # This is a placeholder - in a real implementation, you might save/load 
        # processed data to avoid reprocessing
        print("Warning: --load-data not specified. You must provide processed data.")
        print("For now, will load and process data anyway...")
        return load_or_reuse_data(config, data_dir_override, load_data=True, is_baseline_guided=is_baseline_guided, is_multi_input=False)


def create_test_dataset_automatically(evaluator, model, test_cells_norm, test_vertex_norm, test_times, data_processor, test_baselines=None, test_jets_norm=None, test_tracks_norm=None):
    """Create test dataset automatically based on model type."""
    print("\n3. Creating test dataset for evaluation...")
    
    # Check if this is a multi-input model
    if test_jets_norm is not None and test_tracks_norm is not None:
        print("Creating test dataset for multi-input model...")
        # For multi-input models, create dataset with jets and tracks
        return data_processor.create_multi_input_dataset(
            test_cells_norm, test_vertex_norm, test_jets_norm, test_tracks_norm, test_times, shuffle=False
        )
    elif test_baselines is not None:
        print("Creating test dataset with baseline predictions for baseline-guided model...")
        # For baseline-guided models, we need to create the dataset with baseline predictions
        # Use the data processor's method but with baseline predictions
        return data_processor.create_padded_dataset_with_baseline(
            test_cells_norm, test_vertex_norm, test_times, test_baselines, shuffle=False
        )
    else:
        # Use the standard automatic dataset creation method
        test_dataset = evaluator.create_test_dataset_for_evaluation(
            test_cells_norm, test_vertex_norm, test_times, data_processor, model
        )
        return test_dataset


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
        config, keras_model, is_baseline_guided, is_multi_input = load_config_and_model(args.model_dir)
        
        # Update config with model directory for saving results
        config.models_base_dir = os.path.dirname(args.model_dir)
        config.model_name = os.path.basename(args.model_dir)
        
        # Load or process data
        print("\n2. Loading evaluation data...")
        if is_multi_input:
            test_cells_norm, test_vertex_norm, test_times, data_processor, test_jets_norm, test_tracks_norm = \
                load_or_reuse_data(config, args.data_dir, args.load_data, is_baseline_guided, is_multi_input)
            test_baselines = None
        elif is_baseline_guided:
            test_cells_norm, test_vertex_norm, test_times, data_processor, test_baselines = \
                load_or_reuse_data(config, args.data_dir, args.load_data, is_baseline_guided, is_multi_input)
            test_jets_norm = test_tracks_norm = None
        else:
            test_cells_norm, test_vertex_norm, test_times, data_processor = \
                load_or_reuse_data(config, args.data_dir, args.load_data, is_baseline_guided, is_multi_input)
            test_baselines = None
            test_jets_norm = test_tracks_norm = None
        
        print(f"Test data loaded: {len(test_times)} samples")
        
        # Initialize evaluator
        print("\n3. Initializing evaluator...")
        evaluator = Evaluator(config)
        
        # Create test dataset automatically based on model type
        if is_multi_input:
            test_dataset = create_test_dataset_automatically(
                evaluator, keras_model, test_cells_norm, test_vertex_norm, test_times, data_processor, test_baselines, test_jets_norm, test_tracks_norm
            )
        else:
            test_dataset = create_test_dataset_automatically(
                evaluator, keras_model, test_cells_norm, test_vertex_norm, test_times, data_processor, test_baselines
            )
        
        # Evaluate using Keras
        print("\n4. Evaluating model...")
        keras_metrics = evaluator.evaluate_model(keras_model, test_dataset, args.verbose)
        
        # Make predictions and compute detailed metrics
        print("\n5. Computing detailed metrics...")
        if is_multi_input:
            # For multi-input models, make predictions directly using the dataset
            print("Making predictions for multi-input model...")
            y_pred = keras_model.predict(test_dataset)
            # Flatten predictions to match expected shape for metrics computation
            y_pred = y_pred.flatten()
            detailed_metrics = evaluator.compute_metrics(test_times, y_pred)
        elif is_baseline_guided:
            # For baseline-guided models, make predictions directly using the dataset
            print("Making predictions for baseline-guided model...")
            y_pred = keras_model.predict(test_dataset)
            # Flatten predictions to match expected shape for metrics computation
            y_pred = y_pred.flatten()
            detailed_metrics = evaluator.compute_metrics(test_times, y_pred)
        else:
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
            print("\n6. Creating evaluation plots...")
            visualizer = Visualizer(config)
            
            # Load training history if available
            history_path = os.path.join(args.model_dir, "training_history.npz")
            training_history = None
            if os.path.exists(history_path):
                training_history = Trainer.load_training_history(history_path)
                print(f"Loaded training history from: {history_path}")
            
            # Create all plots with baseline comparison if available
            visualizer.create_comprehensive_evaluation_plots(
                test_times, y_pred, detailed_metrics, training_history,
                baseline_predictions=test_baselines
            )
        
        # Print summary
        print(f"\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {config.model_dir}")
        
        # Print model and configuration info
        if hasattr(config, 'use_attention_mask'):
            mask_status = "enabled" if config.use_attention_mask else "disabled"
            print(f"Attention mask: {mask_status}")
        
        if hasattr(config, 'use_jet_features') and config.use_jet_features:
            print(f"Jet features: included ({config.jet_features})")
            
        if hasattr(config, 'use_cell_jet_matching') and config.use_cell_jet_matching:
            print(f"Cell-jet matching: applied during training")
        
        # Print model type
        if isinstance(config, DNNConfig):
            print(f"Model type: Two-Stage DNN")
        else:
            print(f"Model type: Transformer")
        
        # Print performance summary
        print(f"\nPerformance Summary:")
        print(f"  RMSE: {detailed_metrics['rmse']:.4f}")
        print(f"  MAE: {detailed_metrics['mae']:.4f}")
        print(f"  R²: {detailed_metrics['r_squared']:.4f}")
        print(f"  Correlation: {detailed_metrics['correlation']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
