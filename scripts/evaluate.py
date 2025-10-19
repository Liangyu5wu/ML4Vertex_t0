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
from src.data.hgtd_multi_input_data_loader import HGTDMultiInputDataLoader
from src.data.hgtd_multi_input_data_processor import HGTDMultiInputDataProcessor
from src.data.hgtd_only_data_loader import HGTDOnlyDataLoader
from src.data.hgtd_only_data_processor import HGTDOnlyDataProcessor
from src.models.transformer_model import TransformerModel
from src.models.dnn_model import DNNModel
from src.models.baseline_guided_model import BaselineGuidedDNN
from src.models.multi_input_dnn_model import MultiInputDNNModel
from src.models.multi_input_transformer_model import MultiInputTransformerModel
from src.models.hgtd_multi_input_dnn_model import HGTDMultiInputDNNModel
from src.models.hgtd_only_dnn_model import HGTDOnlyDNNModel
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
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to config file to override evaluation parameters (e.g., fitting method)')

    return parser.parse_args()


def override_evaluation_params(config, config_file_path):
    """
    Override evaluation-specific parameters from a config file.

    This allows updating evaluation parameters (like fitting method) without retraining.
    Only evaluation-related parameters are overridden, not training parameters.

    Args:
        config: Loaded config object from model directory
        config_file_path: Path to YAML config file with updated parameters

    Returns:
        Updated config object
    """
    if not config_file_path or not os.path.exists(config_file_path):
        return config

    import yaml

    try:
        with open(config_file_path, 'r') as f:
            override_params = yaml.safe_load(f)

        # List of evaluation-specific parameters that can be overridden
        evaluation_params = [
            'error_fit_method',
            'error_fit_range',
            'pileup_sigma',
            'fix_pileup_sigma'
        ]

        overridden = []
        for param in evaluation_params:
            if param in override_params:
                old_value = getattr(config, param, None)
                setattr(config, param, override_params[param])
                overridden.append(f"  {param}: {old_value} -> {override_params[param]}")

        if overridden:
            print("\nOverriding evaluation parameters from config file:")
            for msg in overridden:
                print(msg)

    except Exception as e:
        print(f"Warning: Could not override parameters from {config_file_path}: {e}")

    return config


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
                is_hgtd_only = False
                is_hgtd_multi_input = False
            elif yaml_data.get('model_architecture') == 'hgtd_only_dnn':
                config = DNNConfig.load_config(model_dir)
                print(f"Loaded HGTD-only DNN configuration from: {model_dir}")
                is_dnn_model = True
                is_baseline_guided = False
                is_multi_input = False
                is_hgtd_only = True
                is_hgtd_multi_input = False
            elif yaml_data.get('model_architecture') == 'hgtd_multi_input_dnn':
                config = DNNConfig.load_config(model_dir)
                print(f"Loaded HGTD multi-input DNN configuration from: {model_dir}")
                is_dnn_model = True
                is_baseline_guided = False
                is_multi_input = False
                is_hgtd_only = False
                is_hgtd_multi_input = True
            elif yaml_data.get('model_architecture') == 'multi_input_dnn':
                config = DNNConfig.load_config(model_dir)
                print(f"Loaded multi-input DNN configuration from: {model_dir}")
                is_dnn_model = True
                is_baseline_guided = False
                is_multi_input = True
                is_hgtd_only = False
                is_hgtd_multi_input = False
            elif yaml_data.get('model_architecture') == 'multi_input_transformer':
                config = TransformerConfig.load_config(model_dir)
                print(f"Loaded multi-input Transformer configuration from: {model_dir}")
                is_dnn_model = False
                is_baseline_guided = False
                is_multi_input = True
                is_hgtd_only = False
                is_hgtd_multi_input = False
            elif yaml_data.get('model_architecture') == 'two_stage_dnn' or 'cell_encoder_units' in yaml_data:
                config = DNNConfig.load_config(model_dir)
                print(f"Loaded DNN configuration from: {model_dir}")
                is_dnn_model = True
                is_baseline_guided = False
                is_multi_input = False
                is_hgtd_only = False
                is_hgtd_multi_input = False
            else:
                config = TransformerConfig.load_config(model_dir)
                print(f"Loaded Transformer configuration from: {model_dir}")
                is_dnn_model = False
                is_baseline_guided = False
                is_multi_input = False
                is_hgtd_only = False
                is_hgtd_multi_input = False
        else:
            # Fallback to JSON config (assume Transformer for compatibility)
            config = TransformerConfig.load_config(model_dir)
            print(f"Loaded configuration from: {model_dir}")
            is_dnn_model = False
            is_baseline_guided = False
            is_multi_input = False
            is_hgtd_only = False
            is_hgtd_multi_input = False
        
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
        elif is_hgtd_only:
            keras_model = HGTDOnlyDNNModel.load_model(model_path)
        elif is_hgtd_multi_input:
            keras_model = HGTDMultiInputDNNModel.load_model(model_path)
        elif is_multi_input and is_dnn_model:
            keras_model = MultiInputDNNModel.load_model(model_path)
        elif is_multi_input:
            keras_model = MultiInputTransformerModel.load_model(model_path)
        elif is_dnn_model:
            keras_model = DNNModel.load_model(model_path)
        else:
            keras_model = TransformerModel.load_model(model_path)
        print(f"Loaded model from: {model_path}")

        # Display model type information and get corrected multi-input/hgtd status
        is_dnn_model = is_baseline_guided or (hasattr(config, 'model_architecture') and 'dnn' in getattr(config, 'model_architecture', ''))
        actual_is_multi_input, actual_is_hgtd_only, actual_is_hgtd_multi_input = print_model_info(keras_model, is_dnn_model, is_multi_input, is_hgtd_only, is_hgtd_multi_input)

        # Use the corrected multi-input/hgtd status
        is_multi_input = actual_is_multi_input
        is_hgtd_only = actual_is_hgtd_only
        is_hgtd_multi_input = actual_is_hgtd_multi_input

        return config, keras_model, is_baseline_guided, is_multi_input, is_hgtd_only, is_hgtd_multi_input
        
    except Exception as e:
        print(f"Error loading model or config: {e}")
        raise


def print_model_info(model, is_dnn_model, is_multi_input=False, is_hgtd_only=False, is_hgtd_multi_input=False):
    """Print information about the loaded model and return corrected multi-input, HGTD-only, and HGTD multi-input status."""
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

    # Check if the model is actually a multi-input, HGTD-only, or HGTD multi-input model based on actual inputs
    actual_is_hgtd_multi_input = is_hgtd_multi_input and num_inputs == 6
    actual_is_multi_input = is_multi_input and num_inputs == 5
    actual_is_hgtd_only = is_hgtd_only and num_inputs == 2

    if actual_is_hgtd_only:
        model_subtype = "HGTD-Only (HGTD tracks only, no LAr data)"
    elif actual_is_hgtd_multi_input:
        model_subtype = "HGTD Multi-input (cells+jets+LAr tracks+HGTD tracks)"
    elif actual_is_multi_input:
        model_subtype = "Multi-input (cells+jets+tracks)"
    elif num_inputs == 3:
        model_subtype = "Mask-enabled"
        # Override multi-input flags if model only has 3 inputs
        actual_is_multi_input = False
        actual_is_hgtd_only = False
        actual_is_hgtd_multi_input = False
    else:
        model_subtype = "Traditional"
        actual_is_multi_input = False
        actual_is_hgtd_only = False
        actual_is_hgtd_multi_input = False

    print(f"\nModel Information:")
    print(f"  Type: {model_type} ({model_subtype})")
    print(f"  Inputs: {num_inputs} ({', '.join(input_names)})")

    if (is_multi_input and not actual_is_multi_input) or (is_hgtd_only and not actual_is_hgtd_only) or (is_hgtd_multi_input and not actual_is_hgtd_multi_input):
        print(f"  Note: Model was configured differently but only has {num_inputs} inputs. Treating as {model_subtype.lower()}.")

    return actual_is_multi_input, actual_is_hgtd_only, actual_is_hgtd_multi_input


def load_or_reuse_data(config, data_dir_override=None, load_data=False, is_baseline_guided=False, is_multi_input=False, is_hgtd_only=False, is_hgtd_multi_input=False):
    """Load data or try to reuse existing processed data."""
    if data_dir_override:
        config.data_dir = data_dir_override

    if load_data:
        print("Loading and processing data...")

        # Load raw data
        if is_hgtd_only:
            data_loader = HGTDOnlyDataLoader(config)
            hgtd_track_sequences, vertex_features, vertex_times, sequence_lengths = \
                data_loader.load_data_from_files()
            cell_sequences = baseline_predictions = jet_sequences = track_sequences = None
        elif is_hgtd_multi_input:
            data_loader = HGTDMultiInputDataLoader(config)
            cell_sequences, vertex_features, vertex_times, sequence_lengths, jet_sequences, track_sequences, hgtd_track_sequences = \
                data_loader.load_data_from_files()
            baseline_predictions = None
        elif is_multi_input:
            data_loader = MultiInputDataLoader(config)
            cell_sequences, vertex_features, vertex_times, sequence_lengths, jet_sequences, track_sequences = \
                data_loader.load_data_from_files()
            baseline_predictions = None
            hgtd_track_sequences = None
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
        if is_hgtd_only:
            from src.data.hgtd_only_data_processor import HGTDOnlyDataProcessor
            data_processor = HGTDOnlyDataProcessor(config)

            # Split HGTD-only data
            (train_hgtd_tracks, val_hgtd_tracks, test_hgtd_tracks), \
            (train_vertex, val_vertex, test_vertex), \
            (train_times, val_times, test_times) = data_processor.split_data(
                hgtd_track_sequences, vertex_features, vertex_times
            )

            # Normalize HGTD-only features
            (train_hgtd_tracks_norm, val_hgtd_tracks_norm, test_hgtd_tracks_norm), \
            (train_vertex_norm, val_vertex_norm, test_vertex_norm), \
            norm_params = data_processor.normalize_features(
                train_hgtd_tracks, val_hgtd_tracks, test_hgtd_tracks,
                train_vertex, val_vertex, test_vertex,
                train_times, val_times, test_times
            )

            return (test_hgtd_tracks_norm, test_vertex_norm, test_times, data_processor)
        elif is_hgtd_multi_input:
            from src.data.hgtd_multi_input_data_processor import HGTDMultiInputDataProcessor
            data_processor = HGTDMultiInputDataProcessor(config)

            # Split HGTD multi-input data
            (train_cells, val_cells, test_cells), \
            (train_vertex, val_vertex, test_vertex), \
            (train_jets, val_jets, test_jets), \
            (train_tracks, val_tracks, test_tracks), \
            (train_hgtd_tracks, val_hgtd_tracks, test_hgtd_tracks), \
            (train_times, val_times, test_times) = data_processor.split_data(
                cell_sequences, vertex_features, vertex_times, jet_sequences, track_sequences, hgtd_track_sequences
            )

            # Normalize HGTD multi-input features
            (train_cells_norm, val_cells_norm, test_cells_norm), \
            (train_vertex_norm, val_vertex_norm, test_vertex_norm), \
            (train_jets_norm, val_jets_norm, test_jets_norm), \
            (train_tracks_norm, val_tracks_norm, test_tracks_norm), \
            (train_hgtd_tracks_norm, val_hgtd_tracks_norm, test_hgtd_tracks_norm), \
            norm_params = data_processor.normalize_features(
                train_cells, val_cells, test_cells,
                train_vertex, val_vertex, test_vertex,
                train_jets, val_jets, test_jets,
                train_tracks, val_tracks, test_tracks,
                train_hgtd_tracks, val_hgtd_tracks, test_hgtd_tracks,
                train_times, val_times, test_times
            )

            return (test_cells_norm, test_vertex_norm, test_times, data_processor, test_jets_norm, test_tracks_norm, test_hgtd_tracks_norm)
        elif is_multi_input:
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
        return load_or_reuse_data(config, data_dir_override, load_data=True, is_baseline_guided=is_baseline_guided, is_multi_input=is_multi_input, is_hgtd_only=is_hgtd_only, is_hgtd_multi_input=is_hgtd_multi_input)


def create_test_dataset_automatically(evaluator, model, test_cells_norm, test_vertex_norm, test_times, data_processor, test_baselines=None, test_jets_norm=None, test_tracks_norm=None, test_hgtd_tracks_norm=None, is_hgtd_only=False):
    """Create test dataset automatically based on model type."""
    print("\n3. Creating test dataset for evaluation...")

    # Check if this is an HGTD-only model
    if is_hgtd_only:
        print("Creating test dataset for HGTD-only model...")
        # For HGTD-only models, test_cells_norm actually contains HGTD track data
        return data_processor.create_hgtd_only_dataset(
            test_cells_norm, test_vertex_norm, test_times, shuffle=False
        )
    # Check if this is a HGTD multi-input model
    elif test_hgtd_tracks_norm is not None:
        print("Creating test dataset for HGTD multi-input model...")
        # For HGTD multi-input models, create dataset with jets, tracks, and HGTD tracks
        return data_processor.create_hgtd_multi_input_dataset(
            test_cells_norm, test_vertex_norm, test_jets_norm, test_tracks_norm, test_hgtd_tracks_norm, test_times, shuffle=False
        )
    # Check if this is a multi-input model
    elif test_jets_norm is not None and test_tracks_norm is not None:
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
        config, keras_model, is_baseline_guided, is_multi_input, is_hgtd_only, is_hgtd_multi_input = load_config_and_model(args.model_dir)

        # Override evaluation-specific parameters if config file provided
        if args.config_file:
            print(f"\nAttempting to override evaluation parameters from: {args.config_file}")
            config = override_evaluation_params(config, args.config_file)

        # Update config with model directory for saving results
        config.models_base_dir = os.path.dirname(args.model_dir)
        config.model_name = os.path.basename(args.model_dir)
        
        # Load or process data
        print("\n2. Loading evaluation data...")
        if is_hgtd_only:
            test_hgtd_tracks_norm, test_vertex_norm, test_times, data_processor = \
                load_or_reuse_data(config, args.data_dir, args.load_data, is_baseline_guided, is_multi_input, is_hgtd_only, is_hgtd_multi_input)
            test_cells_norm = None
            test_jets_norm = None
            test_tracks_norm = None
            test_baselines = None
        elif is_hgtd_multi_input:
            test_cells_norm, test_vertex_norm, test_times, data_processor, test_jets_norm, test_tracks_norm, test_hgtd_tracks_norm = \
                load_or_reuse_data(config, args.data_dir, args.load_data, is_baseline_guided, is_multi_input, is_hgtd_only, is_hgtd_multi_input)

            # Calculate baseline method predictions for plotting comparison
            print("Calculating baseline method predictions for plotting comparison...")
            try:
                # Load original cell sequences for baseline calculation
                from src.data.hgtd_multi_input_data_loader import HGTDMultiInputDataLoader
                temp_data_loader = HGTDMultiInputDataLoader(config)
                # We need to get the original (before normalization) cell sequences
                cell_sequences, vertex_features, vertex_times, sequence_lengths, jet_sequences, track_sequences, hgtd_track_sequences = \
                    temp_data_loader.load_data_from_files()

                # Split data to get test portion (matching the same split as normalized data)
                from sklearn.model_selection import train_test_split
                indices = np.arange(len(vertex_times))
                train_indices, temp_indices = train_test_split(
                    indices, test_size=config.test_size, random_state=config.random_state
                )
                val_indices, test_indices = train_test_split(
                    temp_indices, test_size=config.val_split, random_state=config.random_state
                )

                test_cell_sequences_orig = [cell_sequences[i] for i in test_indices]
                test_vertex_times_orig = vertex_times[test_indices]

                # Use existing calculate_traditional_t0 method from data processor
                # This reuses the well-tested baseline method calculation
                test_baselines, _ = data_processor.calculate_traditional_t0(
                    test_cell_sequences_orig, test_vertex_times_orig
                )
                print(f"Calculated baseline predictions for {len(test_baselines)} test events using existing method")

            except Exception as e:
                print(f"Warning: Could not calculate baseline method predictions: {e}")
                print("Error distribution plot will not include baseline method comparison")
                test_baselines = None
        elif is_multi_input:
            test_cells_norm, test_vertex_norm, test_times, data_processor, test_jets_norm, test_tracks_norm = \
                load_or_reuse_data(config, args.data_dir, args.load_data, is_baseline_guided, is_multi_input, is_hgtd_multi_input)
            test_hgtd_tracks_norm = None

            # Calculate baseline method predictions for plotting comparison
            print("Calculating baseline method predictions for plotting comparison...")
            try:
                # Load original cell sequences for baseline calculation
                from src.data.multi_input_data_loader import MultiInputDataLoader
                temp_data_loader = MultiInputDataLoader(config)
                # We need to get the original (before normalization) cell sequences
                cell_sequences, vertex_features, vertex_times, sequence_lengths, jet_sequences, track_sequences = \
                    temp_data_loader.load_data_from_files()
                
                # Split data to get test portion (matching the same split as normalized data)
                from sklearn.model_selection import train_test_split
                indices = np.arange(len(vertex_times))
                train_indices, temp_indices = train_test_split(
                    indices, test_size=config.test_size, random_state=config.random_state
                )
                val_indices, test_indices = train_test_split(
                    temp_indices, test_size=config.val_split, random_state=config.random_state
                )
                
                test_cell_sequences_orig = [cell_sequences[i] for i in test_indices]
                test_vertex_times_orig = vertex_times[test_indices]
                
                # Use existing calculate_traditional_t0 method from data processor
                # This reuses the well-tested baseline method calculation
                test_baselines, _ = data_processor.calculate_traditional_t0(
                    test_cell_sequences_orig, test_vertex_times_orig
                )
                print(f"Calculated baseline predictions for {len(test_baselines)} test events using existing method")
                
            except Exception as e:
                print(f"Warning: Could not calculate baseline method predictions: {e}")
                print("Error distribution plot will not include baseline method comparison")
                test_baselines = None
        elif is_baseline_guided:
            test_cells_norm, test_vertex_norm, test_times, data_processor, test_baselines = \
                load_or_reuse_data(config, args.data_dir, args.load_data, is_baseline_guided, is_multi_input, is_hgtd_multi_input)
            test_jets_norm = test_tracks_norm = test_hgtd_tracks_norm = None
        else:
            test_cells_norm, test_vertex_norm, test_times, data_processor = \
                load_or_reuse_data(config, args.data_dir, args.load_data, is_baseline_guided, is_multi_input, is_hgtd_multi_input)
            test_baselines = None
            test_jets_norm = test_tracks_norm = test_hgtd_tracks_norm = None
        
        print(f"Test data loaded: {len(test_times)} samples")
        
        # Initialize evaluator
        print("\n3. Initializing evaluator...")
        evaluator = Evaluator(config)
        
        # Create test dataset automatically based on model type
        if is_hgtd_only:
            test_dataset = create_test_dataset_automatically(
                evaluator, keras_model, test_hgtd_tracks_norm, test_vertex_norm, test_times, data_processor, is_hgtd_only=True
            )
        elif is_hgtd_multi_input:
            test_dataset = create_test_dataset_automatically(
                evaluator, keras_model, test_cells_norm, test_vertex_norm, test_times, data_processor, test_baselines, test_jets_norm, test_tracks_norm, test_hgtd_tracks_norm
            )
        elif is_multi_input:
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
        if is_hgtd_only:
            # For HGTD-only models, make predictions directly using the dataset
            print("Making predictions for HGTD-only model...")
            y_pred = keras_model.predict(test_dataset)
            # Flatten predictions to match expected shape for metrics computation
            y_pred = y_pred.flatten()
            detailed_metrics = evaluator.compute_metrics(test_times, y_pred)
        elif is_hgtd_multi_input:
            # For HGTD multi-input models, make predictions directly using the dataset
            print("Making predictions for HGTD multi-input model...")
            y_pred = keras_model.predict(test_dataset)
            # Flatten predictions to match expected shape for metrics computation
            y_pred = y_pred.flatten()
            detailed_metrics = evaluator.compute_metrics(test_times, y_pred)
        elif is_multi_input:
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
