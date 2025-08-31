"""Main training script for vertex time prediction models with YAML config support and DNN support."""

# python scripts/train.py --config-file config/configs/experiment1.yaml

import os
import sys
import argparse
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config.transformer_config import TransformerConfig
from config.dnn_config import DNNConfig
from config.base_config import BaseConfig
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.data.multi_input_data_loader import MultiInputDataLoader
from src.data.multi_input_data_processor import MultiInputDataProcessor
from src.models.transformer_model import TransformerModel
from src.models.dnn_model import DNNModel
from src.models.baseline_guided_model import BaselineGuidedDNN
from src.models.multi_input_dnn_model import MultiInputDNNModel
from src.models.multi_input_transformer_model import MultiInputTransformerModel
from src.training.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train vertex time prediction model')
    
    # Configuration options
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to YAML configuration file')
    parser.add_argument('--config', type=str, default='transformer',
                       help='Configuration type to use if no YAML file specified (default: transformer)')
    
    # Override parameters (these will override YAML settings if provided)
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing HDF5 data files')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Name for the model (used for saving)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate for optimizer')
    parser.add_argument('--max-cells', type=int, default=None,
                       help='Maximum number of cells per event')
    parser.add_argument('--min-cells', type=int, default=None,
                       help='Minimum number of cells per event')
    parser.add_argument('--use-spatial', type=str, choices=['true', 'false'], default=None,
                       help='Use spatial features (true/false)')
    parser.add_argument('--use-attention-mask', type=str, choices=['true', 'false'], default=None,
                       help='Use attention mask (true/false)')
    
    # Training options
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0, 1, 2)')
    parser.add_argument('--save-config', action='store_true',
                       help='Save final configuration to YAML file')
    parser.add_argument('--skip-jet-validation', action='store_true',
                       help='Skip jet features validation (use with caution)')
    
    return parser.parse_args()


def create_config_from_yaml(yaml_path):
    """Create configuration from YAML file."""
    try:
        # Try to determine config type by checking the content
        import yaml
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Check model architecture type
        if yaml_data.get('model_architecture') == 'baseline_guided_dnn':
            config = DNNConfig.from_yaml(yaml_path)
            logger.info(f"Loaded baseline-guided DNN configuration from: {yaml_path}")
        elif yaml_data.get('model_architecture') == 'multi_input_dnn':
            config = DNNConfig.from_yaml(yaml_path)
            logger.info(f"Loaded multi-input DNN configuration from: {yaml_path}")
        elif yaml_data.get('model_architecture') == 'multi_input_transformer':
            config = TransformerConfig.from_yaml(yaml_path)
            logger.info(f"Loaded multi-input Transformer configuration from: {yaml_path}")
        elif yaml_data.get('model_architecture') == 'two_stage_dnn' or 'cell_encoder_units' in yaml_data:
            config = DNNConfig.from_yaml(yaml_path)
            logger.info(f"Loaded DNN configuration from: {yaml_path}")
        else:
            config = TransformerConfig.from_yaml(yaml_path)
            logger.info(f"Loaded Transformer configuration from: {yaml_path}")
        
        return config
    except Exception as e:
        logger.error(f"Error loading YAML configuration: {e}")
        logger.info("Falling back to default TransformerConfig")
        return TransformerConfig()


def create_config_default(config_type):
    """Create default configuration based on type."""
    if config_type == 'transformer':
        return TransformerConfig()
    elif config_type == 'dnn':
        return DNNConfig()
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def create_config(args):
    """Create configuration based on arguments."""
    # Priority 1: Load from YAML file if provided
    if args.config_file is not None:
        if not os.path.exists(args.config_file):
            raise FileNotFoundError(f"YAML configuration file not found: {args.config_file}")
        
        logger.info(f"Loading configuration from: {args.config_file}")
        config = create_config_from_yaml(args.config_file)
        logger.debug(f"After YAML load, use_spatial_features = {config.use_spatial_features}")
        logger.debug(f"After YAML load, use_attention_mask = {config.use_attention_mask}")
    else:
        # Priority 2: Use default configuration type
        logger.info(f"Using default {args.config} configuration")
        config = create_config_default(args.config)
    
    # Priority 3: Override with command line arguments
    config.update_from_args(args)
    logger.debug(f"After args update, use_spatial_features = {config.use_spatial_features}")
    
    # Handle attention mask argument
    if args.use_attention_mask is not None:
        if args.use_attention_mask == 'true':
            config.use_attention_mask = True
        elif args.use_attention_mask == 'false':
            config.use_attention_mask = False
        logger.debug(f"After args update, use_attention_mask = {config.use_attention_mask}")
    
    return config


def print_training_info(config):
    """Print training information."""
    print("\n" + "="*60)
    print("VERTEX TIME PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Print configuration
    config.print_config()
    
    # Print derived information
    print(f"\nDerived Information:")
    print(f"  Cell features: {len(config.cell_features)} features")
    print(f"  Model directory: {config.model_dir}")
    print(f"  Model path: {config.model_path}")
    print(f"  Using attention mask: {config.use_attention_mask}")
    
    # Print model type
    if isinstance(config, DNNConfig):
        print(f"  Model type: Two-Stage DNN")
    else:
        print(f"  Model type: Transformer")


def validate_jet_features_setup(config, data_loader, skip_validation=False):
    """Validate jet features setup if enabled."""
    if skip_validation:
        logger.warning("Skipping jet features validation as requested.")
        return True
    
    # Check if jet features or jet matching are enabled
    if config.use_jet_features or config.use_cell_jet_matching:
        logger.info("Validating jet features setup...")
        
        # Check if data contains required jet fields
        if not data_loader.validate_jet_features_in_dataset():
            logger.error("Jet features validation failed!")
            logger.error("Your configuration requires jet features, but the dataset doesn't contain them.")
            logger.info("Possible solutions:")
            logger.info("1. Use data processed with cell-jet matching (update your H5 files)")
            logger.info("2. Disable jet features in config: set use_jet_features=false and use_cell_jet_matching=false")
            logger.info("3. Use --skip-jet-validation flag (not recommended)")
            return False
        
        logger.info("✓ Jet features validation passed")
    
    return True


def create_multi_input_datasets_and_model(config, data_processor, train_data, val_data):
    """Create datasets and model for multi-input models."""
    
    (
        train_cells_norm, train_vertex_norm, train_jets_norm, train_tracks_norm, train_times
    ) = train_data
    
    (
        val_cells_norm, val_vertex_norm, val_jets_norm, val_tracks_norm, val_times
    ) = val_data
    
    print(f"\n3. Creating multi-input TensorFlow datasets...")
    print(f"Using attention mask: {config.use_attention_mask}")
    
    # Create multi-input datasets
    train_dataset = data_processor.create_multi_input_dataset(
        train_cells_norm, train_vertex_norm, train_jets_norm, train_tracks_norm, train_times, shuffle=True
    )
    val_dataset = data_processor.create_multi_input_dataset(
        val_cells_norm, val_vertex_norm, val_jets_norm, val_tracks_norm, val_times, shuffle=False
    )
    
    # Create appropriate multi-input model
    feature_dim = len(config.cell_features)
    vertex_dim = train_vertex_norm.shape[1]
    jet_feature_dim = len(config.jet_features)
    track_feature_dim = len(config.track_features)
    
    print(f"Model input dimensions:")
    print(f"  Cell features: {feature_dim}")
    print(f"  Vertex features: {vertex_dim}")
    print(f"  Jet features: {jet_feature_dim}")
    print(f"  Track features: {track_feature_dim}")
    
    is_multi_input_dnn = (getattr(config, 'model_architecture', '') == 'multi_input_dnn')
    is_multi_input_transformer = (getattr(config, 'model_architecture', '') == 'multi_input_transformer')
    
    if is_multi_input_dnn:
        print(f"\n4. Building Multi-Input DNN model...")
        model = MultiInputDNNModel(config)
        keras_model = model.build_model(feature_dim, vertex_dim, jet_feature_dim, track_feature_dim)
    elif is_multi_input_transformer:
        print(f"\n4. Building Multi-Input Transformer model...")
        model = MultiInputTransformerModel(config)
        keras_model = model.build_model(feature_dim, vertex_dim, jet_feature_dim, track_feature_dim)
    else:
        raise ValueError(f"Unsupported multi-input model architecture: {getattr(config, 'model_architecture', '')}")
    
    print(f"Including jet features: {config.jet_features}")
    print(f"Including track features: {config.track_features}")
    
    return train_dataset, val_dataset, model, keras_model


def create_datasets_and_model(config, data_processor, train_cells_norm, val_cells_norm, 
                             train_vertex_norm, val_vertex_norm, train_times, val_times, 
                             train_baselines=None, val_baselines=None):
    """Create datasets and model based on configuration type."""
    
    print(f"\n3. Creating TensorFlow datasets...")
    print(f"Using attention mask: {config.use_attention_mask}")
    
    # Determine model type within function
    is_baseline_guided = (getattr(config, 'model_architecture', '') == 'baseline_guided_dnn')
    is_dnn_model = (isinstance(config, DNNConfig) or 
                   getattr(config, 'model_architecture', '') == 'two_stage_dnn')
    
    if is_baseline_guided:
        print("Creating datasets with baseline predictions for residual learning...")
        train_dataset = data_processor.create_padded_dataset_with_baseline(
            train_cells_norm, train_vertex_norm, train_times, train_baselines, shuffle=True
        )
        val_dataset = data_processor.create_padded_dataset_with_baseline(
            val_cells_norm, val_vertex_norm, val_times, val_baselines, shuffle=False
        )
    elif config.use_attention_mask:
        print("Creating datasets with attention mask support...")
        train_dataset = data_processor.create_padded_dataset_with_mask(
            train_cells_norm, train_vertex_norm, train_times, shuffle=True
        )
        val_dataset = data_processor.create_padded_dataset_with_mask(
            val_cells_norm, val_vertex_norm, val_times, shuffle=False
        )
    else:
        print("Creating datasets with traditional padding...")
        train_dataset = data_processor.create_padded_dataset(
            train_cells_norm, train_vertex_norm, train_times, shuffle=True
        )
        val_dataset = data_processor.create_padded_dataset(
            val_cells_norm, val_vertex_norm, val_times, shuffle=False
        )
    
    # Create appropriate model
    feature_dim = len(config.cell_features)
    print(f"Model input feature dimension: {feature_dim}")
    
    if is_baseline_guided:
        print(f"\n4. Building Baseline-guided DNN model...")
        model = BaselineGuidedDNN(config)
        keras_model = model.build_model(feature_dim, train_vertex_norm.shape[1])
    elif is_dnn_model:
        print(f"\n4. Building DNN model...")
        model = DNNModel(config)
        if config.use_attention_mask:
            keras_model = model.build_model_with_mask(feature_dim, train_vertex_norm.shape[1])
        else:
            keras_model = model.build_model(feature_dim, train_vertex_norm.shape[1])
    else:
        print(f"\n4. Building Transformer model...")
        model = TransformerModel(config)
        if config.use_attention_mask:
            keras_model = model.build_model_with_mask(feature_dim, train_vertex_norm.shape[1])
        else:
            keras_model = model.build_model(feature_dim, train_vertex_norm.shape[1])
    
    if config.use_jet_features:
        print(f"Including jet features: {config.jet_features}")
    
    return train_dataset, val_dataset, model, keras_model


def main():
    """Main training function."""
    args = parse_args()
    
    try:
        # Create configuration
        config = create_config(args)
        
        # Validate configuration
        config.validate_config()
        
        # Print training information
        print_training_info(config)
        
        # Determine model type early for data loading
        is_dnn_model = (isinstance(config, DNNConfig) or 
                       getattr(config, 'model_architecture', '') == 'two_stage_dnn')
        is_baseline_guided = (getattr(config, 'model_architecture', '') == 'baseline_guided_dnn')
        is_multi_input = (getattr(config, 'model_architecture', '') in ['multi_input_dnn', 'multi_input_transformer'])
        
        # Load data
        print(f"\n1. Loading and processing data...")
        if is_multi_input:
            data_loader = MultiInputDataLoader(config)
        else:
            data_loader = DataLoader(config)
        
        # NEW: Validate jet features setup
        if not validate_jet_features_setup(config, data_loader, args.skip_jet_validation):
            return 1
        
        try:
            if is_baseline_guided:
                # Load data with baseline predictions for residual learning
                cell_sequences, vertex_features, vertex_times, sequence_lengths, baseline_predictions = \
                    data_loader.load_data_with_baselines_from_files()
                jet_sequences = track_sequences = None
            elif is_multi_input:
                # Load multi-input data (cells, vertex, jets, tracks)
                cell_sequences, vertex_features, vertex_times, sequence_lengths, jet_sequences, track_sequences = \
                    data_loader.load_multi_input_data_from_files()
                baseline_predictions = None
            else:
                # Standard data loading
                cell_sequences, vertex_features, vertex_times, sequence_lengths = \
                    data_loader.load_data_from_files()
                baseline_predictions = None
                jet_sequences = track_sequences = None
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please check that the data directory exists and contains HDF5 files.")
            return 1
        
        print(f"Loaded {len(vertex_times)} events")
        print(f"Cell feature dimension: {len(config.cell_features)}")
        print(f"Vertex feature dimension: {vertex_features.shape[1]}")
        
        if is_multi_input:
            print(f"Jet feature dimension: {len(config.jet_features)}")
            print(f"Track feature dimension: {len(config.track_features)}")
            print(f"Average jets per event: {sum(len(jets) for jets in jet_sequences) / len(jet_sequences):.1f}")
            print(f"Average tracks per event: {sum(len(tracks) for tracks in track_sequences) / len(track_sequences):.1f}")
        
        # Process data
        print(f"\n2. Splitting and processing data...")
        
        # Add physics features if enabled
        if getattr(config, 'use_physics_informed_features', False):
            print("Adding physics-informed features...")
            from src.data.physics_features import PhysicsFeatureEngineer
            physics_engineer = PhysicsFeatureEngineer(config)
            cell_sequences, feature_names = physics_engineer.add_physics_features(
                cell_sequences, feature_names
            )
            print(f"Enhanced features: {len(feature_names)} total")
        
        if is_multi_input:
            data_processor = MultiInputDataProcessor(config)
        else:
            data_processor = DataProcessor(config)
        
        # Split data
        if is_baseline_guided:
            # For baseline-guided models, we need to split all data together to maintain consistency
            from sklearn.model_selection import train_test_split
            import numpy as np
            
            # Create indices for consistent splitting
            indices = np.arange(len(vertex_times))
            
            # First split: train vs (val + test)
            train_indices, temp_indices = train_test_split(
                indices, test_size=config.test_size + config.val_split, 
                random_state=config.random_state
            )
            
            # Second split: val vs test
            val_indices, test_indices = train_test_split(
                temp_indices, test_size=config.test_size/(config.test_size + config.val_split), 
                random_state=config.random_state
            )
            
            # Split all data using these indices
            train_cells = [cell_sequences[i] for i in train_indices]
            val_cells = [cell_sequences[i] for i in val_indices]
            test_cells = [cell_sequences[i] for i in test_indices]
            
            train_vertex = vertex_features[train_indices]
            val_vertex = vertex_features[val_indices]
            test_vertex = vertex_features[test_indices]
            
            train_times = vertex_times[train_indices]
            val_times = vertex_times[val_indices]
            test_times = vertex_times[test_indices]
            
            train_baselines = baseline_predictions[train_indices]
            val_baselines = baseline_predictions[val_indices]
            test_baselines = baseline_predictions[test_indices]
        elif is_multi_input:
            # Multi-input data splitting
            (train_cells, val_cells, test_cells), \
            (train_vertex, val_vertex, test_vertex), \
            (train_jets, val_jets, test_jets), \
            (train_tracks, val_tracks, test_tracks), \
            (train_times, val_times, test_times) = data_processor.split_data(
                cell_sequences, vertex_features, vertex_times, jet_sequences, track_sequences
            )
            train_baselines = val_baselines = test_baselines = None
        else:
            # Standard data splitting
            (train_cells, val_cells, test_cells), \
            (train_vertex, val_vertex, test_vertex), \
            (train_times, val_times, test_times) = data_processor.split_data(
                cell_sequences, vertex_features, vertex_times
            )
            train_baselines = val_baselines = test_baselines = None
        
        # Normalize features (includes time calibration if enabled)
        if is_multi_input:
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
        else:
            (train_cells_norm, val_cells_norm, test_cells_norm), \
            (train_vertex_norm, val_vertex_norm, test_vertex_norm), \
            norm_params = data_processor.normalize_features(
                train_cells, val_cells, test_cells,
                train_vertex, val_vertex, test_vertex,
                train_times, val_times, test_times
            )
            train_jets_norm = val_jets_norm = test_jets_norm = None
            train_tracks_norm = val_tracks_norm = test_tracks_norm = None
        
        # Create datasets and model based on configuration type
        if is_multi_input:
            train_data = (train_cells_norm, train_vertex_norm, train_jets_norm, train_tracks_norm, train_times)
            val_data = (val_cells_norm, val_vertex_norm, val_jets_norm, val_tracks_norm, val_times)
            train_dataset, val_dataset, model, keras_model = create_multi_input_datasets_and_model(
                config, data_processor, train_data, val_data
            )
        else:
            train_dataset, val_dataset, model, keras_model = create_datasets_and_model(
                config, data_processor, train_cells_norm, val_cells_norm,
                train_vertex_norm, val_vertex_norm, train_times, val_times,
                train_baselines, val_baselines
            )
        
        # Train model
        print(f"\n5. Training model...")
        trainer = Trainer(config, model)
        
        # Validate datasets before training
        trainer.validate_training_data(train_dataset, val_dataset)
        
        try:
            history = trainer.train(train_dataset, val_dataset, verbose=args.verbose)
            
            # Save training history
            trainer.save_training_history()
            
            # Save configuration (both JSON and YAML)
            config.save_config()  # Save JSON (original format)
            
            if args.save_config or args.config_file:
                # Save YAML configuration for reproducibility
                config.save_yaml()
            
            # Print training summary
            summary = trainer.get_training_summary()
            print(f"\n" + "="*60)
            print("TRAINING SUMMARY")
            print("="*60)
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
            
            print(f"\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Model saved to: {config.model_path}")
            print(f"Configuration saved to: {config.model_dir}")
            print(f"Training history saved to: {config.model_dir}")
            
            # Print feature information
            if is_multi_input:
                print(f"\nMulti-input features:")
                print(f"  Cell features: {config.cell_features}")
                print(f"  Jet features: {config.jet_features}")
                print(f"  Track features: {config.track_features}")
            elif config.use_jet_features:
                print(f"\nJet features included: {config.jet_features}")
            if config.use_cell_jet_matching:
                print(f"Cell-jet matching filter applied during training")
            
            # Print mask information
            if config.use_attention_mask:
                print(f"✓ Attention mask enabled for improved performance")
                print(f"✓ Smart padding applied (feature-specific values)")
            else:
                print(f"○ Traditional padding used (compatibility mode)")
            
            # Print model type information
            if is_multi_input:
                model_arch = getattr(config, 'model_architecture', '')
                if model_arch == 'multi_input_dnn':
                    print(f"✓ Multi-Input DNN model with jets and tracks")
                elif model_arch == 'multi_input_transformer':
                    print(f"✓ Multi-Input Transformer model with jets and tracks")
            elif isinstance(config, DNNConfig):
                print(f"✓ Two-Stage DNN model with attention pooling")
            else:
                print(f"✓ Transformer model architecture")
            
            # Print evaluation command
            print(f"\nTo evaluate this model, run:")
            print(f"python scripts/evaluate.py --model-dir {config.model_dir} --load-data")
            
            return 0
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
    except Exception as e:
        print(f"Configuration error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
