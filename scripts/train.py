"""Main training script for vertex time prediction models with YAML config support."""

# python scripts/train.py --config-file config/configs/experiment1.yaml

import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.transformer_config import TransformerConfig
from config.base_config import BaseConfig
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.models.transformer_model import TransformerModel
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
    parser.add_argument('--use-spatial', action='store_true', default=None,
                   help='Use spatial features')
    
    # Training options
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0, 1, 2)')
    parser.add_argument('--save-config', action='store_true',
                       help='Save final configuration to YAML file')
    parser.add_argument('--skip-jet-validation', action='store_true',
                       help='Skip jet features validation (use with caution)')
    
    return parser.parse_args()


def create_config_from_yaml(yaml_path: str):
    """Create configuration from YAML file."""
    try:
        # Try to create TransformerConfig from YAML
        config = TransformerConfig.from_yaml(yaml_path)
        return config
    except Exception as e:
        print(f"Error loading YAML configuration: {e}")
        print("Falling back to default TransformerConfig")
        return TransformerConfig()


def create_config_default(config_type: str):
    """Create default configuration based on type."""
    if config_type == 'transformer':
        return TransformerConfig()
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def create_config(args):
    """Create configuration based on arguments."""
    # Priority 1: Load from YAML file if provided
    if args.config_file is not None:
        if not os.path.exists(args.config_file):
            raise FileNotFoundError(f"YAML configuration file not found: {args.config_file}")
        
        print(f"Loading configuration from: {args.config_file}")
        config = create_config_from_yaml(args.config_file)
        print(f"DEBUG: After YAML load, use_spatial_features = {config.use_spatial_features}")
    else:
        # Priority 2: Use default configuration type
        print(f"Using default {args.config} configuration")
        config = create_config_default(args.config)
    
    # Priority 3: Override with command line arguments
    config.update_from_args(args)
    print(f"DEBUG: After args update, use_spatial_features = {config.use_spatial_features}")
    
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


def validate_jet_features_setup(config, data_loader, skip_validation=False):
    """Validate jet features setup if enabled."""
    if skip_validation:
        print("Skipping jet features validation as requested.")
        return True
    
    # Check if jet features or jet matching are enabled
    if config.use_jet_features or config.use_cell_jet_matching:
        print("\nValidating jet features setup...")
        
        # Check if data contains required jet fields
        if not data_loader.validate_jet_features_in_dataset():
            print("\nError: Jet features validation failed!")
            print("Your configuration requires jet features, but the dataset doesn't contain them.")
            print("\nPossible solutions:")
            print("1. Use data processed with cell-jet matching (update your H5 files)")
            print("2. Disable jet features in config: set use_jet_features=false and use_cell_jet_matching=false")
            print("3. Use --skip-jet-validation flag (not recommended)")
            return False
        
        print("✓ Jet features validation passed")
    
    return True


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
        
        # Load data
        print(f"\n1. Loading and processing data...")
        data_loader = DataLoader(config)
        
        # NEW: Validate jet features setup
        if not validate_jet_features_setup(config, data_loader, args.skip_jet_validation):
            return 1
        
        try:
            cell_sequences, vertex_features, vertex_times, sequence_lengths = \
                data_loader.load_data_from_files()
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please check that the data directory exists and contains HDF5 files.")
            return 1
        
        print(f"Loaded {len(vertex_times)} events")
        print(f"Cell feature dimension: {len(config.cell_features)}")
        print(f"Vertex feature dimension: {vertex_features.shape[1]}")
        
        # Process data
        print(f"\n2. Splitting and processing data...")
        data_processor = DataProcessor(config)
        
        # Split data
        (train_cells, val_cells, test_cells), \
        (train_vertex, val_vertex, test_vertex), \
        (train_times, val_times, test_times) = data_processor.split_data(
            cell_sequences, vertex_features, vertex_times
        )
        
        # Normalize features (includes time calibration if enabled)
        (train_cells_norm, val_cells_norm, test_cells_norm), \
        (train_vertex_norm, val_vertex_norm, test_vertex_norm), \
        norm_params = data_processor.normalize_features(
            train_cells, val_cells, test_cells,
            train_vertex, val_vertex, test_vertex,
            train_times, val_times, test_times
        )
        
        # Create datasets
        print(f"\n3. Creating TensorFlow datasets...")
        train_dataset = data_processor.create_padded_dataset(
            train_cells_norm, train_vertex_norm, train_times, shuffle=True
        )
        val_dataset = data_processor.create_padded_dataset(
            val_cells_norm, val_vertex_norm, val_times, shuffle=False
        )
        
        # Build model
        print(f"\n4. Building model...")
        model = TransformerModel(config)
        
        # Use actual cell features dimension (includes jet features if enabled)
        feature_dim = len(config.cell_features)
        print(f"Model input feature dimension: {feature_dim}")
        
        if config.use_jet_features:
            print(f"Including jet features: {config.jet_features}")
        
        keras_model = model.build_model(feature_dim, train_vertex_norm.shape[1])
        
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
            if config.use_jet_features:
                print(f"\nJet features included: {config.jet_features}")
            if config.use_cell_jet_matching:
                print(f"Cell-jet matching filter applied during training")
            
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
