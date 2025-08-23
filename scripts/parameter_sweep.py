"""Optimized parameter sweep with data caching for transformer and DNN models."""

import os
import sys
import json
import time
import argparse
import itertools
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.transformer_config import TransformerConfig
from config.dnn_config import DNNConfig
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.models.transformer_model import TransformerModel
from src.models.dnn_model import DNNModel
from src.training.trainer import Trainer


class OptimizedParameterSweep:
    """Efficient parameter sweep with data caching and in-memory training."""
    
    def __init__(self, base_config_path: str, output_dir: str = None):
        self.base_config_path = base_config_path
        
        # Setup output directory
        if output_dir is None:
            from config.base_config import get_external_dir
            results_base_dir = get_external_dir("results", "results")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(results_base_dir, f"sweep_{timestamp}")
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cached data (loaded once, used multiple times)
        self.cached_data = None
        self.results = []
        
    def get_parameter_grids(self):
        """Define parameter grids for different sweep types."""
        return {
            'transformer_quick': {
                'd_model': [64, 128],
                'num_heads': [4, 8],
                'learning_rate': [1e-4, 5e-4],
                'dropout_rate': [0.1, 0.2],
                'epochs': [20],  # Reduced for sweeps
            },
            'transformer_full': {
                'd_model': [32, 64, 128, 256],
                'num_heads': [2, 4, 8, 16],
                'num_transformer_blocks': [2, 3, 4],
                'learning_rate': [5e-5, 1e-4, 5e-4],
                'dropout_rate': [0.1, 0.15, 0.2],
                'batch_size': [64, 128],
                'epochs': [20],
            },
            'dnn_quick': {
                'model_type': ['dnn'],
                'cell_encoder_units': [[64, 32], [128, 64]],
                'event_encoder_units': [[256, 128, 64], [512, 256, 128]],
                'learning_rate': [1e-4, 5e-4],
                'attention_hidden_units': [32, 64],
                'epochs': [20],
            },
            'dnn_full': {
                'model_type': ['dnn'],
                'cell_encoder_units': [[32, 16], [64, 32], [128, 64], [256, 128]],
                'event_encoder_units': [[128, 64], [256, 128, 64], [512, 256, 128]],
                'learning_rate': [5e-5, 1e-4, 5e-4, 1e-3],
                'cell_dropout_rate': [0.1, 0.2, 0.3],
                'attention_hidden_units': [16, 32, 64],
                'use_attention_pooling': [True, False],
                'epochs': [20],
            },
            'comparison': {
                'model_type': ['transformer', 'dnn'],
                'learning_rate': [1e-4, 5e-4],
                'use_attention_mask': [True, False],
                'epochs': [15],
                # Transformer params (used only for transformer)
                'd_model': [128],
                'num_heads': [8],
                # DNN params (used only for DNN)  
                'cell_encoder_units': [[64, 32]],
                'event_encoder_units': [[256, 128, 64]],
            }
        }
    
    def validate_params(self, params: Dict) -> bool:
        """Validate parameter combination."""
        model_type = params.get('model_type', 'transformer')
        
        if model_type == 'transformer':
            # Check d_model divisible by num_heads
            if 'd_model' in params and 'num_heads' in params:
                if params['d_model'] % params['num_heads'] != 0:
                    return False
        elif model_type == 'dnn':
            # Check DNN-specific constraints
            if 'event_dropout_rates' in params and 'event_encoder_units' in params:
                if len(params['event_dropout_rates']) != len(params['event_encoder_units']):
                    return False
        
        # General constraints
        if 'learning_rate' in params:
            if not (1e-6 <= params['learning_rate'] <= 1e-2):
                return False
        
        return True
    
    def load_and_cache_data(self, config):
        """Load data once and cache for all experiments."""
        if self.cached_data is not None:
            return self.cached_data
        
        print("Loading and processing data (this will be cached)...")
        start_time = time.time()
        
        # Load raw data
        data_loader = DataLoader(config)
        cell_sequences, vertex_features, vertex_times, sequence_lengths = \
            data_loader.load_data_from_files()
        
        # Process data
        data_processor = DataProcessor(config)
        
        # Split data
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
        
        self.cached_data = {
            'train_cells': train_cells_norm,
            'val_cells': val_cells_norm,
            'test_cells': test_cells_norm,
            'train_vertex': train_vertex_norm,
            'val_vertex': val_vertex_norm,
            'test_vertex': test_vertex_norm,
            'train_times': train_times,
            'val_times': val_times,
            'test_times': test_times,
            'data_processor': data_processor,
            'norm_params': norm_params
        }
        
        load_time = time.time() - start_time
        print(f"Data loading completed in {load_time:.1f}s")
        print(f"Training samples: {len(train_times)}, Validation: {len(val_times)}")
        
        return self.cached_data
    
    def create_config(self, base_config, params: Dict):
        """Create config for specific experiment."""
        model_type = params.get('model_type', 'transformer')
        
        if model_type == 'dnn':
            config = DNNConfig.from_yaml(self.base_config_path)
        else:
            config = TransformerConfig.from_yaml(self.base_config_path)
        
        # Apply parameters
        for key, value in params.items():
            if key != 'model_type' and hasattr(config, key):
                setattr(config, key, value)
        
        # Set model name
        exp_id = len(self.results)
        config.model_name = f"sweep_exp_{exp_id:03d}_{model_type}"
        
        return config
    
    def train_single_model(self, config, data: Dict, exp_id: int) -> Dict:
        """Train a single model with cached data."""
        print(f"[{exp_id:3d}] Training {config.model_name}... ", end="")
        start_time = time.time()
        
        try:
            # Determine model type and create appropriate model
            model_type = getattr(config, 'model_architecture', 'transformer')
            if hasattr(config, 'model_type'):
                model_type = config.model_type
            elif isinstance(config, DNNConfig):
                model_type = 'dnn'
                
            if model_type == 'dnn' or isinstance(config, DNNConfig):
                model_wrapper = DNNModel(config)
                feature_dim = len(config.cell_features)
                
                if config.use_attention_mask:
                    keras_model = model_wrapper.build_model_with_mask(
                        feature_dim, data['train_vertex'].shape[1]
                    )
                    # Create datasets with mask
                    train_dataset = data['data_processor'].create_padded_dataset_with_mask(
                        data['train_cells'], data['train_vertex'], data['train_times']
                    )
                    val_dataset = data['data_processor'].create_padded_dataset_with_mask(
                        data['val_cells'], data['val_vertex'], data['val_times'], shuffle=False
                    )
                else:
                    keras_model = model_wrapper.build_model(
                        feature_dim, data['train_vertex'].shape[1]
                    )
                    # Create datasets without mask
                    train_dataset = data['data_processor'].create_padded_dataset(
                        data['train_cells'], data['train_vertex'], data['train_times']
                    )
                    val_dataset = data['data_processor'].create_padded_dataset(
                        data['val_cells'], data['val_vertex'], data['val_times'], shuffle=False
                    )
            else:
                model_wrapper = TransformerModel(config)
                feature_dim = len(config.cell_features)
                
                if config.use_attention_mask:
                    keras_model = model_wrapper.build_model_with_mask(
                        feature_dim, data['train_vertex'].shape[1]
                    )
                    # Create datasets with mask
                    train_dataset = data['data_processor'].create_padded_dataset_with_mask(
                        data['train_cells'], data['train_vertex'], data['train_times']
                    )
                    val_dataset = data['data_processor'].create_padded_dataset_with_mask(
                        data['val_cells'], data['val_vertex'], data['val_times'], shuffle=False
                    )
                else:
                    keras_model = model_wrapper.build_model(
                        feature_dim, data['train_vertex'].shape[1]
                    )
                    # Create datasets without mask
                    train_dataset = data['data_processor'].create_padded_dataset(
                        data['train_cells'], data['train_vertex'], data['train_times']
                    )
                    val_dataset = data['data_processor'].create_padded_dataset(
                        data['val_cells'], data['val_vertex'], data['val_times'], shuffle=False
                    )
            
            # Train model
            trainer = Trainer(config, model_wrapper)
            history = trainer.train(train_dataset, val_dataset, verbose=0)
            
            # Extract results
            history_dict = history.history
            best_epoch = np.argmin(history_dict['val_loss'])
            
            results = {
                'status': 'success',
                'training_time': time.time() - start_time,
                'best_epoch': best_epoch + 1,
                'best_val_loss': float(history_dict['val_loss'][best_epoch]),
                'best_val_mae': float(history_dict['val_mae'][best_epoch]),
                'final_val_loss': float(history_dict['val_loss'][-1]),
                'final_val_mae': float(history_dict['val_mae'][-1]),
                'total_epochs': len(history_dict['loss'])
            }
            
            if 'val_root_mean_squared_error' in history_dict:
                results['best_val_rmse'] = float(history_dict['val_root_mean_squared_error'][best_epoch])
            
            print(f"✓ {results['training_time']:.1f}s (loss: {results['best_val_loss']:.4f})")
            
        except Exception as e:
            results = {
                'status': 'failed',
                'training_time': time.time() - start_time,
                'error': str(e)[:200]
            }
            print(f"✗ {results['training_time']:.1f}s ({str(e)[:50]})")
        
        return results
    
    def run_sweep(self, grid_type: str = 'transformer_quick', max_experiments: int = None):
        """Run parameter sweep."""
        grids = self.get_parameter_grids()
        if grid_type not in grids:
            raise ValueError(f"Unknown grid type: {grid_type}. Available: {list(grids.keys())}")
        
        parameter_grid = grids[grid_type]
        
        # Generate all valid combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        all_combinations = list(itertools.product(*param_values))
        
        valid_configs = []
        for combo in all_combinations:
            params = dict(zip(param_names, combo))
            if self.validate_params(params):
                valid_configs.append(params)
        
        if max_experiments:
            valid_configs = valid_configs[:max_experiments]
        
        print(f"\nParameter Sweep: {grid_type}")
        print(f"Valid experiments: {len(valid_configs)}")
        print(f"Estimated time: {len(valid_configs) * 5:.1f} min (after initial data loading)")
        
        # Load base config and cache data
        base_config = TransformerConfig.from_yaml(self.base_config_path)
        cached_data = self.load_and_cache_data(base_config)
        
        # Save experiment plan
        plan_path = os.path.join(self.output_dir, "experiment_plan.json")
        with open(plan_path, 'w') as f:
            json.dump(valid_configs, f, indent=2, default=str)
        
        # Run experiments
        print(f"\nStarting {len(valid_configs)} experiments...")
        start_total = time.time()
        
        for i, params in enumerate(valid_configs):
            # Create config for this experiment
            config = self.create_config(base_config, params)
            
            # Train model
            results = self.train_single_model(config, cached_data, i+1)
            
            # Store results
            full_results = {**params, **results, 'exp_id': i+1}
            self.results.append(full_results)
            
            # Save results periodically
            if (i + 1) % 5 == 0 or i == len(valid_configs) - 1:
                results_df = pd.DataFrame(self.results)
                results_df.to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
        
        total_time = time.time() - start_total
        self.print_summary(total_time)
    
    def print_summary(self, total_time: float):
        """Print experiment summary."""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        successful = df[df['status'] == 'success']
        
        print(f"\n{'='*60}")
        print("PARAMETER SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Successful experiments: {len(successful)}/{len(df)}")
        
        if len(successful) > 0:
            best_idx = successful['best_val_loss'].idxmin()
            best_exp = successful.loc[best_idx]
            
            print(f"\nBest experiment:")
            print(f"  Exp ID: {best_exp['exp_id']}")
            print(f"  Val Loss: {best_exp['best_val_loss']:.4f}")
            print(f"  Val MAE: {best_exp['best_val_mae']:.4f}")
            print(f"  Model: {best_exp.get('model_type', 'transformer')}")
            
            # Show key parameters
            key_params = ['learning_rate', 'd_model', 'num_heads', 'cell_encoder_units', 
                         'event_encoder_units', 'use_attention_mask']
            print("  Params:")
            for param in key_params:
                if param in best_exp:
                    print(f"    {param}: {best_exp[param]}")
            
            # Model comparison if available
            if 'model_type' in df.columns and df['model_type'].nunique() > 1:
                print(f"\nModel Comparison:")
                for model_type in df['model_type'].unique():
                    model_results = successful[successful['model_type'] == model_type]
                    if len(model_results) > 0:
                        best_loss = model_results['best_val_loss'].min()
                        avg_loss = model_results['best_val_loss'].mean()
                        print(f"  {model_type}: best={best_loss:.4f}, avg={avg_loss:.4f} ({len(model_results)} exp)")
        
        print(f"\nResults saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Optimized parameter sweep')
    parser.add_argument('--config', type=str, required=True, help='Base config file')
    parser.add_argument('--grid', type=str, default='transformer_quick',
                       choices=['transformer_quick', 'transformer_full', 'dnn_quick', 
                               'dnn_full', 'comparison'],
                       help='Parameter grid type')
    parser.add_argument('--max-exp', type=int, help='Max number of experiments')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return 1
    
    try:
        sweep = OptimizedParameterSweep(args.config, args.output)
        sweep.run_sweep(args.grid, args.max_exp)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
