"""Parameter sweep script for transformer model hyperparameter optimization with mask support."""

import os
import sys
import json
import time
import itertools
import pandas as pd
from datetime import datetime
from typing import Dict, List
import subprocess

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.transformer_config import TransformerConfig


class ParameterSweep:
    """Handle parameter sweep experiments for transformer model with attention mask support."""
    
    def __init__(self, base_config_path: str, output_dir: str = None):
        """
        Initialize parameter sweep.
        
        Args:
            base_config_path: Path to base configuration YAML file
            output_dir: Directory to save sweep results
        """
        self.base_config_path = base_config_path
        
        if output_dir is None:
            # Use external results directory
            from config.base_config import get_external_dir
            results_base_dir = get_external_dir("results", "results")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(results_base_dir, "parameter_sweep_{}".format(timestamp))
        else:
            self.output_dir = output_dir
            
        self.experiments_dir = os.path.join(self.output_dir, "experiments")
        self.configs_dir = os.path.join(self.output_dir, "configs")
        
        # Create directories
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
        
        # Load base configuration
        self.base_config = TransformerConfig.from_yaml(base_config_path)
        
        # Results storage
        self.results = []
        
    def define_parameter_grids(self):
        """Define all parameter grids including mask support with expanded 'full' grid."""
        return {
            'quick': {
                'd_model': [32, 64, 128],
                'learning_rate': [1e-5, 5e-5, 1e-4],
                'num_heads': [2, 4, 8],
                'dropout_rate': [0.1, 0.2],
            },
            'full': {
                # Expanded parameter ranges for comprehensive search
                'd_model': [16, 32, 64, 128, 256, 512],                    # Expanded from [32, 64, 128]
                'num_heads': [2, 4, 8, 16, 32],                        # Expanded from [2, 4, 8]
                'num_transformer_blocks': [1, 2, 3, 4, 5, 6],             # Expanded from [2, 3, 4]
                'dropout_rate': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],        # Expanded from [0.05, 0.1, 0.2]
                'vertex_dense_units': [4, 8, 16, 32],            # Expanded from [8, 16, 32]
                'learning_rate': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],  # Expanded from [1e-5, 5e-5, 1e-4, 2e-4]
                'batch_size': [16, 32, 64, 128, 256],                  # Expanded from [32, 64, 128]
                'max_cells': [20, 40, 60, 80],                           # New parameter
            },
            'architecture': {
                'd_model': [32, 64, 128, 256],
                'num_heads': [2, 4, 8, 16],
                'num_transformer_blocks': [1, 2, 3, 4, 5],
                'dff': [64, 128, 256, 512],
            },
            'training': {
                'learning_rate': [5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4],
                'batch_size': [16, 32, 64, 128, 256],
                'lr_reduction_factor': [0.3, 0.5, 0.7, 0.9],
            },
            'regularization': {
                'dropout_rate': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
                'final_dropout_rates': [
                    [0.1, 0.05, 0.05],
                    [0.2, 0.1, 0.1], 
                    [0.3, 0.2, 0.1],
                    [0.4, 0.3, 0.2]
                ],
            },
            'dense_layers': {
                'vertex_dense_units': [4, 8, 16, 32, 64],
                'final_dense_units': [
                    [16, 8],
                    [32, 16, 8],
                    [64, 32, 16],
                    [32, 16],
                    [64, 32]
                ],
            },
            # NEW: Mask comparison experiments
            'mask_comparison': {
                'use_attention_mask': [True, False],
                'd_model': [64, 128],
                'num_heads': [4, 8],
                'learning_rate': [5e-5, 1e-4],
                'dropout_rate': [0.1, 0.2],
            },
            # NEW: Mask optimization - only for mask-enabled models
            'mask_optimization': {
                'use_attention_mask': [True],  # Only mask models
                'd_model': [32, 64, 128, 256],
                'num_heads': [2, 4, 8, 16],
                'num_transformer_blocks': [2, 3, 4],
                'learning_rate': [1e-5, 5e-5, 1e-4],
                'batch_size': [32, 64, 128],
            },
            # NEW: Features comparison
            'features_comparison': {
                'use_attention_mask': [True],
                'use_spatial_features': [True, False],
                'use_jet_features': [True, False],
                'use_cell_jet_matching': [True, False],
                'd_model': [64, 128],
                'learning_rate': [5e-5, 1e-4],
            }
        }
    
    def print_sweep_info(self, parameter_grid, grid_type):
        """Print parameter sweep information with enhanced details for expanded grids."""
        print("\n" + "="*70)
        print("PARAMETER SWEEP: {}".format(grid_type.upper()))
        print("="*70)
        print("Base config: {}".format(os.path.basename(self.base_config_path)))
        print("Output dir: {}".format(self.output_dir))
        
        print("\nParameters to sweep:")
        total_combinations = 1
        for param, values in parameter_grid.items():
            print("  {}: {} ({} values)".format(param, values, len(values)))
            total_combinations *= len(values)
        
        print("\nCombination Analysis:")
        print("  Total combinations: {:,}".format(total_combinations))
        
        # Estimate valid combinations (rough approximation)
        if grid_type == 'full':
            # For the expanded full grid, expect ~60-70% valid combinations due to constraints
            estimated_valid = int(total_combinations * 0.65)
            print("  Estimated valid combinations: ~{:,} (after constraints)".format(estimated_valid))
            print("  Common invalid cases:")
            print("    - d_model not divisible by num_heads")
            print("    - Very large model + very small batch size")
            print("    - Extreme parameter combinations")
        else:
            estimated_valid = int(total_combinations * 0.85)
            print("  Estimated valid combinations: ~{:,}".format(estimated_valid))
        
        # Time estimation with different scenarios
        print("\nTime Estimation:")
        avg_time_per_exp = 120  # seconds
        fast_time_per_exp = 60   # for quick experiments
        slow_time_per_exp = 300  # for large models
        
        if grid_type == 'quick':
            estimated_hours = estimated_valid * fast_time_per_exp / 3600
        elif grid_type == 'full':
            estimated_hours = estimated_valid * slow_time_per_exp / 3600
        else:
            estimated_hours = estimated_valid * avg_time_per_exp / 3600
        
        print("  Estimated time: ~{:.1f} hours ({:.1f} days)".format(
            estimated_hours, estimated_hours / 24))
        
        if estimated_hours > 48:
            print("  ⚠️  WARNING: This is a very long sweep!")
            print("  💡 Consider using --max-experiments to limit the scope")
            print("  💡 Or use a smaller grid like 'quick' or 'architecture'")
        
        # Print sweep type information
        if 'use_attention_mask' in parameter_grid:
            mask_values = parameter_grid['use_attention_mask']
            if len(set(mask_values)) > 1:
                print("\n*** MASK COMPARISON SWEEP ***")
                print("This sweep will compare mask-enabled vs traditional models")
            elif mask_values[0]:
                print("\n*** MASK-OPTIMIZED SWEEP ***")
                print("This sweep focuses on optimizing mask-enabled models")
            else:
                print("\n*** TRADITIONAL MODEL SWEEP ***")
                print("This sweep uses traditional models only")
        else:
            base_mask_setting = getattr(self.base_config, 'use_attention_mask', 'undefined')
            print("\n*** USING BASE CONFIG MASK SETTING ***")
            print("Mask setting: {}".format(base_mask_setting))
        
        # Special warnings for expanded grids
        if grid_type == 'full':
            print("\n🎯 EXPANDED FULL GRID FEATURES:")
            print("  ✓ Extended d_model range: 16-512")
            print("  ✓ Extended num_heads range: 1-32") 
            print("  ✓ Extended learning_rate range: 1e-6 to 1e-3")
            print("  ✓ New max_cells parameter: 40-100")
            print("  ✓ Enhanced validation constraints")
        
        print("="*70)
    
    def validate_parameter_combination(self, params):
    """Validate parameter combination including mask-specific rules and expanded ranges."""
    # d_model divisible by num_heads
    if 'd_model' in params and 'num_heads' in params:
        if params['d_model'] % params['num_heads'] != 0:
            return False
    
    # Learning rate range validation
    if 'learning_rate' in params:
        if params['learning_rate'] > 2e-3 or params['learning_rate'] < 1e-7:
            return False
    
    # Dropout rate range validation
    if 'dropout_rate' in params:
        if params['dropout_rate'] < 0 or params['dropout_rate'] > 0.5:
            return False
    
    # Model size constraints - avoid extremely large models
    if 'd_model' in params and 'num_transformer_blocks' in params:
        # Limit very large configurations to prevent memory issues
        if params['d_model'] >= 512 and params['num_transformer_blocks'] >= 5:
            return False
    
    # Batch size and model size compatibility
    if 'batch_size' in params and 'd_model' in params:
        # Very small batch sizes with very large models may be unstable
        if params['batch_size'] <= 8 and params['d_model'] >= 256:
            return False
    
    # max_cells validation
    if 'max_cells' in params:
        if params['max_cells'] < 10 or params['max_cells'] > 200:
            return False
    
    # num_heads edge case validation
    if 'num_heads' in params:
        if params['num_heads'] < 1 or params['num_heads'] > 64:
            return False
    
    # Ensure num_transformer_blocks is reasonable
    if 'num_transformer_blocks' in params:
        if params['num_transformer_blocks'] < 1 or params['num_transformer_blocks'] > 8:
            return False
    
    # vertex_dense_units validation
    if 'vertex_dense_units' in params:
        if params['vertex_dense_units'] < 1 or params['vertex_dense_units'] > 256:
            return False
    
    # Mask-specific validations
    if 'use_attention_mask' in params:
        # If using jet features, prefer mask-enabled models (but allow both)
        if 'use_jet_features' in params and params['use_jet_features'] and not params['use_attention_mask']:
            # Allow but this combination may not be optimal
            pass
    
    return True
    
    def generate_experiment_configs(self, parameter_grid):
        """Generate all valid parameter combinations."""
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        combinations = list(itertools.product(*param_values))
        
        experiment_configs = []
        for i, combination in enumerate(combinations):
            config_dict = dict(zip(param_names, combination))
            if self.validate_parameter_combination(config_dict):
                config_dict['experiment_id'] = "exp_{:04d}".format(i)
                experiment_configs.append(config_dict)
        
        print("Generated {} valid configurations".format(len(experiment_configs)))
        return experiment_configs
    
    def create_experiment_config(self, exp_id, params):
        """Create configuration file for experiment."""
        config = TransformerConfig.from_yaml(self.base_config_path)
        
        for param_name, param_value in params.items():
            if param_name != 'experiment_id' and hasattr(config, param_name):
                setattr(config, param_name, param_value)
        
        # Create descriptive model name including mask info
        mask_suffix = ""
        if 'use_attention_mask' in params:
            mask_suffix = "_mask" if params['use_attention_mask'] else "_trad"
        
        config.model_name = "sweep_{}{}".format(exp_id, mask_suffix)
        
        # Set default dff if not specified
        if 'dff' not in params:
            config.dff = config.d_model * 2
        
        config_path = os.path.join(self.configs_dir, "{}.yaml".format(exp_id))
        config.save_yaml(config_path)
        return config_path
    
    def run_single_experiment(self, exp_id, config_path):
        """Run single experiment."""
        print("Running {}...".format(exp_id), end=" ")
        start_time = time.time()
        
        try:
            cmd = [
                "python", "scripts/train.py",
                "--config-file", config_path,
                "--verbose", "0"
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                results = self.extract_training_results(exp_id)
                results['status'] = 'success'
                results['training_time'] = training_time
                print("✓ {:.1f}s".format(training_time))
            else:
                results = {'status': 'failed', 'training_time': training_time}
                print("✗ {:.1f}s".format(training_time))
                # Store error information for debugging
                if result.stderr:
                    results['error_stderr'] = result.stderr[:500]  # First 500 chars
                if result.stdout:
                    results['error_stdout'] = result.stdout[-500:]  # Last 500 chars
                
        except Exception as e:
            results = {'status': 'error', 'error': str(e)}
            print("✗ Error")
        
        return results
    
    def extract_training_results(self, exp_id):
        """Extract results from training."""
        mask_suffix = ""
        # Try to determine mask suffix from experiment config
        config_path = os.path.join(self.configs_dir, "{}.yaml".format(exp_id))
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                if config_data.get('use_attention_mask', True):
                    mask_suffix = "_mask"
                else:
                    mask_suffix = "_trad"
            except:
                pass
        
        model_dir = os.path.join("models", "sweep_{}{}".format(exp_id, mask_suffix))
        results = {}
        
        try:
            history_path = os.path.join(model_dir, "training_history.npz")
            if os.path.exists(history_path):
                import numpy as np
                history_data = dict(np.load(history_path))
                
                results['final_val_loss'] = float(history_data['val_loss'][-1])
                results['final_val_mae'] = float(history_data['val_mae'][-1])
                
                best_epoch = int(np.argmin(history_data['val_loss']))
                results['best_epoch'] = best_epoch
                results['best_val_loss'] = float(history_data['val_loss'][best_epoch])
                results['best_val_mae'] = float(history_data['val_mae'][best_epoch])
                
                # Add RMSE if available
                if 'val_root_mean_squared_error' in history_data:
                    results['best_val_rmse'] = float(history_data['val_root_mean_squared_error'][best_epoch])
                    results['final_val_rmse'] = float(history_data['val_root_mean_squared_error'][-1])
            
            # Check if model file exists (try both .h5 and .keras for compatibility)
            model_h5_path = os.path.join(model_dir, "model.h5")
            model_keras_path = os.path.join(model_dir, "model.keras")
            
            if os.path.exists(model_h5_path):
                results['model_saved'] = True
                results['model_path'] = model_h5_path
            elif os.path.exists(model_keras_path):
                results['model_saved'] = True
                results['model_path'] = model_keras_path
            else:
                results['model_saved'] = False
                
        except Exception as e:
            results['extraction_error'] = str(e)
        
        return results
    
    def run_parameter_sweep(self, grid_type='quick', max_experiments=None):
        """Run parameter sweep."""
        grids = self.define_parameter_grids()
        
        if grid_type not in grids:
            available_grids = list(grids.keys())
            raise ValueError("Unknown grid type: {}. Available: {}".format(grid_type, available_grids))
        
        parameter_grid = grids[grid_type]
        self.print_sweep_info(parameter_grid, grid_type)
        
        experiment_configs = self.generate_experiment_configs(parameter_grid)
        
        if max_experiments is not None:
            experiment_configs = experiment_configs[:max_experiments]
            print("Limited to {} experiments".format(len(experiment_configs)))
        
        # Save experiment plan
        plan_path = os.path.join(self.output_dir, "experiment_plan.json")
        with open(plan_path, 'w') as f:
            json.dump(experiment_configs, f, indent=2)
        
        # Save sweep metadata
        metadata = {
            'grid_type': grid_type,
            'base_config': self.base_config_path,
            'total_experiments': len(experiment_configs),
            'max_experiments': max_experiments,
            'timestamp': datetime.now().isoformat(),
            'base_config_mask_setting': getattr(self.base_config, 'use_attention_mask', None)
        }
        metadata_path = os.path.join(self.output_dir, "sweep_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Run experiments
        print("\nStarting experiments...")
        for i, config in enumerate(experiment_configs):
            exp_id = config['experiment_id']
            print("[{}/{}] ".format(i+1, len(experiment_configs)), end="")
            
            config_path = self.create_experiment_config(exp_id, config)
            results = self.run_single_experiment(exp_id, config_path)
            
            full_results = {**config, **results}
            self.results.append(full_results)
            
            # Save results after each experiment
            results_path = os.path.join(self.output_dir, "results.csv")
            df = pd.DataFrame(self.results)
            df.to_csv(results_path, index=False)
        
        # Print summary
        successful = sum(1 for r in self.results if r.get('status') == 'success')
        failed = sum(1 for r in self.results if r.get('status') == 'failed')
        errors = sum(1 for r in self.results if r.get('status') == 'error')
        
        print("\nSweep completed: {}/{} successful, {} failed, {} errors".format(
            successful, len(self.results), failed, errors))
        
        if successful > 0:
            df_success = pd.DataFrame([r for r in self.results if r.get('status') == 'success'])
            best = df_success.loc[df_success['best_val_loss'].idxmin()]
            print("Best result: {} (val_loss: {:.6f})".format(best['experiment_id'], best['best_val_loss']))
            
            # Print mask comparison if applicable
            if 'use_attention_mask' in df_success.columns and len(df_success['use_attention_mask'].unique()) > 1:
                print("\nMask vs Traditional Comparison:")
                mask_results = df_success[df_success['use_attention_mask'] == True]
                trad_results = df_success[df_success['use_attention_mask'] == False]
                
                if len(mask_results) > 0 and len(trad_results) > 0:
                    mask_best = mask_results['best_val_loss'].min()
                    trad_best = trad_results['best_val_loss'].min()
                    print("  Best mask model: {:.6f}".format(mask_best))
                    print("  Best traditional model: {:.6f}".format(trad_best))
                    print("  Improvement: {:.2f}%".format((trad_best - mask_best) / trad_best * 100))


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parameter sweep with mask support')
    parser.add_argument('--base-config', type=str, required=True)
    parser.add_argument('--grid-type', type=str, default='quick',
                       choices=['full', 'quick', 'architecture', 'training', 'regularization', 
                               'dense_layers', 'mask_comparison', 'mask_optimization', 'features_comparison'])
    parser.add_argument('--max-experiments', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_config):
        print("Error: Base config file not found: {}".format(args.base_config))
        return 1
    
    sweep = ParameterSweep(args.base_config, args.output_dir)
    try:
        sweep.run_parameter_sweep(args.grid_type, args.max_experiments)
        return 0
    except Exception as e:
        print("Error during parameter sweep: {}".format(e))
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
