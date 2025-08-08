"""Parameter sweep script for transformer model hyperparameter optimization."""

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
    """Handle parameter sweep experiments for transformer model."""
    
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
            self.output_dir = os.path.join(results_base_dir, f"parameter_sweep_{timestamp}")
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
        
    def define_parameter_grids(self) -> Dict[str, Dict[str, List]]:
        """Define all parameter grids."""
        return {
            'quick': {
                'd_model': [32, 64, 128],
                'learning_rate': [1e-5, 5e-5, 1e-4],
                'num_heads': [2, 4, 8],
                'dropout_rate': [0.1, 0.2],
            },
            'full': {
                'd_model': [32, 64, 128],
                'num_heads': [2, 4, 8],
                'num_transformer_blocks': [2, 3, 4],
                'dropout_rate': [0.05, 0.1, 0.2],
                'vertex_dense_units': [8, 16, 32],
                'learning_rate': [1e-5, 5e-5, 1e-4, 2e-4],
                'batch_size': [32, 64, 128],
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
            }
        }
    
    def print_sweep_info(self, parameter_grid: Dict[str, List], grid_type: str):
        """Print parameter sweep information."""
        print(f"\n{'='*60}")
        print(f"PARAMETER SWEEP: {grid_type.upper()}")
        print(f"{'='*60}")
        print(f"Base config: {os.path.basename(self.base_config_path)}")
        print(f"Output dir: {self.output_dir}")
        
        print(f"\nParameters to sweep:")
        total_combinations = 1
        for param, values in parameter_grid.items():
            print(f"  {param}: {values} ({len(values)} values)")
            total_combinations *= len(values)
        
        print(f"\nTotal combinations: {total_combinations}")
        estimated_hours = total_combinations * 120 / 3600  # 120s per experiment
        print(f"Estimated time: ~{estimated_hours:.1f} hours")
        print(f"{'='*60}")
    
    def validate_parameter_combination(self, params: Dict) -> bool:
        """Validate parameter combination."""
        # d_model divisible by num_heads
        if 'd_model' in params and 'num_heads' in params:
            if params['d_model'] % params['num_heads'] != 0:
                return False
        
        # Learning rate range
        if 'learning_rate' in params:
            if params['learning_rate'] > 1e-3 or params['learning_rate'] < 1e-6:
                return False
        
        # Dropout rate range
        if 'dropout_rate' in params:
            if params['dropout_rate'] < 0 or params['dropout_rate'] > 0.5:
                return False
                
        return True
    
    def generate_experiment_configs(self, parameter_grid: Dict[str, List]) -> List[Dict]:
        """Generate all valid parameter combinations."""
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        combinations = list(itertools.product(*param_values))
        
        experiment_configs = []
        for i, combination in enumerate(combinations):
            config_dict = dict(zip(param_names, combination))
            if self.validate_parameter_combination(config_dict):
                config_dict['experiment_id'] = f"exp_{i:04d}"
                experiment_configs.append(config_dict)
        
        print(f"Generated {len(experiment_configs)} valid configurations")
        return experiment_configs
    
    def create_experiment_config(self, exp_id: str, params: Dict) -> str:
        """Create configuration file for experiment."""
        config = TransformerConfig.from_yaml(self.base_config_path)
        
        for param_name, param_value in params.items():
            if param_name != 'experiment_id' and hasattr(config, param_name):
                setattr(config, param_name, param_value)
        
        config.model_name = f"sweep_{exp_id}"
        
        if 'dff' not in params:
            config.dff = config.d_model * 2
        
        config_path = os.path.join(self.configs_dir, f"{exp_id}.yaml")
        config.save_yaml(config_path)
        return config_path
    
    def run_single_experiment(self, exp_id: str, config_path: str) -> Dict:
        """Run single experiment."""
        print(f"Running {exp_id}...", end=" ")
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
                print(f"✓ {training_time:.1f}s")
            else:
                results = {'status': 'failed', 'training_time': training_time}
                print(f"✗ {training_time:.1f}s")
                
        except Exception as e:
            results = {'status': 'error', 'error': str(e)}
            print(f"✗ Error")
        
        return results
    
    def extract_training_results(self, exp_id: str) -> Dict:
        """Extract results from training."""
        model_dir = os.path.join("models", f"sweep_{exp_id}")
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
                
        except Exception as e:
            results['extraction_error'] = str(e)
        
        return results
    
    def run_parameter_sweep(self, grid_type: str = 'quick', max_experiments: int = None):
        """Run parameter sweep."""
        grids = self.define_parameter_grids()
        
        if grid_type not in grids:
            raise ValueError(f"Unknown grid type: {grid_type}")
        
        parameter_grid = grids[grid_type]
        self.print_sweep_info(parameter_grid, grid_type)
        
        experiment_configs = self.generate_experiment_configs(parameter_grid)
        
        if max_experiments is not None:
            experiment_configs = experiment_configs[:max_experiments]
            print(f"Limited to {len(experiment_configs)} experiments")
        
        # Save experiment plan
        plan_path = os.path.join(self.output_dir, "experiment_plan.json")
        with open(plan_path, 'w') as f:
            json.dump(experiment_configs, f, indent=2)
        
        # Run experiments
        for i, config in enumerate(experiment_configs):
            exp_id = config['experiment_id']
            print(f"[{i+1}/{len(experiment_configs)}] ", end="")
            
            config_path = self.create_experiment_config(exp_id, config)
            results = self.run_single_experiment(exp_id, config_path)
            
            full_results = {**config, **results}
            self.results.append(full_results)
            
            # Save results
            results_path = os.path.join(self.output_dir, "results.csv")
            df = pd.DataFrame(self.results)
            df.to_csv(results_path, index=False)
        
        # Print summary
        successful = sum(1 for r in self.results if r.get('status') == 'success')
        print(f"\nSweep completed: {successful}/{len(self.results)} successful")
        
        if successful > 0:
            df_success = pd.DataFrame([r for r in self.results if r.get('status') == 'success'])
            best = df_success.loc[df_success['best_val_loss'].idxmin()]
            print(f"Best result: {best['experiment_id']} (val_loss: {best['best_val_loss']:.6f})")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parameter sweep')
    parser.add_argument('--base-config', type=str, required=True)
    parser.add_argument('--grid-type', type=str, default='quick',
                       choices=['full', 'quick', 'architecture', 'training', 'regularization', 'dense_layers'])
    parser.add_argument('--max-experiments', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    sweep = ParameterSweep(args.base_config, args.output_dir)
    sweep.run_parameter_sweep(args.grid_type, args.max_experiments)


if __name__ == "__main__":
    main()