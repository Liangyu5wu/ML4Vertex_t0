"""Easy-to-use script for running parameter sweep experiments."""

import os
import sys
import itertools

# Add src to path  
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.transformer_config import TransformerConfig


def get_parameter_grids():
    """Get all parameter grid definitions."""
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
            'learning_rate': [1e-5, 5e-5, 1e-4, 1e-3],
            'batch_size': [32, 64, 128],
        },
        'architecture': {
            'd_model': [32, 64, 128, 256],
            'num_heads': [2, 4, 8, 16],
            'num_transformer_blocks': [1, 2, 3, 4, 5],
            'dff': [64, 128, 256, 512],
        },
        'training': {
            'learning_rate': [5e-6, 1e-5, 5e-5, 1e-3, 2e-4, 5e-4],
            'batch_size': [32, 64, 128],
            'lr_reduction_factor': [0.3, 0.5, 0.7],
        },
        'regularization': {
            'dropout_rate': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
        },
        'dense_layers': {
            'vertex_dense_units': [4, 8, 16, 32, 64],
        }
    }


def print_sweep_preview(base_config: str, grid_type: str, max_exp: int = None):
    """Print parameter sweep preview."""
    try:
        config = TransformerConfig.from_yaml(base_config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return False
    
    grids = get_parameter_grids()
    if grid_type not in grids:
        print(f"Unknown grid type: {grid_type}")
        return False
    
    parameter_grid = grids[grid_type]
    
    print(f"\n{'='*50}")
    print(f"PARAMETER SWEEP PREVIEW: {grid_type.upper()}")
    print(f"{'='*50}")
    print(f"Base config: {os.path.basename(base_config)}")
    print(f"Model: {config.model_name}")
    print(f"Data: {config.data_dir}")
    print(f"Epochs: {config.epochs}")
    
    print(f"\nParameters to sweep:")
    total_combinations = 1
    for param, values in parameter_grid.items():
        print(f"  {param}: {values}")
        total_combinations *= len(values)
        
        # Show current base value
        if hasattr(config, param):
            base_value = getattr(config, param)
            in_sweep = base_value in values
            status = "✓" if in_sweep else "✗"
            print(f"    Base value: {base_value} {status}")
    
    # Calculate valid combinations
    valid_combinations = 0
    for combination in itertools.product(*parameter_grid.values()):
        param_dict = dict(zip(parameter_grid.keys(), combination))
        
        valid = True
        if 'd_model' in param_dict and 'num_heads' in param_dict:
            if param_dict['d_model'] % param_dict['num_heads'] != 0:
                valid = False
        
        if valid:
            valid_combinations += 1
    
    actual_experiments = valid_combinations
    if max_exp is not None and max_exp < valid_combinations:
        actual_experiments = max_exp
    
    estimated_hours = actual_experiments * 90 / 3600
    
    print(f"\nExperiment info:")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Valid combinations: {valid_combinations}")
    if max_exp is not None:
        print(f"  Limited to: {max_exp}")
    print(f"  Will run: {actual_experiments} experiments")
    print(f"  Estimated time: ~{estimated_hours:.1f} hours")
    print(f"{'='*50}")
    
    return True


def get_user_confirmation(experiments: int) -> bool:
    """Get user confirmation."""
    estimated_hours = experiments * 90 / 3600
    print(f"\nAbout to run {experiments} experiments (~{estimated_hours:.1f} hours)")
    
    while True:
        response = input("Continue? [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no', '']:
            return False
        else:
            print("Please enter 'y' or 'n'")


def main():
    """Main function."""
    base_config = "config/configs/experiment2_fast.yaml"
    
    if not os.path.exists(base_config):
        print(f"Error: Config file not found: {base_config}")
        return
    
    print("Parameter Sweep Runner")
    print("="*30)
    print("1. Quick sweep (4 params)")
    print("2. Architecture sweep")
    print("3. Training sweep")
    print("4. Regularization sweep")
    print("5. Dense layers sweep")
    print("6. Full sweep")
    print("7. Custom")
    
    choice = input("\nSelect sweep type (1-7): ")
    
    sweep_configs = {
        "1": ("quick", 5),
        "2": ("architecture", 50),
        "3": ("training", 200),
        "4": ("regularization", 25),
        "5": ("dense_layers", 15),
        "6": ("full", None),
        "7": ("custom", None)
    }
    
    if choice not in sweep_configs:
        print("Invalid choice!")
        return
    
    grid_type, max_exp = sweep_configs[choice]
    
    if choice == "7":
        grid_type = input("Enter grid type: ")
        max_exp_input = input("Max experiments (or Enter for no limit): ")
        max_exp = int(max_exp_input) if max_exp_input else None
    
    # Show preview
    if not print_sweep_preview(base_config, grid_type, max_exp):
        return
    
    # Get confirmation
    actual_experiments = max_exp if max_exp else 100  # rough estimate
    if not get_user_confirmation(actual_experiments):
        print("Cancelled.")
        return
    
    # Run sweep
    cmd_parts = [
        "python", "scripts/parameter_sweep.py",
        "--base-config", base_config,
        "--grid-type", grid_type
    ]
    
    if max_exp is not None:
        cmd_parts.extend(["--max-experiments", str(max_exp)])
    
    print(f"\nRunning: {' '.join(cmd_parts)}")
    print("Starting sweep...\n")
    
    import subprocess
    result = subprocess.run(cmd_parts)
    
    if result.returncode == 0:
        print(f"\n{'='*50}")
        print("SWEEP COMPLETED!")
        print(f"{'='*50}")
        print("Analyze results with:")
        print("python scripts/analyze_sweep.py results/parameter_sweep_YYYYMMDD_HHMMSS/")
    else:
        print(f"\n{'='*50}")
        print("SWEEP FAILED")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()