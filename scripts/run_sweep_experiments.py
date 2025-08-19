"""Easy-to-use script for running parameter sweep experiments with mask support."""

import os
import sys
import itertools

# Add src to path  
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.transformer_config import TransformerConfig


def get_parameter_grids():
    """Get all parameter grid definitions with mask support."""
    return {
        # Traditional parameter sweeps (updated)
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
        },
        
        # NEW: Mask-specific parameter sweeps
        'mask_comparison': {
            'use_attention_mask': [True, False],
            'd_model': [64, 128],
            'num_heads': [4, 8],
            'learning_rate': [5e-5, 1e-4],
            'dropout_rate': [0.1, 0.2],
        },
        'mask_optimization': {
            'use_attention_mask': [True],  # Only mask models
            'd_model': [32, 64, 128, 256],
            'num_heads': [2, 4, 8, 16],
            'num_transformer_blocks': [2, 3, 4],
            'learning_rate': [1e-5, 5e-5, 1e-4],
            'batch_size': [32, 64, 128],
        },
        'features_comparison': {
            'use_attention_mask': [True],
            'use_spatial_features': [True, False],
            'use_jet_features': [True, False],
            'use_cell_jet_matching': [True, False],
            'd_model': [64, 128],
            'learning_rate': [5e-5, 1e-4],
        },
    }


def get_sweep_descriptions():
    """Get descriptions for each sweep type."""
    return {
        # Traditional sweeps
        'quick': "Fast sweep with 4 key parameters (good for testing)",
        'full': "Comprehensive sweep covering all major parameters",
        'architecture': "Focus on model architecture (d_model, heads, blocks)",
        'training': "Focus on training parameters (LR, batch size, scheduling)",
        'regularization': "Focus on regularization techniques",
        'dense_layers': "Focus on dense layer configurations",
        
        # NEW: Mask-specific sweeps
        'mask_comparison': "Compare mask-enabled vs traditional models",
        'mask_optimization': "Optimize parameters for mask-enabled models only",
        'features_comparison': "Compare different feature combinations with masks",
    }


def print_sweep_preview(base_config, grid_type, max_exp=None):
    """Print parameter sweep preview with enhanced mask information."""
    try:
        config = TransformerConfig.from_yaml(base_config)
    except Exception as e:
        print("Error loading config: {}".format(e))
        return False
    
    grids = get_parameter_grids()
    descriptions = get_sweep_descriptions()
    
    if grid_type not in grids:
        print("Unknown grid type: {}".format(grid_type))
        print("Available types: {}".format(list(grids.keys())))
        return False
    
    parameter_grid = grids[grid_type]
    
    print("\n" + "="*70)
    print("PARAMETER SWEEP PREVIEW: {}".format(grid_type.upper()))
    print("="*70)
    print("Description: {}".format(descriptions.get(grid_type, "No description available")))
    print("Base config: {}".format(os.path.basename(base_config)))
    print("Model: {}".format(config.model_name))
    print("Data: {}".format(config.data_dir))
    print("Epochs: {}".format(config.epochs))
    
    # Display base mask setting
    base_mask_setting = getattr(config, 'use_attention_mask', True)
    print("Base mask setting: {}".format('enabled' if base_mask_setting else 'disabled'))
    
    print("\nParameters to sweep:")
    total_combinations = 1
    mask_analysis = {'has_mask_param': False, 'mask_values': [], 'comparison_type': 'single'}
    
    for param, values in parameter_grid.items():
        print("  {}: {}".format(param, values))
        total_combinations *= len(values)
        
        # Analyze mask parameter
        if param == 'use_attention_mask':
            mask_analysis['has_mask_param'] = True
            mask_analysis['mask_values'] = values
            if len(set(values)) > 1:
                mask_analysis['comparison_type'] = 'comparison'
            elif values[0]:
                mask_analysis['comparison_type'] = 'mask_only'
            else:
                mask_analysis['comparison_type'] = 'traditional_only'
        
        # Show current base value if parameter exists in config
        if hasattr(config, param):
            base_value = getattr(config, param)
            in_sweep = base_value in values
            status = "✓" if in_sweep else "✗"
            print("    Base value: {} {}".format(base_value, status))
    
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
    
    print("\nExperiment info:")
    print("  Total combinations: {}".format(total_combinations))
    print("  Valid combinations: {}".format(valid_combinations))
    if max_exp is not None:
        print("  Limited to: {}".format(max_exp))
    print("  Will run: {} experiments".format(actual_experiments))
    print("  Estimated time: ~{:.1f} hours".format(estimated_hours))
    
    # Print mask analysis
    print("\nMask Analysis:")
    if mask_analysis['has_mask_param']:
        if mask_analysis['comparison_type'] == 'comparison':
            print("  *** MASK vs TRADITIONAL COMPARISON ***")
            print("  This sweep will compare mask-enabled and traditional models")
            print("  Expected: Performance comparison between both approaches")
        elif mask_analysis['comparison_type'] == 'mask_only':
            print("  *** MASK OPTIMIZATION SWEEP ***")
            print("  This sweep focuses on optimizing mask-enabled models")
            print("  Expected: Best parameters for attention mask architecture")
        elif mask_analysis['comparison_type'] == 'traditional_only':
            print("  *** TRADITIONAL MODEL SWEEP ***")
            print("  This sweep uses traditional models only")
            print("  Expected: Optimization without attention masks")
    else:
        print("  Using base config mask setting: {}".format('enabled' if base_mask_setting else 'disabled'))
        print("  All experiments will use the same mask configuration")
    
    print("="*70)
    
    return True


def get_user_confirmation(experiments):
    """Get user confirmation with enhanced information."""
    estimated_hours = experiments * 90 / 3600
    print("\nAbout to run {} experiments (~{:.1f} hours)".format(experiments, estimated_hours))
    print("This will create models and save results to the models/ directory.")
    
    while True:
        response = input("Continue? [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no', '']:
            return False
        else:
            print("Please enter 'y' or 'n'")


def show_sweep_menu():
    """Display enhanced sweep menu with categories."""
    descriptions = get_sweep_descriptions()
    
    print("Parameter Sweep Runner with Mask Support")
    print("="*50)
    
    print("\n🔧 TRADITIONAL PARAMETER SWEEPS:")
    print("1. Quick sweep (4 params) - {}".format(descriptions['quick']))
    print("2. Architecture sweep - {}".format(descriptions['architecture']))
    print("3. Training sweep - {}".format(descriptions['training']))
    print("4. Regularization sweep - {}".format(descriptions['regularization']))
    print("5. Dense layers sweep - {}".format(descriptions['dense_layers']))
    print("6. Full sweep - {}".format(descriptions['full']))
    
    print("\n🎯 MASK-ENHANCED SWEEPS:")
    print("7. Mask comparison - {}".format(descriptions['mask_comparison']))
    print("8. Mask optimization - {}".format(descriptions['mask_optimization']))
    print("9. Features comparison - {}".format(descriptions['features_comparison']))
    
    print("\n🛠️  CUSTOM:")
    print("10. Custom sweep")


def main():
    """Main function with enhanced interface."""
    base_config = "config/configs/experiment_with_jets.yaml"
    
    # Check if base config exists, try alternatives
    if not os.path.exists(base_config):
        alternatives = [
            "config/configs/test_fast.yaml",
            "config/configs/experiment_nersc.yaml",
            "config/configs/experiment2_fast.yaml"
        ]
        
        for alt_config in alternatives:
            if os.path.exists(alt_config):
                base_config = alt_config
                print("Using alternative config: {}".format(base_config))
                break
        else:
            print("Error: No suitable config file found!")
            print("Tried: {}".format([base_config] + alternatives))
            return 1
    
    show_sweep_menu()
    
    choice = input("\nSelect sweep type (1-10): ").strip()
    
    sweep_configs = {
        "1": ("quick", 5),
        "2": ("architecture", 50),
        "3": ("training", 200),
        "4": ("regularization", 25),
        "5": ("dense_layers", 15),
        "6": ("full", None),
        "7": ("mask_comparison", 20),      # NEW
        "8": ("mask_optimization", 100),   # NEW  
        "9": ("features_comparison", 30),  # NEW
        "10": ("custom", None)
    }
    
    if choice not in sweep_configs:
        print("Invalid choice!")
        return 1
    
    grid_type, max_exp = sweep_configs[choice]
    
    if choice == "10":  # Custom
        print("\nAvailable grid types:")
        for grid_name, desc in get_sweep_descriptions().items():
            print("  {}: {}".format(grid_name, desc))
        
        grid_type = input("\nEnter grid type: ").strip()
        if grid_type not in get_parameter_grids():
            print("Invalid grid type!")
            return 1
            
        max_exp_input = input("Max experiments (or Enter for no limit): ").strip()
        max_exp = int(max_exp_input) if max_exp_input else None
    
    # Show preview
    if not print_sweep_preview(base_config, grid_type, max_exp):
        return 1
    
    # Get confirmation
    actual_experiments = max_exp if max_exp else 100  # rough estimate
    if not get_user_confirmation(actual_experiments):
        print("Cancelled.")
        return 0
    
    # Run sweep
    cmd_parts = [
        "python", "scripts/parameter_sweep.py",
        "--base-config", base_config,
        "--grid-type", grid_type
    ]
    
    if max_exp is not None:
        cmd_parts.extend(["--max-experiments", str(max_exp)])
    
    print("\nRunning: {}".format(' '.join(cmd_parts)))
    print("Starting sweep...\n")
    
    import subprocess
    result = subprocess.run(cmd_parts)
    
    if result.returncode == 0:
        print("\n" + "="*70)
        print("SWEEP COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("Analyze results with:")
        print("python scripts/analyze_sweep.py results/parameter_sweep_YYYYMMDD_HHMMSS/")
        
        # Provide specific analysis suggestions based on sweep type
        if grid_type == 'mask_comparison':
            print("\n🎯 MASK COMPARISON ANALYSIS:")
            print("Look for performance differences between mask-enabled and traditional models")
            print("Check if mask models show consistent improvements across different parameters")
        elif grid_type == 'mask_optimization':
            print("\n🎯 MASK OPTIMIZATION ANALYSIS:")
            print("Focus on finding the best parameters for mask-enabled models")
            print("Look for parameter interactions specific to attention mask architecture")
        elif grid_type == 'features_comparison':
            print("\n🎯 FEATURES ANALYSIS:")
            print("Compare how different feature combinations affect performance")
            print("Look for optimal feature sets with mask-enabled models")
            
    else:
        print("\n" + "="*70)
        print("SWEEP FAILED")
        print("="*70)
        print("Check the error messages above for details")
    
    return result.returncode


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
