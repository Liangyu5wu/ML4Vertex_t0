#!/usr/bin/env python3
"""Test script to verify fixes for calibration and evaluation issues."""

import sys
import os
sys.path.append('.')

def test_calibration_loading():
    """Test calibration data loading with fixed configurations."""
    print("Testing calibration data loading...")
    
    try:
        from config.dnn_config import DNNConfig
        from src.data.data_loader import DataLoader
        
        # Test DNN config with calibration enabled
        config = DNNConfig.from_yaml('config/configs/experiment_dnn_with_jets_tracks.yaml')
        print(f"✓ DNN config loaded: {config.model_name}")
        print(f"  use_time_quality_cut: {config.use_time_quality_cut}")
        print(f"  calibration_data_file: {config.calibration_data_file}")
        
        # Test data loader calibration loading
        loader = DataLoader(config)
        try:
            calibration_data = config.load_calibration_data()
            print(f"✓ Calibration data loaded successfully")
            print(f"  Available keys: {list(calibration_data.keys())}")
        except Exception as e:
            print(f"✗ Calibration loading failed: {e}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_input_detection():
    """Test model input detection for different model types."""
    print("\nTesting model input detection...")
    
    try:
        from config.dnn_config import DNNConfig
        from config.transformer_config import TransformerConfig
        from src.models.dnn_model import DNNModel
        from src.models.transformer_model import TransformerModel
        from src.evaluation.evaluator import Evaluator
        
        # Test traditional model (2 inputs)
        config = DNNConfig()
        model = DNNModel(config)
        keras_model = model.build_model(7, 3)  # 7 cell features, 3 vertex features
        
        evaluator = Evaluator(config)
        model_type = evaluator._detect_model_type(keras_model)
        print(f"✓ Traditional model detected as: {model_type}")
        
        # Test mask model (3 inputs)
        keras_model_mask = model.build_model_with_mask(7, 3)
        model_type_mask = evaluator._detect_model_type(keras_model_mask)
        print(f"✓ Mask model detected as: {model_type_mask}")
        
        # Test multi-input model (5 inputs)
        keras_model_multi = model.build_model_with_jets_tracks(7, 3, 4, 5)  # 7 cell, 3 vertex, 4 jet, 5 track
        model_type_multi = evaluator._detect_model_type(keras_model_multi)
        print(f"✓ Multi-input model detected as: {model_type_multi}")
        
        return True
    except Exception as e:
        print(f"✗ Model detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_validation():
    """Test that all configurations are valid."""
    print("\nTesting configuration validation...")
    
    configs_to_test = [
        'config/configs/experiment_baseline_guided_track.yaml',
        'config/configs/experiment_dnn_with_jets_tracks.yaml', 
        'config/configs/experiment_transformer_with_jets_tracks.yaml'
    ]
    
    try:
        from config.dnn_config import DNNConfig
        from config.transformer_config import TransformerConfig
        
        for config_path in configs_to_test:
            print(f"  Testing {os.path.basename(config_path)}...")
            
            if 'transformer' in config_path:
                config = TransformerConfig.from_yaml(config_path)
            else:
                config = DNNConfig.from_yaml(config_path)
            
            # Test calibration loading if enabled
            if getattr(config, 'use_time_quality_cut', False):
                try:
                    calibration_data = config.load_calibration_data()
                    print(f"    ✓ Calibration data loaded")
                except Exception as e:
                    print(f"    ✗ Calibration failed: {e}")
                    return False
            else:
                print(f"    - Time quality cut disabled")
        
        print("✓ All configurations validated")
        return True
        
    except Exception as e:
        print(f"✗ Config validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing fixes for calibration and evaluation issues")
    print("=" * 60)
    
    success = True
    success &= test_calibration_loading()
    success &= test_model_input_detection()
    success &= test_config_validation()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All fixes verified successfully!")
        print("\nNext steps:")
        print("1. Training should now work without calibration warnings")
        print("2. Evaluation will show warnings for multi-input models but won't crash")
        print("3. Traditional and mask models should evaluate normally")
    else:
        print("✗ Some fixes still need work!")
        sys.exit(1)