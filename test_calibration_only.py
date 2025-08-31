#!/usr/bin/env python3
"""Simple test for calibration data loading only."""

import sys
import os
sys.path.append('.')

def test_calibration_simple():
    """Test just calibration loading without other dependencies."""
    print("Testing calibration data loading...")
    
    try:
        from config.base_config import BaseConfig
        
        # Test configurations with different calibration settings
        configs = [
            ('config/configs/experiment_dnn_with_jets_tracks.yaml', 'DNN with jets/tracks'),
            ('config/configs/experiment_transformer_with_jets_tracks.yaml', 'Transformer with jets/tracks'),
            ('config/configs/experiment_baseline_guided_track.yaml', 'Baseline guided with tracks')
        ]
        
        for config_path, desc in configs:
            print(f"\n  Testing {desc}...")
            
            # Load YAML directly to check settings
            import yaml
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            print(f"    use_time_quality_cut: {yaml_data.get('use_time_quality_cut', False)}")
            print(f"    calibration_data_file: '{yaml_data.get('calibration_data_file', '')}'")
            
            # Create config object and test calibration loading
            config = BaseConfig()
            config.use_time_quality_cut = yaml_data.get('use_time_quality_cut', False)
            config.calibration_data_file = yaml_data.get('calibration_data_file', '')
            
            if config.use_time_quality_cut:
                try:
                    calibration_data = config.load_calibration_data()
                    print(f"    ✓ Calibration loaded: {len(calibration_data)} keys")
                except Exception as e:
                    print(f"    ✗ Calibration failed: {e}")
                    return False
            else:
                print(f"    - Time quality cut disabled, no calibration needed")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing calibration data loading fix")
    print("=" * 50)
    
    success = test_calibration_simple()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ Calibration fix verified!")
        print("Training should now work without 'Cannot load calibration data' warnings")
    else:
        print("✗ Calibration fix failed!")
        sys.exit(1)