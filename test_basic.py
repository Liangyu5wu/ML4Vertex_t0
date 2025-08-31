#!/usr/bin/env python3
"""Basic test script for configuration loading."""

import sys
import os
sys.path.append('.')

def test_configs():
    """Test that configuration files can be loaded."""
    print("Testing configuration files...")
    
    try:
        # Test importing config modules
        from config.base_config import BaseConfig
        print("✓ Base config imported")
        
        # Test basic config creation
        config = BaseConfig()
        print(f"✓ Base config created: {config.model_name}")
        
        # Check if new attributes exist
        if hasattr(config, 'use_event_jets'):
            print(f"✓ use_event_jets: {config.use_event_jets}")
        if hasattr(config, 'use_event_tracks'):
            print(f"✓ use_event_tracks: {config.use_event_tracks}")
        if hasattr(config, 'max_jets'):
            print(f"✓ max_jets: {config.max_jets}")
        if hasattr(config, 'max_tracks'):
            print(f"✓ max_tracks: {config.max_tracks}")
            
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yaml_configs():
    """Test YAML configuration loading.""" 
    print("\nTesting YAML configurations...")
    
    try:
        import yaml
        
        # Test DNN config file
        with open('config/configs/experiment_dnn_with_jets_tracks.yaml', 'r') as f:
            dnn_config = yaml.safe_load(f)
        print(f"✓ DNN config loaded: {dnn_config['model_name']}")
        print(f"  use_event_jets: {dnn_config.get('use_event_jets', False)}")
        print(f"  max_jets: {dnn_config.get('max_jets', 'Not set')}")
        
        # Test Transformer config file
        with open('config/configs/experiment_transformer_with_jets_tracks.yaml', 'r') as f:
            transformer_config = yaml.safe_load(f)
        print(f"✓ Transformer config loaded: {transformer_config['model_name']}")
        print(f"  use_event_tracks: {transformer_config.get('use_event_tracks', False)}")
        print(f"  max_tracks: {transformer_config.get('max_tracks', 'Not set')}")
        
        # Test baseline guided config modifications
        with open('config/configs/experiment_baseline_guided_track.yaml', 'r') as f:
            baseline_config = yaml.safe_load(f)
        print(f"✓ Baseline config loaded: {baseline_config['model_name']}")
        print(f"  additional_cell_filters: {baseline_config.get('additional_cell_filters', {})}")
        print(f"  lr_patience: {baseline_config.get('lr_patience', 'Not set')}")
        
        return True
    except Exception as e:
        print(f"✗ YAML config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_structure():
    """Test data file structure without full loading."""
    print("\nTesting data file structure...")
    
    try:
        import h5py
        
        file_path = '../selected_h5_with_jets/output_032.h5'
        if not os.path.exists(file_path):
            print("✗ Test data file not found")
            return False
            
        with h5py.File(file_path, 'r') as f:
            print(f"✓ Data file opened: {file_path}")
            print(f"  Available datasets: {list(f.keys())}")
            
            if 'jets' in f:
                jets_data = f['jets']
                print(f"  Jets shape: {jets_data.shape}")
                print(f"  Jets fields: {list(jets_data.dtype.names)}")
                
            if 'tracks' in f:
                tracks_data = f['tracks']
                print(f"  Tracks shape: {tracks_data.shape}")
                print(f"  Tracks fields: {list(tracks_data.dtype.names)}")
        
        return True
    except Exception as e:
        print(f"✗ Data structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Basic feature test")
    print("=" * 30)
    
    success = True
    success &= test_configs()
    success &= test_yaml_configs()
    success &= test_data_structure()
    
    print("\n" + "=" * 30)
    if success:
        print("✓ Basic tests passed!")
    else:
        print("✗ Some basic tests failed!")
        sys.exit(1)