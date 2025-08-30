#!/usr/bin/env python3
"""
Simple script to test path resolution for baseline analysis.
"""

import os
from pathlib import Path

def test_paths():
    """Test path resolution for calibration and data files."""
    print("Current working directory:", os.getcwd())
    print()
    
    # Test calibration paths
    calib_file = "HStrackmatching_calibration.txt"
    
    paths_to_test = [
        Path("calibration_data") / calib_file,
        Path("../calibration_data") / calib_file,
        Path("../../calibration_data") / calib_file,
    ]
    
    print("Testing calibration data paths:")
    for path in paths_to_test:
        exists = path.exists()
        print(f"  {path} -> {'EXISTS' if exists else 'NOT FOUND'}")
        if exists:
            print(f"    Absolute path: {path.absolute()}")
    
    print()
    
    # Test data directory paths
    data_paths = [
        Path("../selected_h5_with_jets/"),
        Path("../../selected_h5_with_jets/"),
        Path("selected_h5_with_jets/"),
    ]
    
    print("Testing data directory paths:")
    for path in data_paths:
        exists = path.exists()
        print(f"  {path} -> {'EXISTS' if exists else 'NOT FOUND'}")
        if exists:
            print(f"    Absolute path: {path.absolute()}")
            # List a few files
            files = list(path.glob("output_*.h5"))[:3]
            if files:
                print(f"    Sample files: {[f.name for f in files]}")
    
    print()
    
    # Test output directory
    output_paths = [
        Path("../../bad_events_check/"),
        Path("../bad_events_check/"),
        Path("bad_events_check/"),
    ]
    
    print("Testing output directory paths:")
    for path in output_paths:
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  {path} -> CAN CREATE")
            print(f"    Absolute path: {path.absolute()}")
        except Exception as e:
            print(f"  {path} -> CANNOT CREATE: {e}")

if __name__ == "__main__":
    test_paths()