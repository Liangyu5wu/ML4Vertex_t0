"""Debug script to diagnose why no events are loaded."""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.dnn_config import DNNConfig
from src.data.hgtd_multi_input_data_loader import HGTDMultiInputDataLoader


def analyze_filtering_stages(config):
    """Analyze each filtering stage to see where events are lost."""

    data_loader = HGTDMultiInputDataLoader(config)
    file_paths = data_loader.get_file_paths()

    # Use only first file for quick diagnosis
    test_file = file_paths[0]
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC ANALYSIS: {os.path.basename(test_file)}")
    print(f"{'='*70}\n")

    import h5py

    with h5py.File(test_file, 'r') as f:
        vertex_data = f['HSvertex'][:]
        cells_data = f['cells'][:]

        total_events = len(vertex_data)
        print(f"Total events in file: {total_events}")

        # Stage 1: Check valid cells
        events_with_valid_cells = 0
        events_after_time_cut = 0
        events_after_min_cells = 0
        cell_counts = []

        for i in range(min(total_events, 1000)):  # Test first 1000 events
            event_cells = cells_data[i]

            # Stage 1: Valid cells filter
            if config.require_valid_cells:
                valid_mask = event_cells['valid'] == True
                event_cells = event_cells[valid_mask]

            if len(event_cells) > 0:
                events_with_valid_cells += 1

            # Stage 2: Time quality cut
            if config.use_time_quality_cut and len(event_cells) > 0:
                event_cells_before = len(event_cells)
                event_cells = data_loader.apply_time_quality_cut(event_cells)
                event_cells_after = len(event_cells)

                if event_cells_after > 0:
                    events_after_time_cut += 1

                    # Debug: print first event that survives time cut
                    if events_after_time_cut == 1:
                        print(f"\nFirst event passing time cut:")
                        print(f"  Cells before time cut: {event_cells_before}")
                        print(f"  Cells after time cut: {event_cells_after}")
                        print(f"  Cell time range: [{event_cells['Cell_time_TOF_corrected'].min():.1f}, {event_cells['Cell_time_TOF_corrected'].max():.1f}] ps")
            elif not config.use_time_quality_cut and len(event_cells) > 0:
                events_after_time_cut += 1

            # Stage 3: Min cells requirement
            if len(event_cells) >= config.min_cells:
                events_after_min_cells += 1
                cell_counts.append(len(event_cells))

        print(f"\n{'='*70}")
        print(f"FILTERING ANALYSIS (first 1000 events):")
        print(f"{'='*70}")
        print(f"Stage 1 - Valid cells filter:     {events_with_valid_cells:4d} / {min(total_events, 1000)} events ({100*events_with_valid_cells/min(total_events, 1000):.1f}%)")
        print(f"Stage 2 - Time quality cut:       {events_after_time_cut:4d} / {min(total_events, 1000)} events ({100*events_after_time_cut/min(total_events, 1000):.1f}%)")
        print(f"Stage 3 - Min cells requirement:  {events_after_min_cells:4d} / {min(total_events, 1000)} events ({100*events_after_min_cells/min(total_events, 1000):.1f}%)")

        print(f"\n{'='*70}")
        print(f"CONFIGURATION:")
        print(f"{'='*70}")
        print(f"require_valid_cells:      {config.require_valid_cells}")
        print(f"use_time_quality_cut:     {config.use_time_quality_cut}")
        print(f"time_quality_n_sigma:     {config.time_quality_n_sigma}")
        print(f"vertex_time_sigma:        {config.vertex_time_sigma} ps")
        print(f"use_detector_params:      {config.use_detector_params}")
        print(f"min_cells:                {config.min_cells}")

        if cell_counts:
            print(f"\n{'='*70}")
            print(f"CELL STATISTICS (for {len(cell_counts)} passing events):")
            print(f"{'='*70}")
            print(f"Mean cells per event:     {np.mean(cell_counts):.1f}")
            print(f"Median cells per event:   {np.median(cell_counts):.0f}")
            print(f"Min cells per event:      {np.min(cell_counts)}")
            print(f"Max cells per event:      {np.max(cell_counts)}")

        # Analyze time quality cut in detail
        if config.use_time_quality_cut:
            print(f"\n{'='*70}")
            print(f"TIME QUALITY CUT ANALYSIS:")
            print(f"{'='*70}")

            # Sample first event to understand time distribution
            sample_cells = cells_data[0]
            if config.require_valid_cells:
                sample_cells = sample_cells[sample_cells['valid'] == True]

            if len(sample_cells) > 0:
                cell_times = sample_cells['Cell_time_TOF_corrected']
                print(f"Sample event cell times:")
                print(f"  Mean: {np.mean(cell_times):.1f} ps")
                print(f"  Std:  {np.std(cell_times):.1f} ps")
                print(f"  Min:  {np.min(cell_times):.1f} ps")
                print(f"  Max:  {np.max(cell_times):.1f} ps")
                print(f"  Range: {np.max(cell_times) - np.min(cell_times):.1f} ps")

                # Calculate expected cut threshold
                sigma_vertex = config.vertex_time_sigma
                # Without detector params, we use vertex sigma only
                threshold = config.time_quality_n_sigma * sigma_vertex
                print(f"\nTime cut threshold: ±{threshold:.1f} ps ({config.time_quality_n_sigma}σ × {sigma_vertex:.1f} ps)")
                print(f"Cells within threshold: {np.sum(np.abs(cell_times) < threshold)} / {len(cell_times)} ({100*np.sum(np.abs(cell_times) < threshold)/len(cell_times):.1f}%)")

        print(f"\n{'='*70}")
        print(f"RECOMMENDATION:")
        print(f"{'='*70}")

        if events_after_min_cells == 0:
            print("⚠️  NO EVENTS PASS ALL FILTERS!")
            if events_after_time_cut == 0 and config.use_time_quality_cut:
                print("\n🔍 ISSUE: Time quality cut is removing ALL events")
                print("   Solutions:")
                print("   1. Increase time_quality_n_sigma (try 5.0 or higher)")
                print("   2. Disable time quality cut temporarily: use_time_quality_cut: false")
                print("   3. Check if cell times are correctly calibrated")
            elif events_with_valid_cells == 0:
                print("\n🔍 ISSUE: Valid cells filter is removing ALL events")
                print("   Solutions:")
                print("   1. Set require_valid_cells: false")
            elif events_after_min_cells == 0:
                print(f"\n🔍 ISSUE: Min cells requirement (min_cells={config.min_cells}) is too strict")
                print("   Solutions:")
                print("   1. Reduce min_cells value")
        else:
            print(f"✅ {events_after_min_cells} events pass all filters in first 1000 events")
            estimated_total = int(events_after_min_cells * total_events / min(total_events, 1000))
            print(f"   Estimated total passing events in file: ~{estimated_total}")


def main():
    """Main diagnostic function."""
    import argparse

    parser = argparse.ArgumentParser(description='Debug data loading filters')
    parser.add_argument('--config-file', type=str, required=True,
                      help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    config = DNNConfig.from_yaml(args.config_file)

    # Run analysis
    analyze_filtering_stages(config)


if __name__ == '__main__':
    main()
