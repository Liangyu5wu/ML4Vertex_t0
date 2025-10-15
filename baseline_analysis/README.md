# Baseline Time Reconstruction Analysis Tool

This tool performs comprehensive analysis of baseline time reconstruction methods, focusing on understanding why certain events have poor reconstruction performance.

## Features

### Core Analysis
- **Configurable matching**: Support for track-matching or jet-matching via YAML configuration
- **Detailed event logging**: Analysis of worst-performing events with cell-level details
- **Smart padding**: Uses same physics-informed padding strategy as ML models
- **Flexible parameters**: All analysis parameters configurable via YAML

### Generated Outputs

#### 1. Log File (`analysis_log.txt`)
- Complete analysis log with event details
- Shows top N worst events with:
  - Truth time, reconstructed time, error
  - Cell count and energy/time statistics
  - Layer and detector region distribution

#### 2. Baseline Check Plots (`baseline_plots/`)
- `t0_error.png`: Error distribution with Gaussian fit
- `traditional_t0_distribution.png`: Baseline t0 distribution
- `traditional_t0_vs_true_2d.png`: 2D correlation plot

#### 3. Feature Comparison (`feature_comparison/`)
- Histogram comparisons between best and worst events
- Individual feature plots for all cell features:
  - Spatial: `Cell_x`, `Cell_y`, `Cell_z`, `Cell_eta`, `Cell_phi`
  - Detector: `Cell_Barrel`, `Cell_layer`
  - Physics: `Cell_time_TOF_corrected`, `Cell_e`, `Cell_significance`
  - Matching: `matched_track_pt`, `matched_track_deltaR`, etc.
- Statistics (mean, std) shown for each group

#### 4. Additional Analysis (`additional_analysis/`)
- `error_vs_ncells.png`: Reconstruction error vs number of cells
- `error_vs_energy_spread.png`: Error vs cell energy spread
- `error_vs_time_spread.png`: Error vs cell time spread  
- `error_vs_layer1_fraction.png`: Error vs Layer 1 cell fraction

## Usage

### Basic Usage
```bash
cd baseline_analysis
python baseline_analysis.py
```

### With Custom Configuration
```bash
python baseline_analysis.py --config my_config.yaml
```

### Override Parameters
```bash
python baseline_analysis.py --top-events 50 --sample-size 5000
```

## Configuration

The tool uses `baseline_analysis_config.yaml` by default. Key parameters:

### Data Selection
- `use_track_features: true` - Use track matching (default)
- `use_jet_features: false` - Use jet matching instead
- `num_files: 43` - Number of HDF5 files to process
- `calibration_data_file` - Calibration file to use

### Analysis Parameters
- `top_worst_events: 20` - Number of worst events to analyze
- `feature_comparison_sample_size: 2000` - Events for feature comparison
- `comparison_features` - List of features to compare

### Output
- `output_base_dir: "../../bad_events_check"` - Output directory
- `create_timestamp_dir: true` - Create timestamped subdirectory

## Output Structure

```
bad_events_check/
└── baseline_analysis_20241230_143022/
    ├── config_used.yaml          # Configuration used
    ├── analysis_log.txt          # Complete analysis log
    ├── baseline_plots/           # Standard baseline plots
    │   ├── t0_error.png
    │   ├── traditional_t0_distribution.png
    │   └── traditional_t0_vs_true_2d.png
    ├── feature_comparison/       # Best vs worst comparisons
    │   ├── Cell_e_comparison.png
    │   ├── Cell_time_TOF_corrected_comparison.png
    │   ├── matched_track_pt_comparison.png
    │   └── ... (all features)
    └── additional_analysis/      # Correlation analysis
        ├── error_vs_ncells.png
        ├── error_vs_energy_spread.png
        ├── error_vs_time_spread.png
        └── error_vs_layer1_fraction.png
```

## Understanding the Analysis

### Worst Events Analysis
The log shows detailed information for the worst-performing events:
- Cell counts and their energy/time distributions
- Layer composition (EM1, EM2, EM3)
- Barrel vs Endcap distribution
- Helps identify patterns in reconstruction failures

### Feature Comparisons
Compare distributions between best and worst events:
- **Green histograms**: Best reconstructed events
- **Red histograms**: Worst reconstructed events
- Statistics help identify which features differentiate good/bad reconstruction

### Correlation Plots
Show relationships between event properties and reconstruction quality:
- More cells → better/worse reconstruction?
- Energy spread → reconstruction difficulty?
- Time spread → timing challenges?
- Layer composition → detector response issues?

## Configuration Examples

### Track Matching (Default)
```yaml
use_track_features: true
use_jet_features: false
use_cell_track_matching: true
calibration_data_file: "HStrackmatching_calibration.txt"
```

### Jet Matching
```yaml
use_track_features: false
use_jet_features: true
use_cell_jet_matching: true
calibration_data_file: "cell_jet_calibration.txt"
```

### Quick Analysis
```yaml
num_files: 5
top_worst_events: 10
feature_comparison_sample_size: 1000
```