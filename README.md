# Vertex Time Prediction Models

A modular framework for training transformer-based models for vertex t0 prediction with LAr Calorimeter in the ATLAS experiment.

## Project Structure

```
ML4Vertex_t0/
├── config/                     # Configuration files and classes
│   ├── __init__.py            # Configuration module exports
│   ├── base_config.py         # Base configuration class
│   ├── transformer_config.py  # Transformer-specific configuration
│   └── configs/               # YAML configuration files
│       ├── experiment_with_jets.yaml  # Jet matching experimental setup
│       ├── experiment_nersc.yaml      # NERSC cluster configuration
│       └── test_fast.yaml     # Fast testing configuration
├── calibration_data/          # External calibration data files
│   ├── HStrackmatching_calibration.txt    # Cell-track matching calibration
│   └── cell_jet_calibration.txt           # Cell-jet matching calibration 
├── src/                       # Source code
│   ├── __init__.py            # Source package initialization
│   ├── data/                  # Data loading and processing
│   ├── models/                # Model architectures
│   ├── training/              # Training utilities
│   └── evaluation/            # Evaluation and visualization
├── scripts/                   # Main execution scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── parameter_sweep.py    # Hyperparameter optimization
│   ├── run_sweep_experiments.py # Easy sweep interface
│   └── analyze_sweep.py      # Sweep results analysis
├── jobs/                    # SLURM job submission scripts
│   ├── model_V1.sh         # Basic training job
│   ├── model_nersc.sh      # NERSC GPU cluster job
│   ├── sweep_archi.sh      # Architecture parameter sweep
│   └── sweepbatch.sh       # Training parameter sweep
└── process_h5.py            # Data preprocessing utility
```

## Installation

### Using UV (Recommended)
```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv --python 3.9
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Quick Start

### Training
```bash
# Basic fast training with YAML config
python scripts/train.py --config-file config/configs/test_fast.yaml

# With detector calibration (recommended)
python scripts/train.py --config-file config/configs/experiment_with_jets.yaml

# Override specific parameters
python scripts/train.py \
    --config-file config/configs/experiment_with_jets.yaml \
    --epochs 100 \
    --batch-size 32
```

### Evaluation
```bash
python scripts/evaluate.py --model-dir ../models/my_model --load-data
```

### Parameter Sweeps
```bash
# Interactive sweep runner
python scripts/run_sweep_experiments.py

# Direct sweep execution
python scripts/parameter_sweep.py \
    --base-config config/configs/experiment_with_jets.yaml \
    --grid-type quick \
    --max-experiments 50

# Analyze results
python scripts/analyze_sweep.py results/parameter_sweep_YYYYMMDD_HHMMSS/
```

## Configuration

YAML configuration files in `config/configs/`:

### Basic Configuration (`default.yaml`)
```yaml
model_name: "transformer_default"
d_model: 128
num_heads: 8
max_cells: 40
learning_rate: 0.0001
use_spatial_features: false
use_detector_params: false
calibration_data_file: "HStrackmatching_calibration.txt"
```

### With Detector Calibration (`experiment2.yaml`)
```yaml
model_name: "transformer_calibrated"
use_detector_params: true
calibration_data_file: "HStrackmatching_calibration.txt"
calibration_validation: true
validation_detector_type: 1  # 1=barrel, 0=endcap
validation_layer: 1  # 1, 2, or 3
gaussian_fit_range: 120  # Range for Gaussian fitting
```

## Detector Calibration

The framework supports detector-specific time calibration:

### Calibration Data Format
External calibration parameters are stored in `calibration_data/HStrackmatching_calibration.txt`:

```
# Energy bins: 1-1.5, 1.5-2, 2-3, 3-4, 4-5, 5-10, >10 GeV
EMB1_params: 48.5266, 37.56, 28.9393, 23.1505, 18.5468, 13.0141, 8.03724
EMB1_sigma: 416.994, 293.206, 208.321, 148.768, 117.756, 106.804, 57.6545
EMB2_params: 46.2244, 41.5079, 38.5544, 36.9812, 31.2718, 29.7469, 19.331
EMB2_sigma: 2001.56, 1423.38, 1010.24, 720.392, 551.854, 357.594, 144.162
...
```

### Calibration Process
1. **Time Calibration**: `calibrated_time = Cell_time_TOF_corrected - calibration_params[energy_bin]`
2. **Traditional t0 Calculation**: `t0 = Σ(w_i × t_i) / Σ(w_i)` where `w_i = 1/σ_i²`

### Validation and Baseline Checks
When `use_detector_params: true`, the framework automatically generates:

**Calibration Validation** (`calibration_validation.png`):
- 7 subplots showing before/after time distributions by energy bin
- Statistics for specified detector region

**Baseline Checks** (`baseline_check/` directory):
- `traditional_t0_distribution.png` - Traditional method t0 distribution
- `t0_error_distribution.png` - Traditional method error distribution  
- `traditional_t0_vs_true_2d.png` - Traditional vs true t0 comparison

## Key Features

- **YAML Configuration**: Easy parameter management
- **External Calibration Data**: Detector parameters stored in external files
- **Time Calibration**: Energy and layer-dependent time corrections
- **Validation Plots**: Automatic calibration verification
- **Baseline Comparison**: Traditional vs ML method comparison
- **Variable Sequences**: Handles different cell counts per event
- **Parameter Sweeps**: Automated hyperparameter optimization
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **SLURM Integration**: Batch job support

## Model Architecture

Transformer-based sequence model:
1. Cell sequences → Time calibration (if enabled)
2. Dense projection → Positional encoding
3. Multi-head self-attention blocks
4. Global pooling → Fusion with vertex features
5. Dense layers → Time prediction

## Output Structure

Results saved in `../models/[model_name]/`:
- `model.keras`: Trained model
- `config.yaml`: Configuration
- `training_history.npz`: Training metrics
- `evaluation_plots/`: Visualization suite
  - `histogram_comparison.png`: ML prediction distributions (bin width = 10)
  - `prediction_results.png`: Predicted vs actual scatter
  - `calibration_validation.png`: Time calibration verification
- `baseline_check/`: Traditional method analysis
  - `traditional_t0_distribution.png`: Traditional t0 with Gaussian fit
  - `t0_error_distribution.png`: Traditional error analysis
  - `traditional_t0_vs_true_2d.png`: Traditional vs true correlation

## Synchronizing Updates

To sync with latest GitHub updates:
```bash
# Basic sync
git pull

# If you have local changes
git stash        # Save local changes
git pull         # Get updates
git stash pop    # Restore local changes

# Force sync (overwrites local changes)
git fetch origin
git reset --hard origin/main
```

## Environment Setup

```bash
# Activate environment
source setup.sh  # or source .venv/bin/activate

# Set external directories (optional)
export VERTEX_TIME_MODELS_DIR=/path/to/models
export VERTEX_TIME_RESULTS_DIR=/path/to/results
```
