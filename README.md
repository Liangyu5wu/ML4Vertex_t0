# Vertex Time Prediction Models

A modular framework for training transformer-based models for vertex t0 prediction with LAr Calorimeter in the ATLAS experiment.

## Project Structure

```
ML4Vertex_t0/
├── config/                     # Configuration files
│   ├── base_config.py         # Base configuration class
│   ├── transformer_config.py  # Transformer-specific configuration
│   └── configs/               # YAML configuration files
│       ├── default.yaml       # Default model configuration
│       ├── experiment2.yaml   # Main experimental setup
│       └── test_fast.yaml     # Fast testing configuration
├── src/                       # Source code
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
├── jobs/                     # SLURM job scripts
├── models/                   # Saved models and results (external)
└── process_h5.py            # Data preprocessing utility
```

## Installation

### Using UV (Recommended)
```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Using pip
```bash
pip install tensorflow numpy h5py scikit-learn matplotlib seaborn pyyaml
```

## Quick Start

### Training
```bash
# Basic training with YAML config
python scripts/train.py --config-file config/configs/default.yaml

# With detector calibration
python scripts/train.py --config-file config/configs/experiment2.yaml

# Override specific parameters
python scripts/train.py \
    --config-file config/configs/default.yaml \
    --epochs 100 \
    --batch-size 32
```

### Evaluation
```bash
python scripts/evaluate.py --model-dir models/my_model --load-data
```

### Parameter Sweeps
```bash
# Interactive sweep runner
python scripts/run_sweep_experiments.py

# Direct sweep execution
python scripts/parameter_sweep.py \
    --base-config config/configs/experiment2.yaml \
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
```

### With Detector Calibration (`experiment2.yaml`)
```yaml
model_name: "transformer_calibrated"
use_detector_params: true
emb1_params: [48.5266, 37.56, 28.9393, 23.1505, 18.5468, 13.0141, 8.03724]
# ... other detector parameters
```

## Data Format

HDF5 files with:
- **HSvertex**: Vertex coordinates and target time
- **cells**: Variable-length cell sequences per event
- **tracks**: Track information for cell matching

Required cell features:
- Position: `Cell_x`, `Cell_y`, `Cell_z`
- Energy: `Cell_e`, `Cell_significance`
- Detector info: `Cell_Barrel`, `Cell_layer`
- Track matching: `matched_track_pt`, `matched_track_deltaR`

## Key Features

- **YAML Configuration**: Easy parameter management
- **Detector Calibration**: Layer-specific parameter injection
- **Variable Sequences**: Handles different cell counts per event
- **Parameter Sweeps**: Automated hyperparameter optimization
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **SLURM Integration**: Batch job support

## Model Architecture

Transformer-based sequence model:
1. Cell sequences → Dense projection → Positional encoding
2. Multi-head self-attention blocks
3. Global pooling → Fusion with vertex features
4. Dense layers → Time prediction

## Output Structure

Results saved in `models/[model_name]/`:
- `model.keras`: Trained model
- `config.yaml`: Configuration
- `training_history.npz`: Training metrics
- `evaluation_plots/`: Visualization suite

## Environment Setup

```bash
# Activate environment
source setup.sh  # or source .venv/bin/activate

# Set external directories (optional)
export VERTEX_TIME_MODELS_DIR=/path/to/models
export VERTEX_TIME_RESULTS_DIR=/path/to/results
```
