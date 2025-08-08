# Vertex Time Prediction Models

A modular framework for training transformer-based models for vertex t0 prediction with LAr Calorimeter in the ATLAS experiment.

## Project Structure

project_root/
├── config/                     # Configuration files
│   ├── base_config.py         # Base configuration class
│   ├── transformer_config.py  # Transformer-specific configuration
│   └── configs/               # YAML configuration files
│       ├── default.yaml       # Default model configuration
│       ├── large_model.yaml   # Large model configuration
│       └── experiment1.yaml   # Example experimental setup
├── src/                       # Source code
│   ├── data/                  # Data loading and processing
│   ├── models/                # Model architectures
│   ├── training/              # Training utilities
│   └── evaluation/            # Evaluation and visualization
├── scripts/                   # Main execution scripts
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation script
├── models/                   # Saved models and results
│   └── [model_name]/         # Individual model directories
│       ├── model.keras       # Saved model
│       ├── config.json       # Model configuration (JSON)
│       ├── config.yaml       # Model configuration (YAML)
│       └── evaluation_plots/ # Evaluation visualizations
└── README.md

## Quick Start

# Installation
pip install tensorflow numpy h5py scikit-learn matplotlib seaborn pyyaml

# Training with YAML configuration (Recommended)
python scripts/train.py --config-file config/configs/default.yaml

# Training with large model
python scripts/train.py --config-file config/configs/large_model.yaml

# Custom experiment
python scripts/train.py --config-file config/configs/experiment1.yaml

# Override specific parameters
python scripts/train.py \
    --config-file config/configs/default.yaml \
    --model-name my_custom_model \
    --epochs 200 \
    --batch-size 32

# Traditional training (without YAML)
python scripts/train.py \
    --model-name my_transformer \
    --data-dir /path/to/data \
    --epochs 100 \
    --batch-size 128 \
    --max-cells 50 \
    --use-spatial

# Evaluation
python scripts/evaluate.py --model-dir models/my_transformer --load-data

## Configuration

YAML Configuration Files (config/configs/):

# default.yaml - Standard transformer setup
model_name: "transformer_default"
d_model: 128
num_heads: 8
max_cells: 40
learning_rate: 0.0001
use_spatial_features: false

# large_model.yaml - Larger capacity model
model_name: "transformer_large"
d_model: 256
num_heads: 16
max_cells: 60
learning_rate: 0.00005
use_spatial_features: true

# experiment1.yaml - Custom experimental setup
model_name: "transformer_experiment1"
d_model: 192
num_heads: 12
cell_selection_feature: "Cell_significance"
use_spatial_features: true

## Data Format

HDF5 structure:
HSvertex/
├── HSvertex_reco_x, HSvertex_reco_y, HSvertex_reco_z  # Vertex coordinates
└── HSvertex_time                                      # Target vertex time

cells/ (variable length per event)
├── Cell_x, Cell_y, Cell_z                            # Cell coordinates
├── Cell_eta, Cell_phi                                # Cell angles
├── Cell_e, Cell_significance                         # Cell energy and significance
├── Cell_time_TOF_corrected                           # Time-of-flight corrected time
├── matched_track_pt, matched_track_deltaR            # Track matching info
└── valid, matched_track_HS                           # Selection flags

## Features

- YAML Configuration: Easy parameter management through config files
- Modular Design: Easy to extend and modify
- Transformer Architecture: Multi-head attention for sequence modeling
- Variable Sequence Length: Handles events with different numbers of cells
- Comprehensive Evaluation: Multiple metrics and visualizations
- Automatic Saving: Models, configs, and plots organized by experiment
- Resume Training: Continue from saved checkpoints
- Command Line Override: Override YAML parameters via command line

## Creating Custom Configurations

1. Copy an existing YAML file from config/configs/
2. Modify parameters as needed
3. Train with: python scripts/train.py --config-file path/to/your/config.yaml

Example custom configuration:
model_name: "my_experiment"
max_cells: 80
d_model: 320
num_heads: 20
learning_rate: 0.00003
epochs: 150
use_spatial_features: true

## Outputs

All results organized in models/[model_name]/ directory:
- model.keras: Trained model
- config.json & config.yaml: Model configuration
- training_history.npz: Training metrics over epochs
- predictions.npz: Test set predictions
- evaluation_plots/: Comprehensive visualization suite
  - training_metrics.png: Loss and metrics curves
  - prediction_results.png: Predicted vs actual scatter plot
  - histogram_comparison.png: Distribution comparison
  - error_distribution.png: Prediction error analysis
  - residuals_vs_predicted.png: Residual analysis

## Model Architecture

Transformer-based sequence model:
1. Cell sequences → Dense projection → Positional encoding
2. Multi-head self-attention blocks (configurable depth)
3. Global average pooling → Feature fusion with vertex info
4. Dense layers → Vertex time prediction

Key parameters:
- d_model: Hidden dimension (128-256)
- num_heads: Attention heads (8-16)
- num_transformer_blocks: Model depth (3-6)
- max_cells: Maximum cells per event (40-80)
