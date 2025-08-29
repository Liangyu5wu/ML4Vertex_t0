# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

ML4Vertex_t0 is a machine learning framework for vertex time (t0) prediction using LAr Calorimeter data in ATLAS. It implements both Transformer and DNN models with advanced features like attention masks, smart padding, and optimized parameter sweeps.

## Development Environment Setup

### Installation
```bash
# Install UV package manager and create virtual environment
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.9
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Environment Activation
```bash
source setup.sh  # Activates .venv/bin/activate
```

## Common Development Tasks

### Training Models
```bash
# Train Transformer model
python scripts/train.py --config-file config/configs/experiment_with_jets.yaml

# Train DNN model
python scripts/train.py --config-file config/configs/experiment_dnn.yaml

# Train Baseline-Guided DNN model
python scripts/train.py --config-file config/configs/experiment_baseline_guided_track.yaml

# Train with parameter overrides
python scripts/train.py --config-file config/configs/experiment_dnn.yaml --epochs 50 --learning-rate 5e-4
```

### Running Evaluations
```bash
# Evaluate any model (auto-detects Transformer/DNN/Baseline-Guided)
python scripts/evaluate.py --model-dir ../models/your_model --load-data

# Evaluate baseline-guided model
python scripts/evaluate.py --model-dir ../models/baseline_guided_dnn_with_tracks --load-data
```

### Parameter Sweeps
```bash
# Quick parameter sweeps with data caching
python scripts/parameter_sweep.py --config config/configs/experiment_dnn.yaml --grid dnn_quick --max-exp 20
python scripts/parameter_sweep.py --config config/configs/experiment_with_jets.yaml --grid transformer_quick --max-exp 20

# Model comparison sweeps
python scripts/parameter_sweep.py --config config/configs/experiment_with_jets.yaml --grid comparison --max-exp 30

# Analyze sweep results
python scripts/analyze_sweep.py results/sweep_20250101_123456/
```

### SLURM Job Submission (NERSC)
```bash
# Submit parameter sweeps
sbatch jobs/sweep_optimization.sh transformer_full 50
sbatch jobs/sweep_dnn_quick.sh

# Submit training jobs
sbatch jobs/model_dnn_nersc.sh
sbatch jobs/model_nersc.sh
sbatch jobs/model_baseline_guided_track_nersc.sh
```

### Testing
There is no dedicated test suite. Verification is done through:
- Training quick models with `test_fast.yaml`
- Running evaluation scripts
- Parameter sweep validation

## Code Architecture

### Configuration System
- **Base Configuration**: `config/base_config.py` - Shared parameters and YAML loading
- **Model-Specific Configs**: 
  - `config/transformer_config.py` - Transformer architecture parameters
  - `config/dnn_config.py` - Two-stage DNN parameters
- **YAML Configs**: `config/configs/` - Experiment-specific settings

### Model Architecture
Three main model types with advanced features:

#### Transformer Model (`src/models/transformer_model.py`)
```
Variable cells → Attention Mask → Transformer Blocks → Global Pooling → Dense → Vertex Time
```
- Uses `TransformerBlock` and `MaskedGlobalAveragePooling1D`
- Supports attention masks for padding exclusion
- Smart padding with feature-specific values

#### Two-Stage DNN Model (`src/models/dnn_model.py`)
```
Cells → Cell-level MLP → Masked Attention Pooling → Event-level MLP → Vertex Time
```
- Uses `MaskedAttentionPooling` for learned attention weights
- Dual-stage processing: cell-level then event-level
- Alternative to traditional sigma-weighted pooling

#### Baseline-Guided DNN Model (`src/models/baseline_guided_model.py`)
```
Cells → Cell-level MLP → Global Average Pooling → Combine with Baseline → Residual Learning → Vertex Time
                                                        ↑
                                               Baseline Predictions (External)
```
- **Residual Learning Approach**: Learns corrections to existing baseline method predictions
- **Three-Input Architecture**: Cell sequences + Vertex features + Baseline predictions
- **Physics-Informed Design**: Leverages domain knowledge through baseline predictions
- **Simplified Processing**: Uses global average pooling instead of attention mechanisms
- **Robust to Baseline Quality**: Can handle imperfect baseline predictions through residual learning
- **Configuration**: Use `model_architecture: "baseline_guided_dnn"` in config files

### Data Pipeline
- **Data Loading**: `src/data/data_loader.py` - HDF5 file processing with cell filtering
- **Data Processing**: `src/data/data_processor.py` - Feature normalization, train/val/test splits
- **Smart Padding**: Feature-specific padding values instead of zeros

### Training Infrastructure
- **Trainer**: `src/training/trainer.py` - Unified training loop for both model types
- **Parameter Sweeps**: `scripts/parameter_sweep.py` - Optimized data caching and batch training
- **Evaluation**: `src/evaluation/` - Performance analysis and visualization

## Model Features

### Attention Mask System
- **Purpose**: Exclude padding cells from attention computations
- **Implementation**: Boolean masks passed to model layers
- **Benefits**: ~19% RMSE improvement over zero padding
- **Usage**: Set `use_attention_mask: true` in config

### Cell Filtering Options
- **Valid cells**: `require_valid_cells: true`
- **Track matching**: `use_cell_track_matching: true`
- **Jet matching**: `use_cell_jet_matching: true` 
- **Time quality cuts**: `use_time_quality_cut: true`

### Feature Selection
- **Spatial features**: Cell position information
- **Track features**: Cell-track matching data
- **Jet features**: Cell-jet matching data
- **Smart toggling**: Automatic feature list adjustment

### Baseline-Guided Model Features

#### Baseline Method Integration
- **External Predictions**: Requires baseline method predictions as third input
- **Residual Learning**: Model learns `residual = target - baseline` then outputs `baseline + residual`
- **Baseline Method Filtering**: Optional filtering by baseline method performance (`baseline_method_threshold`)
- **Robustness**: Works with imperfect baselines, improving through learned corrections

#### Architectural Differences
- **No Attention Mechanism**: Uses simple global average pooling for computational efficiency
- **Simplified Network**: 3-4 dense layers vs complex attention architectures
- **Physics Integration**: Incorporates domain knowledge through baseline predictions
- **Three-Input Design**: Handles cell sequences, vertex features, and baseline predictions simultaneously

#### Configuration Requirements
- **Data Requirements**: Must have baseline predictions computed and stored
- **Model Architecture**: Set `model_architecture: "baseline_guided_dnn"`
- **Baseline Features**: Often combined with track matching for optimal performance
- **Loss Functions**: Both MSE and Huber loss supported for residual learning

## Performance Optimization

### Data Caching Strategy
Parameter sweeps use optimized data caching:
- **Traditional**: 40min × N experiments = 13+ hours
- **Optimized**: 40min + 5min × N experiments = 2-3 hours
- **Method**: Load data once, train multiple models in memory

### Memory Management
- External directories for models and results (outside repo)
- Environment variables: `VERTEX_TIME_MODELS_DIR`, `VERTEX_TIME_RESULTS_DIR`
- Automatic cleanup of intermediate training states

## Configuration Guidelines

### Model Selection
Choose model type via `model_architecture` parameter:
- `"transformer"` - Use transformer_config.py
- `"two_stage_dnn"` - Use dnn_config.py  
- `"baseline_guided_dnn"` - Use dnn_config.py (baseline-guided variant)

### Loss Functions
- **MSE**: `loss_function: "mse"` (default)
- **Huber**: `loss_function: "huber"` with `huber_delta: 100.0`

### Experiment Naming
Use descriptive model names that reflect configuration:
- `transformer_with_jets` - Transformer with jet features
- `dnn_with_jets` - DNN with jet matching
- `baseline_guided_dnn_with_tracks` - Baseline-guided model with track features
- `baseline_test` - Quick testing configuration

## File Structure Notes

### External Directories
Models and results are stored outside the repository:
- Models: `../models/` (configurable via environment)
- Results: `../results/` (for parameter sweeps)
- Data: `../selected_h5/` or `../selected_h5_with_jets/`

### Job Scripts
SLURM scripts in `jobs/` are configured for NERSC Perlmutter:
- GPU allocation and TensorFlow module loading
- Environment setup and path configuration
- Standardized output/error logging

### Calibration Data
External calibration files in `calibration_data/`:
- `HStrackmatching_calibration.txt` - Cell-track calibration
- `cell_jet_calibration.txt` - Cell-jet calibration

## Development Best Practices

### Configuration Changes
- Always use YAML files for experiments
- Validate configs with `config.validate_config()`
- Save final configs with trained models

### Model Development
- All model classes support various input configurations
- Transformer/DNN: Use `build_model_with_mask()` for attention mask support
- Baseline-guided: Use `build_model()` with three inputs (cells, vertex, baseline)
- Custom layers require registration in `load_model()` methods
- Baseline-guided models require special handling in evaluation scripts

### Parameter Sweeps
- Use optimized sweeps for multiple experiments
- Analyze results with `analyze_sweep.py`
- Check `results.csv` for detailed metrics comparison

### SLURM Usage
- Use appropriate time limits based on experiment size
- Monitor GPU memory usage with TensorFlow
- Check logs in `../logs/` directory