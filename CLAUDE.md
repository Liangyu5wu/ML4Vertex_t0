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

# Train Multi-Input DNN model
python scripts/train.py --config-file config/configs/experiment_dnn_with_jets_tracks.yaml

# Train Multi-Input Transformer model
python scripts/train.py --config-file config/configs/experiment_transformer_with_jets_tracks.yaml

# Train HGTD Multi-Input DNN model
python scripts/train.py --config-file config/configs/experiment_hgtd_dnn_with_jets_tracks.yaml

# Train HGTD-Only DNN model (no LAr data)
python scripts/train.py --config-file config/configs/experiment_hgtd_only.yaml

# Train with parameter overrides
python scripts/train.py --config-file config/configs/experiment_dnn.yaml --epochs 50 --learning-rate 5e-4
```

### Running Evaluations
```bash
# Evaluate any model (auto-detects Transformer/DNN/Multi-Input/Baseline-Guided)
python scripts/evaluate.py --model-dir ../models/your_model --load-data

# Evaluate baseline-guided model
python scripts/evaluate.py --model-dir ../models/baseline_guided_dnn_with_tracks --load-data

# Evaluate multi-input models
python scripts/evaluate.py --model-dir ../models/multi_input_dnn_with_jets_tracks --load-data
python scripts/evaluate.py --model-dir ../models/multi_input_transformer_with_jets_tracks --load-data

# Evaluate HGTD models
python scripts/evaluate.py --model-dir ../models/hgtd_multi_input_dnn_with_jets_tracks --load-data
python scripts/evaluate.py --model-dir ../models/hgtd_only_dnn --load-data
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

### Baseline Method Analysis
```bash
# Comprehensive baseline reconstruction analysis (track matching by default)
cd baseline_analysis
python baseline_analysis.py

# Custom configuration with different parameters
python baseline_analysis.py --config baseline_analysis_config.yaml --top-events 50 --sample-size 5000

# For understanding reconstruction failures and event characteristics
python baseline_analysis.py --top-events 20 --sample-size 2000
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
sbatch jobs/model_multi_input_dnn_nersc.sh
sbatch jobs/model_multi_input_transformer_nersc.sh
sbatch jobs/model_hgtd_multi_input_dnn_nersc.sh
sbatch jobs/model_hgtd_only_dnn_nersc.sh
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

### Model Architecture & Input Structure
Six main model types with different input requirements:

#### Model Input Summary

| Model | Input Count | Input Names | Tensor Shapes |
|-------|------------|-------------|---------------|
| **Transformer** | 2-3 | `cell_sequence`, `vertex_features`, `attention_mask` | (N,F), (V,), (N,) |
| **Two-Stage DNN** | 2-3 | `cell_sequence`, `vertex_features`, `attention_mask` | (N,F), (V,), (N,) |
| **Baseline-Guided DNN** | 3 | `cell_sequence`, `vertex_features`, `baseline_prediction` | (N,F), (V,), (1,) |
| **Multi-Input DNN** | 5 | `cell_inputs`, `vertex_inputs`, `jet_inputs`, `track_inputs`, `mask_inputs` | (N,F), (V,), (J,4), (T,5), (N,) |
| **Multi-Input Transformer** | 5 | `cell_inputs`, `vertex_inputs`, `jet_inputs`, `track_inputs`, `mask_inputs` | (N,F), (V,), (J,4), (T,5), (N,) |
| **HGTD Multi-Input DNN** | 6 | `cell_inputs`, `vertex_inputs`, `jet_inputs`, `track_inputs`, `hgtd_track_inputs`, `mask_inputs` | (N,F), (V,), (J,4), (T,5), (H,7), (N,) |
| **HGTD-Only DNN** | 2 | `hgtd_track_inputs`, `vertex_inputs` | (H,7), (V,) |

**Legend**: N=max_cells(60), F=cell_features(7), V=vertex_features(varies), J=max_jets(7), T=max_tracks(30), H=max_hgtd_tracks(30)

#### Transformer Model (`src/models/transformer_model.py`)
**Inputs**: `cell_sequence` (N×F), `vertex_features` (V), `attention_mask` (N)
```
Variable cells → Attention Mask → Transformer Blocks → Global Pooling → Dense → Vertex Time
```
- Uses `TransformerBlock` and `MaskedGlobalAveragePooling1D`
- Supports attention masks for padding exclusion
- Smart padding with feature-specific values

#### Two-Stage DNN Model (`src/models/dnn_model.py`)
**Inputs**: `cell_sequence` (N×F), `vertex_features` (V), `attention_mask` (N)
```
Cells → Cell-level MLP → Masked Attention Pooling → Event-level MLP → Vertex Time
```
- Uses `MaskedAttentionPooling` for learned attention weights
- Dual-stage processing: cell-level then event-level
- Alternative to traditional sigma-weighted pooling

#### Baseline-Guided DNN Model (`src/models/baseline_guided_model.py`)
**Inputs**: `cell_sequence` (N×F), `vertex_features` (V), `baseline_prediction` (1)
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

#### Multi-Input DNN Model (`src/models/multi_input_dnn_model.py`)
**Inputs**: `cell_inputs` (N×F), `vertex_inputs` (V), `jet_inputs` (J×4), `track_inputs` (T×5), `mask_inputs` (N)
```
Cells → Cell-level MLP → Masked Attention Pooling
Jets → Configurable MLP → Global Average Pooling        } → Concat → Event-level MLP → Vertex Time
Tracks → Configurable MLP → Global Average Pooling
Vertex Features → Dense Processing
Attention Mask → Masking Support
```
- **Five-Input Architecture**: Cell sequences + Vertex features + Jets + Tracks + Attention mask
- **Event-Level Integration**: Processes jets (pt, eta, phi, width) and tracks (pt, eta, phi, d0, z0) separately
- **Configurable Encoders**: Jet/track encoder architecture fully configurable via YAML
- **Specialized Calibration**: Uses `multi_input_calibration.txt` for models without jet/track-matching
- **Flexible Filtering**: Supports 1-7 jets and 1-30 tracks per event with smart padding
- **Advanced Processing**: Combines cell attention pooling with jet/track global pooling
- **Configuration**: Use `model_architecture: "multi_input_dnn"` in config files

#### Multi-Input Transformer Model (`src/models/multi_input_transformer_model.py`)
**Inputs**: `cell_inputs` (N×F), `vertex_inputs` (V), `jet_inputs` (J×4), `track_inputs` (T×5), `mask_inputs` (N)
```
Cells → Dense Projection → Transformer Blocks → Masked Global Pooling
Jets → Dense Processing → Global Average Pooling     } → Concat → Dense Layers → Vertex Time
Tracks → Dense Processing → Global Average Pooling
Vertex Features → Dense Processing
Attention Mask → Transformer Masking
```
- **Five-Input Architecture**: Cell sequences + Vertex features + Jets + Tracks + Attention mask
- **No Positional Encoding**: Relies purely on self-attention without position information
- **Transformer Processing**: Applies transformer blocks to cell sequences with attention masking
- **Event-Level Integration**: Jets and tracks processed through dense layers then global pooled
- **Specialized Calibration**: Uses `multi_input_calibration.txt` for models without jet/track-matching
- **Scalable Design**: Handles variable numbers of jets/tracks through padding and masking
- **Configuration**: Use `model_architecture: "multi_input_transformer"` in config files

#### HGTD Multi-Input DNN Model (`src/models/hgtd_multi_input_dnn_model.py`)
**Inputs**: `cell_inputs` (N×F), `vertex_inputs` (V), `jet_inputs` (J×4), `track_inputs` (T×5), `hgtd_track_inputs` (H×7), `mask_inputs` (N)
```
Cells → Cell-level MLP → Masked Attention Pooling
Jets → Configurable MLP → Global Average Pooling
LAr Tracks → Configurable MLP → Global Average Pooling     } → Concat → Event-level MLP → Vertex Time
HGTD Tracks → Configurable MLP → Global Average Pooling
Vertex Features → Dense Processing
Attention Mask → Masking Support
```
- **Six-Input Architecture**: LAr cell sequences + Vertex features + Jets + LAr tracks + HGTD tracks + Attention mask
- **HGTD Track Features**: pt, eta, phi, d0, z0, time, timeRes (7 features from HGTD timing detector)
- **HGTD Track Filtering**: Selects tracks with `valid==True` & `Track_hasValidTime==1`, sorted by pt
- **Independent Encoders**: Separate configurable encoders for LAr cells, jets, LAr tracks, and HGTD tracks
- **Event-Level Fusion**: All processed features concatenated before event-level prediction
- **Data Directory**: Uses `../Vertex_timing_HGTD_w_LAr/` with 50 HDF5 files
- **Configurable Architecture**: HGTD track encoder units, dropout rates, activation, and batch norm all configurable
- **Smart Padding**: HGTD track padding values (time: 0.0, timeRes: -999.0) transformed to normalized space
- **Configuration**: Use `model_architecture: "hgtd_multi_input_dnn"` in config files

#### HGTD-Only DNN Model (`src/models/hgtd_only_dnn_model.py`)
**Inputs**: `hgtd_track_inputs` (H×7), `vertex_inputs` (V)
```
HGTD Tracks → Track Encoder MLP → Global Average Pooling ┐
Vertex Features → Dense Processing ─────────────────────┤→ Concat → Event MLP → Vertex Time
```
- **Two-Input Architecture**: HGTD tracks + Vertex features only (NO LAr data)
- **Simplified Design**: Uses only HGTD timing detector information without calorimeter cells, jets, or LAr tracks
- **HGTD Track Features**: pt, eta, phi, d0, z0, time, timeRes (7 features from HGTD timing detector)
- **HGTD Track Filtering**: Selects tracks with `valid==True` & `Track_hasValidTime==1`, sorted by pt, top 30 tracks
- **Configurable Encoder**: HGTD track encoder units, dropout rates, activation, and batch norm all configurable
- **Data Directory**: Uses `../Vertex_timing_HGTD_w_LAr/` with 50 HDF5 files (same data as HGTD multi-input)
- **Smart Padding**: HGTD track padding values transformed to normalized space (time: 0.0, timeRes: -999.0)
- **Independent Pipeline**: Completely separate from LAr-based workflows, no cell/jet/track preprocessing
- **Configuration**: Use `model_architecture: "hgtd_only_dnn"` in config files

### Data Pipeline
- **Data Loading**: `src/data/data_loader.py` - HDF5 file processing with cell filtering
- **Multi-Input Loading**: `src/data/multi_input_data_loader.py` - Jets and tracks data processing (returns variable-length sequences)
- **HGTD Multi-Input Loading**: `src/data/hgtd_multi_input_data_loader.py` - LAr cells, jets, LAr tracks, and HGTD tracks data processing
- **HGTD-Only Loading**: `src/data/hgtd_only_data_loader.py` - HGTD tracks only (no LAr data)
- **Data Processing**: `src/data/data_processor.py` - Feature normalization, train/val/test splits
- **Multi-Input Processing**: `src/data/multi_input_data_processor.py` - Jets/tracks normalization and dataset creation
- **HGTD Multi-Input Processing**: `src/data/hgtd_multi_input_data_processor.py` - HGTD tracks normalization and dataset creation
- **HGTD-Only Processing**: `src/data/hgtd_only_data_processor.py` - HGTD tracks normalization and dataset creation (no LAr)
- **Normalization Strategy**: Normalize before padding (statistics computed only from real data)
- **Smart Padding**: Feature-specific padding values transformed to normalized space

### Training Infrastructure
- **Trainer**: `src/training/trainer.py` - Unified training loop for all model types
- **Parameter Sweeps**: `scripts/parameter_sweep.py` - Optimized data caching and batch training
- **Evaluation**: `src/evaluation/` - Performance analysis and visualization with auto-detection

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

### Advanced Filtering Features

#### Time Quality Cut System
- **Two modes available**:
  - `use_detector_params: true` - Full detector calibration + time cut
  - `use_detector_params: false` - Time cut without calibration (raw cell times)
- **Both modes use**: σ_total = √(σ_vertex² + σ_cell²) for precise uncertainty calculation
- **Configuration**: `vertex_time_sigma: 175.0`, `time_quality_n_sigma: 3.0`
- **Benefits**: Removes poor quality cells while preserving data integrity

#### Track Eta Cut (Multi-Input Models)
- **Purpose**: Filter tracks by pseudorapidity for detector acceptance
- **Configuration**: `use_track_eta_cut: true`, `track_eta_cut_value: 2.5`
- **Implementation**: Applied during track processing before model input
- **Usage**: Only affects multi-input models with track data

### Feature Selection
- **Spatial features**: Cell position information
- **Track features**: Cell-track matching data
- **Jet features**: Cell-jet matching data
- **Smart toggling**: Automatic feature list adjustment

### Normalization and Padding Strategy

#### Data Processing Order
For multi-input models (jets and tracks), the data flow follows this optimized order:
1. **Load raw data** from HDF5 files (original feature space)
2. **Split** into train/val/test sets
3. **Normalize** using statistics computed **only from real data** (no padding)
4. **Pad** sequences in normalized space using transformed padding values
5. **Create TensorFlow datasets** for training

#### Key Benefits
- **Correct statistics**: Mean and std computed only from real jets/tracks, not contaminated by padding values
- **Configured padding preserved**: Original padding values from config (`jet_padding_values`, `track_padding_values`) are automatically transformed to normalized space
- **Consistent approach**: All input types (cells, jets, tracks) follow the same normalize-then-pad pattern

#### Implementation Details
- **Variable-length normalization**: `_normalize_variable_length_features()` collects only real features for statistics
- **Smart padding transformation**: Configured padding values (e.g., `-1.0`, `-999.0`) are transformed using:
  ```
  padding_normalized = (padding_original - mean) / std
  ```
- **Example**: If jet pt has mean=50 GeV and std=20 GeV, then padding value `-1.0` becomes `-2.55` in normalized space
- **Model perspective**: Padding values appear as extreme values in normalized space, making them easy to identify and ignore

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

### Multi-Input Model Features

#### Configurable Encoder Architecture
- **Jet Encoder Parameters**: `jet_encoder_units`, `jet_dropout_rates`, `jet_activation`, `jet_use_batch_norm`
- **Track Encoder Parameters**: `track_encoder_units`, `track_dropout_rates`, `track_activation`, `track_use_batch_norm`
- **Flexible Architecture**: Fully configurable hidden layer sizes and dropout rates
- **Default Configuration**: [64, 32] units with [0.1, 0.1] dropout rates for both encoders
- **Batch Normalization**: Optional batch normalization support for improved training stability
- **Activation Functions**: Configurable activation functions (default: ReLU)

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
- `"multi_input_dnn"` - Use dnn_config.py (multi-input DNN variant)
- `"multi_input_transformer"` - Use transformer_config.py (multi-input Transformer variant)
- `"hgtd_multi_input_dnn"` - Use dnn_config.py (HGTD multi-input DNN variant)
- `"hgtd_only_dnn"` - Use dnn_config.py (HGTD-only DNN variant, no LAr data)

### Loss Functions
- **MSE**: `loss_function: "mse"` (default)
- **Huber**: `loss_function: "huber"` with `huber_delta: 100.0`

### Experiment Naming
Use descriptive model names that reflect configuration:
- `transformer_with_jets` - Transformer with jet features
- `dnn_with_jets` - DNN with jet matching
- `baseline_guided_dnn_with_tracks` - Baseline-guided model with track features
- `multi_input_dnn_with_jets_tracks` - Multi-input DNN with jets and tracks
- `multi_input_transformer_with_jets_tracks` - Multi-input Transformer with jets and tracks
- `hgtd_multi_input_dnn_with_jets_tracks` - HGTD multi-input DNN with LAr cells, jets, LAr tracks, and HGTD tracks
- `hgtd_only_dnn` - HGTD-only DNN using only HGTD tracks (no LAr data)
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
- `multi_input_calibration.txt` - Multi-input model calibration (no jet/track matching)
- `sigma_only_test_calibration.txt` - Test file with sigma values only (for validating that mean values are not used when `use_detector_params: false`)

### Baseline Analysis
Standalone analysis tool in `baseline_analysis/`:
- `baseline_analysis.py` - Comprehensive baseline reconstruction analysis
- `baseline_analysis_config.yaml` - Configurable parameters
- `README.md` - Usage documentation
- Outputs to `../../bad_events_check/` with timestamped directories
- Supports both track and jet matching analysis

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

### Baseline Analysis
- Use `baseline_analysis.py` for understanding reconstruction failures
- Configure track vs jet matching via YAML config
- Output includes detailed event logs and multi-level plotting
- Best used for investigating why certain events have poor reconstruction
- Results saved to `../../bad_events_check/` with timestamped directories