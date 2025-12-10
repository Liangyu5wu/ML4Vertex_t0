# Vertex Time Prediction Models

A framework for training models for vertex t0 prediction in ATLAS experiment. Features optimized parameter sweeps, attention mask support, physics-informed residual learning, event-level jets/tracks integration, and HGTD timing detector models.

## Project Structure

```
ML4Vertex_t0/
├── config/                     # Configuration files and classes
│   ├── __init__.py            # Configuration module exports
│   ├── base_config.py         # Base configuration class
│   ├── transformer_config.py  # Transformer-specific configuration
│   ├── dnn_config.py          # DNN-specific configuration
│   └── configs/               # YAML configuration files
│       ├── experiment_with_jets.yaml  # Transformer with jet features
│       ├── experiment_dnn.yaml        # DNN experimental setup
│       ├── experiment_baseline_guided_track.yaml  # Baseline-guided DNN
│       └── experiment_nersc.yaml      # NERSC cluster configuration
├── calibration_data/          # External calibration data files
│   ├── HStrackmatching_calibration.txt    # Cell-track matching calibration
│   ├── cell_jet_calibration.txt           # Cell-jet matching calibration
│   ├── multi_input_calibration.txt        # Multi-input model calibration
│   └── sigma_only_test_calibration.txt    # Test file (sigma only, no mean values)
├── src/                       # Source code
│   ├── __init__.py            # Source package initialization
│   ├── data/                  # Data loading and processing
│   ├── models/                # Model architectures
│   │   ├── common/            # Shared components (base classes, layers, utilities)
│   │   ├── dnn/               # DNN model implementations
│   │   └── transformer/       # Transformer model implementations
│   ├── training/              # Training utilities
│   └── evaluation/            # Evaluation and visualization
├── scripts/                   # Main execution scripts
│   ├── train.py              # Training script (supports all models)
│   ├── evaluate.py           # Evaluation script (auto-detects model type)
│   ├── parameter_sweep.py    # Optimized parameter sweeps with data caching
│   └── analyze_sweep.py      # Simplified results analysis
├── baseline_analysis/        # Standalone baseline method analysis
│   ├── baseline_analysis.py      # Comprehensive baseline reconstruction analysis
│   ├── baseline_analysis_config.yaml  # Configurable parameters for analysis
│   └── README.md             # Usage documentation and examples
├── jobs/                    # SLURM job submission scripts
│   ├── sweep_optimized.sh   # Efficient parameter sweeps
│   ├── sweep_dnn_comparison.sh  # DNN vs Transformer comparison
│   └── sweep_quick_test.sh  # Quick testing (2 hours)
├── setup.sh                 # Virtual environment setup
├── gpu_setup.sh             # GPU environment setup for NERSC Perlmutter
└── process_h5.py            # Data preprocessing utility
```

## Installation

```bash
# Basic setup
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.9
source .venv/bin/activate
uv pip install -r requirements.txt

# For GPU training on NERSC Perlmutter
source gpu_setup.sh
```

## Model Architectures

| Model | Input Types | Key Features |
|-------|-------------|--------------|
| **Transformer** | Cells + Vertex + Mask | Self-attention, global pooling |
| **Two-Stage DNN** | Cells + Vertex + Mask | Learned attention weights, masked pooling |
| **Baseline-Guided DNN** | Cells + Vertex + Baseline | Physics-informed residual learning |
| **Multi-Input (DNN/Transformer)** | Cells + Vertex + Jets + Tracks + Mask | Event-level features integration |
| **HGTD Multi-Input DNN** | Cells + Jets + LAr/HGTD Tracks + Mask | Timing detector fusion |
| **HGTD-Only DNN** | HGTD Tracks + Vertex | Pure timing-based prediction |

### Architecture Details

**Transformer**: Self-attention on cell sequences with masked pooling
**Two-Stage DNN**: Cell-level MLP → Learned attention pooling → Event-level MLP
**Baseline-Guided**: Residual learning on top of physics-based baseline predictions
**Multi-Input**: Separate encoders for cells/jets/tracks → Concatenate → Predict
**HGTD Multi-Input**: Adds HGTD timing tracks to multi-input architecture
**HGTD-Only**: Simple encoder using only HGTD timing information

**Input Dimensions**:
- Cells: N×7 (eta, phi, barrel, layer, time, energy, significance)
- Jets: 7×4 (pt, eta, phi, width)
- LAr Tracks: 30×5 (pt, eta, phi, d0, z0)
- HGTD Tracks: 30×7 (pt, eta, phi, d0, z0, time, timeRes)

## Quick Start

### Training Models

```bash
# Train Transformer model
python scripts/train.py --config-file config/configs/experiment_with_jets.yaml

# Train DNN model  
python scripts/train.py --config-file config/configs/experiment_dnn.yaml

# Train Baseline-Guided DNN model 🆕
python scripts/train.py --config-file config/configs/experiment_baseline_guided_track.yaml

# Train Multi-Input DNN model 🆕
python scripts/train.py --config-file config/configs/experiment_dnn_with_jets_tracks.yaml

# Train Multi-Input Transformer model 🆕
python scripts/train.py --config-file config/configs/experiment_transformer_with_jets_tracks.yaml

# Train HGTD Multi-Input DNN model 🆕
python scripts/train.py --config-file config/configs/experiment_hgtd_dnn_with_jets_tracks.yaml

# Train HGTD-Only DNN model 🆕
python scripts/train.py --config-file config/configs/experiment_hgtd_only.yaml

# Override parameters
python scripts/train.py --config-file config/configs/experiment_dnn.yaml --epochs 50 --learning-rate 5e-4
```

### Evaluation (Auto-detects Model Type)

```bash
# Evaluate any model - automatically detects Transformer/DNN/Multi-Input/Baseline-Guided
python scripts/evaluate.py --model-dir ../models/your_model --load-data

# Evaluate baseline-guided model 🆕
python scripts/evaluate.py --model-dir ../models/baseline_guided_dnn_with_tracks --load-data

# Evaluate multi-input models 🆕
python scripts/evaluate.py --model-dir ../models/multi_input_dnn_with_jets_tracks --load-data
python scripts/evaluate.py --model-dir ../models/multi_input_transformer_with_jets_tracks --load-data

# Evaluate HGTD models 🆕
python scripts/evaluate.py --model-dir ../models/hgtd_multi_input_dnn_with_jets_tracks --load-data
python scripts/evaluate.py --model-dir ../models/hgtd_only_dnn --load-data
```

### Optimized Parameter Sweeps ⚡

```bash
# Quick sweeps (data cached once, multiple experiments)
python scripts/parameter_sweep.py --config config/configs/experiment_dnn.yaml --grid dnn_quick --max-exp 20
python scripts/parameter_sweep.py --config config/configs/experiment_with_jets.yaml --grid transformer_quick --max-exp 20

# Model comparison
python scripts/parameter_sweep.py --config config/configs/experiment_with_jets.yaml --grid comparison --max-exp 30

# Analyze results
python scripts/analyze_sweep.py results/sweep_20250101_123456/
```

### Baseline Method Analysis 🆕

```bash
# Comprehensive baseline reconstruction analysis (track matching by default)
cd baseline_analysis
python baseline_analysis.py

# Custom configuration (e.g., jet matching)
python baseline_analysis.py --config my_config.yaml

# Override parameters
python baseline_analysis.py --top-events 50 --sample-size 5000
```

### SLURM Job Submission

```bash
# Efficient sweeps (10 hours → 80-100 experiments)
sbatch jobs/sweep_optimized.sh transformer_full 50
sbatch jobs/sweep_optimized.sh dnn_full 50

# Quick comparison
sbatch jobs/sweep_dnn_comparison.sh

# Multi-input model training 🆕
sbatch jobs/model_multi_input_dnn_nersc.sh
sbatch jobs/model_multi_input_transformer_nersc.sh

# HGTD model training 🆕
sbatch jobs/model_hgtd_multi_input_dnn_nersc.sh
sbatch jobs/model_hgtd_only_dnn_nersc.sh
```

## Key Features

### 🚀 **Optimized Parameter Sweeps**
- **Data caching**: Load once, train multiple models
- **Time savings**: 40min×N → 40min + 5min×N  
- **Batch training**: In-memory model training
- **Real-time results**: Automatic analysis and plotting

### 🎯 **Smart Model Architecture**
- **Attention masks**: Exclude padding from computations
- **Smart padding**: Feature-specific values (not zeros)
- **Auto-detection**: Scripts automatically handle both model types
- **Calibration**: Built-in detector time calibration

### 📊 **Enhanced Analysis**
- **Model comparison**: Automatic Transformer vs DNN benchmarking
- **Parameter effects**: Visualize parameter impact on performance  
- **Training efficiency**: Time vs performance analysis

### 🧪 **Physics-Informed Learning** 🆕
- **Baseline Integration**: Leverages existing physics-based methods
- **Residual Learning**: Learns corrections instead of predictions from scratch
- **Domain Knowledge**: Incorporates detector calibration and track matching
- **Robust Performance**: Handles imperfect baseline predictions gracefully

### 🔍 **Baseline Analysis Tools** 🆕
- **Standalone Analysis**: Comprehensive baseline method reconstruction analysis
- **Configurable Matching**: Support for track-matching or jet-matching via YAML
- **Event Investigation**: Detailed analysis of worst-performing events
- **Feature Comparisons**: Best vs worst event feature distribution analysis
- **Multi-level Plotting**: Standard baseline plots + feature comparisons + correlation analysis

### ⚡ **HGTD Timing Detector Models** 🆕
- **HGTD Multi-Input**: Combines LAr calorimeter data with HGTD timing tracks
- **HGTD-Only**: Pure timing-based prediction using only HGTD tracks (no LAr data)
- **High-Precision Timing**: Leverages HGTD track timing features (time, timeRes)
- **Independent Workflows**: HGTD-only model completely separate from LAr pipelines

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_attention_mask` | `true` | Enable attention mask and smart padding |
| `use_spatial_features` | `false` | Include cell position features |
| `use_jet_features` | `false` | Include jet matching features |
| `use_cell_jet_matching` | `false` | Filter cells matched to jets |
| `use_detector_params` | `false` | Apply detector-specific time calibration |
| `use_time_quality_cut` | `false` | Enable time-based quality filtering |
| `use_track_eta_cut` | `false` | Filter tracks by pseudorapidity (multi-input models) |
| `track_eta_cut_value` | `2.5` | Track eta cut threshold: |eta| <= value |
| `calibration_validation` | `false` | Generate calibration validation plots |
| `use_baseline_method_filter` | `false` | Filter events by baseline method performance |
| `baseline_method_threshold` | `500.0` | Baseline error threshold in ps (±500 ps default) |
| `model_architecture` | `"two_stage_dnn"` | Model type: `transformer`, `two_stage_dnn`, `baseline_guided_dnn`, `multi_input_dnn`, `multi_input_transformer`, `hgtd_multi_input_dnn`, `hgtd_only_dnn` |
| `max_jets` | `7` | Maximum number of jets per event (multi-input models) |
| `max_tracks` | `30` | Maximum number of LAr tracks per event (multi-input models) |
| `max_hgtd_tracks` | `30` | Maximum number of HGTD tracks per event (HGTD models) |
| `use_event_jets` | `false` | Enable event-level jet features (multi-input models) |
| `use_event_tracks` | `false` | Enable event-level track features (multi-input models) |
| `use_event_hgtd_tracks` | `false` | Enable event-level HGTD track features (HGTD models) |

## Parameter Sweep Types

| Type | Description | Models | Time |
|------|-------------|--------|------|
| `transformer_quick` | Fast Transformer sweep | Transformer | ~2 hours |
| `dnn_quick` | Fast DNN sweep | DNN | ~2 hours |
| `transformer_full` | Full Transformer optimization | Transformer | ~8 hours |
| `dnn_full` | Full DNN optimization | DNN | ~6 hours |
| `comparison` | Model comparison | Both | ~4 hours |

## Output Structure

Results in `../models/[model_name]/`:
- `model.h5`: Trained model (auto-detected type)
- `config.yaml`: Full configuration  
- `training_history.npz`: Training metrics
- `evaluation_plots/`: Performance visualizations
- `baseline_check_*/`: Traditional method comparison

Parameter sweeps in `results/sweep_YYYYMMDD_HHMMSS/`:
- `results.csv`: All experiment results
- `analysis_summary.txt`: Best results and comparisons
- `analysis/`: Performance plots and model comparisons

Baseline analysis in `../../bad_events_check/baseline_analysis_YYYYMMDD_HHMMSS/`: 🆕
- `analysis_log.txt`: Complete analysis log with worst events details
- `baseline_plots/`: Standard baseline check plots (3 plots)
- `feature_comparison/`: Best vs worst events feature distribution comparisons
- `additional_analysis/`: Correlation analysis plots (error vs event properties)

## Examples

```bash
# Train and compare all model types
python scripts/train.py --config-file config/configs/experiment_with_jets.yaml --model-name transformer_test
python scripts/train.py --config-file config/configs/experiment_dnn.yaml --model-name dnn_test
python scripts/train.py --config-file config/configs/experiment_baseline_guided_track.yaml --model-name baseline_guided_test
python scripts/train.py --config-file config/configs/experiment_dnn_with_jets_tracks.yaml --model-name multi_input_dnn_test
python scripts/train.py --config-file config/configs/experiment_transformer_with_jets_tracks.yaml --model-name multi_input_transformer_test
python scripts/train.py --config-file config/configs/experiment_hgtd_dnn_with_jets_tracks.yaml --model-name hgtd_multi_input_test
python scripts/train.py --config-file config/configs/experiment_hgtd_only.yaml --model-name hgtd_only_test

# Run efficient parameter sweep
python scripts/parameter_sweep.py --config config/configs/experiment_dnn.yaml --grid comparison --max-exp 20

# Evaluate results (auto-detects model type)
python scripts/evaluate.py --model-dir ../models/transformer_test --load-data
python scripts/evaluate.py --model-dir ../models/dnn_test --load-data
python scripts/evaluate.py --model-dir ../models/baseline_guided_test --load-data
python scripts/evaluate.py --model-dir ../models/multi_input_dnn_test --load-data
python scripts/evaluate.py --model-dir ../models/multi_input_transformer_test --load-data
python scripts/evaluate.py --model-dir ../models/hgtd_multi_input_test --load-data
python scripts/evaluate.py --model-dir ../models/hgtd_only_test --load-data

# Analyze baseline reconstruction failures 🆕
cd baseline_analysis
python baseline_analysis.py --top-events 20 --sample-size 2000
```