# Vertex Time Prediction Models

A framework for training Transformer, DNN, and Baseline-Guided models for vertex t0 prediction with LAr Calorimeter in ATLAS. Features optimized parameter sweeps, attention mask support, and physics-informed residual learning.

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
│       ├── experiment_nersc.yaml      # NERSC cluster configuration
│       └── test_fast.yaml     # Fast testing configuration
├── calibration_data/          # External calibration data files
│   ├── HStrackmatching_calibration.txt    # Cell-track matching calibration
│   └── cell_jet_calibration.txt           # Cell-jet matching calibration 
├── src/                       # Source code
│   ├── __init__.py            # Source package initialization
│   ├── data/                  # Data loading and processing
│   ├── models/                # Model architectures (Transformer + DNN + Baseline-Guided)
│   ├── training/              # Training utilities
│   └── evaluation/            # Evaluation and visualization
├── scripts/                   # Main execution scripts
│   ├── train.py              # Training script (supports all models)
│   ├── evaluate.py           # Evaluation script (auto-detects model type)
│   ├── parameter_sweep.py    # Optimized parameter sweeps with data caching
│   └── analyze_sweep.py      # Simplified results analysis
├── jobs/                    # SLURM job submission scripts
│   ├── sweep_optimized.sh   # Efficient parameter sweeps
│   ├── sweep_dnn_comparison.sh  # DNN vs Transformer comparison
│   └── sweep_quick_test.sh  # Quick testing (2 hours)
└── process_h5.py            # Data preprocessing utility
```

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.9
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Model Architectures

### Transformer Model
```
Variable cells → Attention Mask → Transformer Blocks → Global Pooling → Dense → Vertex Time
```

### Two-Stage DNN Model
```
Cells → Cell-level MLP → Masked Attention Pooling → Event-level MLP → Vertex Time
```
- Learns attention weights (vs fixed sigma weights in traditional methods)
- Masked attention pooling ignores padding cells
- Smart padding uses feature-specific values

### Baseline-Guided DNN Model 🆕
```
Cells → Cell-level MLP → Global Average Pooling → Combine with Baseline → Residual Learning → Vertex Time
                                                        ↑
                                               Baseline Predictions (External)
```
- **Physics-Informed Design**: Leverages existing baseline method predictions
- **Residual Learning**: Learns corrections to baseline predictions (`target = baseline + residual`)
- **Three-Input Architecture**: Cell sequences + Vertex features + Baseline predictions
- **Simplified Processing**: Global average pooling instead of complex attention
- **Robust Performance**: Works even with imperfect baseline predictions

## Quick Start

### Training Models

```bash
# Train Transformer model
python scripts/train.py --config-file config/configs/experiment_with_jets.yaml

# Train DNN model  
python scripts/train.py --config-file config/configs/experiment_dnn.yaml

# Train Baseline-Guided DNN model 🆕
python scripts/train.py --config-file config/configs/experiment_baseline_guided_track.yaml

# Override parameters
python scripts/train.py --config-file config/configs/experiment_dnn.yaml --epochs 50 --learning-rate 5e-4
```

### Evaluation (Auto-detects Model Type)

```bash
# Evaluate any model - automatically detects Transformer/DNN/Baseline-Guided
python scripts/evaluate.py --model-dir ../models/your_model --load-data

# Evaluate baseline-guided model 🆕
python scripts/evaluate.py --model-dir ../models/baseline_guided_dnn_with_tracks --load-data
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

### SLURM Job Submission

```bash
# Efficient sweeps (10 hours → 80-100 experiments)
sbatch jobs/sweep_optimized.sh transformer_full 50
sbatch jobs/sweep_optimized.sh dnn_full 50

# Quick comparison
sbatch jobs/sweep_dnn_comparison.sh
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

## Configuration

### DNN Configuration
```yaml
model_name: "dnn_with_jets"
model_architecture: "two_stage_dnn"

# Cell-level processing
cell_encoder_units: [64, 32]
cell_dropout_rate: 0.2

# Attention pooling
use_attention_pooling: true
attention_hidden_units: 32

# Event-level processing
event_encoder_units: [256, 128, 64]
event_dropout_rates: [0.3, 0.2, 0.1]

# Features and filtering
use_jet_features: true
use_cell_jet_matching: true
calibration_data_file: "cell_jet_calibration.txt"
```

### Baseline-Guided DNN Configuration 🆕
```yaml
model_name: "baseline_guided_dnn_with_tracks"
model_architecture: "baseline_guided_dnn"
loss_function: "huber"
huber_delta: 100.0

# Simplified network architecture (3-4 dense layers)
cell_encoder_units: [128, 64, 32]
cell_dropout_rate: 0.1
cell_activation: "relu"

# No attention pooling (simple average pooling)
use_attention_pooling: false

# Event-level processing parameters (simplified)
event_encoder_units: [128, 64, 32, 16]
event_dropout_rates: [0.2, 0.2, 0.1, 0.1]
use_batch_norm: true

# Features and filtering (optimized for baseline guidance)
use_spatial_features: false         # Spatial position features
use_track_features: true            # Track matching features  
use_jet_features: false             # Jet matching features
use_physics_informed_features: false # Baseline handles physics weighting

# Baseline method filtering parameters
use_baseline_method_filter: true    # Enable baseline method performance filtering
baseline_method_threshold: 500.0    # Baseline error threshold in ps
```

### Transformer Configuration
```yaml
model_name: "transformer_with_jets"

# Architecture
d_model: 128
num_heads: 8
num_transformer_blocks: 2

# Features and filtering
use_attention_mask: true
use_jet_features: true
use_cell_jet_matching: true
calibration_data_file: "cell_jet_calibration.txt"
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_attention_mask` | `true` | Enable attention mask and smart padding |
| `use_spatial_features` | `false` | Include cell position features |
| `use_jet_features` | `false` | Include jet matching features |
| `use_cell_jet_matching` | `false` | Filter cells matched to jets |
| `use_detector_params` | `false` | Apply detector-specific time calibration |
| `calibration_validation` | `false` | Generate calibration validation plots |
| `use_baseline_method_filter` | `false` | Filter events by baseline method performance |
| `baseline_method_threshold` | `500.0` | Baseline error threshold in ps (±500 ps default) |
| `model_architecture` | `"two_stage_dnn"` | Model type: `transformer`, `two_stage_dnn`, `baseline_guided_dnn` |

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

## Performance Benchmarks

### Attention Mask Benefits
- **RMSE improvement**: ~19% better than traditional padding
- **Training overhead**: ~5% slower, much better accuracy
- **Smart padding**: Avoids semantic issues with zero padding

### Optimized Sweeps
- **Traditional**: 40min × 20 exp = 800min (13.3 hours) ❌
- **Optimized**: 40min + 5min × 20 exp = 140min (2.3 hours) ✅

## Examples

```bash
# Train and compare all three models
python scripts/train.py --config-file config/configs/experiment_with_jets.yaml --model-name transformer_test
python scripts/train.py --config-file config/configs/experiment_dnn.yaml --model-name dnn_test
python scripts/train.py --config-file config/configs/experiment_baseline_guided_track.yaml --model-name baseline_guided_test

# Run efficient parameter sweep  
python scripts/parameter_sweep.py --config config/configs/experiment_dnn.yaml --grid comparison --max-exp 20

# Evaluate results (auto-detects model type)
python scripts/evaluate.py --model-dir ../models/transformer_test --load-data
python scripts/evaluate.py --model-dir ../models/dnn_test --load-data
python scripts/evaluate.py --model-dir ../models/baseline_guided_test --load-data
```
