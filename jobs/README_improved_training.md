# Improved ML Model Training Jobs

This directory contains SLURM job scripts for training the improved ML models with physics-informed features.

## Job Scripts

### 1. `model_improved_nersc.sh` - Fixed Configuration Training
Trains the improved transformer model using the predefined `experiment_improved.yaml` configuration.

**Usage:**
```bash
sbatch jobs/model_improved_nersc.sh
```

**Features:**
- Pre-configured for the improved transformer with physics features
- 12-hour time limit for 50 epochs
- Optimized GPU memory settings
- Automatic model evaluation after training
- Comprehensive error checking and logging

### 2. `train_improved.sh` - Flexible Training Script
More flexible script that allows parameter overrides and different configurations.

**Usage:**
```bash
# Basic usage with default improved config
sbatch jobs/train_improved.sh

# Specify different configuration file
sbatch jobs/train_improved.sh config/configs/experiment_dnn.yaml

# Override training parameters
sbatch jobs/train_improved.sh config/configs/experiment_improved.yaml 30 0.0001 64
#                              ↑config_file                        ↑epochs ↑lr ↑batch_size

# All overrides
sbatch jobs/train_improved.sh config/configs/experiment_improved.yaml 100 0.00005 32
```

**Arguments:**
1. `config_file` (optional): Path to YAML configuration file
   - Default: `config/configs/experiment_improved.yaml`
2. `epochs` (optional): Number of training epochs
3. `learning_rate` (optional): Learning rate override
4. `batch_size` (optional): Batch size override

## Key Improvements

### Enhanced SLURM Configuration
- **Increased time limit**: 12 hours (vs 8 hours) for larger models
- **GPU memory optimization**: Memory growth enabled for physics features
- **Better threading**: Optimized for improved model architecture

### Environment Setup
- **TensorFlow optimization**: GPU memory growth and threading
- **Dependency verification**: Checks for required files and calibration data
- **Configuration validation**: Displays training parameters before execution

### Error Handling & Monitoring
- **Comprehensive logging**: Detailed job information and progress
- **File validation**: Checks for configuration files and scripts
- **Graceful failure**: Proper exit codes and error messages
- **Automatic evaluation**: Runs model evaluation after successful training

## Expected Outputs

### Successful Training
```
✓ Training completed successfully!
✓ Evaluation completed successfully!

Results available at:
  Model: /pscratch/sd/l/liangyu/vertextiming/models/improved_transformer_with_jets
  Plots: /pscratch/sd/l/liangyu/vertextiming/models/improved_transformer_with_jets/plots
```

### Log Files
- **Output log**: `../logs/slurm-improved_training-{JOB_ID}.out`
- **Error log**: `../logs/slurm-improved_training-{JOB_ID}.err`

## Physics-Informed Features

The improved training includes:

### New Features Added (6 per cell)
1. **cell_sigma**: Expected time measurement uncertainty
2. **cell_weight**: Traditional 1/σ² weight for physics-informed pooling
3. **energy_norm_time**: Time normalized by cell energy
4. **log_energy**: Logarithmic energy (physics-motivated)
5. **time_significance**: Time/σ ratio for quality assessment
6. **quality_indicator**: Energy/σ metric for cell quality

### Model Improvements
- **Physics-informed attention**: Uses 1/σ² weights for pooling
- **Larger capacity**: d_model=128, 3 transformer blocks
- **Enhanced training**: Conservative learning rate, gradient clipping
- **Event-level features**: Additional global event characteristics

## Troubleshooting

### Common Issues

1. **Configuration file not found**
   ```bash
   ❌ ERROR: Configuration file not found: config/configs/experiment_improved.yaml
   ```
   Solution: Ensure you're in the project root and the config file exists

2. **Missing calibration data**
   ```bash
   ⚠ WARNING: No calibration data found - physics features may use defaults
   ```
   Solution: Check that `calibration_data/cell_jet_calibration.txt` exists

3. **GPU memory issues**
   ```bash
   ResourceExhaustedError: OOM when allocating tensor
   ```
   Solution: Reduce batch size or use gradient accumulation

4. **Import errors**
   ```bash
   ModuleNotFoundError: No module named 'src.models.improved_transformer'
   ```
   Solution: Ensure all new files are committed and pulled on NERSC

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View live output
tail -f ../logs/slurm-improved_training-{JOB_ID}.out

# Check resource usage
sstat -j {JOB_ID}
```

## Performance Expectations

### Training Time Estimates
- **Improved transformer (50 epochs)**: ~8-10 hours with 43 files
- **DNN model (200 epochs)**: ~4-6 hours with 43 files

### Memory Usage
- **GPU memory**: ~8-12GB for improved transformer
- **RAM**: ~16-24GB for physics feature processing

### Expected Improvements
- Better pred vs truth correlation (physics-informed pooling)
- More stable training (conservative learning rate)
- Enhanced generalization (physics priors reduce overfitting)