#!/bin/bash
#SBATCH --job-name=improved_ml_training
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -c 32         # Required: NERSC gpu_shared_ss11 queue mandates 32 cores per GPU
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=../logs/slurm-improved_training-%j.out
#SBATCH --error=../logs/slurm-improved_training-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liangyu5@stanford.edu

# Usage: sbatch jobs/train_improved.sh [config_file] [epochs] [learning_rate] [batch_size]
# Examples:
#   sbatch jobs/train_improved.sh
#   sbatch jobs/train_improved.sh config/configs/experiment_improved.yaml
#   sbatch jobs/train_improved.sh config/configs/experiment_improved.yaml 30
#   sbatch jobs/train_improved.sh config/configs/experiment_improved.yaml 30 0.0001 64

# Parse command line arguments
CONFIG_FILE=${1:-"config/configs/experiment_improved.yaml"}
EPOCHS=${2:-""}
LEARNING_RATE=${3:-""}
BATCH_SIZE=${4:-""}

# Print job information
echo "=========================================="
echo "IMPROVED ML MODEL TRAINING JOB"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"  
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""
echo "Training parameters:"
echo "  Config file: $CONFIG_FILE"
echo "  Epochs override: ${EPOCHS:-'(from config)'}"
echo "  Learning rate override: ${LEARNING_RATE:-'(from config)'}"
echo "  Batch size override: ${BATCH_SIZE:-'(from config)'}"
echo "=========================================="

# Change to project directory
cd /pscratch/sd/l/liangyu/vertextiming/ML4Vertex_t0
source setup.sh

# Create logs directory
mkdir -p ../logs

echo "Loading NERSC modules..."
module load craype
module load tensorflow/2.12.0

# Set optimized environment variables
export SLURM_CPU_BIND="cores"
export NUMEXPR_MAX_THREADS=128
export CUDA_VISIBLE_DEVICES=0

# Optimized threading for improved model
export TF_NUM_INTEROP_THREADS=8
export TF_NUM_INTRAOP_THREADS=16
export OMP_NUM_THREADS=16

# GPU memory optimization
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private

# Verify GPU access
echo "Checking GPU access..."
nvidia-smi || {
    echo "WARNING: nvidia-smi failed, but continuing..."
}

# Clean up conflicting installations
echo "Cleaning up any pip-installed tensorflow..."
pip uninstall --user -y tensorflow tensorflow-gpu tensorflow-cpu 2>/dev/null || true

# Environment verification
echo "Environment verification:"
echo "  Python version: $(python --version)"
echo "  Python location: $(which python)"

# TensorFlow GPU test
echo "Testing TensorFlow and GPU setup..."
python -c "
import tensorflow as tf
import os
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU devices: {gpus}')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('✓ GPU memory growth enabled')
    except RuntimeError as e:
        print(f'⚠ Memory growth setting failed: {e}')
else:
    print('⚠ No GPUs detected')
print(f'CUDA available: {tf.test.is_built_with_cuda()}')
print(f'Built with XLA: {tf.test.is_built_with_xla()}')
" || {
    echo "WARNING: TensorFlow test failed, but continuing..."
}

# Verify required files
echo ""
echo "Verifying required files..."
if [ -f "$CONFIG_FILE" ]; then
    echo "✓ Configuration file found: $CONFIG_FILE"
else
    echo "❌ ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

if [ -f "scripts/train_improved.py" ]; then
    echo "✓ Improved training script found"
else
    echo "❌ ERROR: Improved training script not found: scripts/train_improved.py"
    exit 1
fi

# Check calibration data
if [ -f "calibration_data/cell_jet_calibration.txt" ]; then
    echo "✓ Calibration data found"
elif [ -f "calibration_data/HStrackmatching_calibration.txt" ]; then
    echo "✓ Track calibration data found"
else
    echo "⚠ WARNING: No calibration data found - physics features may use defaults"
fi

# Display configuration details
echo ""
echo "Configuration details:"
echo "----------------------"
python -c "
import yaml
import os
config_file = '$CONFIG_FILE'
try:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f'Model name: {config.get(\"model_name\", \"N/A\")}')
    print(f'Architecture: {config.get(\"model_architecture\", \"N/A\")}')
    print(f'Data directory: {config.get(\"data_dir\", \"N/A\")}')
    print(f'Number of files: {config.get(\"num_files\", \"N/A\")}')
    print(f'Max cells: {config.get(\"max_cells\", \"N/A\")}')
    print(f'Physics features: {config.get(\"use_physics_informed_features\", \"N/A\")}')
    print(f'Use attention mask: {config.get(\"use_attention_mask\", \"N/A\")}')
    print(f'Loss function: {config.get(\"loss_function\", \"N/A\")}')
    
    # Model-specific parameters
    if config.get('model_architecture') == 'two_stage_dnn':
        print(f'Cell encoder units: {config.get(\"cell_encoder_units\", \"N/A\")}')
        print(f'Event encoder units: {config.get(\"event_encoder_units\", \"N/A\")}')
        print(f'Use attention pooling: {config.get(\"use_attention_pooling\", \"N/A\")}')
    else:  # Transformer
        print(f'd_model: {config.get(\"d_model\", \"N/A\")}')
        print(f'Number of heads: {config.get(\"num_heads\", \"N/A\")}')
        print(f'Transformer blocks: {config.get(\"num_transformer_blocks\", \"N/A\")}')
        print(f'DFF: {config.get(\"dff\", \"N/A\")}')
    
    # Training parameters  
    print(f'Epochs: {config.get(\"epochs\", \"N/A\")}')
    print(f'Batch size: {config.get(\"batch_size\", \"N/A\")}')
    print(f'Learning rate: {config.get(\"learning_rate\", \"N/A\")}')
    print(f'Early stopping patience: {config.get(\"early_stopping_patience\", \"N/A\")}')
    
except Exception as e:
    print(f'Could not parse config: {e}')
"

# Build training command
TRAIN_CMD="python scripts/train_improved.py --config-file $CONFIG_FILE"

if [ -n "$EPOCHS" ]; then
    TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
fi

if [ -n "$LEARNING_RATE" ]; then
    TRAIN_CMD="$TRAIN_CMD --learning-rate $LEARNING_RATE"
fi

if [ -n "$BATCH_SIZE" ]; then
    TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
fi

echo ""
echo "=========================================="
echo "STARTING IMPROVED MODEL TRAINING"
echo "=========================================="
echo "Command: $TRAIN_CMD"
echo "Time started: $(date)"
echo ""

# Execute training
eval $TRAIN_CMD

# Check training result
TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ TRAINING COMPLETED SUCCESSFULLY!"
    
    # Extract model name for evaluation
    MODEL_NAME=$(python -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print(config.get('model_name', 'improved_model'))
except:
    print('improved_model')
")
    
    echo ""
    echo "Starting model evaluation..."
    eval_cmd="python scripts/evaluate.py --model-dir /pscratch/sd/l/liangyu/vertextiming/models/$MODEL_NAME --load-data"
    echo "Evaluation command: $eval_cmd"
    
    eval $eval_cmd
    
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully!"
    else
        echo "⚠ WARNING: Evaluation failed, but training was successful"
    fi
    
    echo ""
    echo "Results available at:"
    echo "  Model: /pscratch/sd/l/liangyu/vertextiming/models/$MODEL_NAME"
    echo "  Plots: /pscratch/sd/l/liangyu/vertextiming/models/$MODEL_NAME/plots"
    
else
    echo "❌ TRAINING FAILED!"
    echo "Check the error logs for details"
fi

echo ""
echo "Job completed at: $(date)"
echo "Log files:"
echo "  Output: ../logs/slurm-improved_training-${SLURM_JOB_ID}.out"
echo "  Error: ../logs/slurm-improved_training-${SLURM_JOB_ID}.err"
echo "=========================================="

exit $TRAIN_EXIT_CODE