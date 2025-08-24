#!/bin/bash
#SBATCH --job-name=improved_transformer_physics
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -c 32         # Required: NERSC gpu_shared_ss11 queue mandates 32 cores per GPU
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00  # Increased time for improved model training (50 epochs)
#SBATCH --output=../logs/slurm-improved_transformer-%j.out
#SBATCH --error=../logs/slurm-improved_transformer-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liangyu5@stanford.edu


# Print job information
echo "=========================================="
echo "IMPROVED TRANSFORMER MODEL TRAINING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"  
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Physics-informed features: ENABLED"
echo "Model improvements: Enhanced attention, larger capacity"
echo "=========================================="


cd /pscratch/sd/l/liangyu/vertextiming/ML4Vertex_t0
source setup.sh

mkdir -p ../logs

echo "Loading NERSC modules..."
module load craype
module load tensorflow/2.12.0

# Set optimized environment variables for improved model
export SLURM_CPU_BIND="cores"
export NUMEXPR_MAX_THREADS=128
export CUDA_VISIBLE_DEVICES=0

# Optimized for larger model and physics features
export TF_NUM_INTEROP_THREADS=8
export TF_NUM_INTRAOP_THREADS=16
export OMP_NUM_THREADS=16

# Memory optimization for physics features
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private

# Verify GPU access
echo "Checking GPU access..."
nvidia-smi || {
    echo "WARNING: nvidia-smi failed, but continuing..."
}

# Remove any conflicting tensorflow installations
echo "Cleaning up any pip-installed tensorflow..."
pip uninstall --user -y tensorflow tensorflow-gpu tensorflow-cpu 2>/dev/null || true

# Verify environment
echo "Python version: $(python --version)"
echo "Python location: $(which python)"
echo "Pip user directory: $(python -m site --user-base)"

# Test TensorFlow functionality with GPU memory growth
echo "Testing TensorFlow functionality..."
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU devices: {gpus}')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('GPU memory growth enabled')
    except RuntimeError as e:
        print(f'Memory growth setting failed: {e}')
print(f'CUDA available: {tf.test.is_built_with_cuda()}')
" || {
    echo "WARNING: TensorFlow test failed, but continuing..."
}

# Verify calibration data exists
echo "Checking calibration data..."
if [ -f "calibration_data/cell_jet_calibration.txt" ]; then
    echo "✓ Calibration data found: calibration_data/cell_jet_calibration.txt"
else
    echo "⚠ WARNING: Calibration data not found - physics features may use fallback values"
fi

# Verify improved configuration exists
if [ -f "config/configs/experiment_improved.yaml" ]; then
    echo "✓ Improved configuration found: config/configs/experiment_improved.yaml"
else
    echo "❌ ERROR: Improved configuration not found!"
    exit 1
fi

# Display configuration summary
echo "Configuration summary:"
echo "----------------------"
python -c "
import yaml
try:
    with open('config/configs/experiment_improved.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f'Model: {config.get(\"model_name\", \"N/A\")}')
    print(f'Epochs: {config.get(\"epochs\", \"N/A\")}')
    print(f'Learning rate: {config.get(\"learning_rate\", \"N/A\")}')
    print(f'Batch size: {config.get(\"batch_size\", \"N/A\")}')
    print(f'Physics features: {config.get(\"use_physics_informed_features\", \"N/A\")}')
    print(f'd_model: {config.get(\"d_model\", \"N/A\")}')
    print(f'Transformer blocks: {config.get(\"num_transformer_blocks\", \"N/A\")}')
except Exception as e:
    print(f'Could not read config: {e}')
"

# Run the improved training pipeline
echo ""
echo "=========================================="
echo "STARTING IMPROVED MODEL TRAINING"
echo "=========================================="
echo "Time started: $(date)"

# Training with improved script and configuration
python scripts/train_improved.py --config-file config/configs/experiment_improved.yaml --verbose 1

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "✓ Training completed successfully!"
    
    # Evaluate the improved model
    echo ""
    echo "Starting model evaluation..."
    python scripts/evaluate.py --model-dir /pscratch/sd/l/liangyu/vertextiming/models/improved_transformer_with_jets --load-data
    
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully!"
    else
        echo "⚠ WARNING: Evaluation failed, but training was successful"
    fi
    
else
    echo "❌ ERROR: Training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "IMPROVED MODEL JOB COMPLETED!"
echo "=========================================="
echo "End time: $(date)"
echo "Model location: /pscratch/sd/l/liangyu/vertextiming/models/improved_transformer_with_jets"
echo "Plots location: /pscratch/sd/l/liangyu/vertextiming/models/improved_transformer_with_jets/plots"
echo "Logs location: ../logs/slurm-improved_transformer-${SLURM_JOB_ID}.out"
echo "=========================================="