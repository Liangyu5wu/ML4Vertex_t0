#!/bin/bash
#SBATCH --job-name=physics_ml
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --output=../logs/slurm-physics-%j.out
#SBATCH --error=../logs/slurm-physics-%j.err

# Usage: sbatch jobs/train_physics.sh [config_file] [epochs]
# Examples:
#   sbatch jobs/train_physics.sh config/configs/experiment_dnn_improved.yaml
#   sbatch jobs/train_physics.sh config/configs/experiment_improved.yaml 30

CONFIG_FILE=${1:-"config/configs/experiment_improved.yaml"}
EPOCHS=${2:-""}

echo "=========================================="
echo "PHYSICS-INFORMED ML TRAINING"
echo "Config: $CONFIG_FILE"
echo "Epochs: ${EPOCHS:-'(from config)'}"
echo "Start: $(date)"
echo "=========================================="

cd /pscratch/sd/l/liangyu/vertextiming/ML4Vertex_t0
source setup.sh
mkdir -p ../logs

# Load modules
module load craype tensorflow/2.12.0

# Environment
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# GPU test
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print(f'TF: {tf.__version__}, GPUs: {len(gpus)}')
"

# Verify files
[ -f "$CONFIG_FILE" ] || { echo "Config not found: $CONFIG_FILE"; exit 1; }
[ -f "scripts/train_physics.py" ] || { echo "Training script not found"; exit 1; }

# Train
CMD="python scripts/train_physics.py --config-file $CONFIG_FILE"
[ -n "$EPOCHS" ] && CMD="$CMD --epochs $EPOCHS"

echo "Command: $CMD"
eval $CMD

if [ $? -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "❌ Training failed!"
    exit 1
fi

echo "Completed: $(date)"