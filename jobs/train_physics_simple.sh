#!/bin/bash
#SBATCH --job-name=physics_ml
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --output=../logs/slurm-physics-%j.out
#SBATCH --error=../logs/slurm-physics-%j.err

# Usage: sbatch jobs/train_physics_simple.sh [config_file]
CONFIG_FILE=${1:-"config/configs/experiment_with_jets.yaml"}

echo "Training with physics features: $CONFIG_FILE"
echo "Start: $(date)"

cd /pscratch/sd/l/liangyu/vertextiming/ML4Vertex_t0
source setup.sh

# Environment
module load tensorflow/2.12.0
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Train
python scripts/train.py --config-file $CONFIG_FILE

echo "Completed: $(date)"