#!/bin/bash
# GPU environment setup script for NERSC Perlmutter
# Usage: source gpu_setup.sh

# 1. Activate virtual environment
source setup.sh

# 2. Load required modules
module load craype
module load tensorflow/2.12.0

# 3. Set environment variables for GPU and threading
export SLURM_CPU_BIND="cores"
export NUMEXPR_MAX_THREADS=128
export CUDA_VISIBLE_DEVICES=0
export TF_NUM_INTEROP_THREADS=8
export TF_NUM_INTRAOP_THREADS=16
export OMP_NUM_THREADS=16

# 4. Verify GPU access
echo "Checking GPU access..."
nvidia-smi || {
    echo "WARNING: nvidia-smi failed, but continuing..."
}

# 5. Set Python to unbuffered mode for better logging
export PYTHONUNBUFFERED=1

echo "GPU environment setup complete!"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
