#!/bin/bash
#SBATCH --job-name=sweep_dnn
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=../logs/sweep/slurm-sweep_dnn-%j.out
#SBATCH --error=../logs/sweep/slurm-sweep_dnn-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liangyu5@stanford.edu

# Print job information
echo "=========================================="
echo "DNN PARAMETER SWEEP WITH PHYSICS FEATURES"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

cd /pscratch/sd/l/liangyu/vertextiming/ML4Vertex_t0_mbp_local
source setup.sh
mkdir -p ../logs/sweep

# Load modules and setup environment
module load craype tensorflow/2.12.0
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# GPU check
nvidia-smi || echo "WARNING: nvidia-smi failed"

# Run DNN parameter sweep
echo "Starting DNN parameter sweep..."
python scripts/parameter_sweep.py \
    --config config/configs/experiment_dnn.yaml \
    --grid dnn_quick \
    --max-exp 20

# Analyze results
if [ $? -eq 0 ]; then
    echo "Sweep completed! Running analysis..."
    RESULTS_DIR=$(find results/ -name "parameter_sweep_*" -type d | sort | tail -1)
    if [ -n "$RESULTS_DIR" ]; then
        python scripts/analyze_sweep.py "$RESULTS_DIR"
        echo "Results available in: $RESULTS_DIR"
    fi
else
    echo "Sweep failed!"
fi

echo "Completed: $(date)"