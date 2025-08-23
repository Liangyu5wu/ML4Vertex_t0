#!/bin/bash
#SBATCH --job-name=opt_sweep
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --output=../logs/sweep/slurm-opt_sweep-%j.out
#SBATCH --error=../logs/sweep/slurm-opt_sweep-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liangyu5@stanford.edu

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

cd /pscratch/sd/l/liangyu/vertextiming/ML4Vertex_t0
source setup.sh

mkdir -p ../logs/sweep

echo "Loading NERSC modules..."
module load craype
module load tensorflow/2.12.0

# Set environment variables
export SLURM_CPU_BIND="cores"
export NUMEXPR_MAX_THREADS=128
export CUDA_VISIBLE_DEVICES=0
export TF_NUM_INTEROP_THREADS=8
export TF_NUM_INTRAOP_THREADS=16
export OMP_NUM_THREADS=16

# Verify GPU access
echo "Checking GPU access..."
nvidia-smi || echo "WARNING: nvidia-smi failed, but continuing..."

# Clean up conflicting tensorflow installations
echo "Cleaning up any pip-installed tensorflow..."
pip uninstall --user -y tensorflow tensorflow-gpu tensorflow-cpu 2>/dev/null || true

# Test TensorFlow
echo "Testing TensorFlow functionality..."
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')
" || echo "WARNING: TensorFlow test failed, but continuing..."

# Configure sweep parameters
GRID_TYPE=${1:-"transformer_quick"}  # Default to transformer_quick
MAX_EXP=${2:-50}                     # Default to 50 experiments
CONFIG_FILE="config/configs/experiment_with_jets.yaml"

echo "=========================================="
echo "OPTIMIZED PARAMETER SWEEP"
echo "=========================================="
echo "Grid type: $GRID_TYPE"
echo "Max experiments: $MAX_EXP"
echo "Config file: $CONFIG_FILE"
echo "Time started: $(date)"

# Run optimized parameter sweep
python scripts/parameter_sweep.py \
    --config "$CONFIG_FILE" \
    --grid "$GRID_TYPE" \
    --max-exp "$MAX_EXP"

# Check if sweep completed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "PARAMETER SWEEP COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    
    # Find the most recent results directory
    RESULTS_DIR=$(find results/ -name "sweep_*" -type d | sort | tail -1)
    
    if [ -n "$RESULTS_DIR" ] && [ -d "$RESULTS_DIR" ]; then
        echo "Results directory: $RESULTS_DIR"
        
        # Show quick summary
        if [ -f "$RESULTS_DIR/results.csv" ]; then
            echo "Quick summary:"
            python -c "
import pandas as pd
df = pd.read_csv('$RESULTS_DIR/results.csv')
successful = df[df['status'] == 'success']
print(f'  Total experiments: {len(df)}')
print(f'  Successful: {len(successful)}')
if len(successful) > 0:
    best = successful.loc[successful['best_val_loss'].idxmin()]
    print(f'  Best loss: {best[\"best_val_loss\"]:.4f}')
    if 'model_type' in best:
        print(f'  Best model: {best[\"model_type\"]}')
"
        fi
        
        # Run automatic analysis
        echo "Running automatic analysis..."
        python scripts/analyze_sweep.py "$RESULTS_DIR"
        
        if [ $? -eq 0 ]; then
            echo "Analysis completed successfully!"
            echo "Check the following files for results:"
            echo "  - $RESULTS_DIR/results.csv"
            echo "  - $RESULTS_DIR/analysis_summary.txt"
            echo "  - $RESULTS_DIR/analysis/*.png"
        else
            echo "Analysis failed, but results are available in: $RESULTS_DIR"
        fi
    else
        echo "WARNING: Could not find results directory for analysis"
        echo "Results should be in: results/sweep_YYYYMMDD_HHMMSS/"
    fi
    
else
    echo "=========================================="
    echo "PARAMETER SWEEP FAILED"
    echo "=========================================="
    echo "Check the error messages above for details"
    echo "Partial results may be available in: results/"
fi

echo "=========================================="
echo "Job completed!"
echo "End time: $(date)"
echo "=========================================="
