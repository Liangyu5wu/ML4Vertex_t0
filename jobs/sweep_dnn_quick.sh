#!/bin/bash
#SBATCH --job-name=sweep_quick
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -c 32         # Required: NERSC gpu_shared_ss11 queue mandates 32 cores per GPU
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=../logs/sweep/slurm-sweep_quick-%j.out
#SBATCH --error=../logs/sweep/slurm-sweep_quick-%j.err
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
# Utilize available cores while keeping GPU workload optimal
export TF_NUM_INTEROP_THREADS=8
export TF_NUM_INTRAOP_THREADS=16
export OMP_NUM_THREADS=16

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

# Test TensorFlow functionality
echo "Testing TensorFlow functionality..."
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')
print(f'CUDA available: {tf.test.is_built_with_cuda()}')
" || {
    echo "WARNING: TensorFlow test failed, but continuing..."
}

# Start parameter sweep
echo "=========================================="
echo "STARTING QUICK PARAMETER SWEEP"
echo "=========================================="
echo "Sweep type: quick"
echo "Base config: config/configs/experiment_nersc.yaml"
echo "Expected experiments: ~24 combinations"
echo "Expected time: ~2-4 hours"
echo "Time started: $(date)"

# Run the parameter sweep
python scripts/parameter_sweep.py \
    --config config/configs/experiment_dnn.yaml \
    --grid dnn_quick

# Check if sweep completed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "PARAMETER SWEEP COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    
    # Find the most recent results directory
    RESULTS_DIR=$(find results/ -name "parameter_sweep_*" -type d | sort | tail -1)
    
    if [ -n "$RESULTS_DIR" ] && [ -d "$RESULTS_DIR" ]; then
        echo "Results directory: $RESULTS_DIR"
        
        # Run analysis on the results
        echo "Running automatic analysis..."
        python scripts/analyze_sweep.py "$RESULTS_DIR"
        
        if [ $? -eq 0 ]; then
            echo "Analysis completed successfully!"
            echo "Check the following files for results:"
            echo "  - $RESULTS_DIR/results.csv"
            echo "  - $RESULTS_DIR/summary_report.txt" 
            echo "  - $RESULTS_DIR/analysis_plots/"
            
            # Print quick summary
            echo ""
            echo "QUICK RESULTS SUMMARY:"
            echo "====================="
            
            # Extract best result from CSV if available
            if [ -f "$RESULTS_DIR/results.csv" ]; then
                echo "Best experiments (by validation loss):"
                python -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$RESULTS_DIR/results.csv')
    successful = df[df['status'] == 'success']
    if len(successful) > 0:
        best_3 = successful.nsmallest(3, 'best_val_loss')
        for i, (_, row) in enumerate(best_3.iterrows()):
            print(f'{i+1}. {row[\"experiment_id\"]} - Val Loss: {row[\"best_val_loss\"]:.6f}')
            print(f'   d_model={row.get(\"d_model\", \"N/A\")}, num_heads={row.get(\"num_heads\", \"N/A\")}, lr={row.get(\"learning_rate\", \"N/A\")}')
        print(f'Success rate: {len(successful)}/{len(df)} ({len(successful)/len(df)*100:.1f}%)')
    else:
        print('No successful experiments found.')
except Exception as e:
    print(f'Error reading results: {e}')
"
            else
                echo "Results CSV not found"
            fi
            
        else
            echo "Analysis failed, but results are available in: $RESULTS_DIR"
        fi
    else
        echo "WARNING: Could not find results directory for analysis"
        echo "Results should be in: results/parameter_sweep_YYYYMMDD_HHMMSS/"
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
