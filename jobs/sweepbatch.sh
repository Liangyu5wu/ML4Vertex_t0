#!/bin/bash
#
#SBATCH --account=atlas:default
#SBATCH --partition=roma
#SBATCH --job-name=test_job
#SBATCH --output=output_test-%j.txt
#SBATCH --error=error_test-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20g
#SBATCH --time=10:00:00

cd /sdf/data/atlas/u/liangyu/vertextiming/Vertex0/ml_algr
source setup.sh

python scripts/parameter_sweep.py \
    --base-config config/configs/experiment2_fast.yaml \
    --grid-type training \
    --max-experiments 200