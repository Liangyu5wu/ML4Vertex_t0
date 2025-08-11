#!/bin/bash
#SBATCH --job-name=transformer_with_jets
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -c 32         # Required: NERSC gpu_shared_ss11 queue mandates 32 cores per GPU
#SBATCH --gpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=../logs/slurm-transformer_with_jets-%j.out
#SBATCH --error=../logs/slurm-transformer_with_jets-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liangyu5@stanford.edu


cd /pscratch/sd/l/liangyu/vertextiming/ML4Vertex_t0
source setup.sh

python scripts/train.py --config-file config/configs/experiment_nersc.yaml
