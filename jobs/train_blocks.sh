#!/bin/bash
# Train a block-config model on Perlmutter.
#
#   sbatch jobs/train_blocks.sh config/blocks/hgtd_multi_input.yaml
#   sbatch jobs/train_blocks.sh config/blocks/hgtd_multi_input.yaml "--datasets ttbar"
#   sbatch --gpus-per-node=4 jobs/train_blocks.sh <config> ""   # MirroredStrategy
#
# $1 config path, $2 extra arguments passed to train_blocks.py.
# setup.sh picks the device layout up from SLURM, so nothing here needs to
# change between a CPU run, one GPU and a whole node.
#SBATCH --job-name=vtx_blocks
#SBATCH --account=m2616_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -c 32                  # gpu_shared_ss11 requires 32 cores per GPU
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=../logs/slurm-blocks-%j.out
#SBATCH --error=../logs/slurm-blocks-%j.err

set -euo pipefail

CONFIG=${1:?usage: sbatch jobs/train_blocks.sh <config.yaml> [extra args]}
EXTRA=${2:-}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
mkdir -p ../logs

echo "=========================================="
echo "Job ID   : ${SLURM_JOB_ID:-interactive}"
echo "Node     : ${SLURM_NODELIST:-$(hostname)}"
echo "Config   : $CONFIG"
echo "Extra    : $EXTRA"
echo "Start    : $(date)"
echo "=========================================="

# shellcheck disable=SC1091
source setup.sh

python -c "import tensorflow as tf; print('TF', tf.__version__, '| GPUs', len(tf.config.list_physical_devices('GPU')))"

# shellcheck disable=SC2086
python scripts/train_blocks.py --config "$CONFIG" $EXTRA

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
