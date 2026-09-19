# shellcheck shell=bash
# Environment setup -- source me, do not execute.
#
#   source setup.sh            # activate, syncing only if the venv is missing
#   source setup.sh --sync     # force `uv sync` (after editing pyproject.toml)
#   source setup.sh --cpu      # ignore any GPUs that are present
#
# Everything comes from uv: no module load, no ~/.local packages.  The same
# script covers a laptop, a single-GPU node and a multi-GPU node -- it counts
# the visible GPUs and exports VERTEX_T0_NUM_GPUS, which the training entry
# point turns into the right tf.distribute strategy.

# --- locate the repo -------------------------------------------------------
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _vt_src="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _vt_src="${(%):-%x}"
else
    _vt_src="$0"
fi
VERTEX_T0_ROOT="$(cd "$(dirname "$_vt_src")" && pwd)"
export VERTEX_T0_ROOT
unset _vt_src

_vt_force_sync=0
_vt_force_cpu=0
for _vt_arg in "$@"; do
    case "$_vt_arg" in
        --sync) _vt_force_sync=1 ;;
        --cpu)  _vt_force_cpu=1 ;;
        *) echo "setup.sh: unknown option $_vt_arg (expected --sync or --cpu)" ;;
    esac
done
unset _vt_arg

# --- uv --------------------------------------------------------------------
# NERSC home directories are small; keep the wheel cache on scratch.
: "${UV_CACHE_DIR:=/pscratch/sd/$(whoami | cut -c1)/$(whoami)/.cache/uv}"
[ -d "$(dirname "$UV_CACHE_DIR")" ] || UV_CACHE_DIR="$HOME/.cache/uv"
export UV_CACHE_DIR

if ! command -v uv >/dev/null 2>&1; then
    for _vt_uv in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        [ -x "$_vt_uv" ] && export PATH="$(dirname "$_vt_uv"):$PATH" && break
    done
    unset _vt_uv
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "setup.sh: uv not found -- install it with:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    return 1 2>/dev/null || exit 1
fi

# --- how many GPUs can we see? --------------------------------------------
_vt_count_gpus() {
    if [ "$_vt_force_cpu" -eq 1 ]; then echo 0; return; fi
    # Inside a job SLURM is authoritative; CUDA_VISIBLE_DEVICES wins if set.
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        if [ "$CUDA_VISIBLE_DEVICES" = "NoDevFiles" ] || [ "$CUDA_VISIBLE_DEVICES" = "-1" ]; then
            echo 0; return
        fi
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c . ; return
    fi
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then echo "$SLURM_GPUS_ON_NODE"; return; fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0
        return
    fi
    echo 0
}
VERTEX_T0_NUM_GPUS="$(_vt_count_gpus)"
export VERTEX_T0_NUM_GPUS
unset -f _vt_count_gpus

# --cpu has to hide the devices from TensorFlow, not just from this script.
[ "$_vt_force_cpu" -eq 1 ] && export CUDA_VISIBLE_DEVICES=""

# --- create / sync the environment ----------------------------------------
_vt_extra=()
[ "$VERTEX_T0_NUM_GPUS" -gt 0 ] && _vt_extra=(--extra cuda)

# A venv first built on a CPU machine has no CUDA wheels; landing on a GPU
# node later must top it up, otherwise TensorFlow silently runs on the CPU.
_vt_need_cuda=0
if [ "$VERTEX_T0_NUM_GPUS" -gt 0 ] && [ -d "$VERTEX_T0_ROOT/.venv" ] && \
   ! compgen -G "$VERTEX_T0_ROOT/.venv/lib/python*/site-packages/nvidia/cudnn" >/dev/null; then
    echo "setup.sh: GPU present but the venv has no CUDA wheels -- adding them"
    _vt_need_cuda=1
fi

if [ ! -x "$VERTEX_T0_ROOT/.venv/bin/python" ] || [ "$_vt_force_sync" -eq 1 ] \
        || [ "$_vt_need_cuda" -eq 1 ]; then
    echo "setup.sh: uv sync ${_vt_extra[*]} (this can take a few minutes the first time)"
    ( cd "$VERTEX_T0_ROOT" && uv sync "${_vt_extra[@]}" ) || {
        echo "setup.sh: uv sync failed"
        return 1 2>/dev/null || exit 1
    }
fi
unset _vt_extra _vt_force_sync _vt_force_cpu _vt_need_cuda

# A loaded NERSC ML module cannot shadow the venv's python (PATH order) or its
# packages (PYTHONNOUSERSITE), but it does leave its own CUDA/cuDNN on
# LD_LIBRARY_PATH, which can shadow the wheels TensorFlow ships with.
if echo "${LOADEDMODULES:-}" | tr ':' '\n' | grep -qiE '^(tensorflow|pytorch|jax)'; then
    echo "setup.sh: warning -- a NERSC ML module is loaded ("
    echo "          $(echo "$LOADEDMODULES" | tr ':' '\n' | grep -iE '^(tensorflow|pytorch|jax)' | tr '\n' ' '))"
    echo "          its CUDA libraries stay on LD_LIBRARY_PATH; run 'module purge' first."
fi

# shellcheck disable=SC1091
. "$VERTEX_T0_ROOT/.venv/bin/activate"

# --- runtime settings ------------------------------------------------------
# Never let ~/.local packages shadow the locked environment.
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    _vt_cores="$SLURM_CPUS_PER_TASK"        # inside a job: use what we were given
else
    # Login nodes are shared -- do not spin up a thread per core.
    _vt_cores="$(nproc 2>/dev/null || echo 8)"
    [ "$_vt_cores" -gt 16 ] && _vt_cores=16
fi
if [ "$VERTEX_T0_NUM_GPUS" -gt 0 ]; then
    # The GPU does the work; a handful of threads is enough to feed it.
    export OMP_NUM_THREADS=$(( _vt_cores > 16 ? 16 : _vt_cores ))
    export TF_NUM_INTRAOP_THREADS="$OMP_NUM_THREADS"
    export TF_NUM_INTEROP_THREADS=8
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    [ "$VERTEX_T0_NUM_GPUS" -gt 1 ] && export NCCL_DEBUG=WARN
else
    export OMP_NUM_THREADS="$_vt_cores"
    export TF_NUM_INTRAOP_THREADS="$_vt_cores"
    export TF_NUM_INTEROP_THREADS=2
fi
[ -n "${SLURM_JOB_ID:-}" ] && export SLURM_CPU_BIND=cores

if [ "$VERTEX_T0_NUM_GPUS" -eq 0 ]; then
    _vt_mode="CPU (${_vt_cores} cores)"
elif [ "$VERTEX_T0_NUM_GPUS" -eq 1 ]; then
    _vt_mode="1 GPU"
else
    _vt_mode="${VERTEX_T0_NUM_GPUS} GPUs (MirroredStrategy)"
fi
echo "ML4Vertex_t0: $_vt_mode | python $(python -V 2>&1 | cut -d' ' -f2) | $VERTEX_T0_ROOT"
unset _vt_cores _vt_mode
