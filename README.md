# ML4Vertex_t0

Hard-scatter vertex time (t0) regression from ATLAS LAr calorimeter cells and
HGTD track timing.

## Data chain

```
ROOT  --R2H5-->  raw h5  --src/pipeline/compact.py-->  compact store  -->  training
                (dense,                               (ragged columns,
                 1 event = 1 row of 1000 cell slots)   ~12x smaller, lossless)
```

R2H5 (separate repo) does the physics-level conversion. Everything from the
raw h5 onwards lives here. Compact stores are kept on CFS:
`/global/cfs/cdirs/m2616/liangyu/vertextiming/compact/{ttbar,vbf_hinv}`.

## Architecture

The model's inputs are declared in YAML as *blocks* (cells, jets, tracks, HGTD
tracks). The same specs drive loading, selection, normalization, padding and
the network, so a new input or a new sample is a config change:

| module | role |
|---|---|
| `src/pipeline/compact.py` | raw h5 → compact store (schema discovered from the file) |
| `src/pipeline/store.py` | ragged read access, vectorised gathers |
| `src/pipeline/blocks.py` | block specs, presets, selection/sorting |
| `src/pipeline/assemble.py` | multi-sample split, normalization, padding, `tf.data` |
| `src/models/block_model.py` | builds the Keras model from the same specs |
| `src/models/layers.py` | masked pooling and transformer block (Keras 3) |
| `src/runtime.py` | CPU / 1 GPU / multi-GPU strategy |
| `scripts/train_blocks.py` | training entry point |
| `scripts/evaluate_blocks.py` | scoring, including on a sample the model never saw |

Config reference: [`src/pipeline/README.md`](src/pipeline/README.md).

## Quickstart

```bash
source setup.sh                    # uv env; detects CPU / GPU / multi-GPU
source setup.sh --sync             # after editing pyproject.toml
source setup.sh --cpu              # force CPU

# raw -> compact (once per sample)
python -m src.pipeline.compact --input-dir ../Vertex_timing_HGTD_w_LAr \
    --output-dir /global/cfs/cdirs/m2616/liangyu/vertextiming/compact/ttbar --sample ttbar

# train (datasets listed in the config; --datasets picks a subset)
python scripts/train_blocks.py --config config/blocks/hgtd_multi_input.yaml
python scripts/train_blocks.py --config config/blocks/hgtd_multi_input.yaml --datasets ttbar

# score an existing model on another sample, reusing its fitted scalers
python scripts/evaluate_blocks.py --model-dir ../models/<name> \
    --dataset vbf_hinv:/global/cfs/.../compact/vbf_hinv --split test
```

On Perlmutter, run the same commands on an interactive node instead of the
login node:

```bash
srun -A m2616_g -C gpu -q interactive -N 1 -n 1 -c 32 --gpus-per-node=1 -t 60 --pty bash
# (--gpus-per-node=4 -c 128 for MirroredStrategy)
```

A training run writes `model.weights.h5`, `model_spec.json`, `norm_params.pkl`,
`config.yaml`, `history.csv`, `metrics.json` and `predictions_test.npz` into
its model directory. Models are rebuilt from the spec rather than
deserialized, so checkpoints survive Keras upgrades.

## Environment

Python 3.12 / TensorFlow 2.20 / Keras 3.15, locked in `pyproject.toml` +
`uv.lock`. Do **not** `module load tensorflow` — the module's CUDA libraries
end up on `LD_LIBRARY_PATH` and can shadow the wheels; `setup.sh` warns if one
is loaded.

## Status

The block pipeline above is the current code path. The previous
loader/processor classes (`src/data/`, `src/models/{common,dnn,transformer}/`,
`scripts/train.py`, `scripts/evaluate.py`, `config/configs/`, `jobs/model_*.sh`)
are still in the tree pending removal.
