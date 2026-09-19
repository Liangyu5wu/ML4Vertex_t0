# ML4Vertex_t0

Hard-scatter vertex time (t0) regression from ATLAS LAr calorimeter cells and
HGTD track timing.

## Data chain

```
ROOT ntuple  --ingest_root-->  event store  --blocks + assemble-->  tensors  -->  model
  138 GB                          23 GB                          cached, 1.1 s
```

The ingest keeps everything the ntuple has (except a track preselection) and
derives two things: the time-of-flight-corrected cell time and a calorimeter
region code. All physics selection happens later, in the block config, so
changing a cut does not mean re-reading the ROOT files.

| | path | size |
|---|---|---|
| ROOT ntuples | `/global/cfs/cdirs/m2616/liangyu/vertextiming/root/{ttbar,vbf_hinv}` | 130 GB |
| event store | `/global/cfs/cdirs/m2616/liangyu/vertextiming/store/{ttbar,vbf_hinv}` | 23 GB, 312k events |
| tensor cache | `/pscratch/sd/l/liangyu/vertextiming/prepared_cache` | ~1 GB per config |
| models | `/pscratch/sd/l/liangyu/vertextiming/models/<name>` | |

The store and the tensor cache are each keyed by a fingerprint of the settings
that produced them: the same config reuses, a changed config rebuilds.

## Structure

```
setup.sh  pyproject.toml  uv.lock        environment, locked
config/blocks/*.yaml                     one file per experiment
calibration_data/*.txt                   per-layer cell time resolutions
scripts/train_blocks.py                  training entry point
scripts/evaluate_blocks.py               scoring, including on an unseen sample
src/pipeline/
    ingest_root.py                       ROOT ntuple -> event store
    event_store.py  store_writer.py      read / write the store
    schema.py                            schema and content validation
    blocks.py                            input-block specs: fields, cuts, sorting
    assemble.py                          split, normalize, pad, tf.data, cache
    inspect_root.py                      what is inside a ROOT file
src/models/block_model.py  layers.py     model built from the same block specs
src/evaluation/summary.py  plots.py  baseline.py
src/runtime.py                           CPU / 1 GPU / multi-GPU strategy
```

Config reference: [`src/pipeline/README.md`](src/pipeline/README.md).

## Environment

```bash
source setup.sh            # create/activate the uv env, detect the devices
source setup.sh --sync     # after editing pyproject.toml
source setup.sh --cpu      # ignore the GPUs
```

Python 3.12 / TensorFlow 2.20 / Keras 3.15, locked in `pyproject.toml` +
`uv.lock`. Do **not** `module load tensorflow`: the module's CUDA libraries
land on `LD_LIBRARY_PATH` and can shadow the wheels.

## Running

On Perlmutter, work on an interactive node:

```bash
srun -A m2616_g -C gpu -q interactive -N 1 -n 1 -c 32 --gpus-per-node=1 -t 60 --pty bash
#   --gpus-per-node=4 -c 128        for the MirroredStrategy path
#   -A m4956 -C cpu (no --gpus)     for CPU work such as the ingest
```

```bash
# ROOT -> event store, once per sample (up-to-date files are skipped)
python -m src.pipeline.ingest_root \
    --input-dir  /global/cfs/cdirs/m2616/liangyu/vertextiming/root/ttbar \
    --output-dir /global/cfs/cdirs/m2616/liangyu/vertextiming/store/ttbar \
    --sample ttbar --shards 8 --workers 8

# train: the config lists the samples, --datasets picks a subset of them
python scripts/train_blocks.py --config config/blocks/lar_hgtd.yaml
python scripts/train_blocks.py --config config/blocks/lar_hgtd.yaml --datasets ttbar

# score a trained model on another sample, reusing its fitted scalers
python scripts/evaluate_blocks.py --model-dir ../models/lar_hgtd \
    --dataset vbf_hinv:/global/cfs/.../store/vbf_hinv --split test

# the traditional t0 estimate, for comparison
python -m src.evaluation.baseline --store /global/cfs/.../store/ttbar --delta-r 0.1
```

Three configs, differing only in their `inputs:` block: `lar_only`,
`hgtd_only`, `lar_hgtd`. A run writes the weights, `model_spec.json`,
`norm_params.pkl`, the config, `history.csv`, `metrics.json`,
`predictions_test.npz` and a plot set into its model directory.
