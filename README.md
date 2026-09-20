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
scripts/sweep.py                         hyper-parameter search
scripts/audit_inputs.py                  one event, stage by stage
config/sweeps/*.yaml                     search spaces, with what each found
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

`audit_inputs.py` prints one event's path through selection, sorting,
truncation, normalization, padding and masking, in physical units before and
normalized after — the way to check a config switch against what it did
rather than against its name.

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

# search hyper-parameters, one trial per GPU, ranked on the validation split
python scripts/sweep.py --config config/blocks/lar_hgtd.yaml \
    --space config/sweeps/optimization.yaml --out ../sweeps/optimization

# the same sweep across two interactive jobs (the per-user limit), 8 GPUs:
#   node A:  ... sweep.py --shard 0/2 --out ../sweeps/opt ...
#   node B:  ... sweep.py --shard 1/2 --out ../sweeps/opt ...
python scripts/sweep.py --report-only --out ../sweeps/opt \
    --space config/sweeps/optimization.yaml       # one ranking over both

# score a trained model on another sample, reusing its fitted scalers
python scripts/evaluate_blocks.py --model-dir ../models/lar_hgtd \
    --dataset vbf_hinv:/global/cfs/.../store/vbf_hinv --split test

# the traditional t0 estimate, for comparison
python -m src.evaluation.baseline --store /global/cfs/.../store/ttbar --delta-r 0.1
```

Three configs, differing only in their `inputs:` block: `lar_only`,
`hgtd_only`, `lar_hgtd`.

Every run writes its own record into the model directory: `record.md` (one
readable page — setup, timing, commit, results), `history.csv` and
`plots/history.png` (loss and RMSE against epoch, train and validation),
`metrics.json` (validation *and* test, split by sample), the weights,
`model_spec.json`, `norm_params.pkl`, `config.yaml` and
`predictions_test.npz`. `--no-plots` skips only the evaluation figures; the
record and the loss curve are always kept.

Sweeps rank on the validation `q68` — the half-width holding 68% of the
errors. A fitted core width is not used for ranking: the fit finds a narrow
core in an untrained model's residuals as well, so it scores the worst trials
best.

For an unattended long run, the same command under `sbatch`:

```bash
sbatch -A m2616_g -C gpu -q shared -N 1 -c 32 --gpus-per-task=1 -t 08:00:00 \
    --wrap "cd $PWD && source setup.sh && python scripts/train_blocks.py --config <cfg>"
```
