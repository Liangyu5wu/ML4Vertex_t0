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
| tensor cache | `/pscratch/sd/l/liangyu/vertextiming/prepared_cache` | 0.5-3 GB per config |
| runs | `/pscratch/sd/l/liangyu/vertextiming/runs/<config>/trial_*` | one directory per training |
| archive | `/global/cfs/cdirs/m2616/liangyu/vertextiming/archive` | runs from before the vertex cut |

The store and the tensor cache are each keyed by a fingerprint of the settings
that produced them: the same config reuses, a changed config rebuilds.

## Structure

```
setup.sh  pyproject.toml  uv.lock        environment, locked
docs/data_chain.md  docs/config.md       the data chain, and every config key
config/blocks/*.yaml                     one file per experiment
calibration_data/*.txt                   per-layer cell time resolutions
scripts/train_blocks.py                  training entry point
scripts/evaluate_blocks.py               scoring, including on an unseen sample
scripts/sweep.py                         hyper-parameter search
scripts/audit_inputs.py                  one event, stage by stage
scripts/cell_coverage.py                 what the cell truncation cuts off
scripts/compare_runs.py                  input sets compared: tables, efficiency, migration
config/sweeps/*.yaml                     search spaces, with what each found
src/pipeline/
    ingest_root.py                       ROOT ntuple -> event store
    event_store.py  store_writer.py      read / write the store
    schema.py                            schema and content validation
    blocks.py                            input-block specs: fields, cuts, sorting
    assemble.py                          split, normalize, pad, tf.data, cache
    inspect_root.py                      what is inside a ROOT file
src/models/block_model.py  layers.py     model built from the same block specs
src/evaluation/summary.py                metrics, fits, sigma cuts
src/evaluation/plots/                    every figure: style, residual, sigma, compare, report
src/evaluation/baseline.py  event_display.py
src/runtime.py                           CPU / 1 GPU / multi-GPU strategy
```

`audit_inputs.py` prints one event's path through selection, sorting,
truncation, normalization, padding and masking, in physical units before and
normalized after — the way to check a config switch against what it did
rather than against its name.

Data chain, stage by stage, with the numbers each stage produces:
[`docs/data_chain.md`](docs/data_chain.md). Config reference:
[`docs/config.md`](docs/config.md). Current results, and what has been tried:
[`CLAUDE.md`](CLAUDE.md), "Where things stand".

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

Work on interactive nodes, never `sbatch`. One node for a single run; two
(the per-user limit, 8 GPUs) for a sweep, each taking a shard:

```bash
srun -A m2616_g -C gpu -q interactive -N 1 -c 32 --gpus-per-node=1 -t 60 --pty bash
#   -A m4956 -C cpu (no --gpus)     for CPU work such as the ingest

salloc -A m2616_g -C gpu -q interactive -N 1 -c 128 --gpus-per-node=4 \
    -t 04:00:00 --no-shell                    # twice; note both job ids
srun --jobid=<A> --overlap -n1 -c 128 --gpus=4 bash -c \
    "cd $PWD && source setup.sh && python scripts/sweep.py ... --shard 0/2"
srun --jobid=<B> ... --shard 1/2
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

# a sweep (here the baselines: 3 training mixtures x 3 seeds), then one ranking
python scripts/sweep.py --config config/blocks/lar_hgtd.yaml \
    --space config/sweeps/physics_lar_hgtd.yaml --out ../runs/lar_hgtd --shard 0/2
python scripts/sweep.py --report-only --space config/sweeps/physics_lar_hgtd.yaml \
    --out ../runs/lar_hgtd

# score a trained model on another sample, reusing its fitted scalers
python scripts/evaluate_blocks.py --model-dir ../runs/lar_hgtd/trial_003 \
    --dataset vbf_hinv:/global/cfs/.../store/vbf_hinv --split test

# input sets compared: resolution against sigma-cut efficiency, and event by
# event where HGTD-only's failures go once LAr is added (LAr-only for reference)
python scripts/compare_runs.py ../runs/lar_hgtd ../runs/hgtd_only \
    --efficiency-plot ../runs/efficiency_lar_hgtd_vs_hgtd_only.png
python scripts/compare_runs.py ../runs/lar_hgtd ../runs/hgtd_only ../runs/lar_only \
    --recovery ../runs/recovery_lar_hgtd.png

# redraw a run's plots, with the sigma working points
python -m src.evaluation.plots ../runs/lar_hgtd/trial_000 --max-sigma 20 40 60

# the traditional t0 estimate, for comparison
python -m src.evaluation.baseline --store /global/cfs/.../store/ttbar --delta-r 0.1
```

Three configs, differing only in their `inputs:` block: `lar_only`,
`hgtd_only`, `lar_hgtd`.

## What a run writes

`record.md` (setup, timing, commit, results on one page), `history.csv` and
`plots/history.png` (loss and RMSE per epoch, train and validation),
`metrics.json` (validation and test, by sample), the weights,
`model_spec.json`, `norm_params.pkl`, `config.yaml`, and
`predictions_val.npz` / `predictions_test.npz`: a cut on the predicted sigma
is chosen from the first and reported on the second. The record and the loss
curve are always written; `--no-plots` skips only the evaluation figures.

`plots/` holds the residual (linear and log), predicted against true,
resolution against the true t0, and for the predicted sigma its spread, its
calibration, the pull, sigma against Delta t0 and resolution against cut
efficiency; each working point in `evaluation.sigma_cut` adds a
`sigma_cut_<N>ps/` with the same residual plots for the kept events.

Sweeps rank on the validation `q68`, the half-width holding 68% of the
errors, never on a fitted core width: the fit finds a narrow core in an
untrained model's residuals too, so it scores the worst trials best.
