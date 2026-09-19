# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Vertex time (t0) regression for ATLAS, from LAr calorimeter cell timing and
HGTD track timing. See `README.md` for the architecture and
`src/pipeline/README.md` for the config reference.

## Environment

`source setup.sh` — that is the whole setup. It creates/activates the uv venv,
adds the CUDA wheels when a GPU is visible, and sizes the thread pools.

- Everything is locked in `pyproject.toml` + `uv.lock` (Python 3.12,
  TensorFlow 2.20, Keras 3.15). Change dependencies there, then
  `source setup.sh --sync`.
- **Never `module load tensorflow`** — the module's CUDA libraries land on
  `LD_LIBRARY_PATH` and can shadow the wheels. `setup.sh` warns if one is loaded.
- Never `pip install --user`; `PYTHONNOUSERSITE=1` is set to keep `~/.local`
  out of the environment.

## Running

Use an interactive node rather than sbatch:

```bash
srun -A m2616_g -C gpu -q interactive -N 1 -n 1 -c 32 --gpus-per-node=1 -t 60 --pty bash
# --gpus-per-node=4 -c 128 for the MirroredStrategy path
```

```bash
python -m src.pipeline.ingest_h5 --input-dir <raw h5> --output-dir <store> --sample <name>
python scripts/train_blocks.py --config config/blocks/lar_hgtd.yaml [--datasets ttbar]
python scripts/evaluate_blocks.py --model-dir <dir> --dataset <name>:<store> --split test
```

Three configs, differing only in `inputs:` — `lar_only`, `hgtd_only`,
`lar_hgtd`. Models and results go outside the repo, under
`/pscratch/sd/l/liangyu/vertextiming/models/`; compact data stores live on CFS
at `/global/cfs/cdirs/m2616/liangyu/vertextiming/compact/`.

## Conventions

- A new input type or sample is a **config** change, not a code change. If it
  cannot be expressed in YAML, extend `src/pipeline/blocks.py` presets rather
  than adding a parallel code path.
- Per-event Python loops over the data are not acceptable; everything is
  vectorised over the flat ragged arrays.
- A feature name that does not resolve against the store must raise, never
  silently become zero.
- All figures go through `src/evaluation/plots.py` so style and palette stay
  consistent. Read the `dataviz` skill before adding a new plot type.
- Models are saved as weights + `model_spec.json` and rebuilt on load; do not
  reintroduce whole-model serialization.

## Known data issues

- **HGTD track truncation.** R2H5 stores at most 200 HGTD tracks per event in
  container order (not pt-sorted); 15.7% of ttbar and 8.8% of VBF events hit
  that cap, so "top 30 by pt" is a top-30 of an arbitrary subset. Raise the cap
  or sort before writing when the samples are regenerated.
- **Forward cells.** 7.3% of cells in EM layers 1-3 have neither
  `Cell_isEM_Barrel` nor `Cell_isEM_EndCap` set — they sit at |eta| up to 4.8
  with high energy (FCal). They are currently treated as endcap, including in
  the time-quality cut, whose calibration table has no FCal entries.
- **Quantised cell energy.** ~10% of cells share an energy exactly with another
  cell in the same event, so any top-N selection needs an explicit tie-break.
- The repository lives on scratch, which purges unused files — this has already
  corrupted the git object store once. Keep the branch pushed.

## Repository state

The legacy pipeline (four loader/processor pairs, per-architecture model
classes, `scripts/train.py`) was removed in this branch's history; the block
pipeline is the only code path. `baseline_analysis/` is a standalone tool that
still references the older LAr dataset and has not been migrated.
