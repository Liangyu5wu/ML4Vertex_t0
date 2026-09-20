# CLAUDE.md

Guidance for Claude Code when working in this repository.

Vertex time (t0) regression for ATLAS from LAr calorimeter and HGTD timing.
`README.md` has the data chain, layout and commands; `src/pipeline/README.md`
is the config reference. Read those first — this file is only what is not
obvious from them.

## Environment

`source setup.sh` is the whole setup: it creates or activates the uv venv,
adds the CUDA wheels when a GPU is visible, and sizes the thread pools.

- Dependencies live in `pyproject.toml` + `uv.lock`; after editing them run
  `source setup.sh --sync`.
- **Never `module load tensorflow`** — the module's CUDA libraries stay on
  `LD_LIBRARY_PATH` and can shadow the ones the wheels ship. `setup.sh` warns.
- Never `pip install --user`; `PYTHONNOUSERSITE=1` keeps `~/.local` out.
- CPU work goes to `-A m4956 -C cpu`; GPU work to `-A m2616_g -C gpu`
  (`m2616`'s CPU hours are exhausted).

## Conventions

- A new input or sample is a **config** change. If it cannot be expressed in
  YAML, extend the presets in `src/pipeline/blocks.py` rather than adding a
  parallel code path.
- No per-event Python loops over data: everything is vectorised over the flat
  ragged arrays.
- A field name that does not resolve against the store must raise, never
  silently become zero. Productions rename branches; that is what the alias
  tuples are for.
- Anything expensive that depends only on settings gets a fingerprint and is
  reused — the event store and the tensor cache both work this way. Extend
  that pattern rather than adding a workflow engine.
- All figures go through `src/evaluation/plots.py`. Read the `dataviz` skill
  before adding a plot type.
- **Every training keeps its record**: `record.md`, `history.csv`,
  `metrics.json` and `plots/history.png` are written unconditionally, even
  under `--no-plots` and for sweep trials. A run whose loss curve was never
  saved cannot be argued about afterwards.
- Anything automated ranks on the **validation** split, and on `q68` rather
  than a fitted core width — a double-Gaussian fit finds a narrow core in an
  untrained model's residuals too, so it rewards models that learned nothing
  (measured: identical degenerate runs fitted anywhere from 5.7 to 46 ps).
- Models are saved as weights + `model_spec.json` and rebuilt on load; do not
  reintroduce whole-model serialization.
- Store files are written to a temporary name and renamed, so an interrupted
  run never leaves a half-written store that still opens.

## Known data issues

Found while validating the ingest; the first two are handled in code, the rest
are open.

- `RecoVtx_isPU` is filled as a running sum across events in both productions
  (the producer never clears the vector), so it is excluded. The
  per-collection count check in `ingest_root` would reject it anyway.
- `BCID` is always 0 and `distFrontBunchTrain` is a constant uninitialised
  value; both are excluded rather than kept as dead columns.
- **Cell energies are quantised**: ~10% of cells share an energy exactly with
  another cell in the same event, so any top-N selection needs an explicit
  tie-break (the cell preset sorts on `(e, significance)`).
- **No calibration outside the EM calorimeter.** `calibration_data/*.txt` has
  sigma only for EMB1-3 and EME1-3. FCal, HEC and Tile cells (18% of the
  store, |eta| up to 4.8) fall back to a 1000 ps resolution, which effectively
  exempts them from the time-quality cut. The default cell selection excludes
  them; whether VBF's forward topology wants them included is open.
- The two samples come from different producers: the VBF ntuple has 241
  branches and ttbar 172, differing in track hit-count details (aliased) and
  VBF-only jet substructure. The 164 common branches cover everything used.

## Repository

The block pipeline is the only code path; the previous loader/processor
classes and per-architecture models were removed in this branch's history
(`git log --diff-filter=D --name-only` to find them). The repository lives on
scratch, which purges unused files and has already corrupted the git object
store once — keep the branch pushed.
