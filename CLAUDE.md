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

## What tuning found

Five rounds of sweeps took the validation q68 from 39 to about 35 ps. Almost
none of it came from tuning, and the null results are worth more than the
wins: do not re-run these.

Where the gain came from:

- **4.6 ps** from a data bug, not a setting. The splits were concatenated by
  sample and the shuffle buffer held 10k of 194k rows, so every batch was
  pure ttbar or pure VBF. Fixed by permuting the training split in
  `prepare()`.
- **3.8 ps** from `loss.beta`: 0.25 and 0.5 tie, 0.0 costs 5.5 ps and 1.0
  costs 3.8. The one hyper-parameter that matters.
- **~2 ps** from removing dropout everywhere. At 194k events against 52k
  parameters there is no train/val gap to close; dropout was regularising a
  model that does not overfit.

Measured and found to do nothing, each within the 1-2 ps run-to-run spread:
pooling (attention, masked average, and the `selection_weighted_time` head),
encoder and head widths beyond [256,128,64] and [256,128,64,32] (wider heads
are *worse*), `head.norm` (layer, batch and none are identical), the event
encoder, batch size, learning rate over 2e-4 to 6e-3, warmup, LR patience,
the cell time-quality cut, the `significance` threshold at 2 against 4,
`max_items` at 120 against 250, and the cell sort key.

Also null: rescaling the time features. Cell time is heavy-tailed enough
that a z-score left the signal region spanning 0.087 sigma, and nine of ten
vertices carry a sentinel resolution that flattened the one real value into
a hundredth of a sigma. Both were fixed -- `transform: {time: {asinh: 100}}`
and `valid_when` -- and the 27 physics runs were repeated: every change fell
between -1.3 and +1.1 ps, while `hgtd_only`, which contains no cells and
should not have moved at all, moved by +3.6. The fixes are kept because
they are right, not because they pay: a first Dense layer can learn a large
weight, and nothing here is optimisation-limited.

One thing was measurably worse: a transformer over the cell set, 37.4 +- 0.3
against 35.2 +- 0.5 for MLP plus attention pooling.

That `max_items` 250 does not beat 120, and that admitting every cell down to
significance 2 does not either, says truncation and selection are not the
constraint -- what the calorimeter can say has saturated. The open lead is
vertex identification. Measured on 27 runs, three seeds each: with the
mixed lar_hgtd training, ttbar reads 32.8 ps and VBF 46.5, but split on
whether the highest-sum-pt^2 vertex is the true hard scatter (|dz| < 0.5 mm,
which fails for 5.9% of ttbar and 20.4% of VBF events) they are 29.5 and
28.4 -- identical. The entire ttbar/VBF gap is that rate. On the events
where the vertex is wrong, q68 is 140-168 ps against a target spread of
175, i.e. nothing is recoverable there, because the target belongs to one
vertex and every input describes another.

One caveat on the null list: everything on it except the transform was
measured before the batch-mixing bug was fixed. Head normalization was
retested afterwards and its verdict flipped, so `selection_weighted_time`
-- the one head with a physics argument behind it, scoring each HGTD track
against a context that contains the calorimeter -- should be retested
before it is believed dead.

Two rules that came out of the process:

- **Repeat before believing.** Weight initialisation is unseeded, so one
  setting run twice spreads by 1-2 ps. Sweeps take `repeats:` and report the
  spread; gaps below it are not findings.
- **Grid, not random search, for a handful of axes.** 24 random points over 7
  parameters returned nothing significant (best p = 0.07); the same budget as
  an exact 4x3x2 grid settled `beta` outright.

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
- **`calibration_data/HStrackmatching_calibration.txt` has no recorded
  provenance** -- no note of who produced it, against which time reference, or
  on which sample. The block presets no longer use it (the cell time-quality
  cut was dropped), but `src/evaluation/baseline.py` still weights by its
  sigmas, so the comparison the whole study is built against rests on a table
  nobody here can source. Settle that before quoting a baseline number.
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
