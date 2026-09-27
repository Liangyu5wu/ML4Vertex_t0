# CLAUDE.md

Guidance for Claude Code when working in this repository.

Vertex time (t0) regression for ATLAS from LAr calorimeter and HGTD timing.
`README.md` has the data chain, layout and commands; `docs/config.md`
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
- All figures go through `src/evaluation/plots/` (`style.py` sets the ATLAS
  house style: closed black frame, ticks inward on all four sides with minors,
  no grid, type at 15-17pt). Read the `dataviz` skill before adding a plot type.
- **Histograms report event counts, never a density.** Label the axis
  `events / <bin width>` and put each sample's count in the legend; do not
  normalise so that two samples overlay neatly.
- **Every training keeps its record**: `record.md`, `history.csv`,
  `metrics.json` and `plots/history.png` are written unconditionally, even
  under `--no-plots` and for sweep trials. A run whose loss curve was never
  saved cannot be argued about afterwards.
- Anything automated ranks on the **validation** split, and on `q68` rather
  than a fitted core width — a double-Gaussian fit finds a narrow core in an
  untrained model's residuals too, so it rewards models that learned nothing
  (measured: identical degenerate runs fitted anywhere from 5.7 to 46 ps).
- A cut on the predicted sigma is chosen on validation (`predictions_val.npz`)
  and reported on test, as a fixed `max_sigma` in ps, per sample.
- Models are saved as weights + `model_spec.json` and rebuilt on load; do not
  reintroduce whole-model serialization.
- Long runs go on two interactive nodes with `sweep.py --shard`, never
  `sbatch` (README, "Running").
- Store files are written to a temporary name and renamed, so an interrupted
  run never leaves a half-written store that still opens.

## Where things stand

Two changes separate current runs from everything earlier: the configs keep
only events with `|hs_vtx_dz| <= 3 mm` (a truth cut standing in for better
vertex identification; it removes 5.2% of ttbar and 18.4% of VBF), and the
vertex time is no longer an input. Earlier runs, `../sweeps` and
`../displays` are archived in
`/global/cfs/cdirs/m2616/liangyu/vertextiming/archive/2026-09-26_before_vertex_cut.tar.gz`
and are no control arm for anything since; `../runs` holds only runs made
after both.

Baselines (`../runs/<config>`, test q68 in ps, three seeds, mixed training
unless noted):

| | ttbar | VBF |
|---|---|---|
| `lar_hgtd` | 27.6 +- 1.2 (26.3 trained on ttbar alone) | 25.1 +- 0.9 (28.9 alone) |
| `hgtd_only` | 50.9 +- 0.9 | 32.8 +- 0.3 (38.7 alone) |
| `lar_only` | 88.8 +- 0.4 | 125.5 +- 1.7 |

- **VBF is no longer the harder sample.** HGTD carries it (32.8 against 50.9
  for ttbar), the calorimeter carries ttbar. Mixed training helps VBF and does
  not hurt ttbar.
- **The combination is the result, and core sigma hides it.** At full
  efficiency LAr+HGTD reads 27.6 against 50.9 ps (ttbar); at q68 <= 20 ps it
  keeps 81% of ttbar against 62%. The fitted core is the same (16.4 against
  16.3 ps): the gain is events moved out of the tails, so quote q68 or the
  efficiency at a fixed q68 (`compare_runs.py --efficiency-plot`).
- **How it works.** HGTD alone is off by more than 60 ps in 26% of events;
  in 92% of those it sits on a pile-up track cluster while a hard-scatter
  track is there too (98%). With LAr, 36% of them come back under 30 ps and
  1.8% of all events get worse; the recovered reach 13 ps where LAr alone
  reaches 40. LAr picks the cluster, HGTD sets the resolution. What makes a
  failure recoverable is open: recovery falls from 43% to 26% as HGTD's error
  grows from 60-100 to past 300 ps, the opposite of "far-apart clusters are
  easy" (`compare_runs.py --recovery`).
- **The old `lar_only` (53 ps) was never calorimeter-only**: its `vertices`
  block carried `RecoVtx_time`, which is built from HGTD tracks, so every
  older statement leaning on it mixed HGTD timing into the "LAr" arm.
- **`lar_only` does not share the others' test split.** The split is drawn
  after `min_items`, and it keeps events with no HGTD track; about a fifth of
  its test events are in theirs. Per-event comparisons match on event number.

The predicted sigma is honest to 5% on the mixture, but ttbar's is slightly
conservative and VBF's 5-25% optimistic, varying by seed. The cut did not
change that, so the wrong vertex was not the cause, and no map from sigma
alone can undo a split between samples. The working points (20 / 40 / 60 ps,
in the configs, rules in `docs/config.md`) leave the kept events far more
Gaussian, but even the tight one keeps ten times a Gaussian's tail beyond 3
sigma (2.7% against 0.27%).

Two known limits of the prediction. It shrinks towards zero across the whole
range -- the median Delta t0 is +18 ps at t0 = -300 and -8 at +200
(`resolution_vs_truth.png`) -- because events far from zero are rare in
training. And `head.norm: layer` bounds it near +-500 ps (the flat lines in
`pred_vs_true.png`, 1% of events): the last LayerNorm fixes the length of
what the read-out sees. Removing it costs 1-3 ps (`none`) or 5.7 ps for
`hgtd_only` (the last one only) and predicts those events no better, so
`layer` stays (`config/sweeps/head_norm*.yaml`).

Null on the cut, within the run-to-run spread: `max_items` 60 / 120 / 250
(`cell_count.yaml`); dropout for `lar_only`, where head dropout 0.2 collapses
half the seeds to a constant (`lar_only_dropout.yaml`); an asinh on `sum_pt2`
and track `pt`, whose z-scores leave them almost no range
(`scale_transforms.yaml`).

Open: the double-Gaussian fit fixes its wide term at 175.74 ps, the spread of
wrong-vertex events. Those are now cut, the tails are narrower, and the core
sigma the fit reports depends on that choice; `fix_pileup_sigma: false` is
undecided.

## Before the vertex cut

Five rounds of sweeps took the validation q68 from 39 to about 35 ps, and
almost none of it came from tuning:

- **4.6 ps** from a data bug: the splits were concatenated by sample and the
  shuffle buffer held 10k of 194k rows, so every batch was one sample. Fixed
  by permuting the training split in `prepare()`.
- **3.8 ps** from `loss.beta`: 0.25 and 0.5 tie, 0.0 costs 5.5 ps and 1.0 3.8.
- **~2 ps** from removing dropout everywhere.

Null, within the 1-2 ps spread: pooling (attention against masked average),
encoder and head widths beyond [256,128,64] and [256,128,64,32] (wider heads
are worse), the event encoder, batch size, learning rate over 2e-4 to 6e-3,
warmup, LR patience, the cell time-quality cut, `significance` 2 against 4,
the cell sort key; rescaling the time features (`transform`, `valid_when`,
kept because they are right); and a head that weighted HGTD tracks by a
probability scored against the calorimeter (0.0 +- 0.6 ps against a masked
average; removed, do not rebuild -- `git log --diff-filter=D`). All but the
last two were measured before the batch-mixing fix, so treat them as
provisional; `head.norm`, once on this list, is not null (above).

Worse: a transformer over the cells (37.4 +- 0.3 against 35.2 +- 0.5), and
`reco_vtx_time` as an event feature (+2.0 ps). The model already reached
the vertex time and declined to follow it, because it is wrong by more than
20 ps in 41% of events; that is what turned 452 ps into 135 in the worst bin.

Vertex identification was then the open lead: split on whether the
sum-pt^2 vertex was right (|dz| < 0.5 mm), ttbar and VBF read the same
(29.5 and 28.4 ps), so the whole gap between them was the wrong-vertex rate
-- which is what the cut above removes.

## Rules the process left


- **Repeat before believing.** Weight initialisation is unseeded, so one
  setting run twice spreads by 1-2 ps. Sweeps take `repeats:` and report the
  spread; gaps below it are not findings.
- **Check whether the control arm already exists before running it.** Compare
  `AssemblySpec.fingerprint()` and the model settings against what is in
  `../runs`; an A/B whose baseline is already on disk should run one arm, not
  two. Since weight initialisation is unseeded, three fresh seeds of an
  existing setting are the same three draws, not a paired comparison.
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
