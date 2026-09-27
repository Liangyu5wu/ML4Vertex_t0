# Config reference

Every key a block config accepts. Adding an input, a sample or a cut is a
YAML change, not a code change. What each stage does with these settings,
and the numbers it produces, is in [`data_chain.md`](data_chain.md).

## 1. Event store

The blocks a config can name as `source`: `cells`, `tracks`, `jets_emtopo`,
`jets_pflow`, `reco_vertices`, `truth_vertices`, `truth_hs_jets`. Field names
are ours, not the ntuple's; each store's `manifest.json` records the
mapping. Layout, ingest and sizes: [`data_chain.md`, Stage 1](data_chain.md).

## 2. Keys

```yaml
model_name: my_model
model_dir: /pscratch/.../models/my_model

data:
  datasets:                      # one entry per sample; order is irrelevant
    - name: ttbar
      path: /global/cfs/.../store/ttbar
      weight: 1.0                # relative loss weight
      fraction: 1.0              # subsample this fraction of events
      max_events: null           # hard cap
      files: null                # null = every file in the manifest
  resample: oversample           # none | oversample | undersample; training split only
  target: truth_vtx_time
  event_features: [reco_vtx_x, reco_vtx_y, reco_vtx_z]   # [] drops the branch
  event_select:                  # whole events, before the split; same operators
    - {field: hs_vtx_dz, abs_max: 3.0}                   # as block selections
  split: {test_size: 0.2, val_split: 0.125, random_state: 42}   # 70/10/20
  cache_dir: /pscratch/.../prepared_cache                # null disables caching

  inputs:                        # any subset of the presets below
    cells:
      preset: lar_cells
      features: [eta, phi, region, layer, time, e, significance]  # optional subset
      max_items: 120
      min_items: 1               # events with fewer objects are dropped
      sort_by: [e, significance] # list = tie-break keys, most significant first
      descending: true
      select: [...]              # replaces the preset's cuts
      select_extra:              # adds to them
        - {field: e, min: 2.0}
      transform: {time: {asinh: 100.0}}    # reshape before fitting the scaler
      valid_when: {time: has_valid_time}   # fit the scale on the real rows only
      padding: {time: 0.0}       # per-feature, in physical units
      skip_normalization: [region, layer]
      pad_in: literal            # literal | normalized (see below)
      encoder: {units: [256, 128, 64], dropout: 0.0, pooling: attention,
                attention_units: 32}

head: {units: [256, 128, 64, 32], dropout: 0.0, norm: layer}   # layer | batch | none
loss: {type: gaussian_nll, beta: 0.25, sigma_init: 100.0}      # mse | mae | huber
optimizer: {type: adam, learning_rate: 0.005}
training: {epochs: 300, batch_size: 1024, early_stopping_patience: 25,
           lr_reduction_factor: 0.5, lr_patience: 8, min_lr: 1.0e-7,
           warmup_epochs: 5}
evaluation:
  fit: {method: double_gaussian, pileup_sigma: 175.74, fix_pileup_sigma: true}
  sigma_cut:                     # one cut or a list of working points
  - {name: tight, max_sigma: 20.0}
  - {name: medium, max_sigma: 40.0}
  - {name: loose, max_sigma: 60.0}
```

`evaluation.sigma_cut` redraws the residual plots for the events whose
predicted sigma passes, one directory per working point
(`plots/sigma_cut_<N>ps/`), with a kept/removed comparison per sample and
`cut_metrics.json`: the threshold, where it came from, and each sample's
efficiency and resolution after the cut.

The rules the working points follow, and why:

- **A fixed `max_sigma` in ps**, the only form a cut in data can take, and
  the same in every config so each reads as an expected resolution. It is
  stable: across three seeds of `lar_hgtd` the medium point keeps 69-73% and
  its q68 moves by 0.2 ps. `lar_only` has no event below 20 ps, so its tight
  point is empty and not drawn (fewer than 50 kept events never are).
- **Raw sigma, not recalibrated.** On the mixture it is honest to within 5%.
  What is off is the split between samples -- ttbar 0.92-1.03, VBF 1.03-1.27
  in q68(|Delta t0| / sigma) -- and a map from sigma alone cannot tell the
  two apart, any more than data can.
- **Chosen on validation, reported on test.** `keep_fraction: F` is turned
  into a `max_sigma` from `predictions_val.npz` when the run has one, and
  `cut_metrics.json` records it; the threshold is then the model's, whichever
  sample is scored.
- **Always reported per sample.** One threshold keeps 3-15 points more of
  VBF than of ttbar.

For an existing run: `python -m src.evaluation.plots <run> --max-sigma 20 40 60`.

### Presets

| preset | source block | |
|---|---|---|
| `lar_cells` | `cells` | EM calorimeter cells |
| `jets_emtopo`, `jets_pflow` | the same name | jets, no selection |
| `hs_tracks` | `tracks` | tracks on the reco HS vertex |
| `hgtd_tracks` | `tracks` | tracks with an HGTD time, near the HS vertex in z |
| `vertices` | `reco_vertices` | reconstructed vertices |

Each preset's features, units, selection, ordering and cap are tabulated in
[`data_chain.md`, Stage 2](data_chain.md#what-the-model-reads).

`vertices` keeps the vertex time, its resolution and `has_valid_time` as
auxiliary fields, not inputs. `features: [z, sum_pt2, is_hs, time, time_res,
has_valid_time]` with `valid_when: {time: has_valid_time, time_res:
has_valid_time}` restores the earlier input.

The jet presets deliberately carry no selection. Truth matching is the only
handle that identifies a jet as hard-scatter and it does not exist in data,
so the match counts stay loaded as auxiliary fields and are never cut on.

`region` is 0 EM barrel, 1 EM endcap, 2 FCal, 3 HEC, 4 Tile. `time` is the
time-of-flight-corrected cell time. Each preset also loads a few auxiliary
fields (positions, quality flags, `dz_hs`, truth-match counts) that cuts can
use without them becoming model inputs.

Names resolve against the fields the store actually has, through the alias
lists in `blocks.py`. A name that resolves to nothing raises — it is never
silently filled with zeros.

Selection operators: `eq`, `ne`, `min`, `max`, `in`, `abs_max`, `abs_min`,
plus a special `time_quality` cut that keeps cells within `n_sigma` of
`sqrt(vertex_sigma^2 + sigma_cell(layer, E)^2)`. All of a block's rules are
combined into one mask and applied in a single pass over the whole sample.

The presets do not use `time_quality`. It was measured at 0.8 +- 0.5 ps, and
buying that in data means deriving a per-layer, per-energy resolution table
against some independent time reference and carrying its systematic. The
model already sees each cell's energy and significance and can discount a
badly measured one without being told to.

### Event selection

`event_select` drops whole events before the split, so every split and every
sample sees the same cut. A rule's `field` is a stored event column or a
derived one from `DERIVED_EVENT_FIELDS` in `blocks.py`:

| derived field | definition |
|---|---|
| `hs_vtx_dz` | `reco_vtx_z - truth_vtx_z`: how far the sum-pt² vertex is from the true hard scatter |

`hs_vtx_dz` is truth. The three configs cut it at 3 mm to stand in for a
vertex identification better than sum-pt², which removes 5.2% of ttbar and
18.4% of VBF. This is not a cut data can make. Every number measured before
it (the tuning record in `CLAUDE.md`, `../runs`) is on all events.

### Pooling

`attention`, `masked_average`, `average`, `max`, `sum`, `flatten`. The masked
variants and `type: transformer` encoders turn the block's attention mask on
automatically. Plain `average` pools the padded slots too, which for a block
that is mostly padding (4 real jets in 7 slots) means the padding dominates —
the presets use `masked_average` everywhere except the cell block, which uses
attention pooling.

### Transform and missing values

`transform` reshapes a feature before its scaler is fitted, so the
statistics and the values cannot disagree. `{asinh: t0}` is linear below
`t0` and logarithmic above it, which is what a heavy-tailed feature needs:
47% of selected cell times are past 1 ns, and a plain z-score left the
200 ps that carries the signal spanning 0.09 of a standard deviation.
`{clip: [lo, hi]}` is the blunt alternative and loses the ordering outside
the window.

`valid_when` names a feature that marks which rows are real. Nine of ten
reco vertices carry a sentinel time resolution rather than a measurement,
and fitting over it flattened the one real value in an event into a
hundredth of a sigma (the vertex time is no longer a default input; this
applies when it is added back). With `valid_when` the scale comes from the real rows
and the others are written at its centre; the validity feature is itself an
input, so the model is told which those are.

Neither measurably changed the result -- see the null list in `CLAUDE.md`
-- but a physics quantity with no dynamic range is a defect either way.

### Padding space

`pad_in: literal` (the default, and what every preset uses) writes the
padding value straight into the scaled tensor, so `0.0` means "the mean" and
is harmless because the mask hides it. `pad_in: normalized` pushes it
through the fitted scaler instead, so `-999` stays an outlier.

Running, environment and outputs: see the [top-level README](../README.md).
