# Block pipeline

Config reference for the data path. Adding an input, a sample or a cut is a
YAML change, not a code change.

```
ROOT ntuple  --ingest_root.py-->  event store  --blocks.py-->  selected ragged
    --assemble.py-->  padded tensors + tf.data  --block_model.py-->  Keras model
```

## 1. Event store

One compressed column per field in a ragged (CSR) layout, so a model that
reads HGTD tracks alone never pays for the calorimeter cells.

```bash
python -m src.pipeline.ingest_root \
    --input-dir  /global/cfs/cdirs/m2616/liangyu/vertextiming/root/ttbar \
    --output-dir /global/cfs/cdirs/m2616/liangyu/vertextiming/store/ttbar \
    --sample ttbar --shards 8 --workers 8
```

Layout inside each file:

```
/events/<field>          (n_events,)      truth_vtx_time, reco_vtx_z, mu, ...
/blocks/<block>/offsets  (n_events+1,)    CSR offsets
/blocks/<block>/<field>  (n_items,)       one column per field
```

Blocks: `cells`, `tracks`, `jets_emtopo`, `jets_pflow`, `reco_vertices`,
`truth_vertices`.

Field names are ours, not the ntuple's; the mapping and the ingest settings
fingerprint are recorded in `manifest.json`. Existing stores:

| sample | events | size |
|---|---|---|
| ttbar | 199,700 | 15 GB (from 83 GB of ROOT) |
| vbf_hinv | 112,400 | 7.6 GB (from 55 GB) |

## 2. Config reference

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
  event_features: [reco_vtx_z]   # [] drops the branch
  split: {test_size: 0.2, val_split: 0.222222, random_state: 42}
  cache_dir: /pscratch/.../prepared_cache                # null disables caching

  inputs:                        # any subset of the presets below
    cells:
      preset: lar_cells
      features: [eta, phi, region, layer, time, e, significance]  # optional subset
      max_items: 60
      min_items: 1               # events with fewer objects are dropped
      sort_by: [e, significance] # list = tie-break keys, most significant first
      descending: true
      select: [...]              # replaces the preset's cuts
      select_extra:              # adds to them
        - time_quality: {n_sigma: 3.0, vertex_sigma: 175.0,
                         calibration: sigma_only_test_calibration.txt,
                         apply_calibration: false}
      padding: {time: 0.0}       # per-feature, in physical units
      skip_normalization: [region, layer]
      pad_in: literal            # literal | normalized (see below)
      encoder: {units: [128, 64, 32], dropout: 0.1, pooling: attention,
                attention_units: 32}

head: {units: [128, 64, 32, 16], dropout: [0.2, 0.1, 0.1, 0.1], batch_norm: false}
loss: {type: huber, delta: 100.0}          # mse | mae | huber
optimizer: {type: adam, learning_rate: 0.001}
training: {epochs: 200, batch_size: 256, early_stopping_patience: 20,
           lr_reduction_factor: 0.5, lr_patience: 8, min_lr: 1.0e-7}
evaluation:
  fit: {method: double_gaussian, pileup_sigma: 175.74, fix_pileup_sigma: true}
```

### Presets

| preset | block | features | default selection | max |
|---|---|---|---|---|
| `lar_cells` | `cells` | eta, phi, region, layer, time, e, significance | region in {EMB, EME}, layer in {1,2,3}, \|significance\| > 4, e > 1 GeV | 60 by (e, significance) |
| `jets_emtopo` | `jets_emtopo` | pt, eta, phi, width | matched to >= 1 truth HS jet | 7 by pt |
| `jets_pflow` | `jets_pflow` | same | same | 7 by pt |
| `hs_tracks` | `tracks` | pt, eta, phi, d0, z0 | `on_hs_vertex == 1` | 30 by pt |
| `hgtd_tracks` | `tracks` | pt, eta, phi, d0, z0, time, time_res | `has_valid_time == 1`, 2.4 < \|eta\| < 4.0 | 30 by pt |

`region` is 0 EM barrel, 1 EM endcap, 2 FCal, 3 HEC, 4 Tile. `time` is the
time-of-flight-corrected cell time. Each preset also loads a few auxiliary
fields (positions, quality flags, `dz_hs`, truth-match counts) that cuts can
use without them becoming model inputs.

Names resolve against the fields the store actually has, through the alias
lists in `blocks.py`. A name that resolves to nothing raises — it is never
silently filled with zeros.

Selection operators: `eq`, `ne`, `min`, `max`, `in`, `abs_max`, `abs_min`,
plus the special `time_quality` cut. All of a block's rules are combined into
one mask and applied in a single pass over the whole sample.

### Pooling

`attention`, `masked_average`, `average`, `max`, `sum`, `flatten`. The masked
variants and `type: transformer` encoders turn the block's attention mask on
automatically. Plain `average` pools the padded slots too, which for a block
that is mostly padding (4 real jets in 7 slots) means the padding dominates —
the presets use `masked_average` everywhere except the cell block, which uses
attention pooling.

### Padding space

`pad_in: normalized` (default) pushes the configured padding value through the
fitted scaler, so `-999` stays an outlier after normalization. `pad_in: literal`
writes the number straight into normalized data — the legacy cell behaviour,
where `0.0` means "the mean" and is harmless because the mask hides it.

## 3. Notes

Each sample is split with its own seed derived from `random_state` and the
sample name, so a sample's test events are the same whether it was trained
alone or in a mixture -- that is what makes cross-sample numbers comparable.

Normalization is fitted on the pooled training split only, after truncation to
`max_items`, i.e. on exactly the objects the model sees. `norm_params.pkl` is
saved with the model and reused by `evaluate_blocks.py`.

Running, environment and outputs: see the [top-level README](../../README.md).
