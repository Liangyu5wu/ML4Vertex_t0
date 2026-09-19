# Block pipeline

Config-driven data path and model assembly, replacing the four loader/processor
pairs and the per-architecture branches in `scripts/train.py`.

Adding an input (or a sample) is a YAML change, not a code change.

```
raw R2H5 h5  --compact.py-->  event store  --blocks.py-->  selected ragged
    --assemble.py-->  padded tensors + tf.data  --block_model.py-->  Keras model
```

## 1. Event store

Raw files keep every collection as a dense `(n_events, n_slots)` float64
structured array: cells use 144 of 1000 slots, tracks 35 of 200, jets 1.8 of 50.
The event store drops invalid slots and stores one compressed column per
field, giving **11–14x smaller files** with bit-identical values.

```bash
python -m src.pipeline.ingest_h5 \
    --input-dir ../Vertex_timing_HGTD_w_LAr \
    --output-dir /global/cfs/cdirs/m2616/liangyu/vertextiming/compact/ttbar \
    --sample ttbar
```

Layout inside each file:

```
/events/<field>          (n_events,)      HSvertex_time, eventNumber, ...
/blocks/<block>/offsets  (n_events+1,)    CSR offsets
/blocks/<block>/<field>  (n_items,)       one column per field
```

The schema is read from the source file, so new R2H5 branches are carried
through without touching the converter. Existing stores:

| sample | path | events | size |
|---|---|---|---|
| ttbar | `.../compact/ttbar` | 46,100 | 502 MB (from 5.8 GB) |
| vbf_hinv | `.../compact/vbf_hinv` | 10,800 | 96 MB (from 1.4 GB) |

## 2. Config reference

```yaml
model_name: my_model
model_dir: /pscratch/.../models/my_model

data:
  datasets:                      # one entry per sample; order is irrelevant
    - name: ttbar
      path: /global/cfs/.../compact/ttbar
      weight: 1.0                # relative loss weight
      fraction: 1.0              # subsample this fraction of events
      max_events: null           # hard cap
      files: null                # null = every file in the manifest
  balance: true                  # equalise each sample's loss contribution
  sample_onehot: false           # append a one-hot sample tag to event features
  target: HSvertex_time
  event_features: [HSvertex_reco_x, HSvertex_reco_y, HSvertex_reco_z]   # [] drops the branch
  split: {test_size: 0.2, val_split: 0.222222, random_state: 42}

  inputs:                        # any subset of the presets below
    cells:
      preset: lar_cells
      features: [eta, phi, barrel, layer, time, e, significance]   # optional subset
      max_items: 60
      min_items: 1               # events with fewer objects are dropped
      sort_by: [e, significance] # list = tie-break keys, most significant first
      descending: true
      select:
        - {field: layer, in: [1, 2, 3]}
        - {field: e, min: 1.0}
        - time_quality: {n_sigma: 3.0, vertex_sigma: 175.0,
                         calibration: sigma_only_test_calibration.txt,
                         apply_calibration: false}
      padding: {time: 0.0}       # per-feature, in physical units
      skip_normalization: [barrel, layer]
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

| preset | store block | features | default selection |
|---|---|---|---|
| `lar_cells` | `cells` | eta, phi, barrel, layer, time, e, significance | layer in {1,2,3} |
| `antikt4_jets` | `jets` | pt, eta, phi, width | `selected == 1` |
| `hs_tracks` | `tracks` | pt, eta, phi, d0, z0 | `is_good_from_hs == 1` |
| `hgtd_tracks` | `hgtd_tracks` | pt, eta, phi, d0, z0, time, time_res | `has_valid_time == 1` |

Logical names resolve against the fields the store actually has (`barrel` finds
either `Cell_Barrel` or `Cell_isEM_Barrel`). A name that resolves to nothing
raises — it is never silently filled with zeros.

Selection operators: `eq`, `ne`, `min`, `max`, `in`, `abs_max`, `abs_min`, plus
the special `time_quality` cut. Every selection is applied to all events at
once, not per event in a Python loop.

### Pooling

`attention`, `masked_average`, `average`, `max`, `sum`, `flatten`. The masked
variants and `type: transformer` encoders automatically turn on the block's
attention mask. `average` on a padded block pools the padding too — that is the
legacy behaviour for jets and tracks, kept for comparability, and switching to
`masked_average` is a one-line experiment.

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
