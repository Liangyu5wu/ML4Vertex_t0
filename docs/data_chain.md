# From ROOT ntuples to model inputs

How a vertex-time training set is built, stage by stage, with the numbers each
stage actually produces. Two samples are in use: a ttbar sample and a VBF
H→invisible sample, both at ⟨μ⟩ = 200.

```
ROOT ntuple  ──ingest_root.py──▶  event store  ──blocks.py──▶  selected ragged
   138 GB                            23 GB                      arrays
                                                                   │
                        model  ◀──block_model.py──  padded tensors ─┘
                                                     + tf.data      assemble.py
```

Two properties are worth stating up front, because they explain most of the
design:

- **The ingest keeps everything.** Physics selection happens downstream, in
  YAML. Changing a cut, a sort order or a multiplicity limit never means
  re-reading the ROOT files.
- **Each expensive stage is keyed by a fingerprint of the settings that
  produced it.** The same settings reuse the previous output; changed settings
  rebuild it. Nothing is versioned by hand.

---

## Stage 1 — ROOT ntuple → event store

`python -m src.pipeline.ingest_root --input-dir <root> --output-dir <store>
--sample ttbar --shards 8 --workers 8`

| sample | ROOT | → | event store | events | ratio |
|---|---|---|---|---|---|
| ttbar | 43 files, 83.1 GB | | 8 files, 15.3 GB | 199,700 | 5.4× |
| vbf_hinv | 33 files, 55.4 GB | | 8 files, 8.1 GB | 112,400 | 6.8× |

The output is not one HDF5 file per ROOT file. Files are merged into 8 shards
per sample so that later stages open 8 handles rather than 43.

### Layout

Each collection is stored as a ragged (CSR) block: one offsets array plus one
compressed column per field.

```
/events/<field>          (n_events,)      one value per event
/blocks/<block>/offsets  (n_events+1,)    where each event's objects start
/blocks/<block>/<field>  (n_items,)       one column per field, gzip-4
```

A model that reads only HGTD tracks therefore never decompresses the
calorimeter cells.

| block | fields | objects/event (mean, median, max) |
|---|---|---|
| `cells` | 15 | 607, 573, 2791 |
| `tracks` | 39 | 890, 886, 1443 |
| `jets_emtopo` | 9 | 9.5, 9, 44 |
| `jets_pflow` | 9 | 10.2, 9, 66 |
| `reco_vertices` | 8 | 102, 102, 145 |
| `truth_vertices` | 5 | 201, 201, 270 |

Event-level fields: `event_number`, `run_number`, `mu`, `weight`,
`n_reco_vtx`, `n_truth_vtx`, `truth_vtx_{x,y,z,time}`,
`reco_vtx_{x,y,z,time,time_res,sum_pt2}`.

### What the ingest computes

Field names in the store are ours, not the ntuple's; the mapping is recorded
in `manifest.json`, so a production that renames a branch is absorbed by an
alias rather than by editing downstream code. Three things are derived.

**1. Time-of-flight-corrected cell time.** The ntuple's `Cell_time` is
referenced to a particle travelling from the detector origin. The hard-scatter
vertex is not at the origin, so the path length differs:

```python
d_vtx    = |r_cell - r_recoHS|          # vertex to cell
d_origin = |r_cell|                     # origin to cell
time_tof = Cell_time - (d_vtx - d_origin) / c        # ps
```

This uses the **reconstructed** vertex position, which is available in data.
No truth information enters the store.

**2. Calorimeter region code**, from the ntuple's per-cell detector flags:

| code | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| region | EM barrel | EM endcap | FCal | HEC | Tile |
| share of cells | 27.6% | 53.9% | 3.9% | 4.2% | 10.6% |

**3. Per-track quantities** relative to the reconstructed hard-scatter vertex:
`on_hs_vertex`, `dz_hs`.

### The one place the ingest does cut

Tracks. The ntuple carries ~1992 per event and the store keeps 890:

```
on reco HS vertex  |  has a valid HGTD time  |  |z0 - z_HS| < 3 mm
```

which is a union of three loose, data-available conditions. For reference,
37.6 tracks/event are on the HS vertex and 803/event carry an HGTD time.

### Validation

`ingest_root` rejects a branch whose per-event counts disagree with the rest
of its collection, which is how `RecoVtx_isPU` was caught: the producer never
clears the vector, so it accumulates across events. Constant and all-zero
columns are reported rather than silently stored.

---

## Stage 2 — event store → selected ragged arrays

Driven entirely by the `inputs:` stanza of the config. Each entry names a
**preset** (a starting point defined in `src/pipeline/blocks.py`) and may
override any part of it.

```yaml
inputs:
  cells:
    preset: lar_cells
    max_items: 120
    min_items: 1
    encoder: {units: [256, 128, 64], dropout: 0.0, pooling: attention}
```

Five blocks are in use. Their model-visible features, selections and limits:

| block | features | selection | keep | order |
|---|---|---|---|---|
| `cells` | eta, phi, region, layer, time, e, significance | region ∈ {EMB, EME}, layer ∈ {1,2,3}, \|significance\| > 4, e > 1 GeV | 120 | (e, significance) ↓ |
| `jets_emtopo` | pt, eta, phi, width | none | 15 | pt ↓ |
| `tracks` | pt, eta, phi, d0, z0 | `on_hs_vertex == 1` | 50 | pt ↓ |
| `hgtd_tracks` | pt, eta, phi, d0, z0, time, time_res | `has_valid_time == 1`, 2.4 < \|eta\| < 4.0, \|dz_hs\| < 2 mm | 55 | pt ↓ |
| `vertices` | z, sum_pt2, time, time_res, is_hs, has_valid_time | none | 10 | sum_pt2 ↓ |

Each preset also loads auxiliary fields — positions, quality flags, truth-match
counts — that selections may use without them becoming model inputs.

Four points about this stage:

- **`reco_vertices/is_hs` is a reconstruction flag, not truth.** It was
  checked against the ordering and equals "highest sum_pt²" in 100% of
  events, so it is available in data and adds nothing beyond the sort order.
- **No truth in any selection.** Jets carry a truth-match count, and it is
  deliberately not cut on: truth matching is the only handle that identifies a
  jet as hard-scatter, and it does not exist in data.
- **Sorting needs a tie-break.** Cell energies are quantised; about 10% of
  cells share an energy exactly with another cell in the same event, so the
  cell preset sorts on `(e, significance)` rather than energy alone.
- **A name that resolves to nothing raises.** Fields are matched against the
  store through alias lists, and an unresolved name is an error, never a
  silent column of zeros.

All of a block's rules are combined into one boolean mask and applied in a
single vectorised pass over the whole sample. There are no per-event Python
loops anywhere in the pipeline.

---

## Stage 3 — ragged arrays → padded tensors

`src/pipeline/assemble.py`, in this order. The order matters and is the point
of the stage.

### 3.1 Split, per sample

```yaml
split: {test_size: 0.2, val_split: 0.125, random_state: 42}
```

giving **70 / 10 / 20**. Each sample is split with its own seed, derived from
`random_state` and the sample name, so a sample's test events are identical
whether it was trained on alone or in a mixture — which is what makes
"train on A, score on B" comparable across runs.

| split | events | of which ttbar / VBF |
|---|---|---|
| train | 218,388 | 139,744 / 78,644 |
| val | 31,199 | 19,964 / 11,235 |
| test | 62,397 | 39,927 / 22,470 |

### 3.2 Equalise the samples

```yaml
resample: oversample      # none | oversample | undersample
```

Applied to the training split only, drawing the smaller samples up to the
largest: 218,388 → **279,488**, with ttbar and VBF at 139,744 each.

### 3.3 Fit scalers, on the training split only

Per-feature mean and standard deviation, computed from **real objects only**
— never padding — and **after truncation**, so the statistics describe
exactly the objects the model will see. Features flagged
`skip_normalization` pass through untouched; `region` and `layer` are
categorical and keep their literal codes.

`norm_params.pkl` is saved beside the model and reused by
`evaluate_blocks.py`, so scoring a new sample applies the scalers the model
was trained with rather than refitting on the new sample.

### 3.4 Normalize, then pad

Normalizing before padding is what keeps padding out of the statistics. The
configured padding value is then pushed through the fitted scaler
(`pad_in: normalized`) or written literally (`pad_in: literal`, the cell
default, where `0.0` lands on the mean).

### 3.5 Emit the mask

A block emits a boolean mask when its encoder needs one — attention pooling,
masked average, a transformer or the selection head. Masked pooling excludes
padded slots from both the weights and the denominator.

Padding has been verified inert: perturbing every padded slot of a batch
changes the model output by exactly 0.0.

### Result

For `lar_hgtd`, the training split:

| tensor | shape | real fraction | mean/limit | events at the limit |
|---|---|---|---|---|
| `cells_input` | (279488, 120, 7) | 84.3% | 101.2 / 120 | 42.5% |
| `jets_emtopo_input` | (279488, 15, 4) | 56.0% | 8.4 / 15 | 4.1% |
| `tracks_input` | (279488, 50, 5) | 59.4% | 29.7 / 50 | 11.4% |
| `hgtd_tracks_input` | (279488, 55, 7) | 51.1% | 28.1 / 55 | 3.7% |
| `vertices_input` | (279488, 10, 6) | 100% | 10.0 / 10 | 100% |
| `event_input` | (279488, 3) | — | — | — |

plus one `<block>_mask` of shape (279488, N) per block. `event_input` carries
the reconstructed vertex position `(x, y, z)`. The regression target is
`truth_vtx_time`, in picoseconds, left unnormalized so the loss and every
reported number are in physical units.

### 3.6 Feed

The training split is permuted once at this point. Without it the split runs
all of one sample and then all of the next — one transition in 218k rows —
which a 10k shuffle buffer cannot bridge, leaving every batch drawn from a
single sample. Validation and test are read in order; their order enters no
metric.

Batches are then formed by shuffling row indices and gathering, rather than
slicing events apart and reassembling them: 4.3 s an epoch becomes 0.6 s, and
the shuffle covers every index rather than a 10k window.

---

## Caching and reuse

| artefact | keyed by | cost | reuse |
|---|---|---|---|
| event store | ingest settings fingerprint | hours | per sample, ~forever |
| prepared tensors | `AssemblySpec.fingerprint()` | ~7 min | every run with the same data config |

The assembly fingerprint covers the datasets, every block's fields,
selections, sort order, limits, padding and mask flag, the event features, the
target, the split and the resampling mode — that is, everything that changes
the tensors. It deliberately does **not** cover the model: architecture,
optimizer and loss changes reuse the cache, which is what makes a
hyper-parameter sweep cost one training per trial and nothing else.

---

## What is not in the chain

- **No manual time calibration.** The model reads
  time-of-flight-corrected cell times and learns the rest. `calibration_data/`
  is used only by the traditional σ-weighted baseline that the model is
  compared against.
- **No target normalization.** Predictions come out in picoseconds.
- **No truth anywhere except the target.** Selections, sorting and features
  are all quantities available in data.

## Reproducing it

```bash
source setup.sh

python -m src.pipeline.ingest_root \
    --input-dir  /global/cfs/cdirs/m2616/liangyu/vertextiming/root/ttbar \
    --output-dir /global/cfs/cdirs/m2616/liangyu/vertextiming/store/ttbar \
    --sample ttbar --shards 8 --workers 8

python scripts/train_blocks.py --config config/blocks/lar_hgtd.yaml

# one event's path through every stage above, in physical and normalized units
python scripts/audit_inputs.py --config config/blocks/lar_hgtd.yaml --event 3
```

Config reference: [`../src/pipeline/README.md`](../src/pipeline/README.md).
