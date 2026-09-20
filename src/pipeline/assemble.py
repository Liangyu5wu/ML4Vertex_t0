"""Assemble one or more event stores into model-ready tensors.

This is the single path from ``datasets:`` + ``inputs:`` in a YAML config to
padded arrays and ``tf.data`` pipelines, for any combination of input blocks
and any number of samples::

    datasets:
      - {name: ttbar,    path: .../store/ttbar,    weight: 1.0}
      - {name: vbf_hinv, path: .../store/vbf_hinv, weight: 1.0}

Samples are split independently and then concatenated, so every split keeps a
``dataset_id`` column and the test set can be scored per sample.  Scalers are
fitted on the pooled training split only, and configured padding values are
pushed through those scalers so padded slots stay where the config put them.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .blocks import BlockSpec, load_block, source_fields, spec_from_config
from .event_store import EventStore, RaggedBlock, pad_ragged

SPLITS = ("train", "val", "test")


@dataclass
class DatasetSource:
    """One sample: where it lives and how much of it to use."""
    name: str
    path: str
    files: Optional[List[str]] = None
    weight: float = 1.0
    fraction: float = 1.0
    max_events: Optional[int] = None

    @classmethod
    def from_config(cls, cfg: dict) -> "DatasetSource":
        cfg = dict(cfg)
        return cls(name=cfg.pop("name"), path=cfg.pop("path"),
                   files=cfg.pop("files", None), weight=float(cfg.pop("weight", 1.0)),
                   fraction=float(cfg.pop("fraction", 1.0)),
                   max_events=cfg.pop("max_events", None))


@dataclass
class SplitConfig:
    """Each sample is split independently, so every split holds the same
    fraction of every sample and the test set stays separable by sample."""
    test_size: float = 0.2
    val_split: float = 0.222222
    random_state: int = 42


@dataclass
class AssemblySpec:
    """Everything needed to turn stores into tensors."""
    datasets: List[DatasetSource]
    blocks: Dict[str, BlockSpec]
    event_features: List[str] = field(default_factory=list)
    target: str = "HSvertex_time"
    split: SplitConfig = field(default_factory=SplitConfig)
    balance: bool = False          # equalise the loss contribution of each sample
    sample_onehot: bool = False    # append a one-hot sample tag to the event features
    cache_dir: Optional[str] = None   # where prepared tensors are kept

    def fingerprint(self) -> str:
        """Hash of everything that changes the tensors this spec produces."""
        payload = {
            "datasets": [(d.name, os.path.abspath(d.path), d.files, d.fraction,
                          d.max_events) for d in self.datasets],
            "blocks": {name: {
                "source": b.source, "features": b.feature_names,
                "selections": b.selections, "sort_by": b.sort_by,
                "descending": b.descending, "max_items": b.max_items,
                "min_items": b.min_items, "pad_in": b.pad_in,
                "pads": b.pad_values, "normalize": b.normalize_flags,
                "emit_mask": b.emit_mask,
            } for name, b in sorted(self.blocks.items())},
            "event_features": self.event_features, "target": self.target,
            "split": (self.split.test_size, self.split.val_split,
                      self.split.random_state),
            "sample_onehot": self.sample_onehot,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True,
                                       default=str).encode()).hexdigest()[:16]

    @classmethod
    def from_config(cls, cfg: dict) -> "AssemblySpec":
        sources = [DatasetSource.from_config(d) for d in cfg["datasets"]]
        blocks = {name: spec_from_config(name, stanza)
                  for name, stanza in (cfg.get("inputs") or {}).items()}
        split = SplitConfig(**(cfg.get("split") or {}))
        return cls(datasets=sources, blocks=blocks,
                   event_features=list(cfg.get("event_features") or []),
                   target=cfg.get("target", "HSvertex_time"), split=split,
                   balance=bool(cfg.get("balance", False)),
                   sample_onehot=bool(cfg.get("sample_onehot", False)),
                   cache_dir=cfg.get("cache_dir") or os.environ.get(
                       "VERTEX_T0_CACHE",
                       "/pscratch/sd/l/liangyu/vertextiming/prepared_cache"))


# --------------------------------------------------------------------------
# Loading and splitting
# --------------------------------------------------------------------------

def _dataset_seed(base: int, name: str) -> int:
    """A per-sample seed that does not depend on the other samples present.

    Splitting each sample with its own seed means a sample's test events are
    the same whether it was trained alone or in a mixture, which is what makes
    "train on A, test on B" comparable across runs.  crc32 is used instead of
    hash() because the latter is randomised per interpreter session.
    """
    return (int(base) * 1000003 + zlib.crc32(name.encode())) % (2 ** 32)


def _split_indices(n: int, split: SplitConfig, rng: np.random.Generator
                   ) -> Dict[str, np.ndarray]:
    """Shuffle ``n`` events into train/val/test with the configured fractions."""
    idx = rng.permutation(n)
    n_test = int(round(n * split.test_size))
    rest = idx[n_test:]
    n_val = int(round(len(rest) * split.val_split))
    return {"train": rest[n_val:], "val": rest[:n_val], "test": idx[:n_test]}


@dataclass
class SampleData:
    """Per-sample arrays before splitting."""
    name: str
    weight: float
    blocks: Dict[str, RaggedBlock]
    events: Dict[str, np.ndarray]     # event-level columns incl. target/provenance

    @property
    def n_events(self) -> int:
        return len(self.events["target"])


def load_sample(source: DatasetSource, spec: AssemblySpec,
                verbose: bool = True) -> SampleData:
    """Read one store, apply block selections, and drop events that fail min_items."""
    store = EventStore(source.path, files=source.files, sample=source.name)

    # Read every source collection once, with the union of the fields the
    # specs using it need, then hand the same array to each of them.
    needed: Dict[str, set] = {}
    for bspec in spec.blocks.values():
        needed.setdefault(bspec.source, set()).update(
            source_fields(store, bspec).values())
    raw = {src: store.block(src, fields=sorted(fields))
           for src, fields in needed.items()}
    blocks = {name: load_block(store, bspec, raw=raw[bspec.source])
              for name, bspec in spec.blocks.items()}
    del raw

    keep = np.ones(store.n_events, dtype=bool)
    for name, bspec in spec.blocks.items():
        if bspec.min_items > 0:
            enough = blocks[name].counts >= bspec.min_items
            if verbose and not enough.all():
                print(f"    {name}: {int((~enough).sum())} event(s) below "
                      f"min_items={bspec.min_items}")
            keep &= enough

    if source.fraction < 1.0 or source.max_events is not None:
        rng = np.random.default_rng(_dataset_seed(spec.split.random_state, source.name))
        eligible = np.flatnonzero(keep)
        n_take = len(eligible)
        if source.fraction < 1.0:
            n_take = int(round(n_take * source.fraction))
        if source.max_events is not None:
            n_take = min(n_take, int(source.max_events))
        chosen = rng.choice(eligible, size=n_take, replace=False)
        keep = np.zeros_like(keep)
        keep[chosen] = True

    event_ids = np.flatnonzero(keep)
    events = {
        "target": store.event_column(spec.target)[event_ids].astype(np.float32),
        "event_number": store.event_column(
            "event_number" if "event_number" in store.event_fields else "eventNumber"
        )[event_ids],
        "file_index": store.file_index()[event_ids],
    }
    if spec.event_features:
        events["features"] = np.stack(
            [store.event_column(f)[event_ids] for f in spec.event_features],
            axis=1).astype(np.float32)
    else:
        events["features"] = np.zeros((len(event_ids), 0), dtype=np.float32)

    blocks = {name: blk.take_events(event_ids) for name, blk in blocks.items()}
    if verbose:
        print(f"  {source.name}: {len(event_ids)}/{store.n_events} events kept "
              f"from {len(store.files)} file(s)")
    return SampleData(source.name, source.weight, blocks, events)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def _fit_stats(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return mean.astype(np.float64), np.where(std > 0, std, 1.0).astype(np.float64)


def fit_normalization(spec: AssemblySpec, train_blocks: Dict[str, RaggedBlock],
                      train_event_features: np.ndarray,
                      event_feature_names: Optional[Sequence[str]] = None
                      ) -> Dict[str, dict]:
    """Fit per-feature mean/std on the training split only (real objects only)."""
    params: Dict[str, dict] = {"blocks": {}, "event": None}
    for name, bspec in spec.blocks.items():
        flat = train_blocks[name].stack(bspec.feature_names).astype(np.float64)
        if len(flat) == 0:
            raise ValueError(f"block {name!r} has no objects in the training split")
        mean, std = _fit_stats(flat)
        # Features flagged normalize=False pass through untouched.
        active = np.array(bspec.normalize_flags, dtype=bool)
        mean = np.where(active, mean, 0.0)
        std = np.where(active, std, 1.0)
        params["blocks"][name] = {"mean": mean, "std": std,
                                  "features": bspec.feature_names}
    if train_event_features.shape[1]:
        mean, std = _fit_stats(train_event_features.astype(np.float64))
        params["event"] = {"mean": mean, "std": std,
                           "features": list(event_feature_names
                                            if event_feature_names is not None
                                            else spec.event_features)}
    return params


def _padding_vector(bspec: BlockSpec, stats: dict) -> np.ndarray:
    """Padding values expressed in the same space as the normalized features."""
    pads = np.asarray(bspec.pad_values, dtype=np.float64)
    if bspec.pad_in == "literal":
        return pads.astype(np.float32)
    if bspec.pad_in != "normalized":
        raise ValueError(f"block {bspec.name!r}: pad_in must be "
                         f"'normalized' or 'literal', got {bspec.pad_in!r}")
    return ((pads - stats["mean"]) / stats["std"]).astype(np.float32)


def build_tensors(spec: AssemblySpec, blocks: Dict[str, RaggedBlock],
                  event_features: np.ndarray, norm: Dict[str, dict]
                  ) -> Dict[str, np.ndarray]:
    """Normalize, pad and mask one split; keys match the model input names."""
    out: Dict[str, np.ndarray] = {}
    for name, bspec in spec.blocks.items():
        stats = norm["blocks"][name]
        blk = blocks[name]
        normalized = RaggedBlock(
            name,
            {f: ((blk[f].astype(np.float64) - m) / s).astype(np.float32)
             for f, m, s in zip(bspec.feature_names, stats["mean"], stats["std"])},
            blk.offsets)
        padded, mask = pad_ragged(normalized, bspec.feature_names, bspec.max_items,
                                  _padding_vector(bspec, stats))
        out[f"{name}_input"] = padded
        if bspec.emit_mask:
            out[f"{name}_mask"] = mask
    if event_features.shape[1]:
        stats = norm["event"]
        out["event_input"] = ((event_features.astype(np.float64) - stats["mean"])
                              / stats["std"]).astype(np.float32)
    return out


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------

@dataclass
class PreparedData:
    """Model-ready arrays plus everything needed to interpret them."""
    inputs: Dict[str, Dict[str, np.ndarray]]      # split -> input name -> array
    targets: Dict[str, np.ndarray]
    weights: Dict[str, np.ndarray]
    provenance: Dict[str, Dict[str, np.ndarray]]  # split -> dataset_id/event_number/...
    norm: Dict[str, dict]
    dataset_names: List[str]
    spec: AssemblySpec
    event_feature_names: List[str] = field(default_factory=list)

    def n_events(self, split: str) -> int:
        return len(self.targets[split])

    def dataset_mask(self, split: str, name: str) -> np.ndarray:
        return self.provenance[split]["dataset_id"] == self.dataset_names.index(name)

    def input_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {k: v.shape[1:] for k, v in self.inputs["train"].items()}

    def merged(self, splits: Sequence[str] = SPLITS) -> Tuple[
            Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
        """Concatenate several splits -- used when scoring a whole sample."""
        inputs = {k: np.concatenate([self.inputs[s][k] for s in splits])
                  for k in self.inputs[splits[0]]}
        target = np.concatenate([self.targets[s] for s in splits])
        prov = {k: np.concatenate([self.provenance[s][k] for s in splits])
                for k in self.provenance[splits[0]]}
        return inputs, target, prov


def _cache_paths(spec: AssemblySpec) -> Optional[Tuple[str, str]]:
    if not spec.cache_dir:
        return None
    key = spec.fingerprint()
    base = os.path.join(spec.cache_dir, key)
    return base + ".npz", base + ".pkl"


def load_cached(spec: AssemblySpec, verbose: bool = True) -> Optional[PreparedData]:
    """Return the prepared tensors for this spec if they are already on disk."""
    paths = _cache_paths(spec)
    if not paths or not all(os.path.exists(p) for p in paths):
        return None
    npz_path, pkl_path = paths
    with open(pkl_path, "rb") as fh:
        meta = pickle.load(fh)
    arrays = np.load(npz_path)
    inputs = {s: {} for s in SPLITS}
    targets, weights, provenance = {}, {}, {s: {} for s in SPLITS}
    for key in arrays.files:
        split, _, name = key.partition("/")
        if name == "__target__":
            targets[split] = arrays[key]
        elif name == "__weight__":
            weights[split] = arrays[key]
        elif name.startswith("prov:"):
            provenance[split][name[5:]] = arrays[key]
        else:
            inputs[split][name] = arrays[key]
    if verbose:
        print(f"  prepared tensors from cache: {npz_path}")
    return PreparedData(inputs, targets, weights, provenance, meta["norm"],
                        meta["dataset_names"], spec, meta["event_feature_names"])


def save_cached(spec: AssemblySpec, data: "PreparedData", verbose: bool = True) -> None:
    paths = _cache_paths(spec)
    if not paths:
        return
    npz_path, pkl_path = paths
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    arrays = {}
    for split in SPLITS:
        for name, value in data.inputs[split].items():
            arrays[f"{split}/{name}"] = value
        arrays[f"{split}/__target__"] = data.targets[split]
        arrays[f"{split}/__weight__"] = data.weights[split]
        for name, value in data.provenance[split].items():
            arrays[f"{split}/prov:{name}"] = value
    # Both files go in atomically, and the sidecar goes first: a sweep starts
    # several trials at once on a cold cache, and load_cached() gates on the
    # npz, so the npz must be the last thing to appear.
    tmp_pkl = f"{pkl_path}.{os.getpid()}.tmp"
    with open(tmp_pkl, "wb") as fh:
        pickle.dump({"norm": data.norm, "dataset_names": data.dataset_names,
                     "event_feature_names": data.event_feature_names}, fh)
    os.replace(tmp_pkl, pkl_path)
    tmp = f"{npz_path}.{os.getpid()}.tmp.npz"     # np.savez appends .npz itself
    np.savez(tmp[:-4], **arrays)
    os.replace(tmp, npz_path)
    if verbose:
        print(f"  prepared tensors cached: {npz_path} "
              f"({os.path.getsize(npz_path) / 1e9:.1f} GB)")


def prepare(spec: AssemblySpec, verbose: bool = True,
            norm: Optional[Dict[str, dict]] = None,
            use_cache: bool = True) -> PreparedData:
    """Load every sample, split, fit scalers on train, and build padded tensors.

    Pass ``norm`` to reuse scalers fitted during training -- that is what makes
    evaluating an existing model on a new sample independent of the sample it
    was trained on.
    """
    # Reusing scalers means the caller is scoring a different sample; that is a
    # one-off, so it does not go through the cache.  Keep the answer now: `norm`
    # itself is reassigned below once the scalers are fitted.
    reuse_norm = norm is not None
    if use_cache and not reuse_norm:
        cached = load_cached(spec, verbose=verbose)
        if cached is not None:
            return cached

    if verbose:
        print(f"Loading {len(spec.datasets)} dataset(s): "
              f"{[d.name for d in spec.datasets]}")
    samples = [load_sample(src, spec, verbose=verbose) for src in spec.datasets]
    names = [s.name for s in samples]

    per_sample_idx = [
        _split_indices(s.n_events, spec.split,
                       np.random.default_rng(_dataset_seed(spec.split.random_state,
                                                           s.name)))
        for s in samples]

    # Per split: gather blocks/events from every sample, then concatenate.
    split_blocks: Dict[str, Dict[str, RaggedBlock]] = {}
    split_events: Dict[str, Dict[str, np.ndarray]] = {}
    for split in SPLITS:
        blocks: Dict[str, RaggedBlock] = {}
        for name in spec.blocks:
            parts = [s.blocks[name].take_events(idx[split])
                     for s, idx in zip(samples, per_sample_idx)]
            blocks[name] = _concat_blocks(name, parts)
        split_blocks[split] = blocks

        ev: Dict[str, np.ndarray] = {}
        for key in ("target", "event_number", "file_index", "features"):
            ev[key] = np.concatenate([s.events[key][idx[split]]
                                      for s, idx in zip(samples, per_sample_idx)])
        ev["dataset_id"] = np.concatenate(
            [np.full(len(idx[split]), i, dtype=np.int32)
             for i, idx in enumerate(per_sample_idx)])
        split_events[split] = ev

    # The sample tag is appended here rather than written back onto the spec:
    # prepare() has to stay callable more than once with the same spec.
    event_feature_names = list(spec.event_features)
    if spec.sample_onehot and len(samples) > 1:
        for split in SPLITS:
            ids = split_events[split]["dataset_id"]
            onehot = np.eye(len(samples), dtype=np.float32)[ids]
            split_events[split]["features"] = np.concatenate(
                [split_events[split]["features"], onehot], axis=1)
        event_feature_names += [f"is_{n}" for n in names]

    if norm is None:
        norm = fit_normalization(spec, split_blocks["train"],
                                 split_events["train"]["features"],
                                 event_feature_names)
    elif verbose:
        print("  using saved normalization parameters (no refit)")

    inputs, targets, weights, provenance = {}, {}, {}, {}
    for split in SPLITS:
        inputs[split] = build_tensors(spec, split_blocks[split],
                                      split_events[split]["features"], norm)
        targets[split] = split_events[split]["target"]
        weights[split] = _sample_weights(spec, samples, split_events[split]["dataset_id"])
        provenance[split] = {k: split_events[split][k]
                             for k in ("dataset_id", "event_number", "file_index")}

    if verbose:
        for split in SPLITS:
            counts = ", ".join(
                f"{n}={int((split_events[split]['dataset_id'] == i).sum())}"
                for i, n in enumerate(names))
            print(f"  {split:5s}: {len(targets[split]):6d} events  ({counts})")
        print("  inputs: " + ", ".join(
            f"{k}{tuple(v.shape[1:])}" for k, v in inputs["train"].items()))

    prepared = PreparedData(inputs, targets, weights, provenance, norm, names,
                            spec, event_feature_names)
    if use_cache and not reuse_norm:
        save_cached(spec, prepared, verbose=verbose)
    return prepared


def _concat_blocks(name: str, parts: Sequence[RaggedBlock]) -> RaggedBlock:
    if len(parts) == 1:
        return parts[0]
    fields = list(parts[0].columns)
    columns = {f: np.concatenate([p[f] for p in parts]) for f in fields}
    offsets = [np.zeros(1, dtype=np.int64)]
    running = 0
    for p in parts:
        offsets.append(p.offsets[1:] + running)
        running += p.n_items
    return RaggedBlock(name, columns, np.concatenate(offsets))


def _sample_weights(spec: AssemblySpec, samples: Sequence[SampleData],
                    dataset_id: np.ndarray) -> np.ndarray:
    """Per-event loss weight: configured weight, optionally size-balanced."""
    weights = np.array([s.weight for s in samples], dtype=np.float32)
    if spec.balance:
        counts = np.array([max((dataset_id == i).sum(), 1)
                           for i in range(len(samples))], dtype=np.float64)
        weights = weights * (counts.mean() / counts).astype(np.float32)
    out = weights[dataset_id]
    return (out / out.mean()).astype(np.float32) if len(out) else out


def make_tf_dataset(prepared: PreparedData, split: str, batch_size: int,
                    shuffle: bool = False, use_weights: bool = True,
                    shuffle_seed: Optional[int] = None):
    """Build a ``tf.data.Dataset`` yielding ``(inputs, target[, weight])``."""
    import tensorflow as tf

    inputs = {k: v for k, v in prepared.inputs[split].items()}
    target = prepared.targets[split]
    weights = prepared.weights[split]
    unweighted = np.allclose(weights, 1.0) if len(weights) else True

    if use_weights and not unweighted:
        ds = tf.data.Dataset.from_tensor_slices((inputs, target, weights))
    else:
        ds = tf.data.Dataset.from_tensor_slices((inputs, target))
    if shuffle:
        ds = ds.shuffle(min(len(target), 10000), seed=shuffle_seed,
                        reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def save_norm(prepared: PreparedData, path: str) -> None:
    """Persist the fitted scalers next to the model so new samples reuse them."""
    import pickle
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(prepared.norm, fh)


def load_norm(path: str) -> Dict[str, dict]:
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)
