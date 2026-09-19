"""Declarative input blocks: the single place where a model's inputs are defined.

A *block* is one collection fed to the network (calorimeter cells, jets, LAr
tracks, HGTD tracks, ...).  Its spec says which store fields to read, how to
select and order the objects, how many to keep, how to pad, and what encoder
to build on top.  Everything downstream -- loading, normalization, tf.data
assembly, model construction -- is driven by these specs, so adding an input
means writing a preset or a YAML stanza, not editing loaders and trainers.

Logical feature names (``pt``, ``eta``, ``time``, ...) are resolved against the
field names actually present in the store, which lets one spec serve samples
whose branches were renamed between R2H5 versions.  A name that resolves to
nothing is an error -- never a silently-zero column.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .event_store import EventStore, RaggedBlock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Feature:
    """One column of a block.

    ``sources`` lists candidate field names in the event store, tried in
    order.  ``pad`` is the value used for padded slots, expressed in the
    original (physical) units.
    """
    name: str
    sources: Tuple[str, ...]
    pad: float = 0.0
    normalize: bool = True

    def resolve(self, available: Sequence[str]) -> str:
        for candidate in self.sources:
            if candidate in available:
                return candidate
        raise KeyError(
            f"feature {self.name!r} matches none of {list(self.sources)}; "
            f"store provides {sorted(available)}")


@dataclass
class BlockSpec:
    """How to turn one store collection into a padded model input."""
    name: str
    source: str
    features: List[Feature]
    aux: List[Feature] = field(default_factory=list)
    selections: List[dict] = field(default_factory=list)
    # A single feature name or a list of them, most significant first.  Cell
    # energies are quantised -- ~10% of cells share an energy with another cell
    # in the same event -- so a secondary key is what makes the truncation to
    # ``max_items`` deterministic rather than dependent on the sort algorithm.
    sort_by: Optional[object] = None
    descending: bool = True
    max_items: int = 30
    min_items: int = 0
    # "normalized": configured pad value is pushed through the fitted scaler.
    # "literal":    configured pad value is written as-is into normalized data
    #               (legacy cell behaviour; harmless when a mask is used).
    pad_in: str = "normalized"
    encoder: dict = field(default_factory=dict)
    emit_mask: bool = False

    @property
    def feature_names(self) -> List[str]:
        return [f.name for f in self.features]

    @property
    def pad_values(self) -> List[float]:
        return [f.pad for f in self.features]

    @property
    def normalize_flags(self) -> List[bool]:
        return [f.normalize for f in self.features]

    def all_features(self) -> List[Feature]:
        """Model features plus the extra columns needed by selections/sorting."""
        seen = {f.name: f for f in self.features}
        for f in self.aux:
            seen.setdefault(f.name, f)
        return list(seen.values())


# --------------------------------------------------------------------------
# Presets: the physics defaults, defined once.
# --------------------------------------------------------------------------

def _cells_preset() -> BlockSpec:
    return BlockSpec(
        name="cells",
        source="cells",
        features=[
            Feature("eta", ("Cell_eta",), pad=0.0),
            Feature("phi", ("Cell_phi",), pad=0.0),
            # Older R2H5 output called this Cell_Barrel.
            Feature("barrel", ("Cell_Barrel", "Cell_isEM_Barrel"), pad=-1.0, normalize=False),
            Feature("layer", ("Cell_layer",), pad=0.0, normalize=False),
            Feature("time", ("Cell_time_TOF_corrected",), pad=0.0),
            Feature("e", ("Cell_e",), pad=0.0),
            Feature("significance", ("Cell_significance",), pad=0.0),
        ],
        aux=[
            Feature("x", ("Cell_x",)), Feature("y", ("Cell_y",)), Feature("z", ("Cell_z",)),
        ],
        selections=[{"field": "layer", "in": [1, 2, 3]}],
        sort_by=["e", "significance"],
        max_items=60,
        min_items=1,
        pad_in="literal",
        emit_mask=True,
        encoder={"units": [128, 64, 32], "dropout": 0.1, "activation": "relu",
                 "batch_norm": False, "pooling": "attention"},
    )


def _jets_preset() -> BlockSpec:
    return BlockSpec(
        name="jets",
        source="jets",
        features=[
            Feature("pt", ("AntiKt4EMTopoJets_pt",), pad=-1.0),
            Feature("eta", ("AntiKt4EMTopoJets_eta",), pad=-999.0),
            Feature("phi", ("AntiKt4EMTopoJets_phi",), pad=-999.0),
            Feature("width", ("AntiKt4EMTopoJets_width",), pad=-1.0),
        ],
        aux=[Feature("selected", ("AntiKt4EMTopoJets_selected",))],
        selections=[{"field": "selected", "eq": 1}],
        sort_by="pt",
        max_items=7,
        encoder={"units": [64, 32], "dropout": 0.1, "activation": "relu",
                 "batch_norm": True, "pooling": "masked_average"},
    )


def _tracks_preset() -> BlockSpec:
    return BlockSpec(
        name="tracks",
        source="tracks",
        features=[
            Feature("pt", ("Track_pt",), pad=-1.0),
            Feature("eta", ("Track_eta",), pad=-999.0),
            Feature("phi", ("Track_phi",), pad=-999.0),
            Feature("d0", ("Track_d0",), pad=-999.0),
            Feature("z0", ("Track_z0",), pad=-999.0),
        ],
        aux=[Feature("is_good_from_hs", ("Track_isGoodFromHS",))],
        selections=[{"field": "is_good_from_hs", "eq": 1}],
        sort_by="pt",
        max_items=30,
        encoder={"units": [64, 32], "dropout": 0.1, "activation": "relu",
                 "batch_norm": True, "pooling": "masked_average"},
    )


def _hgtd_tracks_preset() -> BlockSpec:
    return BlockSpec(
        name="hgtd_tracks",
        source="hgtd_tracks",
        features=[
            Feature("pt", ("Track_pt",), pad=-1.0),
            Feature("eta", ("Track_eta",), pad=-999.0),
            Feature("phi", ("Track_phi",), pad=-999.0),
            Feature("d0", ("Track_d0",), pad=-999.0),
            Feature("z0", ("Track_z0",), pad=-999.0),
            Feature("time", ("Track_time",), pad=0.0),
            Feature("time_res", ("Track_timeRes",), pad=-999.0),
        ],
        aux=[Feature("has_valid_time", ("Track_hasValidTime",)),
             Feature("is_good_from_hs", ("Track_isGoodFromHS",)),
             Feature("has_vtx", ("TrackIsGoodHasVtx",))],
        selections=[{"field": "has_valid_time", "eq": 1}],
        sort_by="pt",
        max_items=30,
        encoder={"units": [64, 32], "dropout": 0.1, "activation": "relu",
                 "batch_norm": True, "pooling": "masked_average"},
    )


PRESETS = {
    "lar_cells": _cells_preset,
    "antikt4_jets": _jets_preset,
    "hs_tracks": _tracks_preset,
    "hgtd_tracks": _hgtd_tracks_preset,
}


def spec_from_config(name: str, cfg: dict) -> BlockSpec:
    """Build a BlockSpec from a YAML stanza.

    ``cfg`` names a preset and may override any scalar field, the feature list
    (by logical name), per-feature padding, selections and the encoder::

        cells:
          preset: lar_cells
          features: [eta, phi, layer, time, e, significance]
          max_items: 120
          select: [{field: e, min: 1.0}]
          encoder: {units: [128, 64], pooling: attention}
    """
    cfg = dict(cfg or {})
    preset_name = cfg.pop("preset", name)
    if preset_name not in PRESETS:
        raise KeyError(f"unknown block preset {preset_name!r}; have {sorted(PRESETS)}")
    spec = PRESETS[preset_name]()
    spec = replace(spec, name=name)

    if "features" in cfg:
        wanted = cfg.pop("features")
        by_name = {f.name: f for f in spec.features + spec.aux}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise KeyError(f"block {name!r}: unknown feature(s) {missing}; "
                           f"preset {preset_name!r} defines {sorted(by_name)}")
        spec.features = [copy.deepcopy(by_name[w]) for w in wanted]

    for feat_name, pad in (cfg.pop("padding", {}) or {}).items():
        for f in spec.features:
            if f.name == feat_name:
                f.pad = float(pad)
                break
        else:
            raise KeyError(f"block {name!r}: padding given for unused feature {feat_name!r}")

    for feat_name in (cfg.pop("skip_normalization", []) or []):
        for f in spec.features:
            if f.name == feat_name:
                f.normalize = False

    if "select" in cfg:
        spec.selections = list(cfg.pop("select") or [])
    for extra in (cfg.pop("select_extra", []) or []):
        spec.selections.append(extra)
    if "encoder" in cfg:
        spec.encoder = {**spec.encoder, **(cfg.pop("encoder") or {})}

    mask_was_explicit = "emit_mask" in cfg
    for key in ("source", "sort_by", "descending", "max_items", "min_items",
                "pad_in", "emit_mask"):
        if key in cfg:
            setattr(spec, key, cfg.pop(key))
    if cfg:
        raise KeyError(f"block {name!r}: unrecognised key(s) {sorted(cfg)}")

    # A mask is only useful if something consumes it, but forgetting to ask for
    # one is an easy mistake, so derive it from the encoder unless set explicitly.
    if not mask_was_explicit:
        pooling = spec.encoder.get("pooling", "average")
        spec.emit_mask = pooling in ("attention", "masked_average") or \
            spec.encoder.get("type") == "transformer"
    return spec


# --------------------------------------------------------------------------
# Loading: store -> selected, sorted RaggedBlock with logical column names
# --------------------------------------------------------------------------

_ENERGY_BINS = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0])
_SIGMA_KEYS = {(1, 1): "EMB1_sigma", (1, 2): "EMB2_sigma", (1, 3): "EMB3_sigma",
               (0, 1): "EME1_sigma", (0, 2): "EME2_sigma", (0, 3): "EME3_sigma"}


def load_calibration(filename: str) -> Dict[str, List[float]]:
    """Read a ``calibration_data/*.txt`` table (resolved against the repo root)."""
    path = filename if os.path.isabs(filename) else \
        os.path.join(REPO_ROOT, "calibration_data", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"calibration file not found: {path}")
    table: Dict[str, List[float]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, values = line.split(":", 1)
            table[key.strip()] = [float(v) for v in values.split(",")]
    return table


def _time_quality_mask(cols: Dict[str, np.ndarray], opts: dict) -> np.ndarray:
    """Keep cells whose |t| is within n_sigma of the combined vertex+cell sigma.

    sigma_total = sqrt(sigma_vertex^2 + sigma_cell(detector, E)^2); with
    ``apply_calibration`` the per-bin mean offset is subtracted from the time
    first.  Vectorised over every cell in the sample at once.
    """
    calib = load_calibration(opts.get("calibration", "sigma_only_test_calibration.txt"))
    barrel = cols["barrel"].astype(np.int32)
    layer = cols["layer"].astype(np.int32)
    energy = cols["e"].astype(np.float64)
    time = cols["time"].astype(np.float64).copy()

    bin_idx = np.clip(np.searchsorted(_ENERGY_BINS, energy, side="right") - 1, 0, 6)
    sigma = np.full(len(time), 1000.0)
    offset = np.zeros(len(time))
    apply_calib = bool(opts.get("apply_calibration", False))

    for (b, l), key in _SIGMA_KEYS.items():
        sel = (barrel == b) & (layer == l)
        if not sel.any():
            continue
        sigma[sel] = np.take(np.asarray(calib[key]), bin_idx[sel], mode="clip")
        if apply_calib:
            param_key = f"EM{'B' if b == 1 else 'E'}{l}_params"
            if param_key not in calib:
                raise KeyError(f"{param_key} missing from calibration file "
                               f"(needed when apply_calibration is true)")
            offset[sel] = np.take(np.asarray(calib[param_key]), bin_idx[sel], mode="clip")

    if apply_calib:
        time -= offset
    sigma_total = np.sqrt(float(opts.get("vertex_sigma", 175.0)) ** 2 + sigma ** 2)
    return np.abs(time) <= float(opts.get("n_sigma", 3.0)) * sigma_total


def _selection_mask(cols: Dict[str, np.ndarray], rule: dict) -> np.ndarray:
    """Turn one selection stanza into a flat boolean mask over objects."""
    rule = dict(rule)
    if "time_quality" in rule:
        return _time_quality_mask(cols, rule["time_quality"] or {})

    fname = rule.pop("field", None)
    if fname is None:
        raise KeyError(f"selection {rule} has no 'field' and no known special form")
    if fname not in cols:
        raise KeyError(f"selection on unknown field {fname!r}; have {sorted(cols)}")
    col = cols[fname]
    mask = np.ones(len(col), dtype=bool)
    for op, value in rule.items():
        if op == "eq":
            mask &= col == value
        elif op == "ne":
            mask &= col != value
        elif op == "min":
            mask &= col >= value
        elif op == "max":
            mask &= col <= value
        elif op == "in":
            mask &= np.isin(col, value)
        elif op == "abs_max":
            mask &= np.abs(col) <= value
        elif op == "abs_min":
            mask &= np.abs(col) >= value
        else:
            raise KeyError(f"unknown selection operator {op!r} on field {fname!r}")
    return mask


def load_block(store: EventStore, spec: BlockSpec) -> RaggedBlock:
    """Read, select and sort one block; columns come back under logical names."""
    available = store.block_fields(spec.source)
    wanted = spec.all_features()
    mapping = {f.name: f.resolve(available) for f in wanted}

    raw = store.block(spec.source, fields=sorted(set(mapping.values())))
    block = RaggedBlock(spec.name,
                        {logical: raw[source] for logical, source in mapping.items()},
                        raw.offsets)

    for rule in spec.selections:
        block = block.select(_selection_mask(block.columns, rule))

    if spec.sort_by:
        keys = [spec.sort_by] if isinstance(spec.sort_by, str) else list(spec.sort_by)
        missing = [k for k in keys if k not in block.columns]
        if missing:
            raise KeyError(f"block {spec.name!r}: sort_by {missing} not among the "
                           f"loaded fields ({sorted(block.columns)})")
        sign = -1.0 if spec.descending else 1.0
        # lexsort applies the last key first, so keys go least- to most-significant.
        lex_keys = [sign * block[k].astype(np.float64) for k in reversed(keys)]
        order = np.lexsort(tuple(lex_keys) + (block.event_index(),))
        block = RaggedBlock(block.name,
                            {f: c[order] for f, c in block.columns.items()},
                            block.offsets)

    # Truncate here rather than at padding time so that normalization statistics
    # are fitted on exactly the objects the model will see.
    block = block.truncate(spec.max_items)

    # Drop the aux columns; only model features continue downstream.
    keep = set(spec.feature_names)
    return RaggedBlock(block.name,
                       {f: c for f, c in block.columns.items() if f in keep},
                       block.offsets)
