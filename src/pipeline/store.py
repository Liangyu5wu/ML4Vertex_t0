"""Read access to the compact event store written by :mod:`src.pipeline.compact`.

A store is a directory of compact HDF5 files plus a ``manifest.json``.  Files
are concatenated transparently, so callers see one flat event index and a
single ragged array per block::

    store = CompactStore("/.../compact/ttbar")
    times = store.event_column("HSvertex_time")          # (n_events,)
    cells = store.block("cells", ["Cell_e", "Cell_eta"]) # ragged, lazy read

Columns are only read from disk when requested, so a model that uses HGTD
tracks alone never pays for the calorimeter cells.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import h5py
import numpy as np


def ragged_arange(counts: np.ndarray, starts: Optional[np.ndarray] = None
                  ) -> np.ndarray:
    """Concatenated ranges, without a Python loop over events.

    ``counts=[3, 2]`` gives ``[0, 1, 2, 0, 1]``; with ``starts=[10, 20]`` it
    gives ``[10, 11, 12, 20, 21]``.  Every ragged gather in this module is
    expressed with this, which keeps them O(total items) in numpy rather than
    O(events) in Python.
    """
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    out = np.arange(total, dtype=np.int64) - offsets
    if starts is not None:
        out += np.repeat(np.asarray(starts, dtype=np.int64), counts)
    return out


class RaggedBlock:
    """One collection in CSR layout: flat columns plus per-event offsets."""

    def __init__(self, name: str, columns: Dict[str, np.ndarray], offsets: np.ndarray):
        self.name = name
        self.columns = columns
        self.offsets = offsets

    @property
    def n_events(self) -> int:
        return len(self.offsets) - 1

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def n_items(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, field: str) -> np.ndarray:
        return self.columns[field]

    def stack(self, fields: Sequence[str]) -> np.ndarray:
        """Return a flat ``(n_items, len(fields))`` float32 view of the columns."""
        return np.stack([self.columns[f].astype(np.float32, copy=False)
                         for f in fields], axis=1)

    def event_index(self) -> np.ndarray:
        """Return ``(n_items,)`` giving the event each object belongs to."""
        return np.repeat(np.arange(self.n_events, dtype=np.int64), self.counts)

    def select(self, mask: np.ndarray) -> "RaggedBlock":
        """Keep the objects flagged in the flat boolean ``mask``, rebuilding offsets."""
        if mask.shape != (self.n_items,):
            raise ValueError(f"mask has shape {mask.shape}, expected ({self.n_items},)")
        counts = np.bincount(self.event_index()[mask], minlength=self.n_events)
        offsets = np.zeros(self.n_events + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return RaggedBlock(self.name,
                           {f: col[mask] for f, col in self.columns.items()},
                           offsets)

    def truncate(self, max_items: int) -> "RaggedBlock":
        """Keep at most ``max_items`` objects per event (sort the block first)."""
        counts = np.minimum(self.counts, max_items)
        if np.array_equal(counts, self.counts):
            return self
        idx = ragged_arange(counts, self.offsets[:-1])
        offsets = np.zeros(self.n_events + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return RaggedBlock(self.name,
                           {f: col[idx] for f, col in self.columns.items()},
                           offsets)

    def take_events(self, event_ids: np.ndarray) -> "RaggedBlock":
        """Return a block holding only ``event_ids`` (in the given order)."""
        starts = self.offsets[event_ids]
        counts = (self.offsets[event_ids + 1] - starts).astype(np.int64)
        idx = ragged_arange(counts, starts)
        offsets = np.zeros(len(event_ids) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return RaggedBlock(self.name,
                           {f: col[idx] for f, col in self.columns.items()},
                           offsets)

    def __repr__(self) -> str:
        return (f"RaggedBlock({self.name!r}, events={self.n_events}, "
                f"items={self.n_items}, fields={sorted(self.columns)})")


class CompactStore:
    """A directory of compact files, presented as one dataset."""

    def __init__(self, path: str, files: Optional[Iterable[str]] = None,
                 sample: Optional[str] = None):
        self.path = os.path.abspath(path)
        manifest_path = os.path.join(self.path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"{manifest_path} not found -- run src.pipeline.compact on the raw files first")
        with open(manifest_path) as fh:
            self.manifest = json.load(fh)

        listed = [m["file"] for m in self.manifest["files"]]
        self.files = list(files) if files is not None else listed
        missing = [f for f in self.files
                   if not os.path.exists(os.path.join(self.path, f))]
        if missing:
            raise FileNotFoundError(f"missing compact files in {self.path}: {missing}")

        self.sample = sample or self.manifest.get("sample", os.path.basename(self.path))
        self._n_per_file = []
        for name in self.files:
            with h5py.File(os.path.join(self.path, name), "r") as f:
                self._n_per_file.append(int(f.attrs["n_events"]))
        self.n_events = int(sum(self._n_per_file))
        self._cache: Dict[str, np.ndarray] = {}

    # -- metadata ---------------------------------------------------------
    @property
    def block_names(self) -> List[str]:
        return sorted(self.manifest["blocks"])

    def block_fields(self, block: str) -> List[str]:
        return list(self.manifest["blocks"][block])

    @property
    def event_fields(self) -> List[str]:
        return list(self.manifest["event_fields"])

    def file_index(self) -> np.ndarray:
        """Return ``(n_events,)`` with the index into ``self.files`` per event."""
        return np.repeat(np.arange(len(self.files), dtype=np.int32),
                         self._n_per_file)

    # -- data -------------------------------------------------------------
    def event_column(self, field: str) -> np.ndarray:
        """Concatenated event-level column across all files."""
        if field in self._cache:
            return self._cache[field]
        parts = []
        for name in self.files:
            with h5py.File(os.path.join(self.path, name), "r") as f:
                parts.append(f["events"][field][:])
        out = np.concatenate(parts) if len(parts) > 1 else parts[0]
        self._cache[field] = out
        return out

    def block(self, block: str, fields: Optional[Sequence[str]] = None) -> RaggedBlock:
        """Read one block, optionally restricted to ``fields``."""
        if block not in self.manifest["blocks"]:
            raise KeyError(f"block {block!r} not in store; have {self.block_names}")
        fields = list(fields) if fields is not None else self.block_fields(block)
        unknown = [f for f in fields if f not in self.block_fields(block)]
        if unknown:
            raise KeyError(f"block {block!r} has no field(s) {unknown}; "
                           f"available: {self.block_fields(block)}")

        col_parts: Dict[str, List[np.ndarray]] = {f: [] for f in fields}
        offset_parts, running = [], 0
        for name in self.files:
            with h5py.File(os.path.join(self.path, name), "r") as f:
                g = f["blocks"][block]
                off = g["offsets"][:]
                offset_parts.append(off[1:] + running)
                running += int(off[-1])
                for field in fields:
                    col_parts[field].append(g[field][:])
        offsets = np.concatenate([np.zeros(1, dtype=np.int64)] + offset_parts)
        columns = {f: (np.concatenate(p) if len(p) > 1 else p[0])
                   for f, p in col_parts.items()}
        return RaggedBlock(block, columns, offsets)

    def __repr__(self) -> str:
        return (f"CompactStore({self.sample!r}, events={self.n_events}, "
                f"files={len(self.files)}, blocks={self.block_names})")


def pad_ragged(block: RaggedBlock, fields: Sequence[str], max_items: int,
               pad_values: Sequence[float]) -> "tuple[np.ndarray, np.ndarray]":
    """Densify a ragged block to ``(n_events, max_items, len(fields))``.

    Objects beyond ``max_items`` are dropped (sort the block first if order
    matters).  Returns the padded array and a ``(n_events, max_items)`` boolean
    mask that is True for real objects.
    """
    n_events = block.n_events
    n_fields = len(fields)
    pad = np.asarray(pad_values, dtype=np.float32)
    if pad.shape != (n_fields,):
        raise ValueError(f"pad_values has shape {pad.shape}, expected ({n_fields},)")

    out = np.broadcast_to(pad, (n_events, max_items, n_fields)).copy()
    mask = np.zeros((n_events, max_items), dtype=bool)

    counts = np.minimum(block.counts, max_items)
    # Flat index of the kept objects, and where each lands in the dense array.
    keep = ragged_arange(counts, block.offsets[:-1])
    rows = np.repeat(np.arange(n_events, dtype=np.int64), counts)
    cols = ragged_arange(counts)

    flat = block.stack(fields)[keep]
    out[rows, cols] = flat
    mask[rows, cols] = True
    return out, mask
