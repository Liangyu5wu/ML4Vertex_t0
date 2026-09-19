"""Write a compact store file.

The compact layout is the boundary between "where the data came from" and
"how the pipeline reads it", so the writer lives on its own: converters hand
it event-level columns and ragged blocks, and it is the only place that knows
about dtypes, compression and the attributes a store file carries.

    write_compact("out.h5",
                  events={"HSvertex_time": times, "eventNumber": numbers},
                  blocks={"cells": (columns_dict, offsets)},
                  attrs={"source_file": src})
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from .schema import check_column

SCHEMA_VERSION = 1

# One block: {field -> flat column}, plus (n_events + 1) CSR offsets.
Block = Tuple[Dict[str, np.ndarray], np.ndarray]


def downcast(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` in the narrowest dtype that holds its values.

    float64 -> float32 is not strictly lossless, but these are detector
    quantities with at most ~7 significant digits and the ROOT branches they
    come from are float32 to begin with.
    """
    kind = arr.dtype.kind
    if kind == "f":
        return arr.astype(np.float32)
    if kind == "b":
        return arr.astype(np.int8)
    if kind in "iu":
        if arr.size == 0:
            return arr.astype(np.int8)
        lo, hi = int(arr.min()), int(arr.max())
        for dtype in (np.int8, np.int16, np.int32):
            info = np.iinfo(dtype)
            if info.min <= lo and hi <= info.max:
                return arr.astype(dtype)
        return arr.astype(np.int64)
    raise TypeError(f"unsupported dtype {arr.dtype}")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _compression_kwargs(compression: Optional[str], complevel: int) -> dict:
    # lzf takes no options; gzip takes a level.  shuffle helps both.
    if not compression:
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": complevel, "shuffle": True}
    return {"compression": compression, "shuffle": True}


def write_compact(dst_path: str, events: Dict[str, np.ndarray],
                  blocks: Dict[str, Block], attrs: Optional[dict] = None,
                  block_attrs: Optional[Dict[str, dict]] = None,
                  compression: Optional[str] = "gzip", complevel: int = 4
                  ) -> Tuple[List[str], Dict[str, dict]]:
    """Write one compact file. Returns (content warnings, per-block summary)."""
    n_events = len(next(iter(events.values()))) if events else 0
    comp_kw = _compression_kwargs(compression, complevel)
    warnings: List[str] = []
    summary: Dict[str, dict] = {}

    with h5py.File(dst_path, "w") as fout:
        ev_group = fout.create_group("events")
        for name, values in events.items():
            col = downcast(np.ascontiguousarray(values))
            warnings.extend(check_column(f"events/{name}", col))
            ev_group.create_dataset(name, data=col, **comp_kw)
        ev_group.attrs["fields"] = json.dumps(list(events))

        blk_group = fout.create_group("blocks")
        for name, (columns, offsets) in blocks.items():
            offsets = np.asarray(offsets, dtype=np.int64)
            if len(offsets) != n_events + 1:
                raise ValueError(
                    f"block {name!r}: offsets has {len(offsets)} entries, "
                    f"expected n_events + 1 = {n_events + 1}")
            counts = np.diff(offsets)

            g = blk_group.create_group(name)
            g.create_dataset("offsets", data=offsets, **comp_kw)
            for field, values in columns.items():
                col = downcast(np.ascontiguousarray(values))
                if len(col) != offsets[-1]:
                    raise ValueError(
                        f"block {name!r} field {field!r}: {len(col)} items, "
                        f"offsets say {offsets[-1]}")
                warnings.extend(check_column(f"{name}/{field}", col))
                g.create_dataset(field, data=col, **comp_kw)

            g.attrs["fields"] = json.dumps(list(columns))
            g.attrs["n_items"] = int(offsets[-1])
            for key, value in (block_attrs or {}).get(name, {}).items():
                g.attrs[key] = value

            summary[name] = {
                "fields": list(columns),
                "n_items": int(offsets[-1]),
                "mean_per_event": float(counts.mean()) if n_events else 0.0,
                "max_per_event": int(counts.max()) if n_events else 0,
                **(block_attrs or {}).get(name, {}),
            }

        fout.attrs["schema_version"] = SCHEMA_VERSION
        fout.attrs["n_events"] = int(n_events)
        fout.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        fout.attrs["git_commit"] = git_commit()
        for key, value in (attrs or {}).items():
            fout.attrs[key] = value

    return warnings, summary


def write_manifest(output_dir: str, sample: str, files_meta: List[dict],
                   source: str, extra: Optional[dict] = None) -> str:
    """Write the directory-level manifest the reader uses to find everything."""
    first = os.path.join(output_dir, files_meta[0]["file"])
    with h5py.File(first, "r") as f:
        blocks = {name: json.loads(g.attrs["fields"]) for name, g in f["blocks"].items()}
        event_fields = json.loads(f["events"].attrs["fields"])

    manifest = {
        "sample": sample,
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "n_events": int(sum(m["n_events"] for m in files_meta)),
        "event_fields": event_fields,
        "blocks": blocks,
        "files": files_meta,
        **(extra or {}),
    }
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path
