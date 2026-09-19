"""Convert raw R2H5 HDF5 output into a compact, ragged event store.

Raw files store every collection as a dense ``(n_events, n_slots)`` structured
array of float64.  In the current samples cells occupy 144 of 1000 slots,
tracks 35 of 200 and jets 1.8 of 50, so most of the file is padding and every
value carries twice the precision it needs.

The compact store drops invalid slots and keeps one 1-D column per field in a
ragged (CSR-style) layout::

    /events/<field>            (n_events,)      event-level scalars
    /blocks/<block>/offsets    (n_events + 1,)  int64
    /blocks/<block>/<field>    (n_items,)       one column per field

Columns are downcast (float64 -> float32, bool/small ints -> int8/int16) and
compressed.  The schema is discovered from the source file, so branches added
on the R2H5 side flow through without code changes here.

Usage::

    python -m src.pipeline.compact \
        --input-dir ../Vertex_timing_HGTD_w_LAr \
        --output-dir /global/cfs/cdirs/m2616/liangyu/vertextiming/compact/ttbar \
        --sample ttbar
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

SCHEMA_VERSION = 1

# Source datasets that hold per-event scalars rather than a collection.
EVENT_TABLE_CANDIDATES = ("HSvertex",)

# Source dataset name -> block name used downstream.
BLOCK_NAMES = {
    "cells": "cells",
    "jets": "jets",
    "tracks": "tracks",
    "tracks_HGTD": "hgtd_tracks",
}


def _downcast(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` in the narrowest dtype that holds its values losslessly.

    float64 -> float32 is not strictly lossless, but these are detector
    quantities with at most ~7 significant digits; float32 keeps ~1e-7 relative
    precision, far below any measurement resolution here.
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


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def discover_layout(f: h5py.File) -> Tuple[str, List[str]]:
    """Return (event_table_name, [collection dataset names]) for an open file."""
    event_table = None
    collections = []
    for name, obj in f.items():
        if not isinstance(obj, h5py.Dataset) or obj.dtype.names is None:
            continue
        if obj.ndim == 1:
            if event_table is None or name in EVENT_TABLE_CANDIDATES:
                event_table = name
        elif obj.ndim == 2:
            collections.append(name)
        else:
            raise ValueError(f"dataset {name} has unsupported ndim {obj.ndim}")
    if event_table is None:
        raise ValueError("no 1-D structured dataset found to use as the event table")
    return event_table, sorted(collections)


def convert_file(src_path: str, dst_path: str, compression: str = "gzip",
                 complevel: int = 4, verbose: bool = True) -> Dict:
    """Convert one raw file to the compact layout. Returns a summary dict."""
    t0 = time.time()
    summary: Dict[str, object] = {
        "source": os.path.abspath(src_path),
        "output": os.path.abspath(dst_path),
    }
    # lzf takes no options; gzip takes a level.  shuffle helps both.
    if not compression:
        comp_kw = {}
    elif compression == "gzip":
        comp_kw = {"compression": "gzip", "compression_opts": complevel, "shuffle": True}
    else:
        comp_kw = {"compression": compression, "shuffle": True}

    with h5py.File(src_path, "r") as fin, h5py.File(dst_path, "w") as fout:
        event_table, collections = discover_layout(fin)
        n_events = fin[event_table].shape[0]
        summary["n_events"] = int(n_events)

        # ---- event-level scalars -------------------------------------------
        ev_group = fout.create_group("events")
        raw_events = fin[event_table][:]
        for field in raw_events.dtype.names:
            col = _downcast(np.ascontiguousarray(raw_events[field]))
            ev_group.create_dataset(field, data=col, **comp_kw)
        ev_group.attrs["fields"] = json.dumps(list(raw_events.dtype.names))
        ev_group.attrs["source_dataset"] = event_table
        del raw_events

        # ---- ragged collections --------------------------------------------
        blk_group = fout.create_group("blocks")
        block_info = {}
        for ds_name in collections:
            block = BLOCK_NAMES.get(ds_name, ds_name)
            raw = fin[ds_name][:]
            n_slots = raw.shape[1]

            if "valid" in raw.dtype.names:
                valid = raw["valid"].astype(bool)
            else:
                if verbose:
                    print(f"  [{block}] no 'valid' field -- keeping all slots")
                valid = np.ones(raw.shape, dtype=bool)

            counts = valid.sum(axis=1).astype(np.int64)
            offsets = np.zeros(n_events + 1, dtype=np.int64)
            np.cumsum(counts, out=offsets[1:])
            n_items = int(offsets[-1])

            g = blk_group.create_group(block)
            g.create_dataset("offsets", data=offsets, **comp_kw)
            fields = [n for n in raw.dtype.names if n != "valid"]
            for field in fields:
                # Boolean-mask indexing flattens row-major, i.e. already in
                # event order -- this is the ragged concatenation we want.
                col = _downcast(raw[field][valid])
                g.create_dataset(field, data=col, **comp_kw)
            g.attrs["fields"] = json.dumps(fields)
            g.attrs["source_dataset"] = ds_name
            g.attrs["n_slots_raw"] = int(n_slots)
            g.attrs["n_items"] = n_items

            block_info[block] = {
                "source_dataset": ds_name,
                "fields": fields,
                "n_items": n_items,
                "n_slots_raw": int(n_slots),
                "mean_per_event": float(counts.mean()) if n_events else 0.0,
                "max_per_event": int(counts.max()) if n_events else 0,
            }
            if verbose:
                print(f"  [{block:12s}] {n_items:9d} objects  "
                      f"mean/event {counts.mean():6.1f}  max {counts.max():4d}  "
                      f"(raw slots {n_slots})")
            del raw, valid

        summary["blocks"] = block_info
        fout.attrs["schema_version"] = SCHEMA_VERSION
        fout.attrs["source_file"] = os.path.abspath(src_path)
        fout.attrs["n_events"] = int(n_events)
        fout.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        fout.attrs["git_commit"] = _git_commit()
        fout.attrs["selection"] = "valid == True (no physics selection applied)"

    src_mb = os.path.getsize(src_path) / 1e6
    dst_mb = os.path.getsize(dst_path) / 1e6
    summary.update({"src_mb": src_mb, "dst_mb": dst_mb,
                    "ratio": src_mb / dst_mb if dst_mb else 0.0,
                    "seconds": time.time() - t0})
    if verbose:
        print(f"  {src_mb:8.1f} MB -> {dst_mb:7.1f} MB  "
              f"({summary['ratio']:.1f}x)  in {summary['seconds']:.1f}s")
    return summary


def _sorted_inputs(input_dir: str, pattern: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    return paths


def convert_directory(input_dir: str, output_dir: str, sample: str,
                      pattern: str = "output_*.h5", limit: Optional[int] = None,
                      compression: str = "gzip", complevel: int = 4,
                      overwrite: bool = False) -> Dict:
    """Convert every raw file in ``input_dir`` and write a manifest."""
    inputs = _sorted_inputs(input_dir, pattern)
    if limit is not None:
        inputs = inputs[:limit]
    if not inputs:
        raise FileNotFoundError(f"no files matching {pattern} in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Converting {len(inputs)} file(s) from {input_dir}")

    files_meta = []
    for i, src in enumerate(inputs):
        dst = os.path.join(output_dir, os.path.basename(src))
        if os.path.exists(dst) and not overwrite:
            print(f"[{i + 1}/{len(inputs)}] {os.path.basename(src)} -- exists, skipping")
            with h5py.File(dst, "r") as f:
                files_meta.append({"file": os.path.basename(dst),
                                   "n_events": int(f.attrs["n_events"]),
                                   "skipped": True})
            continue
        print(f"[{i + 1}/{len(inputs)}] {os.path.basename(src)}")
        summary = convert_file(src, dst, compression=compression, complevel=complevel)
        files_meta.append({
            "file": os.path.basename(dst),
            "n_events": summary["n_events"],
            "src_mb": round(summary["src_mb"], 1),
            "dst_mb": round(summary["dst_mb"], 1),
            "seconds": round(summary["seconds"], 1),
        })

    with h5py.File(os.path.join(output_dir, files_meta[0]["file"]), "r") as f:
        blocks = {name: json.loads(g.attrs["fields"]) for name, g in f["blocks"].items()}
        event_fields = json.loads(f["events"].attrs["fields"])

    manifest = {
        "sample": sample,
        "schema_version": SCHEMA_VERSION,
        "source_dir": os.path.abspath(input_dir),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "n_events": int(sum(m["n_events"] for m in files_meta)),
        "event_fields": event_fields,
        "blocks": blocks,
        "files": files_meta,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    total_src = sum(m.get("src_mb", 0.0) for m in files_meta)
    total_dst = sum(m.get("dst_mb", 0.0) for m in files_meta)
    print(f"\n{sample}: {manifest['n_events']} events in {len(files_meta)} file(s)")
    if total_dst:
        print(f"  {total_src:.0f} MB -> {total_dst:.0f} MB ({total_src / total_dst:.1f}x)")
    print(f"  manifest: {manifest_path}")
    return manifest


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, help="directory of raw R2H5 h5 files")
    p.add_argument("--output-dir", required=True, help="destination for the compact store")
    p.add_argument("--sample", required=True, help="sample name recorded in the manifest")
    p.add_argument("--pattern", default="output_*.h5")
    p.add_argument("--limit", type=int, default=None, help="convert only the first N files")
    p.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    p.add_argument("--complevel", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    convert_directory(
        args.input_dir, args.output_dir, args.sample,
        pattern=args.pattern, limit=args.limit,
        compression=None if args.compression == "none" else args.compression,
        complevel=args.complevel, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
