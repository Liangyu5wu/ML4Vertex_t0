"""Convert dense R2H5 HDF5 output into the event store.

R2H5 writes every collection as a dense ``(n_events, n_slots)`` structured
array of float64: cells occupy 144 of 1000 slots, tracks 35 of 200 and jets
1.8 of 50, so most of the file is padding at twice the precision needed. This
reader drops the invalid slots and hands the ragged result to
:mod:`src.pipeline.store_writer`, which is also what the ROOT reader writes through.

    python -m src.pipeline.ingest_h5 \
        --input-dir ../Vertex_timing_HGTD_w_LAr \
        --output-dir /global/cfs/cdirs/m2616/liangyu/vertextiming/compact/ttbar \
        --sample ttbar
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from .schema import collect_h5, compare
from .store_writer import Block, write_compact, write_manifest

# Source datasets that hold per-event scalars rather than a collection.
EVENT_TABLE_CANDIDATES = ("HSvertex",)

# Source dataset name -> block name used downstream.
BLOCK_NAMES = {
    "cells": "cells",
    "jets": "jets",
    "tracks": "tracks",
    "tracks_HGTD": "hgtd_tracks",
}


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


def convert_file(src_path: str, dst_path: str, compression: Optional[str] = "gzip",
                 complevel: int = 4, verbose: bool = True) -> Dict:
    """Convert one dense file to the compact layout. Returns a summary dict."""
    t0 = time.time()

    with h5py.File(src_path, "r") as fin:
        event_table, collections = discover_layout(fin)
        raw_events = fin[event_table][:]
        events = {name: np.ascontiguousarray(raw_events[name])
                  for name in raw_events.dtype.names}
        n_events = len(raw_events)
        del raw_events

        blocks: Dict[str, Block] = {}
        block_attrs: Dict[str, dict] = {}
        for ds_name in collections:
            block = BLOCK_NAMES.get(ds_name, ds_name)
            raw = fin[ds_name][:]

            if "valid" in raw.dtype.names:
                valid = raw["valid"].astype(bool)
            else:
                if verbose:
                    print(f"  [{block}] no 'valid' field -- keeping all slots")
                valid = np.ones(raw.shape, dtype=bool)

            counts = valid.sum(axis=1).astype(np.int64)
            offsets = np.zeros(n_events + 1, dtype=np.int64)
            np.cumsum(counts, out=offsets[1:])
            # Boolean-mask indexing flattens row-major, i.e. already in event
            # order -- that is exactly the ragged concatenation wanted here.
            columns = {name: raw[name][valid] for name in raw.dtype.names
                       if name != "valid"}
            blocks[block] = (columns, offsets)
            block_attrs[block] = {"source_dataset": ds_name,
                                  "n_slots_raw": int(raw.shape[1])}
            if verbose:
                saturated = int((counts >= raw.shape[1]).sum())
                note = (f"  <-- {saturated} event(s) at the {raw.shape[1]}-slot limit"
                        if saturated else "")
                print(f"  [{block:12s}] {int(offsets[-1]):9d} objects  "
                      f"mean/event {counts.mean():6.1f}  max {counts.max():4d}{note}")
            del raw, valid

    warnings, block_info = write_compact(
        dst_path, events, blocks,
        attrs={"source_file": os.path.abspath(src_path),
               "selection": "valid == True (no physics selection applied)"},
        block_attrs=block_attrs, compression=compression, complevel=complevel)

    src_mb = os.path.getsize(src_path) / 1e6
    dst_mb = os.path.getsize(dst_path) / 1e6
    summary = {"source": os.path.abspath(src_path), "output": os.path.abspath(dst_path),
               "n_events": int(n_events), "blocks": block_info, "warnings": warnings,
               "src_mb": src_mb, "dst_mb": dst_mb,
               "ratio": src_mb / dst_mb if dst_mb else 0.0,
               "seconds": time.time() - t0}
    if verbose:
        print(f"  {src_mb:8.1f} MB -> {dst_mb:7.1f} MB  "
              f"({summary['ratio']:.1f}x)  in {summary['seconds']:.1f}s")
    return summary


def convert_directory(input_dir: str, output_dir: str, sample: str,
                      pattern: str = "output_*.h5", limit: Optional[int] = None,
                      compression: Optional[str] = "gzip", complevel: int = 4,
                      overwrite: bool = False) -> Dict:
    """Convert every dense file in ``input_dir`` and write a manifest."""
    inputs = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if limit is not None:
        inputs = inputs[:limit]
    if not inputs:
        raise FileNotFoundError(f"no files matching {pattern} in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Converting {len(inputs)} file(s) from {input_dir}")

    # Productions drift between files: check before writing anything, so a
    # renamed or missing branch is a loud failure rather than a silent column.
    schemas = [collect_h5(p) for p in inputs]
    differences = compare(schemas)
    if differences:
        print("\nInput files do not share one schema:")
        for d in differences:
            print(f"  ! {d}")
        raise ValueError("inconsistent input schema; convert the groups separately "
                         "or fix the production")
    print(f"schema: {len(schemas[0].columns)} fields, consistent across all files")

    files_meta: List[dict] = []
    all_warnings: set = set()
    for i, src in enumerate(inputs):
        dst = os.path.join(output_dir, os.path.basename(src))
        if os.path.exists(dst) and not overwrite:
            print(f"[{i + 1}/{len(inputs)}] {os.path.basename(src)} -- exists, skipping")
            with h5py.File(dst, "r") as f:
                files_meta.append({"file": os.path.basename(dst),
                                   "n_events": int(f.attrs["n_events"])})
            continue
        print(f"[{i + 1}/{len(inputs)}] {os.path.basename(src)}")
        summary = convert_file(src, dst, compression=compression, complevel=complevel)
        all_warnings.update(summary["warnings"])
        files_meta.append({"file": os.path.basename(dst),
                           "n_events": summary["n_events"],
                           "src_mb": round(summary["src_mb"], 1),
                           "dst_mb": round(summary["dst_mb"], 1),
                           "seconds": round(summary["seconds"], 1)})

    if all_warnings:
        print(f"\ncontent warnings ({len(all_warnings)}):")
        for w in sorted(all_warnings):
            print(f"  ! {w}")

    manifest_path = write_manifest(output_dir, sample, files_meta,
                                   source=os.path.abspath(input_dir))
    total_src = sum(m.get("src_mb", 0.0) for m in files_meta)
    total_dst = sum(m.get("dst_mb", 0.0) for m in files_meta)
    n_events = sum(m["n_events"] for m in files_meta)
    print(f"\n{sample}: {n_events:,} events in {len(files_meta)} file(s)")
    if total_dst:
        print(f"  {total_src:.0f} MB -> {total_dst:.0f} MB ({total_src / total_dst:.1f}x)")
    print(f"  manifest: {manifest_path}")
    return {"manifest": manifest_path, "n_events": n_events, "files": files_meta}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, help="directory of dense R2H5 files")
    p.add_argument("--output-dir", required=True, help="destination event store")
    p.add_argument("--sample", required=True, help="sample name recorded in the manifest")
    p.add_argument("--pattern", default="output_*.h5")
    p.add_argument("--limit", type=int, default=None, help="convert only the first N files")
    p.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    p.add_argument("--complevel", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    convert_directory(args.input_dir, args.output_dir, args.sample,
                      pattern=args.pattern, limit=args.limit,
                      compression=None if args.compression == "none" else args.compression,
                      complevel=args.complevel, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
