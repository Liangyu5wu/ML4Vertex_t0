#!/usr/bin/env python
"""Walk one event from the store to the model input, printing every stage.

Selection, sorting, truncation, normalization, padding and masking each get
a line, in physical units before and normalized after, so a config switch
can be checked against what it actually did rather than against its name.

    python scripts/audit_inputs.py --config config/blocks/lar_hgtd.yaml
    python scripts/audit_inputs.py --config ... --event 7 --block hgtd_tracks
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.pipeline.assemble import AssemblySpec, prepare
from src.pipeline.blocks import load_block, source_fields
from src.pipeline.event_store import EventStore


def audit_block(store, bspec, event, norm, prepared_row, mask_row):
    """One block: raw -> selected -> sorted -> truncated -> normalized -> padded."""
    print(f"\n{'=' * 78}\nblock {bspec.name!r}  (source collection {bspec.source!r})")
    print(f"{'=' * 78}")

    fields = source_fields(store, bspec)
    raw = store.block(bspec.source, fields=sorted(set(fields.values())))
    lo, hi = raw.offsets[event], raw.offsets[event + 1]
    print(f"raw objects in this event: {hi - lo}")

    names = bspec.feature_names
    if bspec.selections:
        print(f"\nselection ({len(bspec.selections)} rule(s)):")
        for rule in bspec.selections:
            print(f"    {rule}")
    else:
        print("\nselection: none")

    block = load_block(store, bspec)          # selection + sort + truncate applied
    blo, bhi = block.offsets[event], block.offsets[event + 1]
    kept = bhi - blo
    print(f"  kept after selection and truncation: {kept} / {hi - lo}"
          f"   (max_items {bspec.max_items}"
          f"{', TRUNCATED' if kept == bspec.max_items else ''})")
    print(f"sort: {bspec.sort_by or 'none'}"
          f"{'  descending' if bspec.descending else '  ascending'}")

    stats = (norm or {}).get("blocks", {}).get(bspec.name)
    show = min(kept, 4)
    print(f"\nfirst {show} of {kept} objects, physical units:")
    print("      " + "".join(f"{n:>13s}" for n in names))
    for i in range(show):
        print(f"  #{i}  " + "".join(f"{block[n][blo + i]:13.4g}" for n in names))

    if stats is not None:
        print("\nnormalization fitted on the training split:")
        for j, n in enumerate(names):
            skipped = not bspec.normalize_flags[j]
            print(f"    {n:>13s}  mean {stats['mean'][j]:11.4g}  "
                  f"std {stats['std'][j]:11.4g}"
                  f"{'   (skip_normalization)' if skipped else ''}")

    print(f"\nmodel input row, shape {prepared_row.shape}, normalized:")
    print("      " + "".join(f"{n:>13s}" for n in names) + "   mask")
    rows = list(range(min(show, len(prepared_row))))
    if kept < len(prepared_row):                 # show the first padded slot too
        rows += [kept] if kept < len(prepared_row) else []
    for i in rows:
        tag = "real" if (mask_row is None or mask_row[i]) else "PAD "
        print(f"  #{i}  " + "".join(f"{prepared_row[i, j]:13.4g}"
                                    for j in range(len(names))) + f"   {tag}")

    if mask_row is not None:
        print(f"\nmask: {int(mask_row.sum())} true of {len(mask_row)}"
              f"   {'AGREES with kept' if int(mask_row.sum()) == kept else 'MISMATCH'}")
        pads = prepared_row[~mask_row]
        if len(pads):
            uniq = np.unique(pads, axis=0)
            print(f"padded rows: {len(pads)}, "
                  f"{'all identical' if len(uniq) == 1 else f'{len(uniq)} distinct'}"
                  f"  value {uniq[0] if len(uniq) == 1 else '(varies)'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--event", type=int, default=0, help="event index within the store")
    p.add_argument("--block", default=None, help="only this block")
    p.add_argument("--dataset", default=None, help="which dataset (default: first)")
    args = p.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    spec = AssemblySpec.from_config(cfg["data"])
    source = next((d for d in spec.datasets if d.name == args.dataset), spec.datasets[0])
    store = EventStore(source.path, files=source.files, sample=source.name)

    print(f"config   {args.config}")
    print(f"dataset  {source.name}  ({store.n_events} events)")
    print(f"event    {args.event}")
    print(f"target   {spec.target} = "
          f"{store.event_column(spec.target)[args.event]:.2f} ps")
    print(f"event features {spec.event_features} = "
          + ", ".join(f"{store.event_column(f)[args.event]:.4g}"
                      for f in spec.event_features))

    # The prepared tensors are keyed by split, so find where this event landed.
    data = prepare(spec, verbose=False)
    ev_no = store.event_column("event_number")[args.event]
    for split in ("train", "val", "test"):
        prov = data.provenance[split]
        hit = np.flatnonzero((prov["event_number"] == ev_no) &
                             (prov["dataset_id"] == spec.datasets.index(source)))
        if len(hit):
            row, found = int(hit[0]), split
            break
    else:
        raise SystemExit(f"event {args.event} was dropped by a min_items cut")
    print(f"landed in the {found!r} split at row {row}")

    for name, bspec in spec.blocks.items():
        if args.block and name != args.block:
            continue
        audit_block(store, bspec, args.event, data.norm,
                    data.inputs[found][f"{name}_input"][row],
                    data.inputs[found].get(f"{name}_mask", [None])[row]
                    if f"{name}_mask" in data.inputs[found] else None)


if __name__ == "__main__":
    main()
