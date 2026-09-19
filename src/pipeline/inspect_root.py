"""Report what is inside a ROOT file, as the first step before converting it.

Writing the ROOT -> compact reader needs to know which trees exist, which
branches they hold, whether a branch is one value per event or a jagged list,
and how big the lists get. This prints exactly that, plus the on-disk cost per
branch so it is obvious what is worth reading.

    python -m src.pipeline.inspect_root file.root
    python -m src.pipeline.inspect_root file.root --tree ntuple --entries 2000
    python -m src.pipeline.inspect_root file.root --grep Cell --values
"""

from __future__ import annotations

import argparse
import fnmatch
import re

import numpy as np


def _fmt(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def describe(path: str, tree_name: str = None, entries: int = 1000,
             pattern: str = None, show_values: bool = False) -> None:
    import awkward as ak

    from .schema import open_tree

    tree = open_tree(path, tree_name)
    print(f"file   : {path}")
    print(f"\nreading '{tree.name}': {tree.num_entries:,} entries, "
          f"{len(tree.keys()):,} branches, first {entries:,} entries sampled\n")

    keys = list(tree.keys())
    if pattern:
        keys = [k for k in keys
                if fnmatch.fnmatch(k, pattern) or re.search(pattern, k)]
        print(f"filtered to {len(keys)} branches matching {pattern!r}\n")

    header = (f"{'branch':<42}{'type':<26}{'depth':>6}{'per event':>22}"
              f"{'compressed':>13}")
    print(header)
    print("-" * len(header))

    total = 0.0
    for key in keys:
        branch = tree[key]
        interp = str(getattr(branch, "interpretation", branch.typename
                             if hasattr(branch, "typename") else "?"))
        size = getattr(branch, "compressed_bytes", 0)
        total += size
        try:
            arr = branch.array(entry_stop=min(entries, tree.num_entries))
        except Exception as exc:                       # unreadable branch type
            print(f"{key:<42}{interp[:25]:<26}{'?':>6}{'unreadable':>22}"
                  f"{_fmt(size):>13}   ({type(exc).__name__})")
            continue

        depth = arr.ndim
        if depth == 1:
            values = ak.to_numpy(arr)
            if values.dtype.kind in "fiub":
                span = f"{np.min(values):.3g} .. {np.max(values):.3g}"
            else:
                span = str(values.dtype)
            per_event = span
        else:
            counts = ak.to_numpy(ak.num(arr, axis=1))
            per_event = (f"n = {counts.mean():.1f} avg, {counts.max()} max")
        print(f"{key:<42}{interp[:25]:<26}{depth:>6}{per_event:>22}{_fmt(size):>13}")

        if show_values and depth > 1:
            flat = ak.to_numpy(ak.flatten(arr))
            if flat.dtype.kind in "fiu" and len(flat):
                print(f"{'':<42}values: {np.min(flat):.4g} .. {np.max(flat):.4g}, "
                      f"{len(np.unique(flat)):,} distinct")

    print("-" * len(header))
    if total:
        print(f"{'total (compressed, listed branches)':<74}{_fmt(total):>13}")
    print(f"file on disk: {_fmt(f.file.source.num_bytes)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--tree", default=None, help="tree name (default: the biggest)")
    p.add_argument("--entries", type=int, default=1000,
                   help="entries to sample for the per-event statistics")
    p.add_argument("--grep", default=None, help="only branches matching this pattern")
    p.add_argument("--values", action="store_true",
                   help="also show the value range of jagged branches")
    args = p.parse_args()
    describe(args.path, args.tree, args.entries, args.grep, args.values)


if __name__ == "__main__":
    main()
