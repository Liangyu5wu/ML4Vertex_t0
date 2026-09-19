"""Schema and content checks for input files.

Productions drift: a branch gets renamed, a new file in the same directory
carries an extra collection, a variable that should vary turns out to be
constant. Every one of those has already bitten this analysis once -- the cell
"barrel" feature was a constant zero for months because a rename went
unnoticed -- so the checks live here and run as part of the conversion rather
than being something to remember.

    python -m src.pipeline.schema file1.root file2.root      # compare files
    python -m src.pipeline.schema store/*.h5 --values        # + content checks
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FileSchema:
    """What a file contains: one entry per branch/field."""
    path: str
    kind: str                                   # "root" | "raw_h5" | "compact"
    columns: Dict[str, str] = field(default_factory=dict)   # name -> type label
    n_events: int = 0

    @property
    def names(self) -> set:
        return set(self.columns)


# --------------------------------------------------------------------------
# Collecting
# --------------------------------------------------------------------------

def collect_root(path: str, tree: Optional[str] = None) -> FileSchema:
    """Branch inventory of a ROOT file (TTree or RNTuple)."""
    import uproot

    f = uproot.open(path)
    trees = {k.split(";")[0]: f[k] for k, cls in f.classnames().items()
             if cls in ("TTree", "ROOT::RNTuple")}
    if not trees:
        raise ValueError(f"{path}: no TTree/RNTuple found")
    name = tree or max(trees, key=lambda k: trees[k].num_entries)
    t = trees[name]
    columns = {}
    for key in t.keys():
        branch = t[key]
        columns[key] = str(getattr(branch, "typename", "?"))
    return FileSchema(path, "root", columns, int(t.num_entries))


def collect_h5(path: str) -> FileSchema:
    """Field inventory of a raw R2H5 file or an event store file."""
    import h5py

    columns: Dict[str, str] = {}
    with h5py.File(path, "r") as f:
        if "blocks" in f and "events" in f:                 # event store
            kind = "compact"
            n_events = int(f.attrs["n_events"])
            for name in f["events"]:
                columns[f"events/{name}"] = str(f["events"][name].dtype)
            for block in f["blocks"]:
                for name in f["blocks"][block]:
                    if name == "offsets":
                        continue
                    columns[f"{block}/{name}"] = str(f["blocks"][block][name].dtype)
        else:                                               # raw R2H5 output
            kind = "raw_h5"
            n_events = 0
            for key, obj in f.items():
                if obj.dtype.names is None:
                    continue
                n_events = max(n_events, obj.shape[0])
                for name in obj.dtype.names:
                    columns[f"{key}/{name}"] = str(obj.dtype[name])
    return FileSchema(path, kind, columns, n_events)


def collect(path: str, tree: Optional[str] = None) -> FileSchema:
    return collect_root(path, tree) if path.endswith(".root") else collect_h5(path)


# --------------------------------------------------------------------------
# Comparing and requiring
# --------------------------------------------------------------------------

def compare(schemas: Sequence[FileSchema]) -> List[str]:
    """Return human-readable differences between files (empty == consistent)."""
    if len(schemas) < 2:
        return []
    reference = schemas[0]
    problems: List[str] = []
    for other in schemas[1:]:
        missing = reference.names - other.names
        extra = other.names - reference.names
        if missing:
            problems.append(f"{os.path.basename(other.path)}: missing "
                            f"{sorted(missing)[:8]}"
                            + (" ..." if len(missing) > 8 else ""))
        if extra:
            problems.append(f"{os.path.basename(other.path)}: has extra "
                            f"{sorted(extra)[:8]}"
                            + (" ..." if len(extra) > 8 else ""))
        for name in sorted(reference.names & other.names):
            if reference.columns[name] != other.columns[name]:
                problems.append(f"{os.path.basename(other.path)}: {name} is "
                                f"{other.columns[name]}, was "
                                f"{reference.columns[name]} in "
                                f"{os.path.basename(reference.path)}")
    return problems


def require(schema: FileSchema, required: Iterable[str],
            aliases: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, str]:
    """Resolve required names against the file, raising with what is available.

    ``aliases`` maps a logical name to the candidate branch names to try, so a
    production that renamed a branch is handled explicitly instead of silently
    producing a column of zeros.
    """
    aliases = aliases or {}
    resolved, missing = {}, []
    for name in required:
        for candidate in aliases.get(name, [name]):
            if candidate in schema.columns:
                resolved[name] = candidate
                break
        else:
            missing.append(name)
    if missing:
        raise KeyError(
            f"{os.path.basename(schema.path)} is missing required input(s) "
            f"{missing}. Tried {[aliases.get(m, [m]) for m in missing]}. "
            f"The file provides {sorted(schema.columns)[:20]}"
            + (" ..." if len(schema.columns) > 20 else ""))
    return resolved


# --------------------------------------------------------------------------
# Content checks
# --------------------------------------------------------------------------

def check_column(name: str, values: np.ndarray) -> List[str]:
    """Warnings about a single column's contents."""
    out: List[str] = []
    if values.size == 0:
        return [f"{name}: empty"]
    if values.dtype.kind == "f":
        n_nan = int(np.isnan(values).sum())
        n_inf = int(np.isinf(values).sum())
        if n_nan:
            out.append(f"{name}: {n_nan} NaN ({100 * n_nan / values.size:.2f}%)")
        if n_inf:
            out.append(f"{name}: {n_inf} Inf")
        finite = values[np.isfinite(values)]
    else:
        finite = values
    if finite.size and finite.min() == finite.max():
        # Exactly how a renamed branch shows up once something fills the gap
        # with a default: the column is there, and it never varies.
        out.append(f"{name}: constant value {finite.flat[0]!r} -- is this the "
                   f"branch you meant?")
    return out


def check_columns(columns: Dict[str, np.ndarray]) -> List[str]:
    warnings: List[str] = []
    for name, values in columns.items():
        warnings.extend(check_column(name, values))
    return warnings


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _report(paths: Sequence[str], tree: Optional[str], show_values: bool) -> int:
    schemas = []
    for path in paths:
        try:
            schemas.append(collect(path, tree))
        except Exception as exc:
            print(f"{path}: FAILED to read -- {type(exc).__name__}: {exc}")
            return 1

    print(f"{len(schemas)} file(s), "
          f"{sum(s.n_events for s in schemas):,} events total")
    for s in schemas:
        print(f"  {os.path.basename(s.path):<28} {s.kind:<9} "
              f"{s.n_events:>9,} events  {len(s.columns):>4} columns")

    problems = compare(schemas)
    print("\nschema consistency: " + ("OK" if not problems else "MISMATCH"))
    for p in problems:
        print(f"  ! {p}")

    if show_values:
        print("\ncontent checks (first file):")
        warnings = _content_warnings(schemas[0])
        if warnings:
            for w in warnings:
                print(f"  ! {w}")
        else:
            print("  no NaN/Inf and no constant columns")
    return 1 if problems else 0


def _content_warnings(schema: FileSchema) -> List[str]:
    import h5py

    if schema.kind == "root":
        import uproot
        import awkward as ak
        f = uproot.open(schema.path)
        name = max((k.split(";")[0] for k, c in f.classnames().items()
                    if c in ("TTree", "ROOT::RNTuple")),
                   key=lambda k: f[k].num_entries)
        tree = f[name]
        out = []
        for key in tree.keys():
            try:
                arr = tree[key].array(entry_stop=min(2000, tree.num_entries))
                flat = ak.to_numpy(ak.ravel(arr))
            except Exception:
                continue
            out.extend(check_column(key, np.asarray(flat)))
        return out

    with h5py.File(schema.path, "r") as f:
        columns = {}
        if schema.kind == "compact":
            for name in f["events"]:
                columns[f"events/{name}"] = f["events"][name][:]
            for block in f["blocks"]:
                for name in f["blocks"][block]:
                    if name != "offsets":
                        columns[f"{block}/{name}"] = f["blocks"][block][name][:]
        else:
            for key, obj in f.items():
                if obj.dtype.names is None:
                    continue
                raw = obj[:2000]
                valid = raw["valid"] if "valid" in raw.dtype.names else None
                for name in raw.dtype.names:
                    values = raw[name][valid] if valid is not None else raw[name]
                    columns[f"{key}/{name}"] = np.asarray(values)
        return check_columns(columns)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="files or globs (.root or .h5)")
    p.add_argument("--tree", default=None, help="ROOT tree name")
    p.add_argument("--values", action="store_true",
                   help="also check for NaN/Inf and constant columns")
    args = p.parse_args()
    paths = [q for p_ in args.paths for q in sorted(glob.glob(p_)) or [p_]]
    raise SystemExit(_report(paths, args.tree, args.values))


if __name__ == "__main__":
    main()
