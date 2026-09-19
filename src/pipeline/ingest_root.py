"""Read ATLAS SuperNtuple ROOT files into the event store.

This is the entry point of the data chain: ROOT ntuple in, ragged event store
out, no intermediate format. What it keeps is deliberately generous -- every
calorimeter cell in the ntuple, both jet collections, every reconstructed and
truth vertex -- because re-reading 130 GB of ROOT is expensive while tightening
a selection in a block config is free. Only tracks are preselected, since
keeping all ~2000 per event would triple the store for objects no model reads.

Store field names are ours, not ROOT's, and the mapping is recorded in the
manifest. Productions that rename a branch are handled by the alias lists
below; a required field that resolves to nothing stops the ingest rather than
quietly becoming a column of zeros.

Several ROOT files are combined into one store file (``--shards``): the store
is read whole for training, so a few large files beat dozens of small ones.

    python -m src.pipeline.ingest_root \
        --input-dir /global/cfs/.../root/ttbar \
        --output-dir /global/cfs/.../store/ttbar --sample ttbar \
        --shards 8 --workers 8
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .schema import collect_root, compare
from .store_writer import Block, write_compact, write_manifest

C_MM_PER_NS = 299.792458

# --- what to read ----------------------------------------------------------
# store field -> ROOT branch (a tuple means "try these in order").

EVENT_FIELDS = {
    "event_number": "eventNumber",
    "run_number": "runNumber",
    "mu": "averageInteractionsPerCrossing",
    "weight": "weight",
    # BCID is 0 and distFrontBunchTrain is uninitialised garbage (a constant
    # 2005374112) in these productions -- both would be dead columns.
}

CELL_FIELDS = {
    "e": "Cell_e", "eta": "Cell_eta", "phi": "Cell_phi",
    "x": "Cell_x", "y": "Cell_y", "z": "Cell_z",
    "layer": "Cell_layer", "sampling": "Cell_sampling",
    "time": "Cell_time", "significance": "Cell_significance",
    "total_noise": "Cell_totalNoise", "quality": "Cell_quality",
    "provenance": "Cell_provenance",
}
# Collapsed into a single `region` column.
CELL_REGION_FLAGS = {
    "Cell_isEM_Barrel": 0, "Cell_isEM_EndCap": 1, "Cell_isFCAL": 2,
    "Cell_isHEC": 3, "Cell_isTile": 4,
}
REGIONS = {0: "EM barrel", 1: "EM endcap", 2: "FCal", 3: "HEC", 4: "Tile",
           -1: "none of the above"}

TRACK_FIELDS = {
    "pt": "Track_pt", "eta": "Track_eta", "phi": "Track_phi",
    "d0": "Track_d0", "z0": "Track_z0", "var_z0": "Track_var_z0",
    "q_over_p": "Track_qOverP", "theta": "Track_theta",
    "chi2": "Track_chi2", "ndof": "Track_ndof", "charge": "Track_charge",
    "time": "Track_time", "time_res": "Track_timeRes",
    "has_valid_time": "Track_hasValidTime", "quality": "Track_quality",
    "reco_vtx_idx": "Track_recoVtx_idx", "reco_vtx_weight": "Track_recoVtx_weight",
    "truth_vtx_idx": "Track_truthVtx_idx", "truth_prob": "Track_truthProb",
    # Renamed between the ttbar and VBF productions.
    "n_pixel_hits": ("Track_nPixelHits", "Track_numberOfPixelHits"),
    "n_strip_hits": ("Track_nStripHits", "Track_numberOfSCTHits"),
}
# Track position extrapolated to each calorimeter layer -- what cell-track
# matching (and the baseline t0 algorithm) needs.  -999 where the track does
# not reach that layer.
TRACK_EXTRAPOLATION = {
    f"{layer.lower()}_{coord}": f"Track_{layer}_{coord}"
    for layer in ("EMB1", "EMB2", "EMB3", "EME1", "EME2", "EME3",
                  "PreSamplerB", "PreSamplerE")
    for coord in ("eta", "phi")
}

JET_FIELDS = {"pt": "pt", "eta": "eta", "phi": "phi", "m": "m",
              "width": "width", "n_constituents": "nConstituents"}
# vector<vector<int>> branches counted per jet rather than stored.
JET_MATCH_COUNTS = {"n_truth_hs_jets": "truthHSJet_idx",
                    "n_truth_itpu_jets": "truthITPUJet_idx",
                    "n_truth_ootpu_jets": "truthOOTPUJet_idx"}
JET_COLLECTIONS = {"jets_emtopo": "AntiKt4EMTopoJets_",
                   "jets_pflow": "AntiKt4EMPFlowJets_"}

RECO_VERTEX_FIELDS = {
    "x": "RecoVtx_x", "y": "RecoVtx_y", "z": "RecoVtx_z",
    "time": "RecoVtx_time", "time_res": "RecoVtx_timeRes",
    "sum_pt2": "RecoVtx_sumPt2", "is_hs": "RecoVtx_isHS",
    "has_valid_time": "RecoVtx_hasValidTime",
    # RecoVtx_isPU is deliberately absent: in both productions its per-event
    # counts are the running sum of the vertex counts, i.e. the producer never
    # clears the vector between events.  _check_counts() would reject it.
}
TRUTH_VERTEX_FIELDS = {
    "x": "TruthVtx_x", "y": "TruthVtx_y", "z": "TruthVtx_z",
    "time": "TruthVtx_time", "is_hs": "TruthVtx_isHS",
}

# Tracks kept at ingest: everything the HS vertex fit claimed, everything HGTD
# timed, and everything close to the HS vertex in z whatever the fit decided.
TRACK_Z0_WINDOW_MM = 3.0
TRACK_SELECTION = ("on the reco HS vertex | valid HGTD time | "
                   f"|z0 - z_HS| < {TRACK_Z0_WINDOW_MM} mm")


@dataclass
class FileArrays:
    """One ROOT file reduced to store form, before anything is written."""
    events: Dict[str, np.ndarray] = field(default_factory=dict)
    blocks: Dict[str, Block] = field(default_factory=dict)
    block_attrs: Dict[str, dict] = field(default_factory=dict)
    mapping: Dict[str, str] = field(default_factory=dict)
    n_events: int = 0


# --- small helpers ---------------------------------------------------------

def _flat(arr) -> np.ndarray:
    import awkward as ak
    return ak.to_numpy(ak.ravel(arr))


def _counts(arr) -> np.ndarray:
    import awkward as ak
    return ak.to_numpy(ak.num(arr, axis=1)).astype(np.int64)


def _offsets(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=out[1:])
    return out


def _resolve(available: Sequence[str], fields: Dict[str, object],
             what: str) -> Dict[str, str]:
    """store field -> branch, following alias tuples; raise on a missing one."""
    resolved, missing = {}, []
    for name, branch in fields.items():
        for candidate in ((branch,) if isinstance(branch, str) else tuple(branch)):
            if candidate in available:
                resolved[name] = candidate
                break
        else:
            missing.append(name)
    if missing:
        raise KeyError(f"{what}: no branch for {missing}; the file has "
                       f"{len(available)} branches")
    return resolved


def _read(tree, mapping: Dict[str, str]) -> Dict[str, object]:
    """Read a group of branches in one pass, keyed by store field name."""
    arrays = tree.arrays(list(mapping.values()), library="ak")
    return {name: arrays[branch] for name, branch in mapping.items()}


def _check_counts(arrays: Dict[str, object], block: str) -> np.ndarray:
    """Every branch of a collection must agree on its objects per event.

    A branch that disagrees means the producer filled it differently -- most
    often a vector that is never cleared, which shows up as a running sum.
    """
    reference = name_ref = None
    for name, arr in arrays.items():
        counts = _counts(arr)
        if reference is None:
            reference, name_ref = counts, name
        elif not np.array_equal(counts, reference):
            hint = ("; it is the running sum of the others, so the producer "
                    "never clears it between events"
                    if np.array_equal(counts, np.cumsum(reference)) else "")
            raise ValueError(
                f"block {block!r}: {name} has {counts[:4]} objects in the first "
                f"events but {name_ref} has {reference[:4]}{hint}")
    return reference if reference is not None else np.zeros(0, dtype=np.int64)


def _ragged(arrays: Dict[str, object], mask=None) -> Block:
    """Jagged arrays (optionally masked) -> ({field: flat column}, offsets)."""
    columns, counts = {}, None
    for name, arr in arrays.items():
        sel = arr if mask is None else arr[mask]
        if counts is None:
            counts = _counts(sel)
        columns[name] = _flat(sel)
    return columns, _offsets(counts if counts is not None else np.zeros(0, np.int64))


def _hs_scalar(values, is_hs) -> np.ndarray:
    """The value belonging to the (single) HS vertex, per event."""
    import awkward as ak
    return np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(values[is_hs]), np.nan)),
                      dtype=np.float64)


def _broadcast(per_event: np.ndarray, counts: np.ndarray):
    """Repeat an event-level value once per object in that event."""
    import awkward as ak
    return ak.unflatten(np.repeat(per_event, counts), counts)


# --- reading one file ------------------------------------------------------

def read_file(src: str, tree_name: str = "ntuple", extrapolations: bool = True
              ) -> FileArrays:
    """Reduce one ROOT file to store form: named columns, ragged blocks."""
    import awkward as ak
    import uproot

    tree = uproot.open(src)[tree_name]
    available = set(tree.keys())
    out = FileArrays(n_events=int(tree.num_entries))

    # ---- event scalars, and the hard-scatter vertex ------------------------
    ev_map = _resolve(available, EVENT_FIELDS, "event fields")
    out.events = {name: tree[branch].array(library="np")
                  for name, branch in ev_map.items()}
    out.mapping.update({f"events/{k}": v for k, v in ev_map.items()})

    rv_map = _resolve(available, RECO_VERTEX_FIELDS, "reco vertices")
    tv_map = _resolve(available, TRUTH_VERTEX_FIELDS, "truth vertices")
    reco_vtx, truth_vtx = _read(tree, rv_map), _read(tree, tv_map)
    reco_is_hs, truth_is_hs = reco_vtx["is_hs"] == 1, truth_vtx["is_hs"] == 1

    n_reco_hs = ak.to_numpy(ak.sum(reco_is_hs, axis=1))
    n_truth_hs = ak.to_numpy(ak.sum(truth_is_hs, axis=1))
    if (n_reco_hs != 1).any() or (n_truth_hs != 1).any():
        raise ValueError(
            f"{os.path.basename(src)}: {(n_reco_hs != 1).sum()} event(s) without "
            f"exactly one reco HS vertex and {(n_truth_hs != 1).sum()} without "
            f"exactly one truth HS vertex; the target and the time-of-flight "
            f"correction are undefined there")

    for key in ("time", "x", "y", "z"):
        out.events[f"truth_vtx_{key}"] = _hs_scalar(truth_vtx[key], truth_is_hs)
    for key in ("time", "time_res", "x", "y", "z", "sum_pt2"):
        out.events[f"reco_vtx_{key}"] = _hs_scalar(reco_vtx[key], reco_is_hs)
    out.events["n_reco_vtx"] = _counts(reco_vtx["z"])
    out.events["n_truth_vtx"] = _counts(truth_vtx["z"])

    # ---- cells -------------------------------------------------------------
    cell_map = _resolve(available, CELL_FIELDS, "cells")
    cells = _read(tree, cell_map)
    counts = _check_counts(cells, "cells")
    out.mapping.update({f"cells/{k}": v for k, v in cell_map.items()})

    # Time of flight relative to a particle from the detector origin, in ps.
    cx, cy, cz = (_flat(cells[k]) for k in "xyz")
    vx, vy, vz = (np.repeat(out.events[f"reco_vtx_{k}"], counts) for k in "xyz")
    d_vtx = np.sqrt((cx - vx) ** 2 + (cy - vy) ** 2 + (cz - vz) ** 2)
    d_origin = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)

    region_map = {b: c for b, c in CELL_REGION_FLAGS.items() if b in available}
    region = np.full(len(cx), -1, dtype=np.int8)
    for branch, code in region_map.items():
        region[_flat(tree[branch].array()).astype(bool)] = code
    out.mapping.update({f"cells/region<-{b}": str(c) for b, c in region_map.items()})

    columns, offsets = _ragged(cells)
    columns["time_tof"] = (_flat(cells["time"])
                           - (d_vtx - d_origin) / C_MM_PER_NS * 1000.0).astype(np.float32)
    columns["region"] = region
    out.blocks["cells"] = (columns, offsets)
    out.block_attrs["cells"] = {"selection": "everything in the ntuple",
                                "regions": str(REGIONS)}

    # ---- tracks ------------------------------------------------------------
    track_fields = dict(TRACK_FIELDS)
    if extrapolations:
        track_fields.update(TRACK_EXTRAPOLATION)
    tr_map = _resolve(available, track_fields, "tracks")
    tracks = _read(tree, tr_map)
    n_tracks = _check_counts(tracks, "tracks")
    out.mapping.update({f"tracks/{k}": v for k, v in tr_map.items()})

    hs_index = ak.to_numpy(ak.fill_none(
        ak.firsts(ak.local_index(reco_is_hs)[reco_is_hs]), -1))
    on_hs = tracks["reco_vtx_idx"] == _broadcast(hs_index, n_tracks)
    dz_hs = tracks["z0"] - _broadcast(out.events["reco_vtx_z"], n_tracks)
    keep = on_hs | (tracks["has_valid_time"] == 1) | (abs(dz_hs) < TRACK_Z0_WINDOW_MM)

    # Derived columns so a block config can select tracks without having to
    # know which vertex index is the hard-scatter one.
    tracks["on_hs_vertex"] = ak.values_astype(on_hs, np.int8)
    tracks["dz_hs"] = dz_hs
    out.blocks["tracks"] = _ragged(tracks, keep)
    out.block_attrs["tracks"] = {
        "selection": TRACK_SELECTION,
        "n_before_selection": int(n_tracks.sum()),
        "extrapolation": "<layer>_eta/phi is -999 where the track does not reach"}

    # ---- jets --------------------------------------------------------------
    for block, prefix in JET_COLLECTIONS.items():
        fields = {k: prefix + v for k, v in JET_FIELDS.items()}
        if not set(fields.values()) <= available:
            continue
        jet_map = _resolve(available, fields, block)
        jets = _read(tree, jet_map)
        out.mapping.update({f"{block}/{k}": v for k, v in jet_map.items()})
        for name, suffix in JET_MATCH_COUNTS.items():
            if prefix + suffix in available:
                jets[name] = ak.num(tree[prefix + suffix].array(), axis=2)
                out.mapping[f"{block}/{name}"] = f"len({prefix + suffix})"
        _check_counts(jets, block)
        out.blocks[block] = _ragged(jets)
        out.block_attrs[block] = {"selection": "everything in the ntuple"}

    # ---- vertices ----------------------------------------------------------
    for block, arrays, mapping in (("reco_vertices", reco_vtx, rv_map),
                                   ("truth_vertices", truth_vtx, tv_map)):
        _check_counts(arrays, block)
        out.blocks[block] = _ragged(arrays)
        out.block_attrs[block] = {"selection": "all vertices"}
        out.mapping.update({f"{block}/{k}": v for k, v in mapping.items()})

    return out


# --- combining and writing -------------------------------------------------

def merge(parts: Sequence[FileArrays]) -> FileArrays:
    """Concatenate several files' arrays into one, fixing up the offsets."""
    if len(parts) == 1:
        return parts[0]
    out = FileArrays(block_attrs=parts[0].block_attrs, mapping=parts[0].mapping,
                     n_events=sum(p.n_events for p in parts))
    out.events = {name: np.concatenate([p.events[name] for p in parts])
                  for name in parts[0].events}
    for block in parts[0].blocks:
        columns = {field: np.concatenate([p.blocks[block][0][field] for p in parts])
                   for field in parts[0].blocks[block][0]}
        offsets, running = [np.zeros(1, dtype=np.int64)], 0
        for p in parts:
            part_offsets = p.blocks[block][1]
            offsets.append(part_offsets[1:] + running)
            running += int(part_offsets[-1])
        out.blocks[block] = (columns, np.concatenate(offsets))
    for block, attrs in parts[0].block_attrs.items():
        total = sum(p.block_attrs[block].get("n_before_selection", 0) for p in parts)
        if total:
            out.block_attrs[block] = {**attrs, "n_before_selection": total}
    return out


def ingest_shard(sources: Sequence[str], dst: str, tree_name: str = "ntuple",
                 extrapolations: bool = True, compression: Optional[str] = "gzip",
                 complevel: int = 4) -> Dict:
    """Read a group of ROOT files and write them as one store file."""
    t0 = time.time()
    data = merge([read_file(s, tree_name, extrapolations) for s in sources])
    warnings, block_info = write_compact(
        dst, data.events, data.blocks,
        attrs={"source_files": ", ".join(os.path.basename(s) for s in sources),
               "tree": tree_name, "ingest": "src.pipeline.ingest_root"},
        block_attrs=data.block_attrs, compression=compression, complevel=complevel)

    src_mb = sum(os.path.getsize(s) for s in sources) / 1e6
    dst_mb = os.path.getsize(dst) / 1e6
    return {"output": os.path.abspath(dst), "n_events": data.n_events,
            "n_sources": len(sources), "blocks": block_info, "warnings": warnings,
            "mapping": data.mapping, "src_mb": src_mb, "dst_mb": dst_mb,
            "seconds": time.time() - t0}


def _shard(sources: Sequence[str], n_shards: int) -> List[List[str]]:
    """Split the inputs into contiguous groups of roughly equal total size."""
    n_shards = max(1, min(n_shards, len(sources)))
    sizes = np.array([os.path.getsize(s) for s in sources], dtype=np.float64)
    edges = np.searchsorted(np.cumsum(sizes),
                            np.linspace(0, sizes.sum(), n_shards + 1)[1:-1])
    return [list(g) for g in np.split(np.array(sources, dtype=object), edges) if len(g)]


def _run_shard(args) -> Dict:
    """Worker entry point (picklable arguments only)."""
    sources, dst, tree_name, extrapolations, compression, complevel = args
    s = ingest_shard(sources, dst, tree_name, extrapolations, compression, complevel)
    print(f"  {os.path.basename(dst):28s} {s['n_sources']:2d} file(s)  "
          f"{s['n_events']:7,d} events  {s['src_mb'] / 1000:5.1f} -> "
          f"{s['dst_mb'] / 1000:5.2f} GB  {s['seconds']:.0f}s", flush=True)
    return s


def ingest_directory(input_dir: str, output_dir: str, sample: str,
                     pattern: str = "*.root", limit: Optional[int] = None,
                     tree_name: str = "ntuple", extrapolations: bool = True,
                     compression: Optional[str] = "gzip", complevel: int = 4,
                     shards: int = 8, workers: int = 1,
                     overwrite: bool = False) -> Dict:
    """Ingest a directory of ROOT files into a sharded store."""
    import h5py

    inputs = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if limit is not None:
        inputs = inputs[:limit]
    if not inputs:
        raise FileNotFoundError(f"no files matching {pattern} in {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    schemas = [collect_root(p, tree_name) for p in inputs]
    differences = compare(schemas)
    if differences:
        print("Input files do not share one schema:")
        for d in differences[:10]:
            print(f"  ! {d}")
        raise ValueError("inconsistent input schema; ingest the groups separately")
    groups = _shard(inputs, shards)
    print(f"{sample}: {len(inputs)} ROOT file(s), "
          f"{sum(s.n_events for s in schemas):,} events, "
          f"{len(schemas[0].columns)} branches (consistent) "
          f"-> {len(groups)} store file(s), {workers} worker(s)")

    todo, files_meta = [], []
    for i, group in enumerate(groups):
        dst = os.path.join(output_dir, f"{sample}_{i:03d}.h5")
        if os.path.exists(dst) and not overwrite:
            with h5py.File(dst, "r") as f:
                files_meta.append({"file": os.path.basename(dst),
                                   "n_events": int(f.attrs["n_events"])})
            print(f"  {os.path.basename(dst)} exists, skipping")
            continue
        todo.append((group, dst, tree_name, extrapolations, compression, complevel))

    all_warnings: set = set()
    mapping: Dict[str, str] = {}
    if todo:
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_run_shard, todo))
        else:
            results = [_run_shard(a) for a in todo]
        for s in results:
            all_warnings.update(s["warnings"])
            mapping = s["mapping"]
            files_meta.append({"file": os.path.basename(s["output"]),
                               "n_events": s["n_events"],
                               "n_sources": s["n_sources"],
                               "src_mb": round(s["src_mb"], 1),
                               "dst_mb": round(s["dst_mb"], 1),
                               "seconds": round(s["seconds"], 1)})
    files_meta.sort(key=lambda m: m["file"])

    if all_warnings:
        print(f"\ncontent warnings ({len(all_warnings)}):")
        for w in sorted(all_warnings):
            print(f"  ! {w}")

    manifest = write_manifest(output_dir, sample, files_meta,
                              source=os.path.abspath(input_dir),
                              extra={"ingest": "ingest_root", "tree": tree_name,
                                     "field_mapping": mapping,
                                     "track_selection": TRACK_SELECTION})
    src_gb = sum(m.get("src_mb", 0.0) for m in files_meta) / 1000
    dst_gb = sum(m.get("dst_mb", 0.0) for m in files_meta) / 1000
    n_events = sum(m["n_events"] for m in files_meta)
    print(f"\n{sample}: {n_events:,} events in {len(files_meta)} store file(s)")
    if dst_gb:
        print(f"  {src_gb:.1f} GB -> {dst_gb:.1f} GB ({src_gb / dst_gb:.1f}x)")
    print(f"  manifest: {manifest}")
    return {"manifest": manifest, "n_events": n_events}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--pattern", default="*.root")
    p.add_argument("--tree", default="ntuple")
    p.add_argument("--limit", type=int, default=None, help="use only the first N files")
    p.add_argument("--shards", type=int, default=8, help="store files to write")
    p.add_argument("--workers", type=int, default=1, help="shards built in parallel")
    p.add_argument("--no-extrapolations", action="store_true",
                   help="skip the per-layer track extrapolations (saves ~25%%)")
    p.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    p.add_argument("--complevel", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    ingest_directory(args.input_dir, args.output_dir, args.sample,
                     pattern=args.pattern, limit=args.limit, tree_name=args.tree,
                     extrapolations=not args.no_extrapolations,
                     compression=None if args.compression == "none" else args.compression,
                     complevel=args.complevel, shards=args.shards,
                     workers=args.workers, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
