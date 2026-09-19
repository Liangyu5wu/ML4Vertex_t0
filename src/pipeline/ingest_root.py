"""Read ATLAS SuperNtuple ROOT files into the event store.

This is the entry point of the data chain: ROOT ntuple in, ragged event store
out, no intermediate format. What it keeps is deliberately generous -- every
calorimeter cell in the ntuple, both jet collections, every reconstructed and
truth vertex -- because re-reading 130 GB of ROOT is expensive while tightening
a selection in a block config is free. Only tracks are preselected, since
keeping all ~2000 per event would triple the store for objects no model uses.

Store field names are ours, not ROOT's, and the mapping is recorded in the
manifest. Productions that rename a branch are handled by the alias lists
below; a required field that resolves to nothing stops the ingest rather than
quietly becoming a column of zeros.

    python -m src.pipeline.ingest_root \
        --input-dir /global/cfs/.../root/ttbar \
        --output-dir /global/cfs/.../store/ttbar --sample ttbar
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

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
# Collapsed into a single `region` column (see REGIONS).
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
}
# Renamed between the ttbar and VBF productions.
TRACK_ALIASES = {
    "n_pixel_hits": ("Track_nPixelHits", "Track_numberOfPixelHits"),
    "n_strip_hits": ("Track_nStripHits", "Track_numberOfSCTHits"),
}
# Track position extrapolated to each calorimeter layer -- what cell-track
# matching (and the baseline t0 algorithm) needs.
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
    # clears the vector between events. _check_counts() below would reject it.
}
TRUTH_VERTEX_FIELDS = {
    "x": "TruthVtx_x", "y": "TruthVtx_y", "z": "TruthVtx_z",
    "time": "TruthVtx_time", "is_hs": "TruthVtx_isHS",
}

# Tracks kept at ingest: everything the HS vertex fit claimed, everything HGTD
# timed, and everything close to the HS vertex in z regardless of assignment.
TRACK_Z0_WINDOW_MM = 3.0


# --- helpers ---------------------------------------------------------------

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


def _check_counts(arrays: Dict[str, object], block: str) -> np.ndarray:
    """Every branch in a collection must have the same objects per event.

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
    picked = ak.firsts(values[is_hs])
    out = ak.to_numpy(ak.fill_none(picked, np.nan))
    return np.asarray(out, dtype=np.float64)


def _resolve(available: Sequence[str], fields: Dict[str, object],
             what: str, required: bool = True) -> Dict[str, str]:
    """store field -> branch, following alias tuples; raise on a missing one."""
    resolved, missing = {}, []
    for name, branch in fields.items():
        candidates = (branch,) if isinstance(branch, str) else tuple(branch)
        for c in candidates:
            if c in available:
                resolved[name] = c
                break
        else:
            missing.append((name, candidates))
    if missing and required:
        raise KeyError(
            f"{what}: no branch for {[m[0] for m in missing]} "
            f"(tried {[list(m[1]) for m in missing]}). "
            f"The file has {len(available)} branches.")
    return resolved


# --- one file --------------------------------------------------------------

def ingest_file(src: str, dst: str, tree_name: str = "ntuple",
                extrapolations: bool = True, compression: Optional[str] = "gzip",
                complevel: int = 4, verbose: bool = True) -> Dict:
    """Convert one ROOT file into one event-store file."""
    import awkward as ak
    import uproot

    t0 = time.time()
    tree = uproot.open(src)[tree_name]
    available = set(tree.keys())
    n_events = int(tree.num_entries)
    mapping: Dict[str, str] = {}

    # ---- event-level scalars and the HS vertex -----------------------------
    ev_map = _resolve(available, EVENT_FIELDS, "event fields")
    events = {name: tree[branch].array(library="np")
              for name, branch in ev_map.items()}
    mapping.update({f"events/{k}": v for k, v in ev_map.items()})

    rv_map = _resolve(available, RECO_VERTEX_FIELDS, "reco vertices")
    tv_map = _resolve(available, TRUTH_VERTEX_FIELDS, "truth vertices")
    reco_vtx = {k: tree[v].array() for k, v in rv_map.items()}
    truth_vtx = {k: tree[v].array() for k, v in tv_map.items()}

    reco_is_hs = reco_vtx["is_hs"] == 1
    truth_is_hs = truth_vtx["is_hs"] == 1
    n_reco_hs = ak.to_numpy(ak.sum(reco_is_hs, axis=1))
    n_truth_hs = ak.to_numpy(ak.sum(truth_is_hs, axis=1))
    if (n_reco_hs != 1).any() or (n_truth_hs != 1).any():
        raise ValueError(
            f"{os.path.basename(src)}: {(n_reco_hs != 1).sum()} event(s) without "
            f"exactly one reco HS vertex and {(n_truth_hs != 1).sum()} without "
            f"exactly one truth HS vertex; the target and the time-of-flight "
            f"correction are undefined for those")

    for key in ("time", "x", "y", "z"):
        events[f"truth_vtx_{key}"] = _hs_scalar(truth_vtx[key], truth_is_hs)
    for key in ("time", "time_res", "x", "y", "z", "sum_pt2"):
        events[f"reco_vtx_{key}"] = _hs_scalar(reco_vtx[key], reco_is_hs)
    events["n_reco_vtx"] = _counts(reco_vtx["z"])
    events["n_truth_vtx"] = _counts(truth_vtx["z"])

    blocks: Dict[str, Block] = {}
    block_attrs: Dict[str, dict] = {}

    # ---- cells -------------------------------------------------------------
    cell_map = _resolve(available, CELL_FIELDS, "cells")
    cells = {name: tree[branch].array() for name, branch in cell_map.items()}
    mapping.update({f"cells/{k}": v for k, v in cell_map.items()})

    # Time of flight from the reconstructed HS vertex, in ps: the extra path
    # length relative to a particle coming from the detector origin.
    counts = _counts(cells["e"])
    vx, vy, vz = (np.repeat(events[f"reco_vtx_{k}"], counts) for k in "xyz")
    cx, cy, cz = (_flat(cells[k]) for k in "xyz")
    d_vtx = np.sqrt((cx - vx) ** 2 + (cy - vy) ** 2 + (cz - vz) ** 2)
    d_origin = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
    time_tof = _flat(cells["time"]) - (d_vtx - d_origin) / C_MM_PER_NS * 1000.0

    region_flat = np.full(len(cx), -1, dtype=np.int8)
    for branch, code in CELL_REGION_FLAGS.items():
        if branch in available:
            flags = _flat(tree[branch].array()).astype(bool)
            region_flat[flags] = code
            mapping[f"cells/region<-{branch}"] = str(code)
    if verbose:
        unknown = int((region_flat < 0).sum())
        if unknown:
            print(f"  note: {unknown} cell(s) match no region flag")

    _check_counts(cells, "cells")
    cell_columns, cell_offsets = _ragged(cells)
    cell_columns["time_tof"] = time_tof.astype(np.float32)
    cell_columns["region"] = region_flat
    blocks["cells"] = (cell_columns, cell_offsets)
    block_attrs["cells"] = {"selection": "everything in the ntuple",
                            "regions": str(REGIONS)}

    # ---- tracks ------------------------------------------------------------
    track_fields = dict(TRACK_FIELDS)
    track_fields.update(TRACK_ALIASES)
    if extrapolations:
        track_fields.update(TRACK_EXTRAPOLATION)
    tr_map = _resolve(available, track_fields, "tracks")
    tracks = {name: tree[branch].array() for name, branch in tr_map.items()}
    mapping.update({f"tracks/{k}": v for k, v in tr_map.items()})

    _check_counts(tracks, "tracks")
    hs_index = ak.to_numpy(ak.fill_none(
        ak.firsts(ak.local_index(reco_is_hs)[reco_is_hs]), -1))
    n_tracks = _counts(tracks["pt"])
    on_hs = tracks["reco_vtx_idx"] == ak.Array(
        [[i] * k for i, k in zip(hs_index, n_tracks)])
    timed = tracks["has_valid_time"] == 1
    dz_hs = tracks["z0"] - ak.Array(
        [[z] * k for z, k in zip(events["reco_vtx_z"], n_tracks)])
    keep = on_hs | timed | (abs(dz_hs) < TRACK_Z0_WINDOW_MM)

    # Derived columns so a block config can select tracks without having to
    # know which vertex index is the hard-scatter one.
    tracks["on_hs_vertex"] = ak.values_astype(on_hs, np.int8)
    tracks["dz_hs"] = dz_hs

    blocks["tracks"] = _ragged(tracks, keep)
    block_attrs["tracks"] = {
        "selection": ("on the reco HS vertex | has a valid HGTD time | "
                      f"|z0 - z_HS| < {TRACK_Z0_WINDOW_MM} mm"),
        "n_before_selection": int(n_tracks.sum()),
        "extrapolation": ("<layer>_eta/phi is -999 when the track does not "
                          "reach that layer")}
    if verbose:
        kept = blocks["tracks"][1][-1]
        print(f"  [tracks      ] {kept:9d} of {int(n_tracks.sum()):9d} "
              f"({100 * kept / max(n_tracks.sum(), 1):.0f}%)  "
              f"{kept / n_events:6.1f}/event")
    del tracks

    # ---- jets --------------------------------------------------------------
    for block, prefix in JET_COLLECTIONS.items():
        fields = {k: prefix + v for k, v in JET_FIELDS.items()}
        if not (set(fields.values()) <= available):
            if verbose:
                print(f"  note: {prefix}* not in this file, skipping {block}")
            continue
        jet_map = _resolve(available, fields, block)
        jets = {name: tree[branch].array() for name, branch in jet_map.items()}
        mapping.update({f"{block}/{k}": v for k, v in jet_map.items()})
        for name, suffix in JET_MATCH_COUNTS.items():
            branch = prefix + suffix
            if branch in available:
                jets[name] = ak.num(tree[branch].array(), axis=2)
                mapping[f"{block}/{name}"] = f"len({branch})"
        _check_counts(jets, block)
        blocks[block] = _ragged(jets)
        block_attrs[block] = {"selection": "everything in the ntuple"}

    # ---- vertices ----------------------------------------------------------
    _check_counts(reco_vtx, "reco_vertices")
    _check_counts(truth_vtx, "truth_vertices")
    blocks["reco_vertices"] = _ragged(reco_vtx)
    blocks["truth_vertices"] = _ragged(truth_vtx)
    mapping.update({f"reco_vertices/{k}": v for k, v in rv_map.items()})
    mapping.update({f"truth_vertices/{k}": v for k, v in tv_map.items()})
    block_attrs["reco_vertices"] = {"selection": "all reconstructed vertices"}
    block_attrs["truth_vertices"] = {"selection": "all truth vertices"}

    warnings, block_info = write_compact(
        dst, events, blocks,
        attrs={"source_file": os.path.abspath(src), "tree": tree_name,
               "ingest": "src.pipeline.ingest_root"},
        block_attrs=block_attrs, compression=compression, complevel=complevel)

    src_mb, dst_mb = os.path.getsize(src) / 1e6, os.path.getsize(dst) / 1e6
    summary = {"source": os.path.abspath(src), "output": os.path.abspath(dst),
               "n_events": n_events, "blocks": block_info, "warnings": warnings,
               "mapping": mapping, "src_mb": src_mb, "dst_mb": dst_mb,
               "seconds": time.time() - t0}
    if verbose:
        print(f"  {src_mb:8.1f} MB -> {dst_mb:7.1f} MB  "
              f"({src_mb / max(dst_mb, 1e-9):.1f}x)  in {summary['seconds']:.0f}s")
    return summary


# --- a directory -----------------------------------------------------------

def _ingest_one(args) -> Dict:
    """Worker entry point: ingest a single file (picklable arguments only)."""
    src, dst, tree_name, extrapolations, compression, complevel = args
    s = ingest_file(src, dst, tree_name=tree_name, extrapolations=extrapolations,
                    compression=compression, complevel=complevel, verbose=False)
    print(f"  done {os.path.basename(src)}  {s['n_events']:6d} events  "
          f"{s['src_mb'] / 1000:.1f} -> {s['dst_mb'] / 1000:.2f} GB  "
          f"{s['seconds']:.0f}s", flush=True)
    return s


def ingest_directory(input_dir: str, output_dir: str, sample: str,
                     pattern: str = "*.root", limit: Optional[int] = None,
                     tree_name: str = "ntuple", extrapolations: bool = True,
                     compression: Optional[str] = "gzip", complevel: int = 4,
                     overwrite: bool = False, workers: int = 1) -> Dict:
    """Ingest every ROOT file in a directory and write the manifest."""
    import h5py

    inputs = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if limit is not None:
        inputs = inputs[:limit]
    if not inputs:
        raise FileNotFoundError(f"no files matching {pattern} in {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ingesting {len(inputs)} ROOT file(s) from {input_dir}")

    schemas = [collect_root(p, tree_name) for p in inputs]
    differences = compare(schemas)
    if differences:
        print("\nInput files do not share one schema:")
        for d in differences[:10]:
            print(f"  ! {d}")
        raise ValueError("inconsistent input schema; ingest the groups separately")
    print(f"schema: {len(schemas[0].columns)} branches, consistent across all files; "
          f"{sum(s.n_events for s in schemas):,} events")

    files_meta: List[dict] = []
    all_warnings: set = set()
    mapping: Dict[str, str] = {}
    todo = []
    for src in inputs:
        dst = os.path.join(output_dir, os.path.basename(src).replace(".root", ".h5"))
        if os.path.exists(dst) and not overwrite:
            with h5py.File(dst, "r") as f:
                files_meta.append({"file": os.path.basename(dst),
                                   "n_events": int(f.attrs["n_events"])})
            print(f"  skip {os.path.basename(src)} -- output exists")
            continue
        todo.append((src, dst, tree_name, extrapolations, compression, complevel))

    if todo:
        print(f"converting {len(todo)} file(s) with {workers} worker(s)")
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_ingest_one, todo))
        else:
            results = [_ingest_one(a) for a in todo]
        for s in results:
            all_warnings.update(s["warnings"])
            mapping = s["mapping"]
            files_meta.append({"file": os.path.basename(s["output"]),
                               "n_events": s["n_events"],
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
                                     "track_selection": block_selection_note()})
    total_src = sum(m.get("src_mb", 0.0) for m in files_meta)
    total_dst = sum(m.get("dst_mb", 0.0) for m in files_meta)
    n_events = sum(m["n_events"] for m in files_meta)
    print(f"\n{sample}: {n_events:,} events in {len(files_meta)} file(s)")
    if total_dst:
        print(f"  {total_src / 1000:.1f} GB -> {total_dst / 1000:.1f} GB "
              f"({total_src / total_dst:.1f}x)")
    print(f"  manifest: {manifest}")
    return {"manifest": manifest, "n_events": n_events}


def block_selection_note() -> str:
    return ("tracks: on the reco HS vertex | valid HGTD time | "
            f"|z0 - z_HS| < {TRACK_Z0_WINDOW_MM} mm; everything else unfiltered")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--pattern", default="*.root")
    p.add_argument("--tree", default="ntuple")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-extrapolations", action="store_true",
                   help="skip the per-layer track extrapolations (saves ~25%%)")
    p.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    p.add_argument("--complevel", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=1,
                   help="convert this many files in parallel")
    args = p.parse_args()
    ingest_directory(args.input_dir, args.output_dir, args.sample,
                     pattern=args.pattern, limit=args.limit, tree_name=args.tree,
                     extrapolations=not args.no_extrapolations,
                     compression=None if args.compression == "none" else args.compression,
                     complevel=args.complevel, overwrite=args.overwrite,
                     workers=args.workers)


if __name__ == "__main__":
    main()
