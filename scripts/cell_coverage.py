#!/usr/bin/env python
"""What the cell truncation throws away, measured against the hard scatter.

    python scripts/cell_coverage.py --config config/blocks/lar_hgtd.yaml

The cell block keeps the top ``max_items`` cells by (e, significance). Nothing
in that ordering knows which vertex a cell came from, so the question is
whether cells from the hard scatter are being cut off, and whether the ones
that are still carry timing information once there.

A cell counts as hard-scatter if it lies within ``--delta-r`` of a track on
the reco HS vertex, extrapolated to the cell's own layer -- the baseline's
matching. For every rank bucket this reports how many such cells an event
has and how well their time-of-flight-corrected time agrees with the truth
t0 (truth is used here to judge, never to select).

The config's cell selection and event_select are applied; its max_items is
ignored, since the point is to see what lies past it.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.evaluation.baseline import EXTRAPOLATION, NO_EXTRAPOLATION
from src.pipeline.assemble import AssemblySpec
from src.pipeline.blocks import event_mask, load_block
from src.pipeline.event_store import EventStore, ragged_arange

LIMITS = (60, 120, 200, 250, 400)
RANK_BUCKETS = (0, 60, 120, 200, 250, 400, 10 ** 9)
PAIRS_PER_CHUNK = 20_000_000


def hs_match(cells, tracks, delta_r: float) -> np.ndarray:
    """Per cell: is it within delta_r of an HS track extrapolated to its layer.

    Every cell is paired with every HS track of its event, in chunks of events
    so the pair arrays stay bounded; no loop over single events.
    """
    matched = np.zeros(cells.n_items, dtype=bool)
    nc, nt = cells.counts, tracks.counts
    cum = np.cumsum(nc * nt)
    edges = np.searchsorted(cum, np.arange(PAIRS_PER_CHUNK, cum[-1], PAIRS_PER_CHUNK),
                            side="right") if len(cum) else []
    edges = np.unique(np.concatenate([[0], edges, [cells.n_events]])).astype(np.int64)
    for start, stop in zip(edges[:-1], edges[1:]):
        cell_ev = np.repeat(np.arange(start, stop), nc[start:stop])
        per_cell = nt[cell_ev]
        pc = np.repeat(np.arange(cells.offsets[start], cells.offsets[stop]),
                       per_cell)                                 # cell of each pair
        pt = ragged_arange(per_cell, tracks.offsets[cell_ev])    # track of each pair
        if not len(pc):
            continue
        region, layer = cells["region"][pc], cells["layer"][pc]
        t_eta = np.full(len(pc), NO_EXTRAPOLATION - 1, dtype=np.float32)
        t_phi = np.zeros(len(pc), dtype=np.float32)
        for (r, l), prefix in EXTRAPOLATION.items():
            sel = (region == r) & (layer == l)
            t_eta[sel] = tracks[f"{prefix}_eta"][pt[sel]]
            t_phi[sel] = tracks[f"{prefix}_phi"][pt[sel]]
        d_phi = np.abs(cells["phi"][pc] - t_phi)
        d_phi = np.where(d_phi > np.pi, 2 * np.pi - d_phi, d_phi)
        hit = ((t_eta > NO_EXTRAPOLATION)
               & ((cells["eta"][pc] - t_eta) ** 2 + d_phi ** 2 < delta_r ** 2))
        matched[np.unique(pc[hit])] = True
    return matched


def q68(x: np.ndarray) -> float:
    return float(np.percentile(np.abs(x), 68)) if len(x) else float("nan")


def study(store: EventStore, spec: AssemblySpec, delta_r: float) -> dict:
    cspec = copy.deepcopy(spec.blocks["cells"])
    cspec.max_items, cspec.transform = 10 ** 9, {}   # everything, in ps
    keep = event_mask(store, spec.event_select)
    ids = np.flatnonzero(keep)

    cells = load_block(store, cspec).take_events(ids)
    track_fields = ["on_hs_vertex"] + [f"{p}_{c}" for p in set(EXTRAPOLATION.values())
                                       for c in ("eta", "phi")]
    tracks = store.block("tracks", track_fields)
    tracks = tracks.select(tracks["on_hs_vertex"] == 1).take_events(ids)

    rank = ragged_arange(cells.counts)
    ev = cells.event_index()
    matched = hs_match(cells, tracks, delta_r)
    resid = cells["time"] - store.event_column("truth_vtx_time")[ids][ev]
    n_ev = len(ids)
    n_sel = cells.counts
    m_per_ev = np.bincount(ev[matched], minlength=n_ev)

    out = {"events": int(n_ev), "events_before_select": int(store.n_events),
           "selected_per_event": {f"p{p}": float(np.percentile(n_sel, p))
                                  for p in (50, 90, 95, 99, 100)},
           "hs_matched_per_event": float(m_per_ev.mean()),
           "hs_matched_fraction_of_selected": float(matched.mean()),
           "limits": {}, "rank_buckets": [], "unmatched_q68_ps": q68(resid[~matched])}

    for n in LIMITS:
        lost = matched & (rank >= n)
        lost_ev = np.bincount(ev[lost], minlength=n_ev)
        kept_ev = m_per_ev - lost_ev
        top = rank < n
        n_top = np.bincount(ev[top], minlength=n_ev)
        n_top_emb = np.bincount(ev[top & (cells["region"] == 0)], minlength=n_ev)
        one_region = (n_top > 0) & ((n_top_emb == 0) | (n_top_emb == n_top))
        top_region = np.where(n_top_emb == n_top, 0, 1)
        lost_other = np.bincount(ev[lost & (cells["region"] != top_region[ev])],
                                 minlength=n_ev) > 0
        out["limits"][n] = {
            "events_truncated": float(np.mean(n_sel > n)),
            "hs_cells_lost": float(lost.sum() / max(matched.sum(), 1)),
            "events_losing_hs_cells": float(np.mean(lost_ev > 0)),
            "events_with_hs_only_past_limit": float(np.mean((kept_ev == 0) & (lost_ev > 0))),
            # the observation that started this: the kept cells are all from
            # one EM region while the HS cells lost are in the other
            "events_one_region_losing_other": float(np.mean(one_region & lost_other)),
        }

    for lo, hi in zip(RANK_BUCKETS[:-1], RANK_BUCKETS[1:]):
        sel = (rank >= lo) & (rank < hi)
        out["rank_buckets"].append({
            "ranks": f"{lo}-{hi if hi < 10 ** 9 else 'end'}",
            "cells_per_event": float(sel.sum() / n_ev),
            "hs_cells_per_event": float((sel & matched).sum() / n_ev),
            "hs_q68_ps": q68(resid[sel & matched]),
            "hs_median_e_gev": float(np.median(cells["e"][sel & matched]))
            if (sel & matched).any() else float("nan"),
        })
    return out


def show(name: str, r: dict) -> None:
    s = r["selected_per_event"]
    print(f"\n=== {name}: {r['events']:,} events (of {r['events_before_select']:,} "
          f"before event_select) ===")
    print(f"selected cells/event  median {s['p50']:.0f}  p90 {s['p90']:.0f}  "
          f"p95 {s['p95']:.0f}  p99 {s['p99']:.0f}  max {s['p100']:.0f}")
    print(f"HS-matched cells/event {r['hs_matched_per_event']:.1f} "
          f"({100 * r['hs_matched_fraction_of_selected']:.1f}% of selected); "
          f"unmatched cells q68 {r['unmatched_q68_ps']:.0f} ps")
    print(f"\n{'max_items':>9} {'ev trunc':>9} {'HS lost':>8} {'ev lose HS':>11} "
          f"{'HS only past':>13} {'1-region':>9}")
    for n, v in r["limits"].items():
        print(f"{n:>9} {100 * v['events_truncated']:8.1f}% {100 * v['hs_cells_lost']:7.1f}% "
              f"{100 * v['events_losing_hs_cells']:10.1f}% "
              f"{100 * v['events_with_hs_only_past_limit']:12.2f}% "
              f"{100 * v['events_one_region_losing_other']:8.2f}%")
    print(f"\n{'ranks':>10} {'cells/ev':>9} {'HS/ev':>7} {'HS q68':>8} {'HS med E':>9}")
    for b in r["rank_buckets"]:
        print(f"{b['ranks']:>10} {b['cells_per_event']:9.1f} {b['hs_cells_per_event']:7.2f} "
              f"{b['hs_q68_ps']:7.0f}  {b['hs_median_e_gev']:7.2f} GeV")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--delta-r", type=float, default=0.05)
    p.add_argument("--out", default=None, help="write the numbers as JSON")
    args = p.parse_args()
    with open(args.config) as fh:
        spec = AssemblySpec.from_config(yaml.safe_load(fh)["data"])
    if "cells" not in spec.blocks:
        raise SystemExit("config has no cells block")
    results = {}
    for d in spec.datasets:
        results[d.name] = study(EventStore(d.path, files=d.files, sample=d.name),
                                spec, args.delta_r)
        show(d.name, results[d.name])
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"config": args.config, "delta_r": args.delta_r,
                       "event_select": spec.event_select, "samples": results}, fh, indent=2)


if __name__ == "__main__":
    main()
