#!/usr/bin/env python
"""Summarise a set of trained runs, split by whether the vertex was found.

    python scripts/compare_runs.py ../runs/lar_hgtd ../runs/lar_only ...

ATLAS takes the highest-sum-pt^2 reconstructed vertex as the hard scatter,
and it is not the right one in 5.9% of ttbar and 20.4% of VBF events. When
it is wrong the target is one vertex's time while every input describes
another, so those events are unpredictable for reasons that have nothing to
do with timing. Reported together, the ttbar/VBF gap is that rate and not a
statement about either detector, which is why everything here is split on
it.

The split is computed by joining each prediction back to the store on
(sample, event number) and comparing the reconstructed and true vertex z.
Runs are grouped by which samples they were trained on, and repeats of one
setting are reported as a mean and a spread.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.evaluation.summary import summarize
from src.pipeline.event_store import EventStore

MATCH_MM = 0.5          # |z_reco - z_truth| below this counts as the right vertex


def vertex_lookup(paths: dict) -> dict:
    """{sample: {event_number: |z_reco - z_truth|}} for every store used."""
    out = {}
    for name, path in paths.items():
        store = EventStore(path, sample=name)
        dz = np.abs(store.event_column("reco_vtx_z")
                    - store.event_column("truth_vtx_z"))
        out[name] = dict(zip(store.event_column("event_number").tolist(),
                             dz.tolist()))
    return out


def load_run(trial: str, lookup: dict) -> dict:
    """One trial: its training samples and its per-event predictions."""
    with open(os.path.join(trial, "trial_config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    z = np.load(os.path.join(trial, "predictions_test.npz"), allow_pickle=False)
    names = [str(n) for n in z["dataset_names"]]
    sample = np.array(names, dtype=object)[z["dataset_id"]]
    dz = np.array([lookup[s].get(int(e), np.nan)
                   for s, e in zip(sample, z["event_number"])])
    return {
        "trained_on": "+".join(d["name"] for d in cfg["data"]["datasets"]),
        "y_true": z["y_true"], "y_pred": z["y_pred"],
        "sigma": z["sigma"] if "sigma" in z else None,
        "sample": sample, "dz": dz,
    }


def report(runs: list, label: str) -> None:
    """Mean and spread over repeats, per sample, split on the vertex."""
    rows = defaultdict(list)
    for r in runs:
        for s in sorted(set(r["sample"])):
            in_s = r["sample"] == s
            for tag, sel in (("vertex ok", r["dz"] < MATCH_MM),
                             ("vertex wrong", r["dz"] >= MATCH_MM),
                             ("all", np.ones(len(r["dz"]), bool))):
                m = in_s & sel & np.isfinite(r["dz"])
                if m.sum() < 100:
                    continue
                sig = None if r["sigma"] is None else r["sigma"][m]
                st = summarize(r["y_true"][m], r["y_pred"][m], sigma=sig)
                rows[(s, tag)].append((st["q68"], st["core_std"],
                                       st["core_fraction"], int(m.sum())))

    print(f"\n{label}")
    print(f"{'sample':10s}{'events':>9s}{'':4s}{'q68 [ps]':>16s}"
          f"{'core std':>16s}{'core frac':>11s}")
    print("-" * 68)
    for (s, tag), vals in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        q = np.array([v[0] for v in vals])
        c = np.array([v[1] for v in vals])
        f = np.array([v[2] for v in vals])
        n = vals[0][3]
        spread = f"+-{(q.max() - q.min()) / 2:.1f}" if len(q) > 1 else ""
        print(f"{s if tag == 'all' else '':10s}{n:9d}  {tag:12s}"
              f"{q.mean():6.1f} {spread:5s}{c.mean():10.1f}"
              f"{100 * f.mean():10.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", help="directories holding trial_* runs")
    p.add_argument("--match-mm", type=float, default=MATCH_MM)
    args = p.parse_args()

    globals()["MATCH_MM"] = args.match_mm

    # Every store any run referenced, read once.
    paths = {}
    for d in args.run_dirs:
        for t in sorted(glob.glob(os.path.join(d, "trial_*"))):
            cfg_path = os.path.join(t, "trial_config.yaml")
            if os.path.exists(cfg_path):
                with open(cfg_path) as fh:
                    for ds in yaml.safe_load(fh)["data"]["datasets"]:
                        paths[ds["name"]] = ds["path"]
    print(f"vertex match: |z_reco - z_truth| < {args.match_mm} mm")
    lookup = vertex_lookup(paths)

    for d in args.run_dirs:
        trials = [t for t in sorted(glob.glob(os.path.join(d, "trial_*")))
                  if os.path.exists(os.path.join(t, "predictions_test.npz"))]
        if not trials:
            print(f"\n{d}: nothing scored yet")
            continue
        runs = [load_run(t, lookup) for t in trials]
        by_training = defaultdict(list)
        for r in runs:
            by_training[r["trained_on"]].append(r)
        name = os.path.basename(os.path.normpath(d))
        for trained_on, group in sorted(by_training.items()):
            report(group, f"=== {name}  trained on {trained_on}  "
                          f"({len(group)} seeds) ===")


if __name__ == "__main__":
    main()
