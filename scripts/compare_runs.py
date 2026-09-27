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


def load_run(trial: str, lookup: dict, predictions: str = "predictions_test.npz"
             ) -> dict:
    """One trial: its training samples and one set of per-event predictions.

    ``predictions`` selects the file, so the same reader serves a model's own
    test split and the cross-sample scores `evaluate_blocks.py` writes beside
    it as ``predictions_xeval_<sample>.npz``.
    """
    with open(os.path.join(trial, "trial_config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    z = np.load(os.path.join(trial, predictions), allow_pickle=False)
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


def matrix(run_dirs, lookup, match_mm: float) -> None:
    """Every (input set, training samples, evaluation sample) cell, together.

    The cells off the diagonal are what say whether the model learned the
    physics or the sample: they come from scoring a single-sample model on
    the other sample, which `evaluate_blocks.py` does with the training
    scalers rather than refitting on what is being measured.
    """
    rows = defaultdict(list)
    for d in run_dirs:
        cfg_name = os.path.basename(os.path.normpath(d))
        for t in sorted(glob.glob(os.path.join(d, "trial_*"))):
            for f in sorted(glob.glob(os.path.join(t, "predictions*.npz"))):
                if not os.path.exists(os.path.join(t, "trial_config.yaml")):
                    continue
                r = load_run(t, lookup, os.path.basename(f))
                for s in sorted(set(r["sample"])):
                    m = (r["sample"] == s) & (r["dz"] < match_mm) & np.isfinite(r["dz"])
                    if m.sum() < 100:
                        continue
                    rows[(cfg_name, r["trained_on"], s)].append(
                        summarize(r["y_true"][m], r["y_pred"][m])["q68"])

    order = ["ttbar", "ttbar+vbf_hinv", "vbf_hinv"]
    evals = ["ttbar", "vbf_hinv"]
    print(f"\n{'=' * 72}\nq68 [ps], events whose vertex was found, "
          f"mean +- half-range over seeds\n{'=' * 72}")
    for cfg_name in [os.path.basename(os.path.normpath(d)) for d in run_dirs]:
        print(f"\n{cfg_name}")
        print(f"  {'trained on \\ scored on':26s}" + "".join(f"{e:>16s}" for e in evals))
        for tr in order:
            line = f"  {tr:26s}"
            for ev in evals:
                v = rows.get((cfg_name, tr, ev))
                line += (f"{np.mean(v):11.1f} +-{(max(v) - min(v)) / 2:3.1f}"
                         if v else f"{'-':>16s}")
            print(line)


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


def efficiency_plot(run_dirs, out: str, trained_on: str = "ttbar+vbf_hinv") -> None:
    """Resolution against sigma-cut efficiency, one line per input set.

    Draws plots.efficiency_comparison and prints the two readings of it: the
    resolution each input set reaches at a given efficiency, and the
    efficiency it keeps at a given resolution. Only trials trained on
    ``trained_on`` are used, so every input set is compared on equal terms.
    """
    import matplotlib
    matplotlib.use("Agg")

    from src.evaluation import plots
    from src.evaluation.summary import efficiency_scan

    scans = {}
    for d in run_dirs:
        label = os.path.basename(os.path.normpath(d))
        for t in sorted(glob.glob(os.path.join(d, "trial_*"))):
            with open(os.path.join(t, "trial_config.yaml")) as fh:
                cfg = yaml.safe_load(fh)
            if "+".join(x["name"] for x in cfg["data"]["datasets"]) != trained_on:
                continue
            z = np.load(os.path.join(t, "predictions_test.npz"))
            fit = (cfg.get("evaluation") or {}).get("fit")
            for i, s in enumerate(str(n) for n in z["dataset_names"]):
                m = z["dataset_id"] == i
                scans.setdefault(label, {}).setdefault(s, []).append(
                    efficiency_scan(z["errors"][m], z["sigma"][m], fit=fit))

    plots.use_style("light")
    fig, _ = plots.efficiency_comparison(
        scans, title=f"Resolution vs $\\sigma$-cut efficiency (trained on {trained_on})")
    fig.savefig(out)
    print(f"wrote {out}")

    def cell(v):
        return f"{np.nanmean(v):6.1f}+-{(np.nanmax(v) - np.nanmin(v)) / 2:3.1f}"

    samples = list(dict.fromkeys(s for per in scans.values() for s in per))
    print(f"\nresolution at a given efficiency [ps], mean +- half-range over seeds")
    for key in ("q68", "core_sigma"):
        for eff in (0.5, 0.7, 0.9, 1.0):
            print(f"  {key:10s} at {eff:4.0%}: " + "   ".join(
                f"{m} {s}: " + cell([np.interp(eff, r['efficiency'], r[key]) for r in per[s]])
                for m, per in scans.items() for s in samples if s in per))
    print(f"\nefficiency at a given q68 [%]")
    for target in (15.0, 20.0, 25.0, 30.0):
        # q68 rises with efficiency; the running maximum makes it monotone
        # for the interpolation, and a target never reached gives 0.
        print(f"  q68 <= {target:4.0f} ps: " + "   ".join(
            f"{m} {s}: " + cell([100 * np.interp(target, np.maximum.accumulate(r["q68"]),
                                                  r["efficiency"], left=0.0, right=1.0)
                                 for r in per[s]])
            for m, per in scans.items() for s in samples if s in per))


def recovery(combined: str, base: str, reference: str, out: str,
             trained_on: str = "ttbar+vbf_hinv") -> None:
    """Match three input sets event by event and draw plots.recovery_plot.

    Trials are paired by index (trial_000 with trial_000, ...), trained on
    ``trained_on`` in all three, and their events pooled. The test split of
    each sample is the same whatever the inputs, so an event number names
    the same event in every run.
    """
    import matplotlib
    matplotlib.use("Agg")

    from src.evaluation import plots

    def trials(d):
        out = []
        for t in sorted(glob.glob(os.path.join(d, "trial_*"))):
            with open(os.path.join(t, "trial_config.yaml")) as fh:
                cfg = yaml.safe_load(fh)
            if "+".join(x["name"] for x in cfg["data"]["datasets"]) == trained_on:
                z = np.load(os.path.join(t, "predictions_test.npz"))
                key = z["dataset_id"].astype(np.int64) * 10 ** 9 + z["event_number"]
                out.append(dict(zip(key.tolist(), z["errors"].tolist())))
        return out

    dirs = {os.path.basename(os.path.normpath(d)): d for d in (combined, base, reference)}
    names = list(dirs)
    runs = {name: trials(d) for name, d in dirs.items()}
    # combined and base share one test split (both require an HGTD track);
    # the reference need not -- lar_only keeps other events, so its split
    # differs -- and is NaN wherever it did not score the event.
    pooled = {name: [] for name in names}
    for rc, rb, rr in zip(*(runs[n] for n in names)):
        common = sorted(set(rc) & set(rb))
        for name, r in zip(names, (rc, rb, rr)):
            pooled[name].extend(r.get(k, np.nan) for k in common)
    errors = {name: np.array(v) for name, v in pooled.items()}
    plots.use_style("light")
    fig, _ = plots.recovery_plot(errors, base=names[1], combined=names[0],
                                 title=f"{names[1]} -> {names[0]}")
    fig.savefig(out)
    print(f"wrote {out}: {len(errors[names[0]]):,} events from "
          f"{len(runs[names[0]])} seed pairs, {int(np.isfinite(errors[names[2]]).sum()):,} "
          f"of them also scored by {names[2]}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", help="directories holding trial_* runs")
    p.add_argument("--match-mm", type=float, default=MATCH_MM)
    p.add_argument("--matrix", action="store_true",
                   help="one table per input set: training samples x scored sample")
    p.add_argument("--efficiency-plot", metavar="PNG",
                   help="resolution against sigma-cut efficiency, one line per input set")
    p.add_argument("--trained-on", default="ttbar+vbf_hinv",
                   help="with --efficiency-plot, which training to compare")
    p.add_argument("--recovery", metavar="PNG",
                   help="event-by-event: where the second run dir's failures go "
                        "under the first; the third is drawn for reference")
    args = p.parse_args()
    if args.recovery:
        if len(args.run_dirs) != 3:
            p.error("--recovery takes three run dirs: combined, base, reference")
        recovery(*args.run_dirs, args.recovery, args.trained_on)
        return
    if args.efficiency_plot:
        efficiency_plot(args.run_dirs, args.efficiency_plot, args.trained_on)
        return

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

    if args.matrix:
        matrix(args.run_dirs, lookup, args.match_mm)
        return

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
