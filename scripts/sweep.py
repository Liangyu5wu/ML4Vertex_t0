#!/usr/bin/env python
"""Search hyper-parameters, ranking on the validation split.

    python scripts/sweep.py --config config/blocks/lar_hgtd.yaml \
        --space config/sweeps/architecture.yaml --out ../sweeps/arch

The search space names parameters by their dotted path in the training config,
so anything the config can express can be scanned::

    trials: 24
    epochs: 60
    space:
      optimizer.learning_rate: {log_uniform: [1e-4, 5e-3]}
      head.dropout:            {uniform: [0.0, 0.3]}
      head.units:              [[128, 64, 32, 16], [256, 128, 64], [64, 32]]
      data.inputs.cells.encoder.pooling: [attention, masked_average]

Trials that change nothing about the data share one prepared-tensor cache, so
each costs a training and nothing else. Trials run in parallel across the
visible GPUs.

A sweep can span several nodes -- NERSC allows two interactive jobs at once,
so eight GPUs. Give each node a shard of the same trial list; the shards are
drawn from the same seed, so they agree on what the trials are without
talking to each other. Rank them together when both are done:

    srun --jobid=$A ... sweep.py --shard 0/2 --out ../sweeps/arch ...   # node A
    srun --jobid=$B ... sweep.py --shard 1/2 --out ../sweeps/arch ...   # node B
    python scripts/sweep.py --report-only --space <space> --out ../sweeps/arch

Trials are ranked by the validation q68 -- the half-width holding 68% of the
errors -- and the test split is never read. Both choices matter: a fitted core
width rewards models that learned nothing (the fit finds a narrow peak in any
distribution), and tuning against the test split would make the final numbers
meaningless.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- the search space ------------------------------------------------------

def sample_value(spec: Any, rng: np.random.Generator) -> Any:
    """One draw: a list is a choice, a dict is a range."""
    if isinstance(spec, list):
        return spec[rng.integers(len(spec))]
    if isinstance(spec, dict):
        if "log_uniform" in spec:
            lo, hi = spec["log_uniform"]
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        if "uniform" in spec:
            lo, hi = spec["uniform"]
            return float(rng.uniform(lo, hi))
        if "int_uniform" in spec:
            lo, hi = spec["int_uniform"]
            return int(rng.integers(lo, hi + 1))
    raise ValueError(f"cannot sample from {spec!r}")


def grid_points(space: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Every combination, if the space is all discrete and small enough."""
    if any(not isinstance(v, list) for v in space.values()):
        return None
    points = [dict(zip(space, values)) for values in itertools.product(*space.values())]
    return points if len(points) <= 256 else None


def set_path(cfg: dict, path: str, value: Any) -> None:
    node = cfg
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in node:
            raise KeyError(f"{path}: no '{key}' in the config")
        node = node[key]
    node[keys[-1]] = value


def get_path(cfg: dict, path: str) -> Any:
    node = cfg
    for key in path.split("."):
        node = node[key]
    return node


# --- running one trial -----------------------------------------------------

def run_trial(index: int, params: Dict[str, Any], base: dict, out_dir: str,
              epochs: Optional[int], gpu: Optional[int]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    for path, value in params.items():
        set_path(cfg, path, value)
    name = f"trial_{index:03d}"
    model_dir = os.path.join(out_dir, name)
    cfg["model_name"] = name
    cfg["model_dir"] = model_dir
    if epochs:
        cfg.setdefault("training", {})["epochs"] = epochs
    os.makedirs(model_dir, exist_ok=True)
    cfg_path = os.path.join(model_dir, "trial_config.yaml")
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = os.path.join(model_dir, "train.log")
    with open(log, "w") as fh:
        code = subprocess.call(
            [sys.executable, os.path.join(REPO, "scripts", "train_blocks.py"),
             "--config", cfg_path, "--no-plots", "--verbose", "0"],
            stdout=fh, stderr=subprocess.STDOUT, cwd=REPO, env=env)

    record: Dict[str, Any] = {"trial": index, "model_dir": model_dir, **params}
    metrics_path = os.path.join(model_dir, "metrics.json")
    if code != 0 or not os.path.exists(metrics_path):
        record["objective"] = float("nan")
        record["error"] = f"exit {code}; see {log}"
        return record
    with open(metrics_path) as fh:
        metrics = json.load(fh)
    val = metrics["val"]["all"]
    record.update({
        # q68, not the fitted core width: a fit finds a narrow core even in an
        # untrained model's error distribution, so ranking on it rewards
        # models that learned nothing. See summarize().
        "objective": float(val["q68"]),
        "val_core_std": float(val["core_std"]),
        "val_core_fraction": float(val["core_fraction"]),
        "val_rmse": float(val["rmse"]),
        "val_fit_sigma": float(val.get("fit", {}).get("sigma", np.nan)),
        "best_val_loss": float(metrics["best_val_loss"]),
        "epochs_run": int(metrics["epochs_run"]),
    })
    return record


def collect(out_dir: str, space: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rebuild every finished trial's record by reading the output directory.

    Reading the directory rather than trusting what this process ran is what
    lets several sweep processes -- one per node -- share one output
    directory and still produce a single ranking.
    """
    import glob

    records = []
    for d in sorted(glob.glob(os.path.join(out_dir, "trial_*"))):
        cfg_path, metrics_path = (os.path.join(d, "trial_config.yaml"),
                                  os.path.join(d, "metrics.json"))
        if not (os.path.exists(cfg_path) and os.path.exists(metrics_path)):
            continue                          # still running, or it failed
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        with open(metrics_path) as fh:
            metrics = json.load(fh)
        val = metrics["val"]["all"]
        records.append({
            "trial": int(os.path.basename(d).rsplit("_", 1)[1]),
            "model_dir": d,
            **{path: get_path(cfg, path) for path in space},
            "objective": float(val["q68"]),
            "val_core_std": float(val["core_std"]),
            "val_core_fraction": float(val["core_fraction"]),
            "val_rmse": float(val["rmse"]),
            "val_fit_sigma": float(val.get("fit", {}).get("sigma", np.nan)),
            "best_val_loss": float(metrics["best_val_loss"]),
            "epochs_run": int(metrics["epochs_run"]),
        })
    return sorted(records, key=lambda r: r["objective"])


def run_sweep(base_config: str, space_file: str, out_dir: str,
              trials: Optional[int], epochs: Optional[int],
              gpus: Sequence[int], shard: tuple = (0, 1)) -> List[Dict[str, Any]]:
    with open(base_config) as fh:
        base = yaml.safe_load(fh)
    with open(space_file) as fh:
        spec = yaml.safe_load(fh)
    space = spec["space"]
    n_trials = trials or spec.get("trials", 16)
    epochs = epochs or spec.get("epochs")
    rng = np.random.default_rng(spec.get("seed", 0))

    points = grid_points(space)
    if points is not None and len(points) <= n_trials:
        settings = points                       # small discrete space: take all
        print(f"grid search: {len(settings)} combinations")
    else:
        settings = [{p: sample_value(s, rng) for p, s in space.items()}
                    for _ in range(n_trials)]
        print(f"random search: {len(settings)} trials over {len(space)} parameters")

    # Every shard samples the same trial list from the same seed and then takes
    # its own slice, so the shards never have to agree on anything at runtime.
    index, n_shards = shard
    mine = [(i, p) for i, p in enumerate(settings) if i % n_shards == index]
    if n_shards > 1:
        print(f"shard {index + 1}/{n_shards}: running trials "
              f"{[i for i, _ in mine]}")

    os.makedirs(out_dir, exist_ok=True)
    work: "queue.Queue" = queue.Queue()
    for i, params in mine:
        work.put((i, params))
    results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                i, params = work.get_nowait()
            except queue.Empty:
                return
            record = run_trial(i, params, base, out_dir, epochs, gpu)
            with lock:
                results.append(record)
                done, total = len(results), len(mine)
                obj = record.get("objective")
                print(f"[{done}/{total}] trial {i:3d} "
                      f"{'q68 %.1f ps' % obj if np.isfinite(obj) else record.get('error')}",
                      flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in (gpus or [None])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(results, key=lambda r: (not np.isfinite(r.get("objective", np.nan)),
                                          r.get("objective", np.inf)))


def _short(value: Any) -> str:
    """A value narrow enough for a table row."""
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, list):
        return "[" + ",".join(_short(v) for v in value) + "]"
    return str(value)


def report(results: List[Dict[str, Any]], space: Dict[str, Any], out_dir: str) -> None:
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluation import plots

    path = os.path.join(out_dir, "results.csv")
    fields = sorted({k for r in results for k in r})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fields})

    finished = [r for r in results if np.isfinite(r.get("objective", np.nan))]
    print(f"\n{len(finished)}/{len(results)} trials finished; best first")
    header = ("rank     q68  core_std  core_frac    rmse  epochs  parameters")
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(finished[:10], 1):
        params = "  ".join(f"{k.split('.')[-1]}={_short(r[k])}" for k in space)
        print(f"{rank:4d}{r['objective']:8.1f}{r['val_core_std']:10.2f}"
              f"{100 * r['val_core_fraction']:9.1f}%{r['val_rmse']:8.1f}"
              f"{r['epochs_run']:8d}  {params[:100]}")

    plots.use_style("light")
    fig, _ = plots.sweep_results(finished, objective="objective",
                                 parameters=list(space),
                                 title=os.path.basename(os.path.normpath(out_dir))
                                 + " sweep")
    if fig is not None:
        fig.savefig(os.path.join(out_dir, "sweep.png"))
        plt.close(fig)
    print(f"\nwrote {path} and sweep.png")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="base training config")
    p.add_argument("--space", required=True, help="search space")
    p.add_argument("--out", required=True, help="directory for the trials")
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None,
                   help="override the per-trial epoch budget")
    p.add_argument("--gpus", type=int, nargs="*", default=None,
                   help="GPU indices to run on (default: all visible)")
    p.add_argument("--shard", default="0/1", metavar="I/N",
                   help="run only trial i where i %% N == I; one shard per node")
    p.add_argument("--report-only", action="store_true",
                   help="rank whatever is already in --out and stop")
    args = p.parse_args()
    if not args.report_only and not args.config:
        p.error("--config is required unless --report-only")

    with open(args.space) as fh:
        space = yaml.safe_load(fh)["space"]

    if not args.report_only:
        gpus = args.gpus
        if gpus is None:
            import tensorflow as tf
            gpus = list(range(len(tf.config.list_physical_devices("GPU")))) or [None]
        index, _, total = args.shard.partition("/")
        run_sweep(args.config, args.space, args.out, args.trials, args.epochs,
                  gpus, shard=(int(index), int(total or 1)))

    # From disk, not from what this process ran, so a sharded sweep still
    # reports one ranking over every trial any shard finished.
    report(collect(args.out, space), space, args.out)


if __name__ == "__main__":
    main()
