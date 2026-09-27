"""The standard plot set of a trained model, and the same again after a sigma cut."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .residual import cut_comparison, error_distribution, prediction_vs_truth, resolution_vs
from .sigma import (MIN_KEPT, pull_distribution, resolution_vs_efficiency, sigma_calibration,
                    sigma_distribution, sigma_vs_residual)
from .style import use_style
from .training import training_history


def _save(figures, outdir: str) -> List[str]:
    """Write and close ``[(file name, figure)]``; return the names."""
    import matplotlib.pyplot as plt

    for fname, fig in figures:
        fig.savefig(os.path.join(outdir, fname))
        plt.close(fig)
    return [f for f, _ in figures]


def _evaluation_cfg(model_dir: str) -> dict:
    """The ``evaluation:`` stanza of the config a model was trained with."""
    path = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(path):
        return {}
    import yaml
    with open(path) as fh:
        return (yaml.safe_load(fh) or {}).get("evaluation") or {}


def _load(path: str) -> dict:
    """A predictions file as arrays, with the sample of each event."""
    z = np.load(path, allow_pickle=False)
    y_true, y_pred = z["y_true"], z["y_pred"]
    return {"y_true": y_true, "y_pred": y_pred, "errors": y_pred - y_true,
            "sigma": z["sigma"] if "sigma" in z.files else None,
            "names": [str(n) for n in z["dataset_names"]] if "dataset_names" in z.files else ["all"],
            "ids": z["dataset_id"] if "dataset_id" in z.files else np.zeros(len(y_true), int)}


def _resolution_vs_truth(y_true, errors, labels, name: str):
    """q68 and the median Delta t0 against the true t0: where the model shrinks."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
    for ax, stat in zip(axes, ("q68", "median")):
        resolution_vs(y_true, errors, np.linspace(-600, 600, 25), ax=ax, stat=stat,
                      labels=labels, xlabel=r"$t_0^{true}$ [ps]" if stat == "median" else "",
                      title=f"{name} -- resolution vs true $t_0$" if stat == "q68" else "")
    if axes[1].get_legend():                 # one legend is enough
        axes[1].get_legend().remove()
    return fig


def _threshold_from_validation(model_dir: str, predictions: str, cut: dict) -> dict:
    """Turn a ``keep_fraction`` into a ``max_sigma`` fitted on the validation split.

    A threshold chosen on the sample it is reported on is tuned on that sample;
    the validation predictions are the model's own, so the threshold is the
    same whichever sample is scored.
    """
    val = os.path.join(model_dir, "predictions_val.npz")
    if "keep_fraction" not in cut or predictions == "predictions_val.npz" or not os.path.exists(val):
        return cut
    cut = dict(cut)
    fraction = cut.pop("keep_fraction")
    return {**cut, "max_sigma": float(np.quantile(np.load(val)["sigma"], fraction)),
            "from": f"keep_fraction {fraction} of predictions_val.npz"}


def _cut_stats(p: dict, keep: np.ndarray, groups: Dict[str, np.ndarray],
               fit: Optional[dict]) -> dict:
    """Per sample: efficiency, the summary of what is kept, and one Gaussian over it.

    The single Gaussian is fitted to the whole kept distribution, not its core,
    and the share beyond 3 sigma of it (0.27% for a Gaussian) says how Gaussian
    the kept events are.
    """
    from src.evaluation.summary import fit_core_resolution, summarize

    samples, gauss = {}, {}
    for n, m in groups.items():
        k = m & keep
        samples[n] = {"efficiency": float(keep[m].mean()),
                      **(summarize(p["y_true"][k], p["y_pred"][k], sigma=p["sigma"][k], fit=fit)
                         if k.sum() >= MIN_KEPT else {})}
        if n == "all" or k.sum() < MIN_KEPT:
            continue
        e = p["errors"][k]
        try:
            g = fit_core_resolution(e, method="single_gaussian", fit_range=np.inf)
        except RuntimeError:                            # fitting is best-effort
            continue
        g["beyond_3sigma"] = float(np.mean(np.abs(e - g["mu"]) > 3 * g["sigma"]))
        gauss[n] = g
    return {"samples": samples, "single_gaussian": gauss}


def cut_report(model_dir: str, p: dict, cut: dict, outdir: str,
               fit: Optional[dict] = None) -> Optional[str]:
    """The residual plots again, for the events a sigma cut keeps.

    Writes, into ``outdir/sigma_cut_<N>ps``: the residual (linear and log,
    refitted on the kept events, and with one Gaussian over the whole
    distribution), predicted against true, a kept/removed comparison per
    sample, and ``cut_metrics.json``. ``cut`` may carry a ``name`` (a working
    point) and a ``from`` (where its threshold came from).
    """
    from src.evaluation.summary import sigma_cut

    cut = dict(cut)
    label, origin = cut.pop("name", None), cut.pop("from", None)
    keep, threshold = sigma_cut(p["sigma"], **cut)
    if keep.sum() < MIN_KEPT:
        # lar_only has no event below 20 ps: an empty working point is a result.
        print(f"{label or 'sigma cut'} at {threshold:.1f} ps keeps {int(keep.sum())} event(s); not drawn")
        return None
    out = os.path.join(outdir, f"sigma_cut_{threshold:.0f}ps")
    os.makedirs(out, exist_ok=True)

    groups = {n: p["ids"] == i for i, n in enumerate(p["names"])}
    if len(groups) > 1:
        groups["all"] = np.ones(len(keep), dtype=bool)
    stats = {"name": label, "cut": cut, "max_sigma": threshold,
             "threshold_from": origin or ("fixed" if "max_sigma" in cut else "the evaluated sample"),
             **_cut_stats(p, keep, groups, fit)}
    with open(os.path.join(out, "cut_metrics.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    e, gauss = p["errors"], stats["single_gaussian"]
    title = (f"{os.path.basename(os.path.normpath(model_dir))}, {label + ', ' if label else ''}"
             rf"$\sigma_{{pred}}$ $\leq$ {threshold:.0f} ps")
    samples = {n: m for n, m in groups.items() if n != "all"}
    kept = {n: e[m & keep] for n, m in samples.items() if (m & keep).sum() >= MIN_KEPT}
    fits = {n: s["fit"] for n, s in stats["samples"].items()
            if isinstance(s.get("fit"), dict) and "sigma" in s["fit"]}
    figures = [("residual.png", error_distribution(kept, fits, title=title)[0]),
               ("residual_log.png", error_distribution(kept, fits, logy=True, title=title)[0]),
               ("pred_vs_true.png", prediction_vs_truth(p["y_true"][keep], p["y_pred"][keep],
                                                        title=title)[0])]
    for logy in (False, True):
        # linear and close in for the core's shape, log and wide for the tails
        fig, ax = error_distribution(kept, gauss, logy=logy, window=600.0 if logy else 150.0,
                                     title=f"{title}: single Gaussian")
        ax.text(0.03, 0.97, "\n".join(f"{n}: {100 * g['beyond_3sigma']:.1f}% beyond 3$\\sigma$"
                                      for n, g in gauss.items()) + "\n(Gaussian: 0.27%)",
                transform=ax.transAxes, va="top", fontsize=13)
        figures.append((f"residual_gauss{'_log' if logy else ''}.png", fig))
    figures += [(f"cut_{n}.png", cut_comparison(e[m], keep[m], title=f"{title}, {n}")[0])
                for n, m in samples.items()]
    _save(figures, out)
    print(f"{label or 'sigma cut'} at {threshold:.1f} ps: "
          + ", ".join(f"{n} keeps {100 * s['efficiency']:.0f}%"
                      + (f" (q68 {s['q68']:.1f})" if "q68" in s else "")
                      for n, s in stats["samples"].items()) + f"; plots in {out}")
    return out


def report(model_dir: str, predictions: str = "predictions_test.npz",
           mode: str = "light", outdir: Optional[str] = None, sigma_cut=None) -> str:
    """Write the standard plot set for a trained model. Returns the directory.

    With a sigma cut -- ``sigma_cut`` here, or ``evaluation.sigma_cut`` in the
    model's config, one cut or a list of working points -- the residual plots
    are drawn again for the kept events, by :func:`cut_report`.
    """
    import matplotlib
    matplotlib.use("Agg")

    use_style(mode)
    outdir = outdir or os.path.join(model_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    p = _load(os.path.join(model_dir, predictions))
    evaluation = _evaluation_cfg(model_dir)
    cuts = sigma_cut if sigma_cut is not None else evaluation.get("sigma_cut")

    metrics = {}
    if os.path.exists(os.path.join(model_dir, "metrics.json")):
        with open(os.path.join(model_dir, "metrics.json")) as fh:
            metrics = json.load(fh).get("test", {})
    fits = {n: metrics[n]["fit"] for n in p["names"]
            if isinstance(metrics.get(n, {}).get("fit"), dict) and "sigma" in metrics[n]["fit"]}

    name = os.path.basename(os.path.normpath(model_dir))
    ids, e = p["ids"], p["errors"]
    per = lambda a: {n: a[ids == i] for i, n in enumerate(p["names"])}  # noqa: E731
    figures = [
        ("residual.png", error_distribution(per(e), fits, title=f"{name} -- residual")[0]),
        ("residual_log.png", error_distribution(per(e), fits, logy=True,
                                                title=f"{name} -- residual (log)")[0]),
        ("pred_vs_true.png", prediction_vs_truth(p["y_true"], p["y_pred"])[0]),
        ("resolution_vs_truth.png", _resolution_vs_truth(p["y_true"], e,
                                                         np.array(p["names"])[ids], name)),
    ]
    written = []
    if p["sigma"] is not None:
        s = p["sigma"]
        figures += [
            ("sigma_distribution.png", sigma_distribution(
                per(s), title=f"{name} -- predicted uncertainty")[0]),
            ("sigma_vs_residual.png", sigma_vs_residual(
                per(e), per(s), title=f"{name} -- predicted uncertainty vs residual")[0]),
            ("resolution_vs_efficiency.png", resolution_vs_efficiency(
                per(e), per(s), fit=evaluation.get("fit"), working_points=cuts,
                title=f"{name} -- resolution vs efficiency")[0]),
            ("sigma_calibration.png", sigma_calibration(
                per(e), per(s), title=f"{name} -- predicted vs achieved")[0]),
            ("pull.png", pull_distribution(e, s, title=f"{name} -- pull")[0]),
        ]
        for cut in [cuts] if isinstance(cuts, dict) else cuts or []:
            out = cut_report(model_dir, p, _threshold_from_validation(model_dir, predictions, cut),
                             outdir, fit=evaluation.get("fit"))
            if out:
                written.append(os.path.basename(out) + "/")
    written = _save(figures, outdir) + written
    if save_training_history(model_dir, mode=mode, outdir=outdir):
        written.append("history.png")
    print(f"plots written to {outdir}: {', '.join(written)}")
    return outdir


def save_training_history(model_dir: str, mode: str = "light",
                          outdir: Optional[str] = None) -> Optional[str]:
    """Write ``plots/history.png`` from ``history.csv``, if there is one.

    Separate from :func:`report` because it costs nothing and every run keeps
    it -- a run without its loss curve cannot be argued about later.
    """
    import matplotlib
    matplotlib.use("Agg")

    history = os.path.join(model_dir, "history.csv")
    if not os.path.exists(history):
        return None
    use_style(mode)
    outdir = outdir or os.path.join(model_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, _save([("history.png", training_history(history)[0])], outdir)[0])


def main(argv: Optional[Sequence[str]] = None):
    import argparse

    p = argparse.ArgumentParser(
        description="Redraw a trained model's plots, optionally after a cut on the predicted "
                    "sigma (default: evaluation.sigma_cut in its config).")
    p.add_argument("model_dir")
    p.add_argument("--predictions", default="predictions_test.npz")
    cut = p.add_mutually_exclusive_group()
    cut.add_argument("--max-sigma", type=float, nargs="+", metavar="PS",
                     help="keep events with a predicted sigma at most this; several values "
                          "draw several working points")
    cut.add_argument("--keep-fraction", type=float, metavar="F",
                     help="keep the fraction F with the smallest predicted sigma, threshold "
                          "taken from predictions_val.npz when present")
    a = p.parse_args(argv)
    report(a.model_dir, predictions=a.predictions,
           sigma_cut=[{"max_sigma": v} for v in a.max_sigma] if a.max_sigma else
                     {"keep_fraction": a.keep_fraction} if a.keep_fraction is not None else None)
