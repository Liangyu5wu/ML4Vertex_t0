"""Plots for t0 regression results -- one style, applied everywhere.

Every figure in the repo goes through this module so that colour, type scale,
grid weight and annotation style are identical across plots and across people.

Colour follows a validated categorical palette: samples take slots in a fixed
order (blue, orange, aqua) and never cycle -- those three slots are the ones
documented to clear the colour-vision-deficiency and contrast gates for
all-pairs comparison, which is what an overlay of distributions needs. Beyond
three samples the extra ones fold into a facet rather than inventing hues.
Continuous density uses a single-hue blue ramp, never a rainbow.

    from src.evaluation import plots
    plots.report("../models/lar_hgtd")           # the standard set
    plots.error_distribution({"ttbar": err}, ax=ax)   # or one at a time
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence

import numpy as np

# --- design tokens ---------------------------------------------------------
# Categorical slots 1-3, assigned in order and never cycled.
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark": ["#3987e5", "#d95926", "#199e70"],
}
# Single-hue sequential ramp (light -> dark) for density.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

INK = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}

_MODE = "light"


def tokens(mode: Optional[str] = None) -> dict:
    return INK[mode or _MODE]


def series_colors(mode: Optional[str] = None) -> Sequence[str]:
    return SERIES[mode or _MODE]


def use_style(mode: str = "light") -> None:
    """Apply the shared matplotlib style. Call once before plotting."""
    global _MODE
    import matplotlib as mpl

    _MODE = mode
    t = INK[mode]
    mpl.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "savefig.bbox": "tight",
        "savefig.dpi": 200, "figure.dpi": 110,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlecolor": t["primary"], "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10, "axes.labelcolor": t["secondary"],
        "text.color": t["primary"],
        # Recessive chrome: hairline y-grid, no top/right spines.
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": t["grid"], "grid.linewidth": 0.8, "grid.alpha": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": t["axis"], "axes.linewidth": 1.0,
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["muted"], "ytick.labelcolor": t["muted"],
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "legend.frameon": False, "legend.labelcolor": t["secondary"],
        "legend.fontsize": 9,
    })


def _ax(ax=None, figsize=(6.4, 4.2)):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def _finish(ax, title: str, xlabel: str, ylabel: str, legend: bool) -> None:
    t = tokens()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    if legend:
        # Identity rides on the coloured handle; the text stays in ink.
        ax.legend(loc="upper right", handlelength=1.6, borderpad=0.2)


# --- the plots -------------------------------------------------------------

def error_distribution(errors: Dict[str, np.ndarray], fits: Optional[Dict[str, dict]] = None,
                       window: float = 600.0, bins: int = 120, ax=None,
                       density: Optional[bool] = None, logy: bool = False,
                       title: str = "Vertex time residual"):
    """Histogram of (predicted - true) per sample, with the fitted core overlaid.

    With more than one sample the histograms are drawn as densities, because
    the samples differ in size by a factor of several and the question being
    asked of the plot is about shape.
    """
    from src.evaluation.summary import _gauss, _two_gauss

    fig, ax = _ax(ax)
    colors = series_colors()
    t = tokens()
    edges = np.linspace(-window, window, bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    width = float(edges[1] - edges[0])
    if density is None:
        density = len(errors) > 1

    for i, (name, err) in enumerate(errors.items()):
        colour = colors[i % len(colors)]
        counts, _ = np.histogram(err, bins=edges)
        scale = 1.0 / (len(err) * width) if density else 1.0
        ax.stairs(counts * scale, edges, color=colour, linewidth=2.0, label=name)

        fit = (fits or {}).get(name)
        if not fit or "sigma" not in fit:
            continue
        if "bkg_amplitude" in fit:
            curve = _two_gauss(centres, fit["core_amplitude"], fit["mu"], fit["sigma"],
                               fit["bkg_amplitude"], fit.get("sigma_bkg", 175.0))
        else:
            curve = _gauss(centres, fit["core_amplitude"], fit["mu"], fit["sigma"])
        if "bin_width" in fit:
            # Amplitudes are counts per fit bin; convert to this plot's binning.
            curve = curve * (width / fit["bin_width"])
        else:
            curve = curve * (counts.max() / max(curve.max(), 1e-9))   # legacy files
        ax.plot(centres, curve * scale, color=colour, linewidth=1.5, linestyle="--",
                label=f"{name} fit  $\\sigma$ = {fit['sigma']:.1f} ps")

    ax.axvline(0.0, color=t["axis"], linewidth=1.0, zorder=0)
    if logy:
        ax.set_yscale("log")
    _finish(ax, title, "predicted - true [ps]",
            "events / ps (normalised)" if density else "events", legend=True)
    return fig, ax


def resolution_vs(x: np.ndarray, errors: np.ndarray, bins: Sequence[float],
                  labels: Optional[np.ndarray] = None, core_window: float = 120.0,
                  ax=None, title: str = "Resolution", xlabel: str = "",
                  min_entries: int = 30):
    """Core resolution in bins of ``x``; one line per sample if ``labels`` given.

    The core width (std of residuals within +-``core_window``) is plotted with a
    bootstrap-free standard error, alongside the fraction of events in the core
    -- for samples like VBF the fraction moves more than the width does.
    """
    fig, ax = _ax(ax)
    colors = series_colors()
    bins = np.asarray(bins, dtype=float)
    centres = 0.5 * (bins[1:] + bins[:-1])
    groups = {"": slice(None)} if labels is None else \
        {str(v): (labels == v) for v in dict.fromkeys(labels)}

    for i, (name, sel) in enumerate(groups.items()):
        colour = colors[i % len(colors)]
        xs, ys, es = [], [], []
        xv, ev = x[sel], errors[sel]
        idx = np.digitize(xv, bins) - 1
        for b in range(len(bins) - 1):
            in_bin = ev[idx == b]
            core = in_bin[np.abs(in_bin) < core_window]
            if len(core) < min_entries:
                continue
            xs.append(centres[b])
            ys.append(core.std())
            es.append(core.std() / np.sqrt(2 * len(core)))
        ax.errorbar(xs, ys, yerr=es, color=colour, marker="o", markersize=6,
                    linewidth=2.0, elinewidth=1.0, capsize=0,
                    label=name or None)

    _finish(ax, title, xlabel, f"core width ($|e|$ < {core_window:.0f} ps) [ps]",
            legend=len(groups) > 1)
    return fig, ax


def sample_comparison(metrics: Dict[str, Dict[str, float]], key: str = "core_std",
                      ax=None, title: Optional[str] = None, unit: str = "ps"):
    """Bar chart of one metric across samples or models, with direct labels."""
    fig, ax = _ax(ax, figsize=(5.6, 3.6))
    colors = series_colors()
    t = tokens()
    names = list(metrics)
    values = [metrics[n][key] for n in names]

    bars = ax.bar(names, values, width=0.6,
                  color=[colors[i % len(colors)] for i in range(len(names))],
                  edgecolor=t["surface"], linewidth=2.0)        # 2px surface gap
    for rect, value in zip(bars, values):
        ax.annotate(f"{value:.1f}", (rect.get_x() + rect.get_width() / 2,
                                     rect.get_height()),
                    ha="center", va="bottom", fontsize=10, color=t["primary"],
                    xytext=(0, 3), textcoords="offset points")
    ax.grid(axis="y")
    ax.set_ylim(0, max(values) * 1.18)
    _finish(ax, title or key.replace("_", " "), "", f"{key.replace('_', ' ')} [{unit}]",
            legend=False)
    return fig, ax


def prediction_vs_truth(y_true: np.ndarray, y_pred: np.ndarray, window: float = 600.0,
                        bins: int = 100, ax=None, title: str = "Predicted vs true"):
    """2-D density on a single-hue ramp, with the y = x reference."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    fig, ax = _ax(ax, figsize=(5.2, 4.6))
    t = tokens()
    cmap = LinearSegmentedColormap.from_list("seq_blue", [t["surface"]] + SEQUENTIAL)
    edges = np.linspace(-window, window, bins + 1)
    h = ax.hist2d(y_true, y_pred, bins=[edges, edges], cmap=cmap,
                  norm=LogNorm(vmin=1))
    ax.plot([-window, window], [-window, window], color=t["axis"], linewidth=1.5,
            linestyle="--", zorder=3)
    cbar = fig.colorbar(h[3], ax=ax, pad=0.02)
    cbar.set_label("events", color=t["secondary"])
    cbar.ax.tick_params(colors=t["muted"])
    cbar.outline.set_edgecolor(t["axis"])
    ax.grid(False)
    _finish(ax, title, "true vertex time [ps]", "predicted vertex time [ps]", legend=False)
    return fig, ax


def training_history(history_csv: str, ax=None, title: str = "Training"):
    """Train/validation loss against epoch."""
    import csv

    fig, ax = _ax(ax)
    colors = series_colors()
    with open(history_csv) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return fig, ax
    epochs = [int(r["epoch"]) + 1 for r in rows]
    for i, key in enumerate(("loss", "val_loss")):
        if key not in rows[0]:
            continue
        ax.plot(epochs, [float(r[key]) for r in rows], color=colors[i],
                linewidth=2.0, label=key.replace("_", " "))
    ax.set_yscale("log")
    ax.grid(axis="both")
    _finish(ax, title, "epoch", "loss", legend=True)
    return fig, ax


# --- the standard set ------------------------------------------------------

def report(model_dir: str, predictions: str = "predictions_test.npz",
           mode: str = "light", outdir: Optional[str] = None) -> str:
    """Write the standard plot set for a trained model. Returns the directory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use_style(mode)
    outdir = outdir or os.path.join(model_dir, "plots")
    os.makedirs(outdir, exist_ok=True)

    data = np.load(os.path.join(model_dir, predictions), allow_pickle=False)
    y_true, y_pred = data["y_true"], data["y_pred"]
    errors = data["errors"] if "errors" in data else y_pred - y_true
    names = [str(n) for n in data["dataset_names"]] if "dataset_names" in data else ["all"]
    ids = data["dataset_id"] if "dataset_id" in data else np.zeros(len(y_true), int)

    metrics_path = os.path.join(model_dir, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            metrics = json.load(fh).get("test", {})

    per_sample = {n: errors[ids == i] for i, n in enumerate(names)}
    fits = {n: metrics[n]["fit"] for n in names
            if n in metrics and isinstance(metrics[n].get("fit"), dict)
            and "sigma" in metrics[n]["fit"]}

    written = []
    name = os.path.basename(os.path.normpath(model_dir))
    fig, _ = error_distribution(per_sample, fits, title=f"{name} -- residual")
    fig.savefig(os.path.join(outdir, "residual.png")); plt.close(fig)
    written.append("residual.png")

    # The tails are the interesting part for VBF, so also show them on a log scale.
    fig, _ = error_distribution(per_sample, fits, logy=True,
                                title=f"{name} -- residual (log)")
    fig.savefig(os.path.join(outdir, "residual_log.png")); plt.close(fig)
    written.append("residual_log.png")

    fig, _ = prediction_vs_truth(y_true, y_pred)
    fig.savefig(os.path.join(outdir, "pred_vs_true.png")); plt.close(fig)
    written.append("pred_vs_true.png")

    comparable = {n: metrics[n] for n in names if n in metrics}
    if len(comparable) > 1:
        for key, unit in (("core_std", "ps"), ("core_fraction", "")):
            fig, _ = sample_comparison(comparable, key=key,
                                       title=key.replace("_", " ") + " by sample",
                                       unit=unit or "-")
            fig.savefig(os.path.join(outdir, f"{key}_by_sample.png")); plt.close(fig)
            written.append(f"{key}_by_sample.png")

    history = os.path.join(model_dir, "history.csv")
    if os.path.exists(history):
        fig, _ = training_history(history)
        fig.savefig(os.path.join(outdir, "history.png")); plt.close(fig)
        written.append("history.png")

    print(f"plots written to {outdir}: {', '.join(written)}")
    return outdir
