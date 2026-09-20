"""Plots for t0 regression results -- one style, applied everywhere.

Every figure in the repo goes through this module so that colour, type scale,
frame weight and annotation style are identical across plots and across
people.

The visual style is the ATLAS house one: a closed black frame, ticks turned
inward on all four sides with minors shown, no grid, and type large enough to
stay readable when the figure is shrunk into a paper column. Only those
parameters are ATLAS; the rules underneath are the usual ones -- categorical
hues assigned in fixed order and never cycled, one hue light-to-dark for
density, one axis per plot, text in ink rather than in a series colour.

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
    "light": {"surface": "#ffffff", "primary": "#000000", "secondary": "#000000",
              "muted": "#000000", "grid": "#e1e0d9", "axis": "#000000"},
    "dark": {"surface": "#1a1a19", "primary": "#ffffff", "secondary": "#ffffff",
             "muted": "#ffffff", "grid": "#2c2c2a", "axis": "#ffffff"},
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
        # ATLAS house style: a closed black frame, ticks turned inward on all
        # four sides with minors shown, no grid, and type large enough to
        # survive being shrunk into a paper column.
        "font.size": 15,
        "axes.titlesize": 16, "axes.titleweight": "bold",
        "axes.titlecolor": t["primary"], "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 17, "axes.labelcolor": t["primary"],
        "text.color": t["primary"],
        "axes.grid": False,
        "grid.color": t["grid"], "grid.linewidth": 0.8, "grid.alpha": 1.0,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.spines.left": True, "axes.spines.bottom": True,
        "axes.edgecolor": t["axis"], "axes.linewidth": 1.3,
        "xtick.color": t["primary"], "ytick.color": t["primary"],
        "xtick.labelcolor": t["primary"], "ytick.labelcolor": t["primary"],
        "xtick.labelsize": 15, "ytick.labelsize": 15,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 9, "ytick.major.size": 9,
        "xtick.minor.size": 4.5, "ytick.minor.size": 4.5,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "xtick.minor.width": 1.0, "ytick.minor.width": 1.0,
        "lines.linewidth": 2.0, "lines.markersize": 7,
        "legend.frameon": False, "legend.labelcolor": t["primary"],
        "legend.fontsize": 14,
    })


def _ax(ax=None, figsize=(7.2, 5.4)):
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


def resolution_vs_efficiency(errors: np.ndarray, sigma: np.ndarray,
                             ax=None, title: str = "Resolution vs efficiency"):
    """Resolution of the events kept, against the fraction kept, cutting on sigma.

    The one plot of the predicted uncertainty that an analysis can act on: it
    says what a tighter selection buys, and the selection needs no truth, only
    the number the model already outputs. The horizontal line is what keeping
    everything gives.
    """
    fig, ax = _ax(ax)
    colors, t = series_colors(), tokens()
    order = np.argsort(sigma)
    err = np.abs(errors[order])
    keep = np.linspace(0.05, 1.0, 96)
    q68 = [np.percentile(err[:max(int(f * len(err)), 20)], 68) for f in keep]

    ax.plot(100 * keep, q68, color=colors[0], linewidth=2.5)
    ax.axhline(q68[-1], color=t["axis"], linewidth=1.2, linestyle="--")
    ax.annotate(f"all events, {q68[-1]:.1f} ps", (8, q68[-1]), va="top",
                xytext=(0, -6), textcoords="offset points",
                fontsize=13, color=t["primary"])
    for f in (0.5, 0.8):
        i = int(np.argmin(np.abs(keep - f)))
        ax.plot([100 * keep[i]], [q68[i]], "o", color=colors[1], zorder=4)
        # Label to the left of the marker: at 80% there is no room to its right.
        ax.annotate(f"{100 * keep[i]:.0f}%: {q68[i]:.1f} ps",
                    (100 * keep[i], q68[i]), textcoords="offset points",
                    xytext=(-10, -20), ha="right",
                    fontsize=13, color=t["primary"])
    ax.set_xlim(0, 104)
    _finish(ax, title, r"events kept, tightest predicted $\sigma$ first [%]",
            "q68 of the kept events [ps]", legend=False)
    return fig, ax


def sigma_calibration(errors: np.ndarray, sigma: np.ndarray, bins: int = 12,
                      ax=None, title: str = "Predicted vs achieved resolution"):
    """Does a predicted sigma mean what it says? Binned, against y = x.

    Points on the diagonal mean the width is honest; above it the model is
    overconfident. Ordering can be right while the scale is not, and the two
    failures want different responses, so they are separated here.
    """
    fig, ax = _ax(ax)
    colors, t = series_colors(), tokens()
    edges = np.quantile(sigma, np.linspace(0, 1, bins + 1))
    pred, got, err_got = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sigma >= lo) & (sigma < hi)
        if m.sum() < 50:
            continue
        e = errors[m]
        pred.append(np.median(sigma[m]))
        # A robust width, so a handful of unrecoverable events cannot set it.
        got.append(0.7413 * (np.percentile(e, 75) - np.percentile(e, 25)))
        err_got.append(got[-1] / np.sqrt(2 * m.sum()))

    lim = [0, 1.15 * max(max(pred), max(got))]
    ax.plot(lim, lim, color=t["axis"], linewidth=1.2, linestyle="--",
            label="perfectly calibrated")
    ax.errorbar(pred, got, yerr=err_got, fmt="o", color=colors[0],
                markersize=8, linewidth=1.5, capsize=3, label="measured")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    _finish(ax, title, r"predicted $\sigma$ [ps]",
            "achieved width of those events [ps]", legend=True)
    return fig, ax


def pull_distribution(errors: np.ndarray, sigma: np.ndarray, window: float = 5.0,
                      bins: int = 100, ax=None, title: str = "Pull"):
    """(predicted - true) / predicted sigma, against a unit Gaussian.

    The compact form of the calibration question: if the widths are honest
    this is a standard normal, and its own width is the factor they are out
    by, in one number rather than a curve.
    """
    fig, ax = _ax(ax)
    colors, t = series_colors(), tokens()
    pull = errors / np.maximum(sigma, 1e-9)
    inside = pull[np.abs(pull) < window]
    edges = np.linspace(-window, window, bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])

    counts, _ = np.histogram(pull, bins=edges)
    ax.stairs(counts / (len(pull) * (edges[1] - edges[0])), edges,
              color=colors[0], linewidth=2.0,
              label=f"pull, width {inside.std():.2f}")
    ax.plot(centres, np.exp(-0.5 * centres ** 2) / np.sqrt(2 * np.pi),
            color=t["axis"], linewidth=1.5, linestyle="--", label="unit Gaussian")
    ax.axvline(0.0, color=t["axis"], linewidth=1.0, zorder=0)
    _finish(ax, title, r"(predicted - true) / predicted $\sigma$",
            "events (normalised)", legend=True)
    return fig, ax


def training_history(history_csv: str, axes=None, title: str = "Training"):
    """Loss and the physical error against epoch, train and validation.

    Two panels because with a likelihood loss the loss itself is no longer in
    picoseconds: the left panel says whether the fit is converging, the right
    one says what it is worth. The chosen epoch is the one early stopping
    restored, not the last.
    """
    import csv

    import matplotlib.pyplot as plt

    with open(history_csv) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None, None
    epochs = [int(r["epoch"]) + 1 for r in rows]
    colors, t = series_colors(), tokens()

    # Whichever error metric this run recorded; Keras keeps the metric's own
    # name, which for the ones defined here starts with an underscore.
    def recorded(kind):
        return next((c for c in rows[0] if c.lstrip("_").startswith(kind)
                     and not c.startswith("val_")), None)

    metric = recorded("rmse") or recorded("mae")
    unit = "RMSE [ps]" if (metric or "").lstrip("_").startswith("rmse") else "MAE [ps]"
    panels = [("loss", "loss")] + ([(metric, unit)] if metric else [])

    if axes is None:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 3.8))
    axes = np.atleast_1d(axes)
    fig = axes[0].figure

    best = int(np.argmin([float(r["val_loss"]) for r in rows]))
    for ax, (key, label) in zip(axes, panels):
        shown = []
        for i, prefix in enumerate(("", "val_")):
            if prefix + key not in rows[0]:
                continue
            values = [float(r[prefix + key]) for r in rows]
            shown += values
            ax.plot(epochs, values, color=colors[i], linewidth=2.0,
                    label="validation" if prefix else "training")
        # A log axis is right when the curve falls by orders of magnitude and
        # unreadable when it does not.
        if shown and min(shown) > 0 and max(shown) / min(shown) > 5:
            ax.set_yscale("log")
        ax.axvline(epochs[best], color=t["axis"], linewidth=1.0, linestyle="--",
                   zorder=0)
        # Along the line rather than above it: the curves fill the bottom and
        # the title and legend fill the top, but the line's own track is free.
        ax.annotate(f"best epoch {epochs[best]} ", (epochs[best], 0.5),
                    xycoords=("data", "axes fraction"), rotation=90,
                    ha="right", va="center", fontsize=9, color=t["muted"])
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(axis="both")
        _finish(ax, title if key == "loss" else "", "epoch", label,
                legend=ax is axes[0])
    fig.tight_layout()
    return fig, axes


def sweep_results(trials: Sequence[dict], objective: str = "objective",
                  parameters: Optional[Sequence[str]] = None, title: str = "Sweep",
                  ylabel: str = "validation q68 [ps]"):
    """One panel per hyper-parameter: its value against the objective.

    A scan is usually asked to answer two things -- which setting is best, and
    which settings matter at all. Reading the spread along each axis answers
    the second, which is the one that generalises.
    """
    import matplotlib.pyplot as plt

    finished = [t for t in trials if np.isfinite(t.get(objective, np.nan))]
    if not finished:
        return None, None
    parameters = list(parameters or [k for k in finished[0]
                                     if k not in (objective, "trial", "model_dir")])
    n = len(parameters)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                             squeeze=False)
    colors, t = series_colors(), tokens()
    y = np.array([f[objective] for f in finished], dtype=float)
    best = int(np.argmin(y))

    for k, (ax, param) in enumerate(zip(axes.flat, parameters)):
        raw = [f.get(param) for f in finished]
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v in raw)
        if numeric:
            x = np.array(raw, dtype=float)
            if x.min() > 0 and x.max() / max(x.min(), 1e-12) > 50:
                ax.set_xscale("log")
        else:                                   # categorical: one column each
            levels = list(dict.fromkeys(str(v) for v in raw))
            x = np.array([levels.index(str(v)) for v in raw], dtype=float)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels, rotation=20, ha="right", fontsize=8)
        ax.scatter(x, y, s=26, color=colors[0], alpha=0.75, edgecolor=t["surface"],
                   linewidth=0.5, zorder=3)
        ax.scatter([x[best]], [y[best]], s=90, color=colors[1], zorder=4,
                   edgecolor=t["surface"], linewidth=1.0)
        # A few trials that never converged otherwise squash every good one
        # into the bottom centimetre of the panel.
        if y.min() > 0 and y.max() / y.min() > 3:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(plt.ScalarFormatter())   # ps, not 6x10^1
            ax.yaxis.set_minor_formatter(plt.ScalarFormatter())
        ax.set_axisbelow(True)
        ax.set_xlabel(param.split(".")[-1] if "." in param else param)
        if k % cols == 0:
            ax.set_ylabel(ylabel)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{title}  ({len(finished)} trials, best marked)", x=0.02,
                 ha="left", fontsize=12, fontweight="bold", color=t["primary"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes


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

    # What the predicted sigma is worth: what a cut on it buys, whether its
    # scale is honest, and the same question in one number.
    if "sigma" in data.files:
        sigma = data["sigma"]
        for fn, fname, kw in (
                (resolution_vs_efficiency, "resolution_vs_efficiency.png",
                 {"title": f"{name} -- resolution vs efficiency"}),
                (sigma_calibration, "sigma_calibration.png",
                 {"title": f"{name} -- predicted vs achieved"}),
                (pull_distribution, "pull.png", {"title": f"{name} -- pull"})):
            fig, _ = fn(errors, sigma, **kw)
            fig.savefig(os.path.join(outdir, fname)); plt.close(fig)
            written.append(fname)

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
    import matplotlib.pyplot as plt

    history = os.path.join(model_dir, "history.csv")
    if not os.path.exists(history):
        return None
    use_style(mode)
    outdir = outdir or os.path.join(model_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "history.png")
    fig, _ = training_history(history)
    fig.savefig(path)
    plt.close(fig)
    return path
