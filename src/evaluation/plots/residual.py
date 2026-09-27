"""The residual and what it depends on: Delta t0 distributions, binned resolution,
predicted against true, and what a selection removed."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .style import SEQUENTIAL, _ax, _finish, series_colors, tokens


def error_distribution(errors: Dict[str, np.ndarray], fits: Optional[Dict[str, dict]] = None,
                       window: float = 600.0, bins: int = 120, ax=None,
                       logy: bool = False,
                       title: str = "Vertex time residual"):
    """Histogram of (predicted - true) per sample, with the fitted core overlaid.

    Raw event counts, never a density: a histogram in this project reports how
    many events it has. Samples differ in size by a factor of two, so the
    legend carries each one's count and the reader scales by eye rather than
    being handed a normalisation they did not ask for.
    """
    from src.evaluation.summary import _gauss, _two_gauss

    fig, ax = _ax(ax)
    colors = series_colors()
    t = tokens()
    edges = np.linspace(-window, window, bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    width = float(edges[1] - edges[0])

    top = 1.0
    for i, (name, err) in enumerate(errors.items()):
        colour = colors[i % len(colors)]
        counts, _ = np.histogram(err, bins=edges)
        top = max(top, counts.max())
        ax.stairs(counts, edges, color=colour, linewidth=2.0,
                  label=f"{name}  ({len(err):,})")

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
        ax.plot(centres, curve, color=colour, linewidth=1.5, linestyle="--",
                label=f"{name} fit  $\\sigma$ = {fit['sigma']:.1f} ps")

    ax.axvline(0.0, color=t["axis"], linewidth=1.0, zorder=0)
    if logy:
        ax.set_yscale("log")
        # Below one event is nothing; a single-Gaussian curve would otherwise
        # drag the axis to 1e-300 and its margins to 1e20.
        ax.set_ylim(0.5, 20 * top)
    _finish(ax, title, r"$\Delta t_0$ [ps]", f"events / {width:.0f} ps",
            legend=True)
    return fig, ax


def resolution_vs(x: np.ndarray, errors: np.ndarray, bins: Sequence[float],
                  labels: Optional[np.ndarray] = None, core_window: float = 120.0,
                  ax=None, title: str = "Resolution", xlabel: str = "",
                  min_entries: int = 30, stat: str = "core"):
    """One statistic of the residual in bins of ``x``; a line per sample if ``labels``.

    ``stat`` is ``core`` (std within +-``core_window``), ``q68`` (of the
    absolute residual) or ``median`` (the bias). The core width hides a bin
    whose events have all moved out of the window, which is what happens far
    from t0 = 0, so binned against the truth use ``q68`` and ``median``.
    """
    fig, ax = _ax(ax)
    colors, t = series_colors(), tokens()
    bins = np.asarray(bins, dtype=float)
    centres = 0.5 * (bins[1:] + bins[:-1])
    groups = {"": slice(None)} if labels is None else \
        {str(v): (labels == v) for v in dict.fromkeys(labels)}

    for i, (name, sel) in enumerate(groups.items()):
        xs, ys, es = [], [], []
        xv, ev = x[sel], errors[sel]
        idx = np.digitize(xv, bins) - 1
        for b in range(len(bins) - 1):
            e = ev[idx == b]
            if stat == "core":
                e = e[np.abs(e) < core_window]
            if len(e) < min_entries:
                continue
            if stat == "core":
                y, err = e.std(), e.std() / np.sqrt(2 * len(e))
            elif stat == "q68":
                y = np.percentile(np.abs(e), 68)
                err = y / np.sqrt(2 * len(e))
            elif stat == "median":
                y = np.median(e)
                err = 1.2533 * 0.7413 * np.subtract(*np.percentile(e, [75, 25])) / np.sqrt(len(e))
            else:
                raise ValueError(f"resolution_vs: unknown stat {stat!r}")
            xs.append(centres[b]); ys.append(y); es.append(err)
        ax.errorbar(xs, ys, yerr=es, color=colors[i % len(colors)], marker="o",
                    markersize=6, linewidth=2.0, elinewidth=1.0, capsize=0,
                    label=name or None)

    if stat == "median":
        ax.axhline(0.0, color=t["axis"], linewidth=1.0, zorder=0)
    ylabel = {"core": rf"core width ($|\Delta t_0|$ < {core_window:.0f} ps) [ps]",
              "q68": r"q68 of $|\Delta t_0|$ [ps]",
              "median": r"median $\Delta t_0$ [ps]"}[stat]
    _finish(ax, title, xlabel, ylabel, legend=len(groups) > 1)
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


def cut_comparison(errors: np.ndarray, keep: np.ndarray, window: float = 600.0,
                   bins: int = 60, axes=None, label: str = "kept",
                   title: str = "Effect of the selection"):
    """What a selection removed, and how well it targeted what it should.

    Top: the residual before the cut, with the surviving and removed events
    drawn over it, on a log scale because the tails are the whole question.
    Bottom: the fraction surviving in each bin -- a cut that only removes
    badly predicted events shows a dip in the middle and wings near zero.

    Two panels rather than two y-axes on one: a ratio and a count do not
    share a scale.
    """
    import matplotlib.pyplot as plt

    colors, t = series_colors(), tokens()
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1],
                                              "hspace": 0.08})
    axes = np.atleast_1d(axes)
    fig = axes[0].figure
    edges = np.linspace(-window, window, bins + 1)
    width = float(edges[1] - edges[0])

    total, _ = np.histogram(errors, bins=edges)
    kept, _ = np.histogram(errors[keep], bins=edges)
    axes[0].stairs(total, edges, color=t["primary"], linewidth=2.0,
                   label=f"all ({len(errors):,})")
    axes[0].stairs(kept, edges, color=colors[0], linewidth=2.0,
                   fill=True, alpha=0.30,
                   label=f"{label} ({int(keep.sum()):,}, "
                         f"{100 * keep.mean():.0f}%)")
    axes[0].stairs(total - kept, edges, color=colors[1], linewidth=2.0,
                   label=f"removed ({int((~keep).sum()):,})")
    axes[0].set_yscale("log")
    _finish(axes[0], title, "", f"events / {width:.0f} ps", legend=True)

    # Only where there is something to divide by.
    ok = total > 0
    centres = 0.5 * (edges[1:] + edges[:-1])[ok]
    frac = kept[ok] / total[ok]
    err = np.sqrt(np.maximum(frac * (1 - frac), 0) / total[ok])
    axes[1].errorbar(centres, 100 * frac, yerr=100 * err, fmt="o",
                     color=colors[0], markersize=4, linewidth=1.0)
    axes[1].axhline(100 * keep.mean(), color=t["axis"], linewidth=1.2,
                    linestyle="--")
    axes[1].set_ylim(0, 105)
    _finish(axes[1], "", r"$\Delta t_0$ [ps]", f"{label} [%]", legend=False)
    return fig, axes
