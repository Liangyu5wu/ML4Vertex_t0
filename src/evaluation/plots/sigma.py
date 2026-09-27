"""What the predicted sigma is worth: its spread, its calibration, and what a cut on it keeps."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .style import SEQUENTIAL, _ax, _finish, series_colors, tokens


MIN_KEPT = 50      # fewer kept events than this and a working point is not drawn


def _working_point_labels(working_points) -> list:
    """[(label, max_sigma)] from a sigma_cut stanza: one dict or a list of them."""
    cuts = [working_points] if isinstance(working_points, dict) else working_points or []
    return [(c.get("name") or f"{c['max_sigma']:.0f} ps", float(c["max_sigma"]))
            for c in cuts if "max_sigma" in c]


def resolution_vs_efficiency(errors: Dict[str, np.ndarray], sigma: Dict[str, np.ndarray],
                             fit: Optional[dict] = None, working_points=None,
                             ax=None, title: str = "Resolution vs efficiency"):
    """What every threshold on sigma keeps, and how well those events are measured.

    Each point is one threshold: x is the fraction it keeps (tightest sigma
    first), y the q68 (solid) and, with ``fit``, the fitted core sigma
    (dashed) of the kept events -- both in ps, on one axis. The working
    points are marked where they fall. One colour per sample, since one
    threshold keeps a different fraction of each.
    """
    import matplotlib.lines as mlines

    from src.evaluation.summary import efficiency_scan

    fig, ax = _ax(ax)
    colors, t = series_colors(), tokens()
    wps = _working_point_labels(working_points)
    handles = []
    for i, (name, e) in enumerate(errors.items()):
        colour, s = colors[i % len(colors)], sigma[name]
        scan = efficiency_scan(e, s, np.linspace(0.05, 1.0, 39), fit=fit)
        ax.plot(100 * scan["efficiency"], scan["q68"], color=colour, linewidth=2.5)
        if fit:
            ax.plot(100 * scan["efficiency"], scan["core_sigma"], color=colour,
                    linewidth=2.0, linestyle="--")
        for label, cut in wps:
            keep = s <= cut
            if keep.sum() < MIN_KEPT:
                continue
            x, y = 100 * keep.mean(), np.percentile(np.abs(e[keep]), 68)
            ax.plot([x], [y], "o", color=colour, markersize=8, zorder=4)
            if i == 0:                     # one label per working point is enough
                ax.annotate(f"{label}, {cut:.0f} ps", (x, y), xytext=(-8, 8),
                            textcoords="offset points", ha="right", fontsize=12,
                            color=t["primary"])
        handles.append(mlines.Line2D([], [], color=colour, linewidth=2.5,
                                     label=f"{name} ({len(e):,})"))
    handles.append(mlines.Line2D([], [], color=t["primary"], linewidth=2.0, label="q68"))
    if fit:
        handles.append(mlines.Line2D([], [], color=t["primary"], linewidth=2.0,
                                     linestyle="--", label="core $\\sigma$ (fit)"))
    ax.set_xlim(0, 104)
    ax.set_ylim(bottom=0)
    _finish(ax, title, r"events kept, tightest predicted $\sigma$ first [%]",
            "resolution of the kept events [ps]", legend=False)
    ax.legend(handles=handles, loc="upper left", handlelength=1.8, borderpad=0.2)
    return fig, ax


def sigma_calibration(errors: Dict[str, np.ndarray], sigma: Dict[str, np.ndarray],
                      bins: int = 12, ax=None,
                      title: str = "Predicted vs achieved resolution"):
    """Does a predicted sigma mean what it says? Binned, against y = x.

    Points on the diagonal mean the width is honest; above it the model is
    overconfident. Ordering can be right while the scale is not, and the two
    failures want different responses, so they are separated here. One series
    per sample: a single sigma can be honest for one and not the other.
    """
    fig, ax = _ax(ax)
    colors, t = series_colors(), tokens()
    top = 0.0
    for i, (name, e_all) in enumerate(errors.items()):
        s_all = sigma[name]
        edges = np.quantile(s_all, np.linspace(0, 1, bins + 1))
        pred, got, err_got = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (s_all >= lo) & (s_all < hi)
            if m.sum() < 50:
                continue
            e = e_all[m]
            pred.append(np.median(s_all[m]))
            # A robust width, so a handful of unrecoverable events cannot set it.
            got.append(0.7413 * (np.percentile(e, 75) - np.percentile(e, 25)))
            err_got.append(got[-1] / np.sqrt(2 * m.sum()))
        ax.errorbar(pred, got, yerr=err_got, fmt="o", color=colors[i % len(colors)],
                    markersize=8, linewidth=1.5, capsize=3, label=name)
        top = max(top, max(pred), max(got))

    lim = [0, 1.15 * top]
    ax.plot(lim, lim, color=t["axis"], linewidth=1.2, linestyle="--",
            label="perfectly calibrated", zorder=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    _finish(ax, title, r"predicted $\sigma$ [ps]",
            "achieved width of those events [ps]", legend=False)
    ax.legend(loc="upper left", handlelength=1.6, borderpad=0.2)
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

    width = float(edges[1] - edges[0])
    counts, _ = np.histogram(pull, bins=edges)
    ax.stairs(counts, edges, color=colors[0], linewidth=2.0,
              label=f"pull ({len(pull):,}), width {inside.std():.2f}")
    # The reference is scaled to this histogram rather than the histogram to
    # it, so the y axis stays a count.
    ax.plot(centres, len(pull) * width * np.exp(-0.5 * centres ** 2)
            / np.sqrt(2 * np.pi),
            color=t["axis"], linewidth=1.5, linestyle="--", label="unit Gaussian")
    ax.axvline(0.0, color=t["axis"], linewidth=1.0, zorder=0)
    _finish(ax, title, r"$\Delta t_0 / \sigma$",
            f"events / {width:.2f}", legend=True)
    return fig, ax


def sigma_vs_residual(errors: Dict[str, np.ndarray], sigma: Dict[str, np.ndarray],
                      window: float = 600.0, bins: int = 100,
                      title: str = "Predicted uncertainty vs residual"):
    """Predicted sigma against (predicted - true), one panel per sample.

    Small multiples on shared axes and one colour scale, so the samples
    compare by position. Sigma is on a log axis -- it spans two orders of
    magnitude -- and the dashed lines are |residual| = sigma: an honest sigma
    puts 68% of each row between them.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    t = tokens()
    cmap = LinearSegmentedColormap.from_list("seq_blue", [t["surface"]] + SEQUENTIAL)
    every = np.concatenate(list(sigma.values()))
    x_edges = np.linspace(-window, window, bins + 1)
    y_edges = np.geomspace(max(every.min(), 1.0), every.max(), bins + 1)
    counts = {n: np.histogram2d(errors[n], sigma[n], bins=[x_edges, y_edges])[0]
              for n in errors}
    norm = LogNorm(vmin=1, vmax=max(c.max() for c in counts.values()))

    fig, axes = plt.subplots(1, len(errors), figsize=(5.6 * len(errors) + 1.2, 5.4),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    for ax, (name, h) in zip(axes[0], counts.items()):
        mesh = ax.pcolormesh(x_edges, y_edges, h.T, cmap=cmap, norm=norm)
        for sign in (-1, 1):
            ax.plot(sign * y_edges, y_edges, color=t["axis"], linewidth=1.3,
                    linestyle="--")
        ax.set_yscale("log")
        ax.set_xlim(-window, window)
        _finish(ax, f"{name}  ({len(errors[name]):,})", r"$\Delta t_0$ [ps]",
                r"predicted $\sigma$ [ps]" if ax is axes[0][0] else "", legend=False)
    cbar = fig.colorbar(mesh, ax=axes[0].tolist())
    cbar.set_label("events", color=t["secondary"])
    cbar.outline.set_edgecolor(t["axis"])
    fig.suptitle(title, x=0.0, ha="left", fontsize=16, fontweight="bold",
                 color=t["primary"])
    return fig, axes[0]


def sigma_distribution(sigma: Dict[str, np.ndarray], bins: int = 60, ax=None,
                       title: str = "Predicted uncertainty"):
    """How the predicted sigma is spread, per sample and over all of them.

    What the model thinks it knows, before any question of whether it is
    right. The spread is the useful part -- a model that returned one width
    for every event would be reporting an average, not a per-event
    uncertainty -- and comparing samples shows where the easy events are.

    Log-spaced bins, because sigma runs over nearly two orders of magnitude.
    """
    fig, ax = _ax(ax)
    colors, t = series_colors(), tokens()
    every = np.concatenate(list(sigma.values()))
    edges = np.geomspace(max(every.min(), 1e-3), every.max(), bins + 1)

    series = dict(sigma)
    if len(sigma) > 1:
        series["total"] = every
    for i, (name, v) in enumerate(series.items()):
        counts, _ = np.histogram(v, bins=edges)
        # "total" is the sum of the others, so it takes the neutral ink rather
        # than a categorical slot -- it is not a fourth sample.
        colour = t["primary"] if name == "total" else colors[i % len(colors)]
        ax.stairs(counts, edges, color=colour, linewidth=2.0,
                  linestyle="--" if name == "total" else "-",
                  label=f"{name}  ({len(v):,}), median {np.median(v):.0f} ps")
    ax.set_xscale("log")
    _finish(ax, title, r"predicted $\sigma$ [ps]", "events / bin", legend=True)
    return fig, ax
