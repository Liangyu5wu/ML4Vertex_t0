"""Several models at once: resolution against sigma-cut efficiency, and where one
model's failures go under another."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .style import SEQUENTIAL, _finish, series_colors, tokens


def efficiency_comparison(scans: Dict[str, Dict[str, list]],
                          title: str = "Resolution vs efficiency, by input"):
    """Several models, one panel per sample: q68 solid, core sigma dashed.

    ``scans`` is {model: {sample: [efficiency_scan of each seed]}}. Both
    statistics are in ps, so they share one axis rather than two; colour is
    the model and line style the statistic. Each line is its seeds' mean with
    their range shaded -- a gap narrower than the band is not a difference --
    and the q68 at full efficiency is labelled where the models differ most.
    """
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    colors, t = series_colors(), tokens()
    samples = list(dict.fromkeys(s for per in scans.values() for s in per))
    fig, axes = plt.subplots(1, len(samples), figsize=(5.6 * len(samples), 5.0),
                             sharey=True, squeeze=False, layout="constrained")
    for ax, sample in zip(axes[0], samples):
        ends = {}
        for i, (model, per) in enumerate(scans.items()):
            runs, colour = per.get(sample), colors[i % len(colors)]
            if not runs:
                continue
            x = 100 * runs[0]["efficiency"]
            for key, style in (("q68", "-"), ("core_sigma", "--")):
                y = np.array([r[key] for r in runs])
                ax.fill_between(x, np.nanmin(y, 0), np.nanmax(y, 0), color=colour,
                                alpha=0.22, linewidth=0)
                ax.plot(x, np.nanmean(y, 0), color=colour, linewidth=2.2, linestyle=style)
            ends[model] = (x[-1], np.nanmean([r["q68"][-1] for r in runs]))
        for xe, v in ends.values():
            # past the end of the line, where nothing else is drawn
            ax.annotate(f"{v:.1f}", (xe, v), xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=12, color=t["primary"])
        ax.set_xlim(0, 116)
        ax.set_xticks(range(0, 101, 20))
        ax.set_ylim(bottom=0)
        _finish(ax, sample, r"kept, tightest $\sigma$ first [%]",
                "resolution of the kept events [ps]" if ax is axes[0][0] else "",
                legend=False)
    handles = [mlines.Line2D([], [], color=colors[i % len(colors)], linewidth=2.2,
                             label=f"{m} ({len(next(iter(per.values())))} seeds)")
               for i, (m, per) in enumerate(scans.items())]
    handles += [mlines.Line2D([], [], color=t["primary"], linewidth=2.2, label="q68"),
                mlines.Line2D([], [], color=t["primary"], linewidth=2.2, linestyle="--",
                              label=r"core $\sigma$ (fit)")]
    axes[0][0].legend(handles=handles, loc="upper left", handlelength=1.8,
                      borderpad=0.2, fontsize=13)
    fig.suptitle(title, x=0.0, ha="left", fontsize=16, fontweight="bold",
                 color=t["primary"])
    return fig, axes[0]


def recovery_plot(errors: Dict[str, np.ndarray], base: str, combined: str,
                  fail: float = 60.0, bands: Sequence[float] = (20.0, 60.0, 150.0),
                  title: str = "What the combination recovers"):
    """Where one model's events go under another, and what it does to its failures.

    ``errors`` is {model: Delta t0 of the same events, in the same order}.
    Left: a migration matrix -- rows are ``base``'s |Delta t0| band, columns
    ``combined``'s, each cell the share of its row -- so the events made
    better sit below the diagonal and those made worse above it. Right:
    Delta t0 of every model for the events ``base`` gets wrong
    (> ``fail``), which shows whether the recovered ones reach the
    precision of the combined model or only that of a model without
    ``base``'s inputs.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    colors, t = series_colors(), tokens()
    a, b = np.abs(errors[base]), np.abs(errors[combined])
    failed = a > fail
    edges = np.array([0.0, *bands, np.inf])
    ra, rb = np.digitize(a, edges) - 1, np.digitize(b, edges) - 1
    n = len(edges) - 1
    counts = np.array([[np.sum((ra == i) & (rb == j)) for j in range(n)] for i in range(n)])
    share = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    names = [f"< {bands[0]:.0f}"] + [f"{lo:.0f}-{hi:.0f}" for lo, hi in
                                      zip(bands[:-1], bands[1:])] + [f"> {bands[-1]:.0f}"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.4), layout="constrained",
                                  gridspec_kw={"width_ratios": [1.0, 1.2]})
    cmap = LinearSegmentedColormap.from_list("seq_blue", [t["surface"]] + SEQUENTIAL)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1, origin="upper")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{100 * share[i, j]:.0f}%\n{counts[i, j]:,}", ha="center",
                    va="center", fontsize=11,
                    color=t["surface"] if share[i, j] > 0.55 else t["primary"])
    ax.set_xticks(range(n), names)
    ax.set_yticks(range(n), names)
    ax.minorticks_off()
    ax.tick_params(top=False, right=False)
    _finish(ax, f"Event migration, {base} $\\rightarrow$ {combined}",
            rf"$|\Delta t_0|$, {combined} [ps]", rf"$|\Delta t_0|$, {base} [ps]",
            legend=False)

    window, bins = 300.0, 60
    e_edges = np.linspace(-window, window, bins + 1)
    top = 0
    for i, (name, e) in enumerate(errors.items()):
        sel = e[failed]
        sel = sel[np.isfinite(sel)]              # a reference may not score every event
        counts, _ = np.histogram(sel, bins=e_edges)
        top = max(top, counts.max())
        note = (" (selection)" if name == base else
                f" ({len(sel):,} ev.)" if len(sel) < failed.sum() else "")
        ax2.stairs(counts, e_edges, color=colors[i % len(colors)], linewidth=2.0,
                   label=f"{name}{note}: {np.median(np.abs(sel)):.0f} ps")
    ax2.set_ylim(0, 1.45 * top)                  # room for the legend above the data
    ax2.axvline(0.0, color=t["axis"], linewidth=1.0, zorder=0)
    _finish(ax2, f"{base} $|\\Delta t_0|$ > {fail:.0f} ps: {int(failed.sum()):,} events",
            r"$\Delta t_0$ [ps]", f"events / {2 * window / bins:.0f} ps", legend=True)
    ax2.legend(loc="upper left", title=r"median $|\Delta t_0|$", handlelength=1.6,
               borderpad=0.2)
    return fig, (ax, ax2)
