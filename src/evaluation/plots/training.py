"""How a training went, and how a sweep ranked its trials."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .style import _finish, series_colors, tokens


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
