"""The one plotting style: ATLAS house frame, a validated palette, ink for text.

The visual style is the ATLAS house one: a closed black frame, ticks turned
inward on all four sides with minors shown, no grid, and type large enough to
stay readable when the figure is shrunk into a paper column. Samples take the
categorical slots in a fixed order (blue, orange, aqua) and never cycle --
the three documented to clear the colour-vision-deficiency and contrast gates
for an overlay -- density uses one blue ramp, never a rainbow, and text stays
in ink rather than a series colour.
"""

from __future__ import annotations

from typing import Optional, Sequence

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
