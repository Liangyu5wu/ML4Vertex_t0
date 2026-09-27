"""Plots for t0 regression results -- one style, applied everywhere.

Every figure in the repo goes through this package, so colour, type, frame
and annotation are the same on every plot (``style``). The residual is
written Delta t0 = t0_pred - t0_true on every axis.

    style      the ATLAS house style and the palette
    residual   Delta t0 distributions, binned resolution, predicted vs true
    sigma      the predicted sigma: spread, calibration, what a cut keeps
    compare    several models: efficiency comparison, event migration
    training   the loss curve and a sweep's ranking
    report     the standard set of a trained model, and after a sigma cut

    from src.evaluation import plots
    plots.report("../runs/lar_hgtd/trial_000")        # the standard set
    plots.error_distribution({"ttbar": err}, ax=ax)   # or one at a time

    python -m src.evaluation.plots ../runs/lar_hgtd/trial_000 --max-sigma 20 40 60
"""

from .compare import efficiency_comparison, recovery_plot
from .report import cut_report, main, report, save_training_history
from .residual import (cut_comparison, error_distribution, prediction_vs_truth,
                       resolution_vs, sample_comparison)
from .sigma import (MIN_KEPT, pull_distribution, resolution_vs_efficiency, sigma_calibration,
                    sigma_distribution, sigma_vs_residual)
from .style import SEQUENTIAL, SERIES, series_colors, tokens, use_style
from .training import sweep_results, training_history

__all__ = [
    "MIN_KEPT", "SEQUENTIAL", "SERIES", "cut_comparison", "cut_report", "efficiency_comparison",
    "error_distribution", "main", "prediction_vs_truth", "pull_distribution", "recovery_plot",
    "report", "resolution_vs", "resolution_vs_efficiency", "sample_comparison",
    "save_training_history", "series_colors", "sigma_calibration", "sigma_distribution",
    "sigma_vs_residual", "sweep_results", "tokens", "training_history", "use_style",
]
