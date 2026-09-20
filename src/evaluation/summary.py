"""Prediction metrics shared by training, evaluation and cross-sample studies."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _gauss(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _two_gauss(x, a_core, mu, sigma_core, a_bkg, sigma_bkg):
    return _gauss(x, a_core, mu, sigma_core) + _gauss(x, a_bkg, mu, sigma_bkg)


def fit_core_resolution(errors: np.ndarray, method: str = "double_gaussian",
                        fit_range: float = 120.0, pileup_sigma: float = 175.74,
                        fix_pileup_sigma: bool = True, bins: int = 200
                        ) -> Dict[str, float]:
    """Fit the error distribution and return the core width in ps.

    ``single_gaussian`` fits within +-``fit_range``; ``double_gaussian`` fits a
    narrow core plus a wide pile-up term over the full range, optionally with
    the background width fixed.
    """
    from scipy.optimize import curve_fit

    lo, hi = np.percentile(errors, [0.5, 99.5])
    span = max(abs(lo), abs(hi))
    counts, edges = np.histogram(errors, bins=bins, range=(-span, span))
    centres = 0.5 * (edges[1:] + edges[:-1])

    bin_width = float(edges[1] - edges[0])

    if method == "single_gaussian":
        sel = np.abs(centres) <= fit_range
        p0 = [counts.max(), 0.0, min(fit_range / 2, errors.std())]
        popt, _ = curve_fit(_gauss, centres[sel], counts[sel], p0=p0, maxfev=20000)
        return {"method": method, "mu": float(popt[1]), "sigma": float(abs(popt[2])),
                "core_amplitude": float(popt[0]), "bin_width": bin_width,
                "n_events": int(len(errors))}

    if method != "double_gaussian":
        raise ValueError(f"unknown fit method {method!r}")

    if fix_pileup_sigma:
        def model(x, a_core, mu, sigma_core, a_bkg):
            return _two_gauss(x, a_core, mu, sigma_core, a_bkg, pileup_sigma)
        p0 = [counts.max(), 0.0, 40.0, counts.max() * 0.1]
        bounds = ([0, -1000, 1, 0], [np.inf, 1000, 500, np.inf])
    else:
        def model(x, a_core, mu, sigma_core, a_bkg, sigma_bkg):
            return _two_gauss(x, a_core, mu, sigma_core, a_bkg, sigma_bkg)
        p0 = [counts.max(), 0.0, 40.0, counts.max() * 0.1, pileup_sigma]
        bounds = ([0, -1000, 1, 0, 50], [np.inf, 1000, 500, np.inf, 1000])

    popt, _ = curve_fit(model, centres, counts, p0=p0, bounds=bounds, maxfev=40000)
    out = {"method": method, "mu": float(popt[1]), "sigma": float(abs(popt[2])),
           "core_amplitude": float(popt[0]), "bkg_amplitude": float(popt[3]),
           # The amplitudes are counts per fit bin -- a plot with different
           # binning has to rescale them, so record what they refer to.
           "bin_width": bin_width, "n_events": int(len(errors))}
    out["sigma_bkg"] = float(pileup_sigma if fix_pileup_sigma else abs(popt[4]))
    return out


def split_prediction(y_pred: np.ndarray):
    """(mean, sigma) from a model output; sigma is None unless it predicts one."""
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
        return y_pred[:, 0], np.exp(0.5 * y_pred[:, 1])
    return y_pred.reshape(-1), None


def summarize(y_true: np.ndarray, y_pred: np.ndarray,
              core_window: float = 120.0, sigma: Optional[np.ndarray] = None,
              fit: Optional[dict] = None) -> Dict[str, float]:
    """RMSE/MAE plus the core fraction, core width and (optionally) a Gaussian fit.

    With ``sigma`` the summary also reports whether the predicted uncertainty
    is honest: the pull (error / sigma) should have unit width if it is.
    """
    errors = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    core = errors[np.abs(errors) < core_window]
    out = {
        "n_events": int(len(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        # The half-width holding 68% of the events. Unlike a fitted core width
        # it cannot be improved by a worse model -- a fit will happily find a
        # narrow peak inside a distribution that is nothing but the target's
        # own spread -- so this is what automated ranking uses.
        "q68": float(np.percentile(np.abs(errors), 68)) if len(errors) else float("nan"),
        "q95": float(np.percentile(np.abs(errors), 95)) if len(errors) else float("nan"),
        "core_fraction": float(len(core) / len(errors)) if len(errors) else 0.0,
        "core_std": float(core.std()) if len(core) else float("nan"),
    }
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64)
        pull = errors / np.maximum(sigma, 1e-9)
        out["sigma_median"] = float(np.median(sigma))
        out["pull_std"] = float(np.std(pull[np.abs(pull) < 5]))
        # the events the model says it knows best
        good = sigma < np.quantile(sigma, 0.5)
        out["best_half_core_std"] = float(errors[good][np.abs(errors[good]) < core_window].std())
    if fit:
        try:
            out["fit"] = fit_core_resolution(errors, **fit)
        except Exception as exc:                      # fitting is best-effort
            out["fit"] = {"error": str(exc)}
    return out


def format_summary(name: str, stats: Dict[str, float]) -> str:
    line = (f"{name:>16s}  n={stats['n_events']:6d}  q68={stats['q68']:6.1f}  "
            f"RMSE={stats['rmse']:7.2f}  bias={stats['bias']:+6.2f}  "
            f"core({stats['core_fraction'] * 100:.0f}%)_std={stats['core_std']:6.2f}")
    if "sigma_median" in stats:
        line += (f"  pred_sigma={stats['sigma_median']:6.1f}"
                 f"  pull={stats['pull_std']:4.2f}"
                 f"  best50%_std={stats['best_half_core_std']:5.1f}")
    fit = stats.get("fit")
    if isinstance(fit, dict) and "sigma" in fit:
        line += f"  fit_sigma={fit['sigma']:6.2f}"
    return line
