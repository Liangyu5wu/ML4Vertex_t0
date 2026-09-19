"""The traditional t0 estimate, computed from the event store.

The baseline the network is compared against: take the calorimeter cells that
sit on a hard-scatter track, and average their time-of-flight-corrected times
weighted by the per-cell time resolution.

    sigma_i  = sigma(detector region, layer, cell energy)      [calibration table]
    t0       = sum(t_i / sigma_i^2) / sum(1 / sigma_i^2)

Cell-track matching uses the track positions extrapolated to each calorimeter
layer, which is why the ingest keeps them. This module exists as much to
prove the store carries what the baseline needs as to produce the numbers.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..pipeline.blocks import load_calibration
from ..pipeline.event_store import EventStore

# Energy bin edges of the calibration table, in GeV.
ENERGY_BINS = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0])
# (region, layer) -> table key. Region 0 is the EM barrel, 1 the EM endcap.
SIGMA_KEYS = {(0, 1): "EMB1_sigma", (0, 2): "EMB2_sigma", (0, 3): "EMB3_sigma",
              (1, 1): "EME1_sigma", (1, 2): "EME2_sigma", (1, 3): "EME3_sigma"}
# Extrapolated track position to use for each (region, layer).
EXTRAPOLATION = {(0, 1): "emb1", (0, 2): "emb2", (0, 3): "emb3",
                 (1, 1): "eme1", (1, 2): "eme2", (1, 3): "eme3"}
NO_EXTRAPOLATION = -998.0          # the store writes -999 where a track stops short


def cell_sigma(region: np.ndarray, layer: np.ndarray, energy: np.ndarray,
               calibration: Dict[str, list], fallback: float = 1000.0) -> np.ndarray:
    """Per-cell time resolution in ps from the calibration table."""
    bin_idx = np.clip(np.searchsorted(ENERGY_BINS, energy, side="right") - 1, 0, 6)
    sigma = np.full(len(energy), fallback, dtype=np.float64)
    for (r, l), key in SIGMA_KEYS.items():
        sel = (region == r) & (layer == l)
        if sel.any():
            sigma[sel] = np.take(np.asarray(calibration[key]), bin_idx[sel], mode="clip")
    return sigma


def _delta_r2(eta_a, phi_a, eta_b, phi_b):
    """Squared dR between every a and every b (outer product)."""
    d_eta = eta_a[:, None] - eta_b[None, :]
    d_phi = np.abs(phi_a[:, None] - phi_b[None, :])
    d_phi = np.where(d_phi > np.pi, 2 * np.pi - d_phi, d_phi)
    return d_eta ** 2 + d_phi ** 2


def baseline_t0(store: EventStore, max_events: Optional[int] = None,
                delta_r: float = 0.05, calibration: str = "HStrackmatching_calibration.txt",
                min_cells: int = 1, verbose: bool = True
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t0, truth time, number of matched cells) per event.

    Events with no matched cell get NaN. ``delta_r`` is the matching cone
    between a cell and a hard-scatter track extrapolated to that cell's layer.
    """
    table = load_calibration(calibration)

    cells = store.block("cells", ["eta", "phi", "region", "layer", "e",
                                  "significance", "time_tof"])
    track_fields = ["eta", "on_hs_vertex"] + \
        [f"{p}_{c}" for p in set(EXTRAPOLATION.values()) for c in ("eta", "phi")]
    tracks = store.block("tracks", track_fields)

    # The cells the baseline considers: the same selection the models use.
    keep = ((np.isin(cells["region"], [0, 1])) & (np.isin(cells["layer"], [1, 2, 3]))
            & (np.abs(cells["significance"]) > 4) & (cells["e"] > 1.0))
    cells = cells.select(keep)
    tracks = tracks.select(tracks["on_hs_vertex"] == 1)

    n_events = store.n_events if max_events is None else min(max_events, store.n_events)
    sigma_all = cell_sigma(cells["region"], cells["layer"], cells["e"], table)

    t0 = np.full(n_events, np.nan)
    n_matched = np.zeros(n_events, dtype=np.int32)
    for i in range(n_events):
        cs, ce = cells.offsets[i], cells.offsets[i + 1]
        ts, te = tracks.offsets[i], tracks.offsets[i + 1]
        if ce == cs or te == ts:
            continue
        region, layer = cells["region"][cs:ce], cells["layer"][cs:ce]
        matched = np.zeros(ce - cs, dtype=bool)
        for (r, l), prefix in EXTRAPOLATION.items():
            in_layer = (region == r) & (layer == l)
            if not in_layer.any():
                continue
            t_eta = tracks[f"{prefix}_eta"][ts:te]
            t_phi = tracks[f"{prefix}_phi"][ts:te]
            reached = t_eta > NO_EXTRAPOLATION
            if not reached.any():
                continue
            d2 = _delta_r2(cells["eta"][cs:ce][in_layer], cells["phi"][cs:ce][in_layer],
                           t_eta[reached], t_phi[reached])
            matched[in_layer] = d2.min(axis=1) < delta_r ** 2

        if matched.sum() < min_cells:
            continue
        sigma = sigma_all[cs:ce][matched]
        times = cells["time_tof"][cs:ce][matched]
        weights = 1.0 / sigma ** 2
        t0[i] = np.sum(times * weights) / np.sum(weights)
        n_matched[i] = matched.sum()

    truth = store.event_column("truth_vtx_time")[:n_events].astype(np.float64)
    if verbose:
        ok = np.isfinite(t0)
        print(f"baseline t0 on {n_events} events: {ok.sum()} reconstructed "
              f"({100 * ok.mean():.1f}%), {n_matched[ok].mean():.1f} matched cells/event")
    return t0, truth, n_matched
