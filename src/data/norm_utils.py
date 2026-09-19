"""Helpers for loading saved normalization parameters and transforming new data.

Training writes ``<model_dir>/norm_params.pkl`` with the fitted scalers. The
evaluation scripts use these helpers to apply those scalers without refitting.
"""

import os
import pickle
import numpy as np


def load_saved_norm_params(model_dir):
    """Return saved norm_params dict if ``model_dir/norm_params.pkl`` exists, else None."""
    path = os.path.join(model_dir, 'norm_params.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def apply_var_len_scaler(sequences, scaler):
    """Apply a fitted StandardScaler to a list of variable-length event sequences."""
    out = []
    for ev in sequences:
        if len(ev) == 0:
            out.append([])
        else:
            arr = np.asarray(ev, dtype=np.float64)
            out.append(scaler.transform(arr).tolist())
    return out


def apply_saved_norm(processor, norm_params, *,
                     cells=None, vertex=None,
                     jets=None, tracks=None, hgtd_tracks=None):
    """Apply previously-fitted normalization to new data using a saved norm_params dict.

    Also attaches the variable-length scalers to the processor so its
    ``_pad_*`` methods can transform configured padding values into normalized
    space the same way they would right after fitting.

    Returns a dict containing only the inputs that were provided, normalized.
    """
    out = {}

    if cells is not None:
        # Time calibration must match what normalize_features did during training.
        # It's a no-op when use_detector_params=False, but call it for correctness.
        calibrated = processor.apply_time_calibration(cells)
        out['cells'] = processor._apply_cell_normalization(
            calibrated, norm_params['cell_means'], norm_params['cell_stds']
        )

    if vertex is not None:
        # HGTDOnly stores a fitted vertex_scaler; everything else stores means/stds.
        if 'vertex_means' in norm_params:
            means = np.asarray(norm_params['vertex_means'])
            stds = np.asarray(norm_params['vertex_stds'])
        else:
            scaler = norm_params['vertex_scaler']
            means = scaler.mean_
            stds = scaler.scale_
        stds = np.where(stds > 0, stds, 1.0)
        out['vertex'] = (vertex - means) / stds

    if jets is not None:
        scaler = norm_params['jet_scaler']
        processor.jet_scaler = scaler
        out['jets'] = apply_var_len_scaler(jets, scaler)

    if tracks is not None:
        scaler = norm_params['track_scaler']
        processor.track_scaler = scaler
        out['tracks'] = apply_var_len_scaler(tracks, scaler)

    if hgtd_tracks is not None:
        scaler = norm_params['hgtd_track_scaler']
        processor.hgtd_track_scaler = scaler
        out['hgtd_tracks'] = apply_var_len_scaler(hgtd_tracks, scaler)

    return out
