"""Evaluate a previously trained model on a new HDF5 dataset (e.g., VBF samples).

If <model_dir>/norm_params.pkl exists (new models save it during training), the
fitted scalers are loaded directly and applied to the new data — no need to
reload the training data.

Otherwise, fall back to reloading the original training data (data_dir from the
saved config) and refitting scalers with the same random_state, then replacing
the test split with the new dataset so it gets normalized using training stats.

Usage:
    python scripts/evaluate_new_samples.py \\
        --model-dir ../models/hgtd_multi_input_dnn_with_jets_tracks \\
        --new-data-dir ../VBF_Hinv_Vertex_timing_HGTD_w_LAr/ \\
        --new-num-files 1
"""

import os
import pickle
import sys
import argparse
import numpy as np
from copy import deepcopy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml

from config.dnn_config import DNNConfig
from config.transformer_config import TransformerConfig

from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.data.multi_input_data_loader import MultiInputDataLoader
from src.data.multi_input_data_processor import MultiInputDataProcessor
from src.data.hgtd_multi_input_data_loader import HGTDMultiInputDataLoader
from src.data.hgtd_multi_input_data_processor import HGTDMultiInputDataProcessor
from src.data.hgtd_only_data_loader import HGTDOnlyDataLoader
from src.data.hgtd_only_data_processor import HGTDOnlyDataProcessor
from src.data.norm_utils import load_saved_norm_params, apply_saved_norm
from src.models.dnn import MultiInputDNNModel, HGTDMultiInputDNNModel, HGTDOnlyDNNModel


def _rebuild_and_load_weights(config, arch, model_h5):
    """Rebuild model architecture from config and load weights from h5.

    Some saved h5 files cannot be deserialized with newer Keras versions
    ('inputs not connected to outputs'). Rebuilding the architecture from
    config and calling load_weights() bypasses the deserialization step.
    """
    feature_dim = len(config.cell_features)
    vertex_dim = 3
    jet_feature_dim = len(config.jet_features)
    track_feature_dim = len(config.track_features)
    hgtd_dim = len(config.hgtd_track_features)

    if arch == 'multi_input_dnn':
        builder = MultiInputDNNModel(config)
        builder.build_model(feature_dim, vertex_dim, jet_feature_dim, track_feature_dim)
    elif arch == 'hgtd_multi_input_dnn':
        builder = HGTDMultiInputDNNModel(config)
        builder.build_model(feature_dim, vertex_dim, jet_feature_dim, track_feature_dim, hgtd_dim)
    elif arch == 'hgtd_only_dnn':
        builder = HGTDOnlyDNNModel(config)
        builder.build_model(hgtd_dim, vertex_dim)
    else:
        raise NotImplementedError(f"Architecture {arch} not supported by rebuild fallback")

    model = builder.get_model()
    model.load_weights(model_h5)
    print(f"Rebuilt {arch} model from config and loaded weights from {model_h5}")
    return model


def load_config_and_model_robust(model_dir):
    """Load config + model, with fallback to rebuilding architecture from config."""
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    arch = yaml_data.get('model_architecture', '')

    if arch in ('multi_input_dnn', 'hgtd_only_dnn', 'hgtd_multi_input_dnn',
                'two_stage_dnn', 'baseline_guided_dnn'):
        config = DNNConfig.load_config(model_dir)
    else:
        config = TransformerConfig.load_config(model_dir)

    model_h5 = os.path.join(model_dir, 'model.h5')
    model = _rebuild_and_load_weights(config, arch, model_h5)

    is_multi_input = arch == 'multi_input_dnn'
    is_hgtd_only = arch == 'hgtd_only_dnn'
    is_hgtd_multi_input = arch == 'hgtd_multi_input_dnn'
    is_baseline_guided = arch == 'baseline_guided_dnn'

    print(f"Loaded {arch} configuration from: {model_dir}")
    return config, model, is_baseline_guided, is_multi_input, is_hgtd_only, is_hgtd_multi_input


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained model on a new dataset'
    )
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing saved model and config')
    parser.add_argument('--new-data-dir', type=str, required=True,
                        help='Directory containing the new HDF5 files to evaluate')
    parser.add_argument('--new-num-files', type=int, default=1,
                        help='Number of files to read from --new-data-dir (default: 1)')
    parser.add_argument('--training-data-dir', type=str, default=None,
                        help='Override training data dir (default: from saved config)')
    parser.add_argument('--training-num-files', type=int, default=None,
                        help='Override number of training files (default: from saved config)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output npz path. Default: <model_dir>/predictions_<basename>.npz')
    parser.add_argument('--force-refit', action='store_true',
                        help='Ignore saved norm_params.pkl and refit scalers from training data')
    parser.add_argument('--verbose', type=int, default=1)
    return parser.parse_args()


def _make_loader(config, is_baseline_guided, is_multi_input, is_hgtd_only, is_hgtd_multi_input):
    if is_hgtd_only:
        return HGTDOnlyDataLoader(config)
    if is_hgtd_multi_input:
        return HGTDMultiInputDataLoader(config)
    if is_multi_input:
        return MultiInputDataLoader(config)
    return DataLoader(config)


def _make_processor(config, is_hgtd_only, is_hgtd_multi_input, is_multi_input):
    if is_hgtd_only:
        return HGTDOnlyDataProcessor(config)
    if is_hgtd_multi_input:
        return HGTDMultiInputDataProcessor(config)
    if is_multi_input:
        return MultiInputDataProcessor(config)
    return DataProcessor(config)


def evaluate_hgtd_only(config, keras_model, train_cfg, new_cfg, output_file, saved_norm=None):
    """Pipeline for hgtd_only_dnn models."""
    processor = HGTDOnlyDataProcessor(train_cfg)

    new_loader = HGTDOnlyDataLoader(new_cfg)
    print(f"\nLoading new data from {new_cfg.data_dir} ({new_cfg.num_files} files)...")
    hgtd_seq_n, vertex_n, vtime_n, _ = new_loader.load_data_from_files()
    new_meta = new_loader._event_metadata

    if saved_norm is not None:
        applied = apply_saved_norm(
            processor, saved_norm,
            vertex=vertex_n, hgtd_tracks=hgtd_seq_n,
        )
        hgtd_n_norm = applied['hgtd_tracks']
        vertex_n_norm = applied['vertex']
    else:
        print(f"Loading training data from {train_cfg.data_dir} ({train_cfg.num_files} files) "
              f"to refit scalers...")
        train_loader = HGTDOnlyDataLoader(train_cfg)
        hgtd_seq_t, vertex_t, vtime_t, _ = train_loader.load_data_from_files()
        (train_hgtd, val_hgtd, _), \
        (train_vertex, val_vertex, _), \
        (train_times, val_times, _) = processor.split_data(hgtd_seq_t, vertex_t, vtime_t)

        print("Normalizing new data using freshly-fit scalers...")
        (_, _, hgtd_n_norm), \
        (_, _, vertex_n_norm), \
        _ = processor.normalize_features(
            train_hgtd, val_hgtd, hgtd_seq_n,
            train_vertex, val_vertex, vertex_n,
            train_times, val_times, vtime_n,
        )

    print(f"\nCreating dataset and predicting on {len(vtime_n)} events...")
    test_dataset = processor.create_hgtd_only_dataset(
        hgtd_n_norm, vertex_n_norm, vtime_n, shuffle=False
    )
    y_pred = keras_model.predict(test_dataset, verbose=1).flatten()

    _save(output_file, vtime_n, y_pred, new_meta)


def evaluate_hgtd_multi(config, keras_model, train_cfg, new_cfg, output_file, saved_norm=None):
    """Pipeline for hgtd_multi_input_dnn models."""
    processor = HGTDMultiInputDataProcessor(train_cfg)

    new_loader = HGTDMultiInputDataLoader(new_cfg)
    print(f"\nLoading new data from {new_cfg.data_dir} ({new_cfg.num_files} files)...")
    cells_n, vertex_n, vtime_n, _, jets_n, tracks_n, hgtd_n = new_loader.load_data_from_files()
    new_meta = new_loader._event_metadata

    if saved_norm is not None:
        applied = apply_saved_norm(
            processor, saved_norm,
            cells=cells_n, vertex=vertex_n,
            jets=jets_n, tracks=tracks_n, hgtd_tracks=hgtd_n,
        )
        cells_n_norm = applied['cells']
        vertex_n_norm = applied['vertex']
        jets_n_norm = applied['jets']
        tracks_n_norm = applied['tracks']
        hgtd_n_norm = applied['hgtd_tracks']
    else:
        print(f"Loading training data from {train_cfg.data_dir} ({train_cfg.num_files} files) "
              f"to refit scalers...")
        train_loader = HGTDMultiInputDataLoader(train_cfg)
        cells_t, vertex_t, vtime_t, _, jets_t, tracks_t, hgtd_t = train_loader.load_data_from_files()
        (train_cells, val_cells, _), \
        (train_vertex, val_vertex, _), \
        (train_jets, val_jets, _), \
        (train_tracks, val_tracks, _), \
        (train_hgtd, val_hgtd, _), \
        (train_times, val_times, _) = processor.split_data(
            cells_t, vertex_t, vtime_t, jets_t, tracks_t, hgtd_t
        )
        print("Normalizing new data using freshly-fit scalers...")
        (_, _, cells_n_norm), \
        (_, _, vertex_n_norm), \
        (_, _, jets_n_norm), \
        (_, _, tracks_n_norm), \
        (_, _, hgtd_n_norm), \
        _ = processor.normalize_features(
            train_cells, val_cells, cells_n,
            train_vertex, val_vertex, vertex_n,
            train_jets, val_jets, jets_n,
            train_tracks, val_tracks, tracks_n,
            train_hgtd, val_hgtd, hgtd_n,
            train_times, val_times, vtime_n,
        )

    print(f"\nCreating dataset and predicting on {len(vtime_n)} events...")
    test_dataset = processor.create_hgtd_multi_input_dataset(
        cells_n_norm, vertex_n_norm, jets_n_norm, tracks_n_norm, hgtd_n_norm,
        vtime_n, shuffle=False
    )
    y_pred = keras_model.predict(test_dataset, verbose=1).flatten()

    _save(output_file, vtime_n, y_pred, new_meta)


def evaluate_multi_input(config, keras_model, train_cfg, new_cfg, output_file, saved_norm=None):
    """Pipeline for multi_input_dnn / multi_input_transformer models."""
    processor = MultiInputDataProcessor(train_cfg)

    new_loader = MultiInputDataLoader(new_cfg)
    print(f"\nLoading new data from {new_cfg.data_dir} ({new_cfg.num_files} files)...")
    cells_n, vertex_n, vtime_n, _, jets_n, tracks_n = new_loader.load_data_from_files()
    new_meta = new_loader._event_metadata

    if saved_norm is not None:
        applied = apply_saved_norm(
            processor, saved_norm,
            cells=cells_n, vertex=vertex_n, jets=jets_n, tracks=tracks_n,
        )
        cells_n_norm = applied['cells']
        vertex_n_norm = applied['vertex']
        jets_n_norm = applied['jets']
        tracks_n_norm = applied['tracks']
    else:
        print(f"Loading training data from {train_cfg.data_dir} ({train_cfg.num_files} files) "
              f"to refit scalers...")
        train_loader = MultiInputDataLoader(train_cfg)
        cells_t, vertex_t, vtime_t, _, jets_t, tracks_t = train_loader.load_data_from_files()
        (train_cells, val_cells, _), \
        (train_vertex, val_vertex, _), \
        (train_jets, val_jets, _), \
        (train_tracks, val_tracks, _), \
        (train_times, val_times, _) = processor.split_data(
            cells_t, vertex_t, vtime_t, jets_t, tracks_t
        )
        print("Normalizing new data using freshly-fit scalers...")
        (_, _, cells_n_norm), \
        (_, _, vertex_n_norm), \
        (_, _, jets_n_norm), \
        (_, _, tracks_n_norm), \
        _ = processor.normalize_features(
            train_cells, val_cells, cells_n,
            train_vertex, val_vertex, vertex_n,
            train_jets, val_jets, jets_n,
            train_tracks, val_tracks, tracks_n,
            train_times, val_times, vtime_n,
        )

    print(f"\nCreating dataset and predicting on {len(vtime_n)} events...")
    test_dataset = processor.create_multi_input_dataset(
        cells_n_norm, vertex_n_norm, jets_n_norm, tracks_n_norm, vtime_n, shuffle=False
    )
    y_pred = keras_model.predict(test_dataset, verbose=1).flatten()

    _save(output_file, vtime_n, y_pred, new_meta)


def _save(output_file, y_true, y_pred, new_meta):
    save_dict = {
        'y_true': y_true,
        'y_pred': y_pred,
        'errors': y_pred - y_true,
    }
    if new_meta is not None:
        save_dict['event_numbers'] = new_meta['event_numbers']
        save_dict['file_indices'] = new_meta['file_indices']

    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
    np.savez(output_file, **save_dict)

    print("\n" + "=" * 60)
    print(f"Predictions saved to: {output_file}")
    print("=" * 60)
    print(f"  events:             {len(y_true)}")
    print(f"  y_true mean / std:  {np.mean(y_true):.3f} / {np.std(y_true):.3f}")
    print(f"  y_pred mean / std:  {np.mean(y_pred):.3f} / {np.std(y_pred):.3f}")
    err = y_pred - y_true
    print(f"  RMSE:               {np.sqrt(np.mean(err ** 2)):.3f}")
    print(f"  MAE:                {np.mean(np.abs(err)):.3f}")


def main():
    args = parse_args()

    print("=" * 60)
    print("EVALUATE TRAINED MODEL ON NEW DATA")
    print("=" * 60)
    print(f"Model dir:    {args.model_dir}")
    print(f"New data dir: {args.new_data_dir} ({args.new_num_files} files)")

    # Load model + config
    config, keras_model, is_baseline_guided, is_multi_input, is_hgtd_only, is_hgtd_multi_input = \
        load_config_and_model_robust(args.model_dir)

    if is_baseline_guided:
        raise NotImplementedError(
            "baseline-guided models are not supported by this script "
            "(they need precomputed baseline predictions)."
        )

    # Two config copies: one for original training data (to fit scalers),
    # one with the new data dir / num_files (to load the new dataset).
    train_cfg = deepcopy(config)
    if args.training_data_dir is not None:
        train_cfg.data_dir = args.training_data_dir
    if args.training_num_files is not None:
        train_cfg.num_files = args.training_num_files

    new_cfg = deepcopy(config)
    new_cfg.data_dir = args.new_data_dir
    new_cfg.num_files = args.new_num_files

    # Default output path
    output_file = args.output_file
    if output_file is None:
        tag = os.path.basename(os.path.normpath(args.new_data_dir))
        output_file = os.path.join(args.model_dir, f"predictions_{tag}.npz")

    # Auto-detect saved normalization params
    saved_norm = None if args.force_refit else load_saved_norm_params(args.model_dir)
    if saved_norm is not None:
        print(f"\nFound saved normalization params at {args.model_dir}/norm_params.pkl "
              f"-- skipping training-data reload.")
    else:
        print(f"\nNo norm_params.pkl in {args.model_dir} -- will reload training data to refit scalers.")

    if is_hgtd_only:
        evaluate_hgtd_only(config, keras_model, train_cfg, new_cfg, output_file, saved_norm)
    elif is_hgtd_multi_input:
        evaluate_hgtd_multi(config, keras_model, train_cfg, new_cfg, output_file, saved_norm)
    elif is_multi_input:
        evaluate_multi_input(config, keras_model, train_cfg, new_cfg, output_file, saved_norm)
    else:
        raise NotImplementedError(
            "Plain transformer / two-stage DNN paths are not implemented in this script."
        )


if __name__ == "__main__":
    main()
