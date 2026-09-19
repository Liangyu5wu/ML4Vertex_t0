#!/usr/bin/env python
"""Train a vertex-time model from a block config -- one entry point, any inputs.

    python scripts/train_blocks.py --config config/blocks/hgtd_mixed.yaml
    python scripts/train_blocks.py --config ... --epochs 5 --model-dir /tmp/try

The config declares which samples to read and which input blocks to build; the
architecture follows from those, so switching between ttbar-only, VBF-only and
a mixed training is a change to the ``datasets:`` list and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.summary import format_summary, split_prediction, summarize
from src.models.block_model import build_model, model_spec_from_assembly, save_model
from src.pipeline.assemble import (AssemblySpec, make_tf_dataset, prepare, save_norm)
from src.runtime import get_strategy, resolve_batch_size


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model-dir", default=None, help="override output directory")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="keep only these dataset names from the config")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-plots", action="store_true", help="skip the plot set")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def apply_overrides(cfg: dict, args) -> dict:
    train = cfg.setdefault("training", {})
    if args.epochs is not None:
        train["epochs"] = args.epochs
    if args.batch_size is not None:
        train["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg.setdefault("optimizer", {})["learning_rate"] = args.learning_rate
    if args.model_dir is not None:
        cfg["model_dir"] = args.model_dir
    if args.seed is not None:
        cfg.setdefault("data", {}).setdefault("split", {})["random_state"] = args.seed
    if args.datasets:
        keep = set(args.datasets)
        chosen = [d for d in cfg["data"]["datasets"] if d["name"] in keep]
        missing = keep - {d["name"] for d in chosen}
        if missing:
            raise SystemExit(f"--datasets: no such dataset(s) in config: {sorted(missing)}")
        cfg["data"]["datasets"] = chosen
    return cfg


def build_callbacks(cfg: dict, model_dir: str):
    import tensorflow as tf
    train = cfg.get("training", {})
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=int(train.get("early_stopping_patience", 20)),
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=float(train.get("lr_reduction_factor", 0.5)),
            patience=int(train.get("lr_patience", 8)),
            min_lr=float(train.get("min_lr", 1e-7)), verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(model_dir, "history.csv")),
    ]
    return cbs


def main():
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    cfg = apply_overrides(cfg, args)

    model_dir = cfg.get("model_dir") or os.path.join("../models", cfg["model_name"])
    os.makedirs(model_dir, exist_ok=True)
    print("=" * 74)
    print(f"model      : {cfg['model_name']}")
    print(f"output dir : {os.path.abspath(model_dir)}")
    print("=" * 74)

    t0 = time.time()
    spec = AssemblySpec.from_config(cfg["data"])
    data = prepare(spec)
    print(f"data ready in {time.time() - t0:.1f}s")

    model_spec = model_spec_from_assembly(
        spec, head=cfg.get("head", {}), loss=cfg.get("loss", {}),
        optimizer=cfg.get("optimizer", {}), event_encoder=cfg.get("event_encoder"),
        name=cfg["model_name"], event_dim=len(data.event_feature_names))

    train_cfg = cfg.get("training", {})
    strategy, n_replicas = get_strategy()
    with strategy.scope():
        model = build_model(model_spec)
    print(f"model parameters: {model.count_params():,}")
    if args.verbose > 1:
        model.summary()

    batch_size = resolve_batch_size(
        int(train_cfg.get("batch_size", 256)), n_replicas,
        bool(train_cfg.get("batch_size_per_replica", False)))
    train_ds = make_tf_dataset(data, "train", batch_size, shuffle=True,
                               shuffle_seed=spec.split.random_state)
    val_ds = make_tf_dataset(data, "val", batch_size)

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=int(train_cfg.get("epochs", 200)),
                        shuffle=False,        # the tf.data pipeline shuffles
                        callbacks=build_callbacks(cfg, model_dir),
                        verbose=args.verbose)

    save_model(model, model_spec, model_dir)
    save_norm(data, os.path.join(model_dir, "norm_params.pkl"))
    with open(os.path.join(model_dir, "config.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    # Score the test split as a whole and per sample.
    test_ds = make_tf_dataset(data, "test", batch_size, use_weights=False)
    y_pred, sigma = split_prediction(model.predict(test_ds, verbose=0))
    y_true = data.targets["test"]
    fit_cfg = cfg.get("evaluation", {}).get("fit")

    print("\n" + "=" * 74)
    metrics = {"all": summarize(y_true, y_pred, sigma=sigma, fit=fit_cfg)}
    print(format_summary("all", metrics["all"]))
    if len(data.dataset_names) > 1:
        for name in data.dataset_names:
            m = data.dataset_mask("test", name)
            metrics[name] = summarize(y_true[m], y_pred[m],
                                      sigma=None if sigma is None else sigma[m],
                                      fit=fit_cfg)
            print(format_summary(name, metrics[name]))
    print("=" * 74)

    np.savez(os.path.join(model_dir, "predictions_test.npz"),
             y_true=y_true, y_pred=y_pred, errors=y_pred - y_true,
             **({} if sigma is None else {"sigma": sigma}),
             dataset_id=data.provenance["test"]["dataset_id"],
             event_number=data.provenance["test"]["event_number"],
             file_index=data.provenance["test"]["file_index"],
             dataset_names=np.array(data.dataset_names))
    with open(os.path.join(model_dir, "metrics.json"), "w") as fh:
        json.dump({"test": metrics,
                   "epochs_run": len(history.history.get("loss", [])),
                   "best_val_loss": float(min(history.history["val_loss"]))}, fh, indent=2)
    print(f"\nsaved model, norm params, predictions and metrics to {model_dir}")

    if not args.no_plots:
        from src.evaluation import plots
        plots.report(model_dir)


if __name__ == "__main__":
    main()
