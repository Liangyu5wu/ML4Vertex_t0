#!/usr/bin/env python
"""Score a trained block model, on its own test split or on another sample.

    # test split the model was trained against
    python scripts/evaluate_blocks.py --model-dir ../models/hgtd_blocks_mixed

    # cross-sample: a ttbar-trained model on VBF, reusing the training scalers
    python scripts/evaluate_blocks.py --model-dir ../models/hgtd_blocks_ttbar \
        --dataset vbf_hinv:/global/cfs/.../compact/vbf_hinv --split all

The saved ``norm_params.pkl`` is always reused, so the new sample is
transformed exactly as the training sample was -- no reloading of training
data, and no accidental refit on the sample being measured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.summary import format_summary, split_prediction, summarize
from src.models.block_model import load_model
from src.pipeline.assemble import AssemblySpec, load_norm, prepare


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--dataset", action="append", default=[], metavar="NAME:PATH",
                   help="evaluate on this sample instead of the configured ones "
                        "(repeatable)")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="keep only these names from the model's own config")
    p.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    p.add_argument("--tag", default=None, help="suffix for the output files")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--no-plots", action="store_true", help="skip the plot set")
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir

    with open(os.path.join(model_dir, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    data_cfg = cfg["data"]

    if args.dataset:
        entries = []
        for item in args.dataset:
            if ":" not in item:
                raise SystemExit(f"--dataset expects NAME:PATH, got {item!r}")
            name, path = item.split(":", 1)
            entries.append({"name": name, "path": path})
        data_cfg["datasets"] = entries
    elif args.datasets:
        keep = set(args.datasets)
        data_cfg["datasets"] = [d for d in data_cfg["datasets"] if d["name"] in keep]
    # Resampling only makes sense while training; scoring reads each sample as
    # it is.
    data_cfg["resample"] = "none"

    spec = AssemblySpec.from_config(data_cfg)
    norm = load_norm(os.path.join(model_dir, "norm_params.pkl"))
    data = prepare(spec, norm=norm)

    model = load_model(model_dir)
    if args.split == "all":
        inputs, y_true, prov = data.merged()
    else:
        inputs = data.inputs[args.split]
        y_true = data.targets[args.split]
        prov = data.provenance[args.split]

    y_pred, sigma = split_prediction(
        model.predict(inputs, batch_size=args.batch_size, verbose=0))
    fit_cfg = cfg.get("evaluation", {}).get("fit")

    names = data.dataset_names
    print("\n" + "=" * 74)
    print(f"{os.path.basename(os.path.normpath(model_dir))}  "
          f"split={args.split}  samples={names}")
    metrics = {"all": summarize(y_true, y_pred, sigma=sigma, fit=fit_cfg)}
    print(format_summary("all", metrics["all"]))
    if len(names) > 1:
        for i, name in enumerate(names):
            m = prov["dataset_id"] == i
            metrics[name] = summarize(y_true[m], y_pred[m],
                                      sigma=None if sigma is None else sigma[m],
                                      fit=fit_cfg)
            print(format_summary(name, metrics[name]))
    print("=" * 74)

    tag = args.tag or f"{'_'.join(names)}_{args.split}"
    np.savez(os.path.join(model_dir, f"predictions_{tag}.npz"),
             y_true=y_true, y_pred=y_pred, errors=y_pred - y_true,
             **({} if sigma is None else {"sigma": sigma}),
             dataset_id=prov["dataset_id"], event_number=prov["event_number"],
             file_index=prov["file_index"], dataset_names=np.array(names))
    with open(os.path.join(model_dir, f"metrics_{tag}.json"), "w") as fh:
        json.dump({"split": args.split, "datasets": names, "metrics": metrics}, fh,
                  indent=2)
    print(f"wrote predictions_{tag}.npz and metrics_{tag}.json to {model_dir}")

    if not args.no_plots:
        from src.evaluation import plots
        plots.report(model_dir, predictions=f"predictions_{tag}.npz",
                     outdir=os.path.join(model_dir, f"plots_{tag}"))


if __name__ == "__main__":
    main()
