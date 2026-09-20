"""Build a Keras model from input-block specs.

One builder covers every architecture the repo used to keep as a separate
class: which blocks exist, what encoder sits on each and how it is pooled all
come from the same specs that drive the data pipeline, so a "multi-input DNN
with HGTD" and an "HGTD-only DNN" differ only by their ``inputs:`` stanza.

Models are saved as weights plus a JSON spec and rebuilt on load, which keeps
checkpoints readable across Keras versions -- the repo's older ``model.h5``
files can no longer be deserialized at all.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import keras
import numpy as np
from keras import layers, ops

from .layers import (AttentionPooling, MaskedAveragePooling,
                     TransformerBlock)

POOLINGS = ("attention", "average", "masked_average", "max", "sum", "flatten")
MASKED_POOLINGS = ("attention", "masked_average")
WEIGHTS_FILE = "model.weights.h5"
SPEC_FILE = "model_spec.json"


def _as_list(value, n: int, what: str) -> List:
    """Broadcast a scalar to ``n`` entries, or check an explicit list's length."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{what}: expected {n} entries, got {len(value)}")
        return list(value)
    return [value] * n


def _mlp(x, cfg: dict, name: str):
    """Dense stack. `norm` is "layer" (default), "batch" or "none".

    Layer normalization is the default for a reason: batch normalization over
    a padded set normalizes across the item axis too, so its statistics are
    computed over the padded slots as well -- and a jet block is two thirds
    padding. Layer norm acts on each object alone and is blind to padding.
    """
    units = list(cfg.get("units", [64, 32]))
    dropouts = _as_list(cfg.get("dropout", 0.1), len(units), f"{name} dropout")
    activation = cfg.get("activation", "relu")
    norm = cfg.get("norm", "batch" if cfg.get("batch_norm") else "layer")
    for i, (u, d) in enumerate(zip(units, dropouts)):
        x = layers.Dense(u, activation=activation, name=f"{name}_dense_{i}")(x)
        if norm == "layer":
            x = layers.LayerNormalization(name=f"{name}_ln_{i}")(x)
        elif norm == "batch":
            x = layers.BatchNormalization(name=f"{name}_bn_{i}")(x)
        elif norm not in (None, "none"):
            raise ValueError(f"{name}: unknown norm {norm!r}")
        if d:
            x = layers.Dropout(d, name=f"{name}_dropout_{i}")(x)
    return x


def _transformer(x, cfg: dict, name: str, mask=None):
    d_model = int(cfg.get("d_model", 64))
    x = layers.Dense(d_model, name=f"{name}_projection")(x)
    for i in range(int(cfg.get("num_blocks", 2))):
        x = TransformerBlock(d_model=d_model,
                             num_heads=int(cfg.get("num_heads", 4)),
                             dff=int(cfg.get("dff", 4 * d_model)),
                             dropout=float(cfg.get("dropout", 0.1)),
                             name=f"{name}_block_{i}")(x, mask=mask)
    return x


def _occupancy(mask, max_items: int, name: str):
    """How many real objects the block had, as a fraction of its capacity.

    Pooling averages away the count, yet the count is exactly what says how
    well an event can be measured -- two timed tracks and twenty are not the
    same evidence. Without this the model cannot tell them apart, nor tell an
    empty block from one whose objects happen to average to zero.
    """
    return layers.Lambda(
        lambda m, n=float(max_items): ops.sum(ops.cast(m, "float32"), axis=1,
                                              keepdims=True) / n,
        output_shape=lambda sh: (sh[0], 1), name=f"{name}_occupancy")(mask)


def _pool(x, how: str, cfg: dict, name: str, mask=None):
    if how == "attention":
        return AttentionPooling(hidden_units=int(cfg.get("attention_units", 32)),
                                name=f"{name}_attention_pool")(x, mask=mask)
    if how == "masked_average":
        return MaskedAveragePooling(name=f"{name}_masked_avg_pool")(x, mask=mask)
    if how == "average":
        return layers.GlobalAveragePooling1D(name=f"{name}_avg_pool")(x)
    if how == "max":
        return layers.GlobalMaxPooling1D(name=f"{name}_max_pool")(x)
    if how == "sum":
        return layers.Lambda(lambda t: ops.sum(t, axis=1),
                             output_shape=lambda s: (s[0], s[-1]),
                             name=f"{name}_sum_pool")(x)
    if how == "flatten":
        return layers.Flatten(name=f"{name}_flatten")(x)
    raise ValueError(f"{name}: unknown pooling {how!r}; choose from {POOLINGS}")


def build_model(model_spec: dict) -> keras.Model:
    """Build and compile a model from a plain-dict spec.

    The spec is JSON-serialisable so it can be saved next to the weights::

        {"blocks": {"cells": {"shape": [60, 7], "mask": true,
                              "encoder": {...}}, ...},
         "event_dim": 3,
         "head": {"units": [...], "dropout": [...]},
         "loss": {"type": "huber", "delta": 100.0},
         "optimizer": {"type": "adam", "learning_rate": 1e-3}}
    """
    inputs: Dict[str, keras.KerasTensor] = {}
    branches = []

    def encode(name, block):
        """Inputs and per-object embeddings for one block."""
        max_items, n_features = block["shape"]
        x_in = layers.Input(shape=(max_items, n_features), name=f"{name}_input")
        inputs[f"{name}_input"] = x_in
        mask = None
        if block.get("mask"):
            mask = layers.Input(shape=(max_items,), dtype="bool", name=f"{name}_mask")
            inputs[f"{name}_mask"] = mask
        encoder = dict(block.get("encoder") or {})
        kind = encoder.get("type", "mlp")
        if kind == "mlp":
            x = _mlp(x_in, encoder, name)
        elif kind == "transformer":
            x = _transformer(x_in, encoder, name, mask=mask)
        else:
            raise ValueError(f"{name}: unknown encoder type {kind!r}")
        return x_in, x, mask, encoder

    for name, block in model_spec["blocks"].items():
        x_in, x, mask, encoder = encode(name, block)
        pooling = encoder.get("pooling", "average")
        if pooling in MASKED_POOLINGS and mask is None:
            raise ValueError(f"block {name!r} uses {pooling} pooling but emits no "
                             f"mask; set emit_mask: true on the block")
        pooled = _pool(x, pooling, encoder, name, mask=mask)
        if mask is not None:
            pooled = layers.Concatenate(name=f"{name}_pooled")(
                [pooled, _occupancy(mask, block["shape"][0], name)])
        branches.append(pooled)

    event_dim = int(model_spec.get("event_dim", 0))
    if event_dim:
        event_in = layers.Input(shape=(event_dim,), name="event_input")
        inputs["event_input"] = event_in
        event_cfg = model_spec.get("event_encoder") or {}
        branches.append(_mlp(event_in, event_cfg, "event")
                        if event_cfg.get("units") else event_in)

    if not branches:
        raise ValueError("model spec defines no inputs")

    x = branches[0] if len(branches) == 1 else layers.Concatenate(name="combine")(branches)

    head = model_spec.get("head") or {}
    x = _mlp(x, {"units": head.get("units", [128, 64, 32, 16]),
                 "dropout": head.get("dropout", 0.1),
                 "activation": head.get("activation", "relu"),
                 "norm": head.get("norm", "none")}, "head")

    loss_cfg = dict(model_spec.get("loss") or {})
    heteroscedastic = loss_cfg.get("type") == "gaussian_nll"
    if heteroscedastic:
        # Two units: the mean and log(sigma^2).  The width starts at the scale
        # of the target rather than at 1, because a model that begins by
        # claiming picosecond precision on hundred-picosecond residuals spends
        # its first epochs undoing that instead of learning.
        sigma0 = float(loss_cfg.get("sigma_init", 100.0))
        lo, hi = (float(loss_cfg.get("sigma_min", 5.0)),
                  float(loss_cfg.get("sigma_max", 2000.0)))
        raw = layers.Dense(
            2, name="vertex_time_raw",
            bias_initializer=keras.initializers.Constant([0.0, 2 * np.log(sigma0)]))(x)
        # Hold the width inside a range anyone would believe. Without a ceiling
        # the beta-weighted likelihood is unbounded in log-variance, and a head
        # whose mean starts far from the target can walk off to infinity
        # before it ever learns to predict the mean.
        output = layers.Lambda(
            lambda t, a=2 * np.log(lo), b=2 * np.log(hi):
                ops.stack([t[..., 0], ops.clip(t[..., 1], a, b)], axis=-1),
            output_shape=lambda sh: sh, name="vertex_time")(raw)
    else:
        output = layers.Dense(1, name="vertex_time")(x)

    model = keras.Model(inputs=inputs, outputs=output,
                        name=model_spec.get("name", "block_model"))
    model.compile(optimizer=_make_optimizer(model_spec.get("optimizer") or {}),
                  loss=_make_loss(loss_cfg),
                  metrics=[_rmse_of_mean, _mae_of_mean] if heteroscedastic else
                          [keras.metrics.RootMeanSquaredError(name="rmse"),
                           keras.metrics.MeanAbsoluteError(name="mae")],
                  # Sample weights steer the loss; reported metrics stay
                  # unweighted so they remain comparable across mixtures.
                  weighted_metrics=[])
    return model


@keras.saving.register_keras_serializable(package="ml4vertex")
def gaussian_nll(y_true, y_pred, beta: float = 0.5):
    """Negative log-likelihood of a Gaussian whose width the model predicts.

    ``y_pred`` is (mean, log variance). The beta-weighting of Seitzer et al.
    multiplies each term by a detached sigma^(2*beta): at beta=0 this is the
    plain likelihood, which early in training is minimised by declaring
    everything uncertain; at beta=1 it reduces to mean squared error. The
    default 0.5 trains without a warm-up phase.
    """
    mean, log_var = y_pred[..., :1], y_pred[..., 1:]
    y_true = ops.reshape(y_true, ops.shape(mean))
    nll = 0.5 * (ops.exp(-log_var) * ops.square(y_true - mean) + log_var)
    if beta:
        nll = nll * ops.stop_gradient(ops.exp(beta * log_var))
    return ops.mean(nll, axis=-1)


@keras.saving.register_keras_serializable(package="ml4vertex")
def _rmse_of_mean(y_true, y_pred):
    return ops.sqrt(ops.mean(ops.square(
        ops.reshape(y_true, (-1,)) - y_pred[..., 0])))


@keras.saving.register_keras_serializable(package="ml4vertex")
def _mae_of_mean(y_true, y_pred):
    return ops.mean(ops.abs(ops.reshape(y_true, (-1,)) - y_pred[..., 0]))


def _make_loss(cfg: dict):
    kind = cfg.get("type", "mse")
    if kind in ("mse", "mae"):
        return kind
    if kind == "huber":
        return keras.losses.Huber(delta=float(cfg.get("delta", 100.0)))
    if kind == "gaussian_nll":
        beta = float(cfg.get("beta", 0.5))
        return lambda y, p: gaussian_nll(y, p, beta=beta)
    raise ValueError(f"unsupported loss {kind!r}")


def _make_optimizer(cfg: dict):
    kind = cfg.get("type", "adam")
    lr = float(cfg.get("learning_rate", 1e-3))
    if kind == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if kind == "adamw":
        return keras.optimizers.AdamW(learning_rate=lr,
                                      weight_decay=float(cfg.get("weight_decay", 1e-4)))
    if kind == "sgd":
        return keras.optimizers.SGD(learning_rate=lr,
                                    momentum=float(cfg.get("momentum", 0.9)))
    raise ValueError(f"unsupported optimizer {kind!r}")


def model_spec_from_assembly(assembly, head: dict, loss: dict, optimizer: dict,
                             event_encoder: Optional[dict] = None,
                             name: str = "block_model",
                             event_dim: Optional[int] = None,
                             norm: Optional[dict] = None) -> dict:
    """Derive a model spec from an :class:`AssemblySpec` plus head/loss settings.

    ``event_dim`` overrides the count taken from the spec; pass the prepared
    data's ``event_feature_names`` length when a sample tag was appended.
    """
    blocks = {}
    for bname, bspec in assembly.blocks.items():
        entry = {"shape": [bspec.max_items, len(bspec.features)],
                 "mask": bool(bspec.emit_mask), "encoder": dict(bspec.encoder)}
        blocks[bname] = entry
    return {"name": name, "blocks": blocks,
            "event_dim": (len(assembly.event_features) if event_dim is None
                          else int(event_dim)),
            "event_encoder": event_encoder or {},
            "head": head, "loss": loss, "optimizer": optimizer}


def save_model(model: keras.Model, model_spec: dict, directory: str) -> None:
    """Write weights + spec so the model can be rebuilt on any Keras version."""
    os.makedirs(directory, exist_ok=True)
    model.save_weights(os.path.join(directory, WEIGHTS_FILE))
    with open(os.path.join(directory, SPEC_FILE), "w") as fh:
        json.dump(model_spec, fh, indent=2)


def load_model(directory: str) -> keras.Model:
    """Rebuild a model from its spec and load the saved weights."""
    with open(os.path.join(directory, SPEC_FILE)) as fh:
        model_spec = json.load(fh)
    model = build_model(model_spec)
    model.load_weights(os.path.join(directory, WEIGHTS_FILE))
    return model
