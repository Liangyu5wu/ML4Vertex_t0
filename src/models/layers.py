"""Keras 3 layers used by the block model.

Written against ``keras.ops`` rather than raw TensorFlow so the same code runs
on any Keras backend, and kept deliberately small: three layers cover every
architecture in this repo.

These replace the Keras 2 implementations in ``src/models/common/``, which do
not work under Keras 3 (hand-rolled multi-head attention, ``tf.*`` ops in
``call``).
"""

from __future__ import annotations

import keras
from keras import layers, ops

NEG_INF = -1e9


def _mask_to_float(mask, dtype):
    """(batch, items) boolean mask -> (batch, items, 1) float multiplier."""
    return ops.expand_dims(ops.cast(mask, dtype), axis=-1)


@keras.saving.register_keras_serializable(package="ml4vertex")
class MaskedAveragePooling(layers.Layer):
    """Average over the item axis, counting only unmasked items."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None          # the item axis is gone; the mask is consumed here

    def call(self, inputs, mask=None):
        if mask is None:
            return ops.mean(inputs, axis=1)
        m = _mask_to_float(mask, inputs.dtype)
        total = ops.sum(inputs * m, axis=1)
        count = ops.maximum(ops.sum(m, axis=1), 1.0)
        return total / count

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


@keras.saving.register_keras_serializable(package="ml4vertex")
class AttentionPooling(layers.Layer):
    """Pool over the item axis with learned, mask-aware attention weights."""

    def __init__(self, hidden_units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.hidden_units = hidden_units
        self.hidden = layers.Dense(hidden_units, activation="relu", name="score_hidden")
        self.score = layers.Dense(1, name="score")

    def compute_mask(self, inputs, mask=None):
        return None          # the item axis is gone; the mask is consumed here

    def call(self, inputs, mask=None):
        scores = self.score(self.hidden(inputs))               # (batch, items, 1)
        if mask is not None:
            scores += (1.0 - _mask_to_float(mask, scores.dtype)) * NEG_INF
        weights = ops.softmax(scores, axis=1)
        return ops.sum(inputs * weights, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return {**super().get_config(), "hidden_units": self.hidden_units}


@keras.saving.register_keras_serializable(package="ml4vertex")
class TransformerBlock(layers.Layer):
    """Pre-norm self-attention block with a feed-forward projection.

    Pre-norm (rather than the post-norm of the old implementation) trains
    stably without a warm-up schedule, which matters here because the runs are
    short.
    """

    def __init__(self, d_model: int, num_heads: int, dff: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True     # the item axis survives; keep propagating
        self.d_model, self.num_heads, self.dff, self.dropout = \
            d_model, num_heads, dff, dropout
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=max(d_model // num_heads, 1), dropout=dropout)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation="gelu"),
            layers.Dense(d_model),
        ], name="ffn")
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=None):
        # MultiHeadAttention wants (batch, queries, keys); (batch, 1, keys)
        # broadcasts across queries, which is what a padding mask needs.
        attn_mask = None if mask is None else ops.expand_dims(ops.cast(mask, "bool"), 1)

        h = self.norm1(inputs)
        h = self.attention(h, h, attention_mask=attn_mask, training=training)
        x = inputs + self.drop1(h, training=training)

        h = self.ffn(self.norm2(x))
        return x + self.drop2(h, training=training)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.d_model,)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model,
                "num_heads": self.num_heads, "dff": self.dff,
                "dropout": self.dropout}


@keras.saving.register_keras_serializable(package="ml4vertex")
class SelectionWeightedTime(layers.Layer):
    """Pick the objects that belong to the hard scatter, then average their times.

    Each object carries a time measurement and its resolution. If we knew which
    objects came from the hard-scatter vertex we would average just those,
    weighted by resolution; we do not, so the layer scores every object

        p_i = sigmoid( MLP([ embedding_i , context ]) )

    and forms the weighted mean that a known selection would have given::

        t = sum_i p_i w_i t_i / sum_i p_i w_i        w_i = 1 / sigma_i^2

    The context is what makes this more than a re-weighting of one detector:
    with the calorimeter's own time estimate in it, an object whose time
    disagrees with the calorimeter can be down-weighted. A uniform p reproduces
    the masked average this replaces.

    Times arrive normalized, so the layer is given the affine constants and
    works in physical units; it returns the time, the effective number of
    selected objects, and the per-object probabilities.
    """

    def __init__(self, time_scale=(0.0, 1.0), res_scale=(0.0, 1.0),
                 hidden_units: int = 32, min_resolution: float = 5.0,
                 min_probability: float = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.time_scale = tuple(float(v) for v in time_scale)
        self.res_scale = tuple(float(v) for v in res_scale)
        self.hidden_units = hidden_units
        self.min_resolution = min_resolution
        # A floor on the probability keeps the denominator away from zero. The
        # gradient of a weighted mean goes as one over the total weight, so a
        # model that briefly decides to select nothing would otherwise take an
        # unbounded step and never come back.
        self.min_probability = min_probability
        self.norm = layers.LayerNormalization(name="score_norm")
        self.hidden = layers.Dense(hidden_units, activation="relu", name="score_hidden")
        self.score = layers.Dense(1, name="score")

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        # The mask travels as an ordinary input: Keras hands a list of masks to
        # a layer with list inputs, which is not what this one means by it.
        embeddings, raw, context, mask = inputs
        n_items = ops.shape(embeddings)[1]
        context = ops.repeat(ops.expand_dims(context, 1), n_items, axis=1)
        features = self.norm(ops.concatenate([embeddings, context], axis=-1))
        p = ops.sigmoid(self.score(self.hidden(features)))            # (b, n, 1)
        p = (self.min_probability + (1.0 - self.min_probability) * p) \
            * _mask_to_float(mask, p.dtype)

        t_mean, t_std = self.time_scale
        r_mean, r_std = self.res_scale
        time = raw[..., :1] * t_std + t_mean
        resolution = ops.maximum(raw[..., 1:2] * r_std + r_mean, self.min_resolution)

        weight = p / ops.square(resolution)
        total = ops.sum(weight, axis=1)
        estimate = ops.sum(weight * time, axis=1) / ops.maximum(total, 1e-12)
        n_eff = ops.sum(p, axis=1)
        # A time is only meaningful if something was selected; report both so
        # the head downstream can discount events where nothing was.
        return [ops.concatenate([estimate, n_eff], axis=-1), ops.squeeze(p, -1)]

    def compute_output_shape(self, input_shape):
        emb = input_shape[0]
        return [(emb[0], 2), (emb[0], emb[1])]

    def get_config(self):
        return {**super().get_config(), "time_scale": self.time_scale,
                "res_scale": self.res_scale, "hidden_units": self.hidden_units,
                "min_resolution": self.min_resolution,
                "min_probability": self.min_probability}
