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
