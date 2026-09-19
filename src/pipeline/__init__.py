"""Compact event store and config-driven data pipeline."""

from .store import CompactStore, RaggedBlock, pad_ragged

__all__ = ["CompactStore", "RaggedBlock", "pad_ragged"]
