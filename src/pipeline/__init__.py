"""Compact event store and config-driven data pipeline."""

from .event_store import EventStore, RaggedBlock, pad_ragged

__all__ = ["EventStore", "RaggedBlock", "pad_ragged"]
