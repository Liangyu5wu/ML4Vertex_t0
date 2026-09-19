"""Device selection: the same entry point runs on CPU, one GPU or many.

``setup.sh`` exports ``VERTEX_T0_NUM_GPUS``; this module asks TensorFlow what
it can actually see, so the two disagreeing (a stale environment variable, a
job that got fewer GPUs than requested) is reported rather than silently
producing a single-GPU run.
"""

from __future__ import annotations

import os
from typing import Tuple


def configure_devices(verbose: bool = True) -> int:
    """Enable memory growth and return the number of usable GPUs."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass          # already initialised; harmless

    expected = os.environ.get("VERTEX_T0_NUM_GPUS")
    if verbose and expected is not None and int(expected) != len(gpus):
        print(f"note: setup.sh saw {expected} GPU(s), TensorFlow sees {len(gpus)}")
    return len(gpus)


def get_strategy(verbose: bool = True) -> Tuple[object, int]:
    """Return ``(strategy, n_replicas)`` matching the visible devices."""
    import tensorflow as tf

    n_gpus = configure_devices(verbose=verbose)
    if n_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        if verbose:
            print(f"devices: {n_gpus} GPUs, MirroredStrategy "
                  f"({strategy.num_replicas_in_sync} replicas)")
        return strategy, strategy.num_replicas_in_sync

    strategy = tf.distribute.get_strategy()      # default, single device
    if verbose:
        print(f"devices: {'1 GPU' if n_gpus else 'CPU only'}")
    return strategy, 1


def resolve_batch_size(batch_size: int, n_replicas: int,
                       scale_with_replicas: bool = False,
                       verbose: bool = True) -> int:
    """Global batch size; optionally treat the configured value as per-replica."""
    if n_replicas <= 1:
        return batch_size
    if scale_with_replicas:
        total = batch_size * n_replicas
        if verbose:
            print(f"batch size: {batch_size} per replica -> {total} global")
        return total
    if batch_size % n_replicas and verbose:
        print(f"note: global batch {batch_size} is not divisible by "
              f"{n_replicas} replicas; TensorFlow will pad the last shard")
    if verbose:
        print(f"batch size: {batch_size} global -> "
              f"{batch_size // n_replicas} per replica")
    return batch_size
