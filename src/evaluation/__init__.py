"""Metrics and plots.

``plots`` is imported lazily so that computing metrics does not pull in
matplotlib.
"""

from .summary import format_summary, summarize


def __getattr__(name):
    if name == "plots":
        import importlib
        # importlib, not `from . import plots`: the latter re-enters __getattr__.
        return importlib.import_module(".plots", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["summarize", "format_summary", "plots"]
