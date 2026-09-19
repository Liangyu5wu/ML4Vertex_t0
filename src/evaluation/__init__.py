"""Evaluation modules.

Imported lazily: pulling in metrics must not require the plotting stack, which
drags in matplotlib/seaborn and is not always present on a compute node.
"""


def __getattr__(name):
    if name == "Evaluator":
        from .evaluator import Evaluator
        return Evaluator
    if name == "Visualizer":
        from .visualizer import Visualizer
        return Visualizer
    if name == "plots":
        import importlib
        # importlib, not `from . import plots`: the latter re-enters __getattr__.
        return importlib.import_module(".plots", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Evaluator", "Visualizer", "plots"]
