from .format_selector import select_format, FORMAT_RATIONALE
from .metrics import timed_generate, compute_metrics

# InferencePipeline imports torch; import only when needed to avoid PyTorch load on Python 3.13
def __getattr__(name):
    if name == "InferencePipeline":
        from .pipeline import InferencePipeline
        return InferencePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "select_format",
    "FORMAT_RATIONALE",
    "InferencePipeline",
    "timed_generate",
    "compute_metrics",
]
