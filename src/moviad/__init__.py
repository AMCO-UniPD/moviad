"""MoViAD package."""

from importlib import import_module
from typing import Any

__all__ = ["evaluator"]


def __getattr__(name: str) -> Any:
    if name != "evaluator":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = import_module(".utilities.evaluation.evaluator", __name__)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
