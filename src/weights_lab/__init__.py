"""A persistent workspace for inspecting language models."""

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"
__all__ = ["Lab", "Worker", "__version__"]

if TYPE_CHECKING:
    from .lab import Lab
    from .worker import Worker


def __getattr__(name: str) -> Any:
    # CLI commands that inspect a session should not import the model stack.
    if name == "Lab":
        from .lab import Lab

        return Lab
    if name == "Worker":
        from .worker import Worker

        return Worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
