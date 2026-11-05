"""A persistent workspace for inspecting language models."""

from typing import TYPE_CHECKING, Any

__version__ = "0.2.1"
__all__ = ["BudgetExceeded", "EditBook", "Lab", "PassBudget", "Worker", "__version__"]

if TYPE_CHECKING:
    from .budget import BudgetExceeded, PassBudget
    from .editing import EditBook
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
    if name in {"PassBudget", "BudgetExceeded"}:
        from . import budget

        return getattr(budget, name)
    if name == "EditBook":
        from .editing import EditBook

        return EditBook
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
