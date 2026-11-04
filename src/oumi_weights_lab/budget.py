"""Forward-call accounting independent of model and tokenizer dependencies."""

from threading import RLock


class BudgetExceeded(RuntimeError):
    """A forward call was rejected before the model executed."""


class PassBudget:
    """Count accepted model forwards, optionally rejecting calls beyond a limit.

    A batch costs one pass. Generation costs one pass for every base-model call.
    A rejected call consumes nothing; an accepted call that fails still counts.
    """

    def __init__(self, limit: int | None = None):
        self._lock = RLock()
        self._used = 0
        self.limit = limit

    @staticmethod
    def _validate(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")

    @property
    def limit(self) -> int | None:
        return self._limit

    @limit.setter
    def limit(self, value: int | None) -> None:
        if value is not None:
            self._validate(value, "pass limit")
        with self._lock:
            self._limit = value

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.used)

    def check(self, count: int = 1) -> None:
        self._validate(count, "pass count")
        with self._lock:
            if self.limit is not None and self.used + count > self.limit:
                raise BudgetExceeded(
                    f"forward-pass budget exhausted: used {self.used}, "
                    f"limit {self.limit}, requested {count}"
                )

    def consume(self, count: int = 1) -> int:
        with self._lock:
            self.check(count)
            self._used += count
            return self._used
