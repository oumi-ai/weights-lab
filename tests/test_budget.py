from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from oumi_weights_lab.budget import BudgetExceeded, PassBudget, sequences_in_call


def Shaped(*shape):
    """A torch-free stand-in for a tensor with the given shape."""
    return SimpleNamespace(shape=shape)


@pytest.mark.parametrize("limit", [-1, 1.5, True, "2"])
def test_pass_limit_requires_nonnegative_integer(limit):
    with pytest.raises(ValueError, match="nonnegative integer"):
        PassBudget(limit)


def test_zero_limit_and_rejected_consumption_do_not_spend_passes():
    budget = PassBudget(0)
    with pytest.raises(BudgetExceeded):
        budget.consume()
    assert budget.used == 0
    assert budget.remaining == 0
    budget.limit = 2
    assert budget.consume(2) == 2
    with pytest.raises(BudgetExceeded):
        budget.consume()
    assert budget.used == 2
    assert budget.remaining == 0


def test_budget_is_safe_for_concurrent_forward_accounting():
    budget = PassBudget(17)

    def try_consume(_):
        try:
            budget.consume()
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        accepted = list(executor.map(try_consume, range(100)))
    assert sum(accepted) == budget.used == 17


def test_unlimited_budget_still_counts_calls():
    budget = PassBudget()
    assert budget.consume(1000) == 1000
    assert budget.remaining is None


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        ((), {"input_ids": Shaped(3, 5)}, 3),
        ((), {"input_ids": None, "inputs_embeds": Shaped(2, 4, 8)}, 2),
        ((), {"inputs_embeds": Shaped(4, 1, 8)}, 4),
        ((Shaped(5, 2),), {}, 5),
        ((Shaped(7),), {}, 1),  # unbatched 1-D ids
        ((), {"input_ids": Shaped(0, 4)}, 0),  # zero-row call
        ((), {"input_ids": Shaped(2, 1), "inputs_embeds": Shaped(9, 1, 8)}, 2),  # ids win
        (("text",), {"attention_mask": None}, 1),  # nothing shaped
    ],
)
def test_sequences_in_call_reads_batch_size(args, kwargs, expected):
    assert sequences_in_call(args, kwargs) == expected


def test_zero_sequence_call_passes_an_exhausted_budget():
    budget = PassBudget(0)
    budget.check(sequences_in_call((), {"input_ids": Shaped(0, 4)}))
    assert budget.consume(0) == 0


def test_budget_module_stays_torch_free():
    import inspect

    from oumi_weights_lab import budget as module

    source = inspect.getsource(module)
    assert "import torch" not in source and "from torch" not in source
