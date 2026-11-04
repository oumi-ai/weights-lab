from concurrent.futures import ThreadPoolExecutor

import pytest

from oumi_weights_lab.budget import BudgetExceeded, PassBudget


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
