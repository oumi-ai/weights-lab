from datetime import date, timedelta

import pytest
import torch

from weights_lab.circuits import (
    Compare,
    DateCalculator,
    DateFacts,
    DigitAdder,
    LookupTable,
    NearestMatch,
    TokenMatcher,
)


def test_fixed_token_lookup_and_nearest_match():
    matcher = TokenMatcher(5, [1, 3])
    assert matcher(torch.tensor([0, 1, 3, 1])).tolist() == [[0, 0], [1, 0], [0, 1], [1, 0]]
    assert not any(parameter.requires_grad for parameter in matcher.parameters())
    tokens = torch.tensor([[0, 1, 4, 1, 2], [2, 2, 1, 0, 0]])
    assert NearestMatch(1)(tokens).tolist() == [[-1, 1, 1, 3, 3], [-1, -1, 2, 2, 2]]
    assert NearestMatch(1, inclusive=False)(tokens).tolist() == [
        [-1, -1, 1, 1, 3],
        [-1, -1, -1, 2, 2],
    ]
    assert LookupTable([[2, 4], [6, 8]])([1, 0]).tolist() == [[6, 8], [2, 4]]
    with pytest.raises(ValueError):
        matcher(torch.tensor([5]))


def test_digit_carries_for_every_digit_pair_and_multiple_columns():
    circuit = DigitAdder()
    for a in range(10):
        for b in range(10):
            for carry in range(2):
                assert circuit(a, b, carry).tolist() == [
                    (a + b + carry) % 10,
                    (a + b + carry) // 10,
                ]
    assert circuit.add([9, 9, 9], [1, 0, 0]).tolist() == [0, 0, 0, 1]
    assert circuit.add([[9, 9], [2, 1]], [[1, 0], [8, 2]]).tolist() == [[0, 0, 1], [0, 4, 0]]
    assert Compare()([2**62, -(2**62), 4], [-(2**62), 2**62, 4]).tolist() == [1, -1, 0]
    with pytest.raises(ValueError):
        circuit(1.5, 2)


def test_calendar_is_exact_for_every_day_in_2000_through_2099():
    start, end = date(2000, 1, 1), date(2100, 1, 1)
    days = [start + timedelta(days=index) for index in range((end - start).days)]
    dates = torch.tensor([[day.year, day.month, day.day] for day in days])
    assert DateCalculator()(dates).tolist() == [day.weekday() for day in days]
    for invalid in ([2001, 2, 29], [2100, 1, 1], [2000, 13, 1], [2000, 1, 0]):
        with pytest.raises(ValueError):
            DateCalculator()(invalid)


def test_date_facts_validate_dates_and_preserve_question():
    question = "Which day is 2026-11-03? Ignore invalid 2026-02-30."
    result = DateFacts()(question)
    assert "2026-11-03 is Tuesday." in result
    assert "2026-02-30 is" not in result
    assert result.endswith(question)
    assert DateFacts()("No date") == "No date"
