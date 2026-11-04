"""Experimental exact tensor circuits with fixed tables and no training.

These modules use explicit indexing, comparisons, and integer arithmetic as well
as frozen weights. They are not claims of learned neural arithmetic.
"""

from __future__ import annotations

import re
from datetime import date

import torch
from torch import nn


def _integers(value, *, minimum=None, maximum=None):
    value = torch.as_tensor(value)
    if value.is_floating_point() and (
        not torch.isfinite(value).all() or (value != value.round()).any()
    ):
        raise ValueError("inputs must contain finite integers")
    value = value.long()
    if minimum is not None and (value < minimum).any():
        raise ValueError(f"inputs must be at least {minimum}")
    if maximum is not None and (value > maximum).any():
        raise ValueError(f"inputs must be at most {maximum}")
    return value


class TokenMatcher(nn.Module):
    """Map token IDs to fixed indicator channels for selected tokens."""

    def __init__(self, vocabulary_size, tokens):
        super().__init__()
        selected = _integers(tokens, minimum=0, maximum=vocabulary_size - 1)
        if selected.ndim != 1 or len(selected.unique()) != len(selected):
            raise ValueError("tokens must be a distinct one-dimensional list")
        weights = torch.zeros(vocabulary_size, len(selected))
        weights[selected, torch.arange(len(selected))] = 1
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)

    def forward(self, tokens):
        return self.embedding(
            _integers(tokens, minimum=0, maximum=self.embedding.num_embeddings - 1)
        )


class NearestMatch(nn.Module):
    """Index the nearest matching token at or before each sequence position.

    With inclusive=False the current position cannot match. Missing matches are -1.
    """

    def __init__(self, token, inclusive=True):
        super().__init__()
        self.token, self.inclusive = token, inclusive

    def forward(self, tokens):
        tokens = _integers(tokens)
        indices = torch.arange(tokens.shape[-1], device=tokens.device).expand_as(tokens)
        matched = torch.where(tokens == self.token, indices, -1).cummax(dim=-1).values
        if not self.inclusive:
            matched = torch.cat((torch.full_like(matched[..., :1], -1), matched[..., :-1]), dim=-1)
        return matched


class LookupTable(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values).clone())
        if self.values.ndim < 1 or not len(self.values):
            raise ValueError("lookup table must contain rows")

    def forward(self, indices):
        return self.values[_integers(indices, minimum=0, maximum=len(self.values) - 1)]


class DigitAdder(nn.Module):
    """Add digits using a fixed [a,b,carry]->[digit,carry] lookup table."""

    def __init__(self, base=10):
        super().__init__()
        if not isinstance(base, int) or not 2 <= base <= 36:
            raise ValueError("base must be an integer from 2 to 36")
        self.base = base
        table = [
            [((a + b + carry) % base, (a + b + carry) // base) for carry in range(2)]
            for a in range(base)
            for b in range(base)
        ]
        self.register_buffer("table", torch.tensor(table).reshape(base, base, 2, 2))

    def forward(self, left, right, carry=0):
        left = _integers(left, minimum=0, maximum=self.base - 1).to(self.table.device)
        right = _integers(right, minimum=0, maximum=self.base - 1).to(self.table.device)
        carry = _integers(carry, minimum=0, maximum=1).to(self.table.device)
        return self.table[left, right, carry]

    def add(self, left, right):
        """Add equally sized, least-significant-first digit vectors; keep carry."""
        left, right = torch.as_tensor(left), torch.as_tensor(right)
        if left.shape != right.shape or left.ndim < 1 or left.shape[-1] == 0:
            raise ValueError("digit arrays must have equal nonempty shapes")
        carry, digits = torch.zeros(left.shape[:-1], dtype=torch.long, device=self.table.device), []
        for index in range(left.shape[-1]):
            output = self(left[..., index], right[..., index], carry)
            digits.append(output[..., 0])
            carry = output[..., 1]
        return torch.stack([*digits, carry], dim=-1)


class Compare(nn.Module):
    """Return -1, 0, or 1 for exact integer ordering (no float subtraction)."""

    def forward(self, left, right):
        left, right = _integers(left), _integers(right)
        return (left > right).long() - (left < right).long()


class DateCalculator(nn.Module):
    """Gregorian weekdays for valid dates in 2000..2099; Monday=0."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "month_offsets", torch.tensor([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        )
        self.register_buffer(
            "month_lengths", torch.tensor([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        )

    def forward(self, dates):
        values = _integers(dates).to(self.month_offsets.device)
        if values.ndim < 1 or values.shape[-1] != 3:
            raise ValueError("dates must have final dimension [year, month, day]")
        year, month, day = values.unbind(-1)
        if ((year < 2000) | (year > 2099) | (month < 1) | (month > 12)).any():
            raise ValueError("dates must be in 2000..2099 with months 1..12")
        leap = year.remainder(4) == 0
        length = self.month_lengths[month - 1] + (leap & (month == 2)).long()
        if ((day < 1) | (day > length)).any():
            raise ValueError("invalid day for month")
        years = year - 2000
        days = years * 365 + (years + 3) // 4 + self.month_offsets[month - 1]
        days += (leap & (month > 2)).long() + day - 1
        return (days + 5).remainder(7)  # 2000-01-01 was Saturday.


class DateFacts:
    """Insert checked ISO-date weekday facts before an unchanged question."""

    def __call__(self, prompt):
        facts = []
        for text in dict.fromkeys(re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", prompt)):
            try:
                value = date.fromisoformat(text)
            except ValueError:
                continue
            names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            facts.append(f"{text} is {names[value.weekday()]}.")
        return "Calendar facts: " + " ".join(facts) + "\n" + prompt if facts else prompt
