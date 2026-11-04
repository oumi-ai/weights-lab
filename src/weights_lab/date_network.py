"""A finite calendar implemented with frozen embeddings, linear maps, and ReLU.

Python builds calendar tables and encodes ISO inputs. Inference uses those weights;
it does not call datetime to compute an answer. This is constructed computation,
not learned arithmetic or a language-model tokenizer.
"""

from __future__ import annotations

from datetime import date, timedelta

import torch
from torch import nn


def _linear(weights):
    # Discarding random initialization must not perturb a lab's generation seed.
    with torch.random.fork_rng(devices=[]):
        layer = nn.Linear(len(weights), 1, bias=False)
    layer.weight = nn.Parameter(torch.tensor([weights], dtype=torch.float32), requires_grad=False)
    return layer


class FixedDateNetwork(nn.Module):
    """Exact day arithmetic over an inclusive, finite date domain in float32.

    Inputs and outputs are integer day IDs relative to ``start``. ``encode`` and
    ``render`` translate ISO strings. All parameters are frozen. Business days
    mean Monday through Friday, with no public holidays and the start excluded.
    """

    vocabulary = "0123456789-"

    def __init__(self, start="2000-01-01", end="2099-12-31"):
        super().__init__()
        self.start, self.end = date.fromisoformat(start), date.fromisoformat(end)
        self.day_count = (self.end - self.start).days + 1
        if not 1 <= self.day_count <= 36600:
            raise ValueError("domain must contain 1..36600 days in ascending order")
        features, characters, business_dates = [], [], []
        rank = -1
        for index in range(self.day_count):
            day = self.start + timedelta(days=index)
            weekday = day.weekday()
            weekend = weekday >= 5
            if not weekend:
                rank += 1
                business_dates.append([index])
            forward = index + (7 - weekday if weekend else 0)
            backward = index - (weekday - 4 if weekend else 0)
            features.append([index, weekday, rank, rank + weekend, forward, backward])
            characters.append([self.vocabulary.index(char) for char in day.isoformat()])
        self.dates = nn.Embedding.from_pretrained(torch.tensor(features).float(), freeze=True)
        self.characters = nn.Embedding.from_pretrained(
            torch.tensor(characters).float(), freeze=True
        )
        # Keep a dummy row for domains containing only a weekend; checks reject
        # nonzero business offsets before that row can become an answer.
        self.business_count = len(business_dates)
        self.business_dates = nn.Embedding.from_pretrained(
            torch.tensor(business_dates or [[0]]).float(), freeze=True
        )
        self.character_logits = nn.Embedding.from_pretrained(torch.eye(11), freeze=True)
        self.add = _linear([1, 1])
        self.subtract = _linear([-1, 1])
        self.relu = nn.ReLU()

    def _integers(self, values, minimum, maximum):
        if self.dates.weight.dtype != torch.float32:
            raise ValueError("FixedDateNetwork requires float32 weights for exact day IDs")
        values = torch.as_tensor(values, device=self.dates.weight.device)
        if values.is_complex() or (
            values.is_floating_point()
            and (not torch.isfinite(values).all() or (values != values.round()).any())
        ):
            raise ValueError("inputs must contain finite integers")
        original_dtype = values.dtype
        # Compare in a type that represents the domain bounds exactly. Narrow
        # integer comparisons can otherwise overflow their Python scalar bound.
        values = values.float() if values.is_floating_point() else values.long()
        # uint64 -> int64 wraps above 2**63-1. Those are outside every supported
        # range, even when the wrapped value would look like a valid offset.
        if original_dtype == torch.uint64 and (values < 0).any():
            raise ValueError("unsigned integer input exceeds the supported range")
        if ((values < minimum) | (values > maximum)).any():
            raise ValueError(f"input or result outside supported range {minimum}..{maximum}")
        return values.long()

    def _ids(self, values):
        return self._integers(values, 0, self.day_count - 1)

    def encode(self, values):
        """Encode an ISO string/date or a flat sequence of them as day IDs."""
        scalar = isinstance(values, (str, date))
        days = [values] if scalar else values
        indices = [
            ((date.fromisoformat(day) if isinstance(day, str) else day) - self.start).days
            for day in days
        ]
        result = self._ids(indices)
        return result[0] if scalar else result

    def forward(self, day_ids, offsets=0, *, business=False):
        """Add signed calendar or business days; input shapes broadcast.

        Zero always preserves the date, including weekends. Nonzero business
        offsets count only visited weekdays, in either direction. Results beyond
        the configured domain raise ValueError rather than wrapping or clipping.
        """
        indices, offsets = torch.broadcast_tensors(
            self._ids(day_ids), self._integers(offsets, -self.day_count, self.day_count)
        )
        features = self.dates(indices)
        if not business:
            result = self.add(torch.stack((features[..., 0], offsets.float()), dim=-1))
            return self._ids(result.squeeze(-1))
        base = torch.where(offsets < 0, features[..., 3], features[..., 2])
        ranks = self.add(torch.stack((base, offsets.float()), dim=-1)).squeeze(-1)
        invalid = (ranks < 0) | (ranks >= self.business_count)
        if ((offsets != 0) & invalid).any():
            raise ValueError("business-day result outside supported date domain")
        safe = ranks.clamp(0, max(0, self.business_count - 1)).long()
        result = self.business_dates(safe).squeeze(-1).long()
        return torch.where(offsets == 0, indices, result)

    def roll_weekend(self, day_ids, *, direction="forward"):
        """Keep weekdays; move weekends to Monday or Friday, respectively."""
        if direction not in {"forward", "backward"}:
            raise ValueError("direction must be forward or backward")
        column = 4 if direction == "forward" else 5
        return self._ids(self.dates(self._ids(day_ids))[..., column])

    def weekday(self, day_ids):
        """Return Monday=0 through Sunday=6."""
        return self.dates(self._ids(day_ids))[..., 1].long()

    def gap(self, start_ids, end_ids):
        """Signed calendar days from start to end (end minus start)."""
        start, end = torch.broadcast_tensors(self._ids(start_ids), self._ids(end_ids))
        values = torch.stack((self.dates(start)[..., 0], self.dates(end)[..., 0]), dim=-1)
        return self.subtract(values).squeeze(-1).long()

    def compare(self, left_ids, right_ids):
        """Return -1, 0, or 1 when left is before, equal to, or after right."""
        difference = self.gap(right_ids, left_ids).float()
        positive = self.relu(difference) - self.relu(difference - 1)
        negative = self.relu(-difference) - self.relu(-difference - 1)
        return (positive - negative).long()

    def token_ids(self, day_ids):
        """Return ten ISO character IDs per date, using ``vocabulary``."""
        return self.characters(self._ids(day_ids)).long()

    def logits(self, day_ids):
        """Return [..., 10, 11] fixed character logits; argmax is exact."""
        return self.character_logits(self.token_ids(day_ids))

    def decode(self, token_ids):
        """Decode ten-character sequences; preserve batch dimensions as lists."""
        tokens = self._integers(token_ids, 0, len(self.vocabulary) - 1)
        if tokens.ndim == 0 or tokens.shape[-1] != 10:
            raise ValueError("ISO token sequences must have final dimension 10")

        def decode_rows(rows):
            if not rows or isinstance(rows[0], list):
                return [decode_rows(row) for row in rows]
            return "".join(self.vocabulary[token] for token in rows)

        return decode_rows(tokens.tolist())

    def render(self, day_ids):
        """Render computed IDs through the fixed character-logit readout."""
        return self.decode(self.logits(day_ids).argmax(dim=-1))
