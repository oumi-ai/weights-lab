"""Directions from contrasting prompts, with a held-out check."""

from __future__ import annotations

import math
import random
from typing import Any

from .inspection import _accepts, _is_int
from .scoring import _check_batch_size, _pad_token

_METHODS = ("mean_diff", "pca")
_SITES = ("output", "input")
_STACKS = ("layers", "h")


def _text(value: Any, role: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"each {role} prompt must be text")
    if not value.strip():
        raise ValueError(f"{role} prompts must not be empty")
    return value


def _normalized_pairs(pairs: Any) -> list[tuple[str, str]]:
    if isinstance(pairs, (str, dict)) or not hasattr(pairs, "__iter__"):
        raise ValueError("pairs must be a list of (positive, negative) prompts")
    result = []
    for item in pairs:
        if isinstance(item, dict):
            if set(item) != {"positive", "negative"}:
                raise ValueError('a pair dict needs exactly the keys "positive" and "negative"')
            positive, negative = item["positive"], item["negative"]
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            positive, negative = item
        else:
            raise ValueError("each pair must be (positive, negative) or a positive/negative dict")
        pair = (_text(positive, "positive"), _text(negative, "negative"))
        if pair[0] == pair[1]:
            raise ValueError("the positive and negative prompt of a pair must differ")
        result.append(pair)
    if len(result) < 4:
        raise ValueError("contrast_directions needs at least 4 pairs")
    if len(set(result)) != len(result):
        raise ValueError("pairs must not contain duplicates")
    if {(negative, positive) for positive, negative in result} & set(result):
        raise ValueError(
            "a pair also appears reversed: (a, b) with (b, a) would put the same contrast "
            "on both sides of the split"
        )
    return result


def _decoder_layers(model: Any) -> dict[int, str]:
    """Names of the decoder blocks, by layer index: modules called ``layers.N`` or ``h.N``."""
    found: dict[int, str] = {}
    for name, _ in model.named_modules():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-1].isdigit() and parts[-2] in _STACKS:
            index = int(parts[-1])
            if index in found:
                raise ValueError(
                    f"{found[index]} and {name} have the same layer index {index}; "
                    "name the modules with record and patch instead"
                )
            found[index] = name
    if not found:
        raise ValueError(
            "no decoder layers were found: expected modules named like model.layers.N "
            "or transformer.h.N"
        )
    return found


def _groups(pairs: list[tuple[str, str]]) -> list[list[int]]:
    """Pair indexes grouped so pairs connected by a shared prompt are in one group."""
    parent: dict[str, str] = {}

    def find(prompt: str) -> str:
        parent.setdefault(prompt, prompt)
        while parent[prompt] != prompt:
            parent[prompt] = parent[parent[prompt]]
            prompt = parent[prompt]
        return prompt

    for positive, negative in pairs:
        parent[find(positive)] = find(negative)
    members: dict[str, list[int]] = {}
    for index, (positive, _) in enumerate(pairs):
        members.setdefault(find(positive), []).append(index)
    return list(members.values())


def _split(
    groups: list[list[int]], count: int, held_out: float, seed: int
) -> tuple[list[int], list[int]]:
    """Train and held-out pair indexes; a group goes whole to one side.

    Groups are taken in a shuffled order for the held-out side until it reaches
    ``ceil(held_out * count)`` pairs, never past half the pairs, so at least half train.
    """
    target = math.ceil(held_out * count - 1e-9)
    cap = count // 2
    order = list(groups)
    random.Random(seed).shuffle(order)
    held: list[int] = []
    for group in order:
        if len(held) < target and len(held) + len(group) <= cap:
            held.extend(group)
    taken = set(held)
    return [index for index in range(count) if index not in taken], sorted(held)


def _auc(positive: Any, negative: Any) -> float:
    wins = (positive[:, None] > negative[None, :]).double().sum()
    ties = (positive[:, None] == negative[None, :]).double().sum()
    return ((wins + 0.5 * ties) / (len(positive) * len(negative))).item()


def _effect_size(positive: Any, negative: Any) -> float | None:
    if len(positive) < 2:
        return None
    pooled = ((positive.var() + negative.var()) / 2).sqrt().item()
    gap = (positive.mean() - negative.mean()).item()
    if pooled == 0:
        return 0.0 if gap == 0 else math.copysign(math.inf, gap)
    return gap / pooled


def _direction(differences: Any, method: str) -> tuple[Any, float, float | None]:
    """The unit direction, the raw mean-difference norm, and PCA's explained variance."""
    import torch

    mean = differences.mean(0)
    norm = mean.norm().item()
    if method == "mean_diff":
        return (mean / norm if norm > 0 else mean), norm, None
    # Uncentered, so the consistent offset between positive and negative is kept.
    _, singular, components = torch.linalg.svd(differences.double(), full_matrices=False)
    total = (singular**2).sum().item()
    if total == 0:
        return torch.zeros_like(mean), norm, None
    direction = components[0].float()
    if torch.dot(direction, mean).item() < 0:
        direction = -direction
    return direction, norm, (singular[0] ** 2).item() / total


class DirectionsMixin:
    def contrast_directions(
        self,
        pairs: Any,
        *,
        layers: list[int] | None = None,
        method: str = "mean_diff",
        positions: int = -1,
        at: str = "output",
        held_out: float = 0.25,
        seed: int = 0,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """Compute a direction per layer from contrasting prompts and test it on held-out pairs.

        ``pairs`` are ``(positive, negative)`` tuples or ``{"positive", "negative"}`` dicts.
        Each prompt is run once, so a call costs two passes per pair. The activation at
        ``positions`` (a negative int; the default -1 is the last token) stands for a prompt,
        read from the residual stream after each decoder block (``at="output"``) or before it
        (``at="input"``). ``layers`` are decoder layer indexes; None means every layer.

        Pairs that share a prompt, directly or through a chain, form a group, and a group
        goes whole to one side of the split, so no prompt is seen on both sides. Groups are
        chosen from ``seed`` until the held-out side reaches ``held_out`` (a fraction from 0
        to 0.5) of the pairs, rounded up, but never more than half the pairs, so at least half
        train. With ``held_out`` above 0 there must be at least two groups. ``"mean_diff"`` averages
        positive minus negative over the training pairs. ``"pca"`` takes the top right
        singular vector of the uncentered training differences (the first principal
        component after randomly flipping each pair's sign), signed so the mean difference
        projects positively. ``explained_variance`` is the share of the differences' second
        moment it carries, sigma1^2 over the sum of sigma^2; it is None when every
        difference is zero, and the direction is then all zeros.

        Each row holds the unit ``direction`` (float32, CPU), the raw mean-difference
        ``norm`` to scale steering by, the pair indexes in ``train_pairs`` and
        ``heldout_pairs``, and three checks on the held-out pairs: ``heldout_pair_accuracy``
        (the share of pairs whose positive projects above its negative), ``heldout_auc``
        (separating positive from negative prompts by projection) and
        ``heldout_effect_size`` (difference of mean projections over the pooled standard
        deviation; None with fewer than two held-out pairs). They are None when
        ``held_out`` is 0. A zero ``norm`` means the contrast is invisible at that site, and
        its direction is all zeros.

        Each row also has ``pair_groups``, the number of groups. Refuses fewer than 4 pairs,
        empty or duplicate pairs, a pair that also appears reversed, one group of pairs with
        ``held_out`` above 0, a pair whose two prompts are
        equal, a position that left padding would move (anything but a negative int, or one
        a prompt is too short for), unknown layers, and a call that would exceed a pass cap.
        Prompts pass through any prefix attachment, as scoring does.
        """
        import torch

        pairs = _normalized_pairs(pairs)
        if method not in _METHODS:
            raise ValueError(f"method must be one of {', '.join(map(repr, _METHODS))}")
        if at not in _SITES:
            raise ValueError(f"at must be one of {', '.join(map(repr, _SITES))}")
        if not _is_int(positions) or positions >= 0:
            raise ValueError(
                "positions must be a negative integer such as -1: prompts are left-padded in "
                "batches, so a position counted from the left would select padding"
            )
        if isinstance(held_out, bool) or not isinstance(held_out, (int, float)):
            raise ValueError("held_out must be a number from 0 to 0.5")
        if not 0 <= held_out <= 0.5:
            raise ValueError("held_out must be a number from 0 to 0.5")
        if not _is_int(seed):
            raise ValueError("seed must be an integer")
        _check_batch_size(batch_size)
        self._check_batching(batch_size)
        blocks = _decoder_layers(self.model)
        if layers is None:
            chosen = sorted(blocks)
        else:
            if (
                not isinstance(layers, (list, tuple))
                or not layers
                or not all(_is_int(layer) for layer in layers)
                or len(set(layers)) != len(layers)
                or not set(layers) <= set(blocks)
            ):
                raise ValueError(
                    f"layers must be distinct decoder layer indexes from {sorted(blocks)}"
                )
            chosen = sorted(layers)
        modules = [blocks[layer] for layer in chosen]

        with self._prompts_prepared_once():
            tokens = [
                [self._prompt_ids(self._scoring_prompt(text)) for text in pair] for pair in pairs
            ]
        shortest = min(len(ids) for pair in tokens for ids in pair)
        if shortest < -positions:
            raise ValueError(
                f"every prompt needs at least {-positions} tokens for position {positions}, "
                f"but one has {shortest}; a shorter prompt would select padding"
            )
        groups = _groups(pairs)
        if held_out > 0 and len(groups) < 2:
            raise ValueError(
                "all pairs share prompts, directly or through a chain, so they form one group "
                "and cannot be split without leaking; use pairs with distinct prompts, or "
                "held_out=0"
            )
        train, held = _split(groups, len(pairs), held_out, seed)
        rows = [ids for pair in tokens for ids in pair]
        self._check_pass_cap(len(rows))

        pad = _pad_token(self.tokenizer)
        takes_positions = _accepts(self.model, "position_ids")
        keeps_logits = _accepts(self.model, "logits_to_keep")
        captured = {name: [] for name in modules}
        with self._evaluation_mode(), self.record(modules, positions=positions, at=at) as seen:
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                length = max(len(ids) for ids in batch)
                ids = torch.tensor(
                    [[pad] * (length - len(row)) + row for row in batch], device=self.device
                )
                mask = torch.tensor(
                    [[0] * (length - len(row)) + [1] * len(row) for row in batch],
                    device=self.device,
                )
                inputs = {"input_ids": ids, "attention_mask": mask}
                if takes_positions:
                    inputs["position_ids"] = (mask.cumsum(-1) - 1).clamp(min=0)
                if keeps_logits:
                    inputs["logits_to_keep"] = 1
                self.model(**inputs)
                for name in modules:
                    captured[name].append(seen[name].pop()[:, 0].float())
        result = []
        for layer, name in zip(chosen, modules, strict=True):
            activations = torch.cat(captured[name]).cpu().reshape(len(pairs), 2, -1)
            positive, negative = activations[:, 0], activations[:, 1]
            direction, norm, explained = _direction((positive - negative)[train], method)
            row = {
                "layer": layer,
                "module": name,
                "at": at,
                "method": method,
                "direction": direction.clone(),
                "norm": norm,
                "explained_variance": explained,
                "pair_groups": len(groups),
                "train_pairs": list(train),
                "heldout_pairs": list(held),
                "heldout_pair_accuracy": None,
                "heldout_auc": None,
                "heldout_effect_size": None,
            }
            if held:
                up, down = positive[held] @ direction, negative[held] @ direction
                row["heldout_pair_accuracy"] = (up > down).double().mean().item()
                row["heldout_auc"] = _auc(up, down)
                row["heldout_effect_size"] = _effect_size(up, down)
            result.append(row)
        return result

    def _check_pass_cap(self, count: int) -> None:
        """Raise BudgetExceeded before any pass runs when ``count`` passes would not fit."""
        self._budget_state()
        for budget in [self.budget, *self._evaluation_budgets]:
            budget.check(count)
