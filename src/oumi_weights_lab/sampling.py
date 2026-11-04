"""Per-sequence sampling and finish tracking for batched generation.

generate() runs custom logits processors before its own sampling warpers, so the
sampler applies those warpers itself, draws each row from that row's own CPU
generator, and returns a one-hot that greedy decoding then selects. A row's
tokens therefore depend only on its seed and its own logits, never on which
other rows share its batch.

This module imports torch and transformers at the top for its base classes, so
import it only inside functions; the CLI must stay importable without torch.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from transformers import GenerationConfig
from transformers.generation import logits_process
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    StoppingCriteria,
    StopStringCriteria,
)

# GenerationConfig fields consumed by sampling warpers rather than by generate() itself.
SAMPLING_FIELDS = (
    "temperature",
    "top_h",
    "top_k",
    "top_p",
    "min_p",
    "typical_p",
    "epsilon_cutoff",
    "eta_cutoff",
)


def effective_config(model: Any, settings: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """The GenerationConfig generate() would use, and the settings it does not consume.

    Priority matches generate(): explicit settings, then the model's generation
    config, then transformers' global defaults for anything still unset.
    """
    base = getattr(model, "generation_config", None)
    config = copy.deepcopy(base) if base is not None else GenerationConfig()
    defaults = getattr(type(config), "_get_default_generation_params", None)
    if defaults is not None:
        config.update(**defaults(), defaults_only=True)
    unused = config.update(**copy.deepcopy(settings))
    return config, unused


def neutral_sampling_settings() -> dict[str, Any]:
    """Settings for the greedy pass that takes the sampler's draw: no warpers, plain greedy.

    penalty_alpha is ignored when sampling, but in a greedy pass it would switch
    generate() to contrastive search.
    """
    defaults = getattr(GenerationConfig, "_get_default_generation_params", None)
    values = defaults() if defaults is not None else vars(GenerationConfig())
    return {**{field: values.get(field) for field in SAMPLING_FIELDS}, "penalty_alpha": None}


def warpers(config: Any) -> list[LogitsProcessor]:
    """The warpers generate() would apply when sampling one sequence, in its order."""
    out: list[LogitsProcessor] = []
    value = getattr(config, "temperature", None)
    if value is not None and value != 1.0:
        out.append(logits_process.TemperatureLogitsWarper(value))
    value = getattr(config, "top_h", None)
    if value is not None:
        out.append(logits_process.TopHLogitsWarper(top_h=value))
    value = getattr(config, "top_k", None)
    if value is not None and value != 0:
        out.append(logits_process.TopKLogitsWarper(top_k=value, min_tokens_to_keep=1))
    value = getattr(config, "top_p", None)
    if value is not None and value < 1.0:
        out.append(logits_process.TopPLogitsWarper(top_p=value, min_tokens_to_keep=1))
    value = getattr(config, "min_p", None)
    if value is not None:
        out.append(logits_process.MinPLogitsWarper(min_p=value, min_tokens_to_keep=1))
    value = getattr(config, "typical_p", None)
    if value is not None and value < 1.0:
        out.append(logits_process.TypicalLogitsWarper(mass=value, min_tokens_to_keep=1))
    value = getattr(config, "epsilon_cutoff", None)
    if value is not None and 0.0 < value < 1.0:
        out.append(logits_process.EpsilonLogitsWarper(epsilon=value, min_tokens_to_keep=1))
    value = getattr(config, "eta_cutoff", None)
    if value is not None and 0.0 < value < 1.0:
        out.append(logits_process.EtaLogitsWarper(epsilon=value, min_tokens_to_keep=1))
    return out


class PerSequenceSampler(LogitsProcessor):
    """Draw each row's next token from its own seeded CPU generator, by inverse CDF."""

    def __init__(self, seeds: list[int], processors: list[LogitsProcessor]):
        self.generators = [torch.Generator().manual_seed(int(seed)) for seed in seeds]
        self.processors = processors

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if scores.shape[0] != len(self.generators):
            raise ValueError("the sampler needs exactly one seed per sequence")
        for processor in self.processors:
            scores = processor(input_ids, scores)
        probs = scores.float().softmax(-1)
        cdf = probs.cumsum(-1)
        draws = torch.stack(
            [
                torch.rand((), generator=generator, dtype=torch.float64)
                for generator in self.generators
            ]
        ).to(device=cdf.device, dtype=cdf.dtype)
        tokens = torch.searchsorted(cdf, (draws * cdf[:, -1])[:, None], right=True).squeeze(-1)
        # Rounding can put a draw at the very top; fall back to the last token with mass.
        last = probs.shape[-1] - 1 - (probs > 0).flip(-1).int().argmax(-1)
        tokens = torch.minimum(tokens, last)
        chosen = torch.full_like(scores, float("-inf"))
        return chosen.scatter(-1, tokens[:, None], 0.0)


class FinishTracker(StoppingCriteria):
    """Record when and why each row finished. It never stops generation itself."""

    def __init__(
        self,
        batch: int,
        prompt_length: int,
        *,
        eos_token_id: Any = None,
        stop_strings: Any = None,
        tokenizer: Any = None,
    ):
        self.prompt_length = prompt_length
        self.criteria: list[tuple[str, StoppingCriteria]] = []
        if eos_token_id is not None:
            self.criteria.append(("eos", EosTokenCriteria(eos_token_id)))
        if stop_strings:
            self.criteria.append(("stop_string", StopStringCriteria(tokenizer, stop_strings)))
        self.finished: list[tuple[int, str] | None] = [None] * batch

    @property
    def unfinished(self) -> int:
        return sum(entry is None for entry in self.finished)

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        steps = input_ids.shape[1] - self.prompt_length
        for reason, criterion in self.criteria:  # EOS first, so it wins a tie.
            for row, hit in enumerate(criterion(input_ids, scores).tolist()):
                if hit and self.finished[row] is None:
                    self.finished[row] = (steps, reason)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    def results(self, steps: int) -> list[tuple[int, str]]:
        """Each row's (generated tokens, finish reason); unfinished rows ran to the limit."""
        return [entry or (steps, "max_tokens") for entry in self.finished]
