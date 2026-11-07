"""Auditable local evaluation, continuation likelihoods, and pass limits.

Continuation policy: encode prompt and answer separately, with special tokens
disabled, then concatenate their token IDs. No whitespace is inserted. Only
answer tokens contribute log probability; an empty prompt uses BOS, or EOS if
BOS is unavailable. This explicit token boundary avoids silently scoring a
prompt suffix when tokenization merges across the text boundary. Include any
desired leading whitespace in the answer itself. Scores are sums, not averages.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from .budget import PassBudget, sequences_in_call
from .evaluation import EvaluationMixin, majority, pass_at_k
from .evaluation_data import _choices, _digest, _reserve, _safe_name, _tasks
from .session import write_json

# The scorer supplies these itself; a caller's copy would collide inside generate().
_OWNED_GENERATE_ARGUMENTS = {
    "inputs",
    "input_ids",
    "attention_mask",
    "generation_config",
    "tokenizer",
}
# GenerationConfig.validate() accepts these, but generate() raises on the first call.
_SAMPLING_RANGES = {
    "temperature": (lambda value: value > 0, "must be positive when sampling"),
    "top_k": (lambda value: value >= 0, "must be nonnegative"),
    "top_p": (lambda value: 0 < value <= 1, "must be in (0, 1]"),
    "typical_p": (lambda value: 0 < value <= 1, "must be in (0, 1]"),
    "min_p": (lambda value: 0 <= value <= 1, "must be in [0, 1]"),
}


DEFAULT_BATCH_SIZE = 8
# apply_chat_template arguments the lab sets itself.
_OWNED_CHAT_ARGUMENTS = {
    "conversation",
    "tokenize",
    "add_generation_prompt",
    "continue_final_message",
    "return_dict",
    "return_tensors",
    "return_assistant_tokens_mask",
}
# Rendered once per chat setting to fingerprint what the template produces.
_CHAT_PROBE = "Weights Lab prompt format probe."


def _prediction(scores: dict[str, float]) -> str | None:
    """The best-scoring option's label, or None on an exact tie, which is never correct."""
    best = max(scores.values())
    leaders = [label for label, value in scores.items() if value == best]
    return leaders[0] if len(leaders) == 1 else None


def _check_batch_size(batch_size: Any) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


def _pack(rows: list[list[Any]], batch_size: int) -> Iterator[list[Any]]:
    """Group rows' sequences into batches: whole rows in order, splitting only oversized rows."""
    batch: list[Any] = []
    for row in rows:
        if len(row) > batch_size:
            if batch:
                yield batch
                batch = []
            for start in range(0, len(row), batch_size):
                yield row[start : start + batch_size]
            continue
        if len(batch) + len(row) > batch_size:
            yield batch
            batch = []
        batch.extend(row)
    if batch:
        yield batch


def _pass_at_values(pass_at: Any, num_samples: int) -> list[int]:
    """Validated, sorted pass@k values; each k must be between 1 and num_samples."""
    if pass_at is None:
        return []
    values = [pass_at] if isinstance(pass_at, int) and not isinstance(pass_at, bool) else pass_at
    if (
        not isinstance(values, (list, tuple))
        or not values
        or any(
            isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= num_samples
            for k in values
        )
    ):
        raise ValueError("pass_at values must be integers between 1 and num_samples")
    return sorted(set(values))


def _checker_call(checker: Any) -> Any:
    """A grader taking (generated, answer, row); the row goes only to checkers that name it.

    A checker receives a copy of the row only when its signature has a parameter
    named ``row``. ``**kwargs`` alone does not count, so two-argument checkers that
    forward keyword arguments keep working; everything else is called as
    checker(generated, answer).
    """
    if checker is None:
        return None
    import inspect

    try:
        parameters = inspect.signature(checker).parameters
    except (TypeError, ValueError):  # Some builtins and C callables have no signature.
        parameters = {}
    row = parameters.get("row")
    if row is None or row.kind in (row.VAR_POSITIONAL, row.VAR_KEYWORD):
        return lambda generated, answer, task: checker(generated, answer)
    if row.kind is row.POSITIONAL_ONLY:
        return lambda generated, answer, task: checker(generated, answer, copy.deepcopy(task))
    return lambda generated, answer, task: checker(generated, answer, row=copy.deepcopy(task))


def _graded(
    samples: list[dict[str, Any]],
    task: dict[str, Any],
    grade: Any,
    answer_key: Any,
    pass_ks: list[int],
) -> dict[str, Any]:
    """Item fields for one row's samples: correctness, and optional pass@k and vote."""
    graded = []
    for sample in samples:
        generated = sample["generated"]
        correct = (
            grade(generated, task["answer"], task)
            if grade is not None
            else generated.strip() == str(task["answer"]).strip()
        )
        entry = {**sample, "correct": bool(correct)}
        if answer_key is not None:
            entry["key"] = answer_key(generated, copy.deepcopy(task))
            try:
                json.dumps(entry["key"], allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"answer_key must return JSON values; got {entry['key']!r}"
                ) from error
        graded.append(entry)
    count = sum(entry["correct"] for entry in graded)
    if len(graded) == 1:
        (only,) = graded
        fields = {
            "generated": only["generated"],
            "prediction": only["generated"],
            "correct": only["correct"],
            "finish_reason": only["finish_reason"],
            "generated_tokens": only["generated_tokens"],
            "sample_seed": only["sample_seed"],
        }
    else:
        fields = {
            "samples": graded,
            "prediction": None,
            "correct": count / len(graded),
            "correct_samples": count,
        }
    if pass_ks:
        fields["pass_at"] = {str(k): pass_at_k(len(graded), count, k) for k in pass_ks}
    if answer_key is not None:
        # A key of None means the extractor found no answer: that sample abstains.
        keys = [entry["key"] for entry in graded if entry["key"] is not None]
        vote, tie = majority(keys) if keys else (None, False)
        fields.update(
            vote=vote,
            vote_tie=tie,
            # A tie names no answer, so it is never correct.
            vote_correct=vote is not None
            and not tie
            and any(entry["correct"] for entry in graded if entry["key"] == vote),
        )
        if len(graded) > 1:
            fields["prediction"] = vote
    return fields


def _padding_sensitive(config: Any) -> list[str]:
    """Effective generation settings whose meaning batching or left padding would change.

    Penalties that read the whole sequence would count the padding, and the pad
    token is often EOS.
    """

    def value(key: str) -> Any:
        return getattr(config, key, None)

    keys = ["num_beams"] if (value("num_beams") or 1) > 1 else []
    keys += [
        key
        for key in ("max_time", "prefill_chunk_size", "watermarking_config")
        if value(key) is not None
    ]
    if (value("min_length") or 0) > 0:
        keys.append("min_length")
    if value("repetition_penalty") not in (None, 1.0):
        keys.append("repetition_penalty")
    if (value("no_repeat_ngram_size") or 0) > 0:
        keys.append("no_repeat_ngram_size")
    if value("max_length") is not None and value("max_new_tokens") is None:
        keys.append("max_length")
    return keys


def _assisted(config: Any) -> list[str]:
    """Assisted-generation settings, whose verification forwards cover several tokens each."""
    keys = ("prompt_lookup_num_tokens", "assistant_early_exit", "use_mtp")
    return [key for key in keys if getattr(config, key, None) not in (None, False)]


def _sampler_name(config: Any) -> str:
    """Which decoder generate() runs for this effective config."""
    if (getattr(config, "num_beams", None) or 1) > 1:
        return "hf-beam-v0.1"
    if not getattr(config, "do_sample", False):
        return "greedy-v1"
    if getattr(config, "watermarking_config", None) is not None:
        # The watermark biases generate()'s own distribution, after custom processors.
        return "hf-sample-v0.1"
    return "per-sequence-uniform-v1"


def _summed(logits: Any, answer_ids: list[int]) -> float:
    """Sum the answer tokens' log probabilities from the logits that predict them."""
    import torch

    logits = logits.float()
    target = torch.tensor(answer_ids, device=logits.device)
    return float(logits.log_softmax(-1).gather(-1, target[:, None]).squeeze(-1).sum().item())


def _pad_token(tokenizer: Any) -> int:
    for name in ("pad_token_id", "eos_token_id"):
        token = getattr(tokenizer, name, None)
        if token is not None:
            return int(token)
    return 0


def _generation_kwargs(
    generation_config: dict[str, Any] | None, generate: Any = None
) -> dict[str, Any]:
    """Greedy defaults plus the caller's settings, refusing what generate() would reject."""
    import inspect

    from transformers import GenerationConfig

    config = {"max_new_tokens": 32, "do_sample": False, **(generation_config or {})}
    fields = {key for key in vars(GenerationConfig()) if not key.startswith("_")}
    known = set(fields)
    if generate is not None:
        known |= {
            name
            for name, parameter in inspect.signature(generate).parameters.items()
            if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        }
    unknown = sorted(set(config) - (known - _OWNED_GENERATE_ARGUMENTS))
    if unknown:
        raise ValueError(f"generation_config has unknown keys: {unknown}")
    if config.get("num_return_sequences", 1) != 1:
        raise ValueError("sampled scoring requires num_return_sequences=1")
    sampling = bool(config.get("do_sample"))
    for key, (valid, rule) in _SAMPLING_RANGES.items():
        if sampling and config.get(key) is not None and not valid(config[key]):
            raise ValueError(f"invalid generation_config: {key} {rule}")
    penalty = config.get("repetition_penalty")
    if penalty is not None and not penalty > 0:
        raise ValueError("invalid generation_config: repetition_penalty must be positive")
    try:
        # Arguments that only generate() accepts are not GenerationConfig fields.
        GenerationConfig(**{key: config[key] for key in config if key in fields}).validate()
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid generation_config: {error}") from error
    return config


class ScoringMixin(EvaluationMixin):
    """Evaluation helpers and the pass-counting hook.

    The host registers ``_count_forward`` as a forward pre-hook with kwargs, passes the
    counted model to ``_count_beams_once``, owns ``_forward_passes``, and provides
    ``device``, where input tensors are created.
    """

    @property
    def budget(self) -> PassBudget:
        if not hasattr(self, "_budget"):
            self._budget = PassBudget()
        return self._budget

    def _budget_state(self) -> None:
        if not hasattr(self, "_evaluation_budgets"):
            self._evaluation_budgets: list[PassBudget] = []
            self._budget_guard = threading.RLock()

    def _before_forward(self, count: int = 1) -> None:
        self._budget_state()
        with self._budget_guard:
            budgets = [self.budget, *self._evaluation_budgets]
            for budget in budgets:
                budget.check(count)
            for budget in budgets:
                budget.consume(count)

    def _count_forward(self, module: Any, args: tuple, kwargs: dict) -> None:
        """Forward pre-hook: charge one pass per sequence before the model runs.

        Inside beam search, a prompt's beams count as one sequence. Inside
        ``_metered``, the meter decides the cost instead, such as batched
        generation charging only its unfinished rows.
        """
        meter = getattr(self, "_meter", None)
        if meter is not None:
            count = meter()
        else:
            count = -(-sequences_in_call(args, kwargs) // getattr(self, "_beam_width", 1))
        self._before_forward(count)
        self._forward_passes += count

    @contextmanager
    def _metered(self, cost: Any):
        """While active, charge each base-model call ``cost()`` passes instead of its batch size."""
        previous = getattr(self, "_meter", None)
        self._meter = cost
        try:
            yield
        finally:
            self._meter = previous

    def _count_beams_once(self, model: Any) -> None:
        """Wrap ``model.generate`` so the counter divides beam-search batches by the beam width."""
        original = getattr(model, "generate", None)
        if original is None:
            return

        @functools.wraps(original)
        def generate(*args: Any, **kwargs: Any) -> Any:
            config = kwargs.get("generation_config") or getattr(model, "generation_config", None)
            width = kwargs.get("num_beams") or getattr(config, "num_beams", None) or 1
            previous = getattr(self, "_beam_width", 1)
            self._beam_width = width
            try:
                return original(*args, **kwargs)
            finally:
                self._beam_width = previous

        model.generate = generate

    @contextmanager
    def limit_passes(self, max_passes: int | None):
        """Limit underlying forwards within this context, in addition to the session budget."""
        local = PassBudget(max_passes)
        self._budget_state()
        self._evaluation_budgets.append(local)
        try:
            yield local
        finally:
            self._evaluation_budgets.remove(local)

    @contextmanager
    def _evaluation_mode(self):
        import torch

        training_states = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            for module, was_training in training_states:
                module.training = was_training

    def _prompt_ids(self, prompt: str) -> list[int]:
        ids = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        if not ids:
            token = self.tokenizer.bos_token_id
            if token is None:
                token = self.tokenizer.eos_token_id
            if token is None:
                raise ValueError("empty prompts require a tokenizer BOS or EOS token")
            ids = [token]
        return ids

    def _encode(
        self, prompt: str, choices: dict[str, str]
    ) -> tuple[list[int], dict[str, list[int]]]:
        prompt_ids = self._prompt_ids(self._scoring_prompt(prompt))
        answers = {}
        for label, continuation in choices.items():
            answers[label] = list(self.tokenizer.encode(continuation, add_special_tokens=False))
            if not answers[label]:
                raise ValueError(f"option {label!r} encodes to no continuation tokens")
        return prompt_ids, answers

    def _check_inputs(
        self, tasks: list[dict[str, Any]], mode: str, generation_config: dict[str, Any]
    ) -> None:
        """Tokenize every row, so input errors fail before a held-out reservation."""
        max_length = generation_config.get("max_length")
        limited = generation_config.get("max_new_tokens") is None and max_length is not None
        for task in tasks:
            if mode == "sampled":
                prompt_ids = self._prompt_ids(self._scoring_prompt(task["prompt"]))
                if limited and len(prompt_ids) >= max_length:
                    raise ValueError(
                        f"prompt has {len(prompt_ids)} tokens, which max_length={max_length} "
                        "leaves no room to generate"
                    )
            else:
                _, answers = self._encode(task["prompt"], task["options"])
                # Options with the same tokens always tie, so the row could never be correct.
                if len({tuple(ids) for ids in answers.values()}) != len(answers):
                    raise ValueError("options must encode to distinct token sequences")

    def score(
        self,
        prompt: str,
        options: list[str] | dict[str, str],
        *,
        max_passes: int | None = None,
        batch_size: int | None = None,
        chat_template: Any = None,
    ) -> dict[str, float]:
        """Return summed continuation log probabilities using the module's token-boundary policy.

        Options are scored in left-padded batches of up to ``batch_size`` sequences
        (default 8, or 1 for a model that cannot take position ids); ``batch_size=1``
        scores each option alone, exactly as v0.1 did. ``chat_template=True`` (or
        ``{"system": ..., "kwargs": {...}}``) renders the prompt as a user turn with
        the tokenizer's chat template first; options follow the assistant header.
        """
        batch_size = self._default_batch_size() if batch_size is None else batch_size
        _check_batch_size(batch_size)
        self._check_batching(batch_size)
        chat = self._chat_settings(chat_template)
        with self.limit_passes(max_passes), self._evaluation_mode(), self._chat_prompts(chat):
            for _, scores in self._score_rows(
                [{"prompt": prompt, "options": _choices(options)}], batch_size
            ):
                return scores
        raise AssertionError("unreachable: one row always yields scores")

    def _takes_position_ids(self) -> bool:
        from .inspection import _accepts

        model = getattr(self, "model", None)
        return model is None or _accepts(model, "position_ids")

    def _default_batch_size(self) -> int:
        """Batch by default only where left padding can be corrected with position ids."""
        return DEFAULT_BATCH_SIZE if self._takes_position_ids() else 1

    def _hook_positions(self) -> list[int]:
        return [index for _, indices in getattr(self, "_position_hooks", []) for index in indices]

    def _check_batching(self, batch_size: int) -> None:
        """Refuse padded batches where left padding would change what a model or hook sees."""
        if batch_size == 1:
            return
        if not self._takes_position_ids():
            raise ValueError(
                "batch_size>1 needs a model whose forward accepts position_ids; use batch_size=1"
            )
        same = "use batch_size=1 for this run and the runs you compare it with"
        if any(item.active and item.mode == "logits" for item in getattr(self, "_attachments", [])):
            raise ValueError(f"a logit attachment would see padded batches; {same}")
        if any(index >= 0 for index in self._hook_positions()):
            raise ValueError(
                f"an active record or patch uses token positions that left padding shifts; {same}"
            )

    def _setup(self) -> dict[str, Any]:
        """Describe where and how the model runs, so cross-session reports can show what differs."""
        import torch
        import transformers

        model = getattr(self, "model", None)
        parameters = list(model.parameters()) if model is not None else []
        adapter = hasattr(model, "peft_config")
        peft = None
        if adapter:
            import peft as peft_module

            peft = peft_module.__version__
        config = getattr(model, "config", None)
        return {
            "devices": sorted({str(p.device) for p in parameters}),
            "dtypes": sorted({str(p.dtype) for p in parameters}),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft,
            "adapter": adapter,
            "attention": getattr(config, "_attn_implementation", None),
            "device_map": getattr(self, "device_map", None) is not None,
            "quantization": getattr(self, "quantization", None),
            "allow_hub_kernels": bool(getattr(self, "allow_hub_kernels", False)),
            "model_id": getattr(self, "model_id", ""),
            "model_commit": getattr(self, "model_commit", None),
        }

    def _scoring_config(
        self, mode: str, batch_size: int, decoder: Any = None, stops: list[int] | None = None
    ) -> dict[str, Any]:
        """Describe how a run scores, so only runs scored the same way pair."""
        if mode == "sampled":
            config = {
                "sampler": _sampler_name(decoder),
                "batch_size": batch_size,
                "padding_side": "left" if batch_size > 1 else "none",
            }
            # Recorded only when the tokenizer's EOS was added, so runs that stop where
            # they did before still pair with earlier runs.
            if stops is not None:
                config["stop_token_ids"] = stops
            return config
        from .inspection import _accepts

        model = getattr(self, "model", None)
        keeps = batch_size > 1 and model is not None and _accepts(model, "logits_to_keep")
        return {
            "scorer": "flat-left-pad-v1",
            "batch_size": batch_size,
            "padding_side": "left" if batch_size > 1 else "none",
            "logits": "last_k" if keeps else "full",
        }

    def _option_tokens(self, options: dict[str, str]) -> dict[str, int]:
        return {
            label: len(self.tokenizer.encode(text, add_special_tokens=False))
            for label, text in options.items()
        }

    def _score_rows(
        self, tasks: list[dict[str, Any]], batch_size: int
    ) -> Iterator[tuple[int, dict[str, float]]]:
        """Yield each row's option scores, in row order, as soon as the row is complete.

        Whole rows are packed in order while they fit within ``batch_size``
        sequences; a row with more options than that is split across forwards.
        Batches are never reordered, so the same rows and batch size always run
        the same forwards.
        """
        yield from self._score_encoded(self._encoded_rows(tasks), batch_size)

    def _encoded_rows(self, tasks: list[dict[str, Any]]) -> list[list[tuple]]:
        """Each row's sequences as ``(row, label, prompt tokens, answer tokens)``."""
        sequences = []
        for index, task in enumerate(tasks):
            prompt_ids, answers = self._encode(task["prompt"], task["options"])
            sequences.append(
                [(index, label, prompt_ids, answer_ids) for label, answer_ids in answers.items()]
            )
        return sequences

    def _score_encoded(
        self, sequences: list[list[tuple]], batch_size: int
    ) -> Iterator[tuple[int, dict[str, float]]]:
        """``_score_rows`` for rows already encoded, so a caller can score them repeatedly."""
        pending: dict[int, dict[str, float]] = {}
        remaining = {index: len(row) for index, row in enumerate(sequences)}
        next_row = 0
        for batch in _pack(sequences, batch_size):
            values = self._score_batch(
                [(prompt_ids, answer_ids) for _, _, prompt_ids, answer_ids in batch],
                padded=batch_size > 1,
            )
            for (index, label, _, _), value in zip(batch, values, strict=True):
                pending.setdefault(index, {})[label] = value
                remaining[index] -= 1
            while next_row < len(sequences) and remaining[next_row] == 0:
                # Return labels in the row's option order, however its batches were split.
                order = [label for _, label, _, _ in sequences[next_row]]
                row_scores = pending.pop(next_row)
                yield next_row, {label: row_scores[label] for label in order}
                next_row += 1

    def _score_batch(
        self, batch: list[tuple[list[int], list[int]]], *, padded: bool
    ) -> list[float]:
        """Run one forward over (prompt, answer) token pairs and sum each answer's log probs."""
        import torch

        from .inspection import _accepts

        device = self.device
        if not padded:
            ((prompt_ids, answer_ids),) = batch
            ids = torch.tensor([prompt_ids + answer_ids], device=device)
            output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids))
            start = len(prompt_ids) - 1
            return [_summed(output.logits[0, start : start + len(answer_ids)], answer_ids)]
        length = max(len(prompt_ids) + len(answer_ids) for prompt_ids, answer_ids in batch)
        shortest = min(len(prompt_ids) + len(answer_ids) for prompt_ids, answer_ids in batch)
        # A negative hook position past a short sequence would select left padding;
        # refuse it the way an unpadded forward does.
        if any(-index > shortest for index in self._hook_positions() if index < 0):
            raise ValueError("token position is outside the module output")
        keep = max(len(answer_ids) for _, answer_ids in batch) + 1
        pad = _pad_token(self.tokenizer)
        rows, masks = [], []
        for prompt_ids, answer_ids in batch:
            padding = length - len(prompt_ids) - len(answer_ids)
            rows.append([pad] * padding + prompt_ids + answer_ids)
            masks.append([0] * padding + [1] * (length - padding))
        ids = torch.tensor(rows, device=device)
        mask = torch.tensor(masks, device=device)
        inputs = {
            "input_ids": ids,
            "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
        }
        if _accepts(self.model, "logits_to_keep"):
            inputs["logits_to_keep"] = keep
        # Left padding ends every answer at the right edge, so the last `keep` logits cover all.
        logits = self.model(**inputs).logits[:, -keep:]
        return [
            _summed(logits[row, keep - len(answer_ids) - 1 : keep - 1], answer_ids)
            for row, (_, answer_ids) in enumerate(batch)
        ]

    @contextmanager
    def _prompts_prepared_once(self):
        """Within one evaluation, apply prefix attachments once per distinct prompt.

        The input check prepares every prompt before reserving data, and scoring
        reuses that text, so a costly or stateful transform is not applied twice.
        """
        self._prepared_prompts: dict[str, str] | None = {}
        try:
            yield
        finally:
            self._prepared_prompts = None

    def _scoring_prompt(self, prompt: str) -> str:
        """The text a prompt is scored as: prefix attachments first, then any chat template."""
        chat = getattr(self, "_chat", None)
        # A nested call with other template settings must not reuse this rendering.
        key = (prompt, None if chat is None else chat["format"]["rendered_sha256"])
        cache = getattr(self, "_prepared_prompts", None)
        if cache is not None and key in cache:
            return cache[key]
        prepare = getattr(self, "_prepare_prompt", None)
        text = prepare(prompt) if prepare is not None else prompt
        if chat is not None:
            text = self._render_chat(text, chat)
        if cache is not None:
            cache[key] = text
        return text

    def _chat_settings(self, chat_template: Any) -> dict[str, Any] | None:
        """Validate a chat_template argument: None or False, True, or {system, kwargs}."""
        if chat_template is None or chat_template is False:
            return None
        if chat_template is True:
            settings: dict[str, Any] = {}
        elif isinstance(chat_template, dict):
            settings = dict(chat_template)
        else:
            raise ValueError("chat_template must be True, False, None, or {system, kwargs}")
        if set(settings) - {"system", "kwargs"}:
            raise ValueError("chat_template settings take only 'system' and 'kwargs'")
        system, kwargs = settings.get("system"), settings.get("kwargs", {})
        if system is not None and not isinstance(system, str):
            raise ValueError("chat_template system must be text")
        if not isinstance(kwargs, dict):
            raise ValueError("chat_template kwargs must be a dictionary")
        try:
            json.dumps(kwargs, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("chat_template kwargs must be JSON values") from error
        owned = sorted(set(kwargs) & _OWNED_CHAT_ARGUMENTS)
        if owned:
            raise ValueError(f"chat_template kwargs cannot set {owned}; the lab controls them")
        if not getattr(self.tokenizer, "chat_template", None) and "chat_template" not in kwargs:
            raise ValueError("this tokenizer has no chat template; pass chat_template=None")
        # The template actually used: tokenizers may hold several named ones.
        template = self.tokenizer.get_chat_template(
            kwargs.get("chat_template"), kwargs.get("tools")
        )
        chat = {"system": system, "kwargs": kwargs}
        # Render a fixed probe now, so template errors (such as a template that rejects a
        # system message) surface before any reservation, and so templates that insert
        # changing text, such as today's date, record a different fingerprint.
        probe = self._render_chat(_CHAT_PROBE, chat)
        chat["format"] = {
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "system": system,
            "kwargs": kwargs,
            "rendered_sha256": hashlib.sha256(probe.encode()).hexdigest(),
        }
        return chat

    def _render_chat(self, text: str, chat: dict[str, Any]) -> str:
        """Render text as the user turn, ready for the assistant's reply."""
        system = chat["system"]
        messages = [] if system is None else [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": text})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat["kwargs"]
        )

    def _prompt_format(self, chat: dict[str, Any] | None) -> dict[str, Any] | None:
        """What a run's summary records about how prompts were rendered."""
        return None if chat is None else chat["format"]

    @contextmanager
    def _chat_prompts(self, chat: dict[str, Any] | None):
        """While active, render every scored or sampled prompt with the chat template."""
        previous = getattr(self, "_chat", None)
        self._chat = chat
        try:
            yield
        finally:
            self._chat = previous

    def _sample_rows(
        self,
        tasks: list[dict[str, Any]],
        generation_config: dict[str, Any],
        seed: int,
        batch_size: int,
        num_samples: int = 1,
    ) -> Iterator[tuple[int, list[dict[str, Any]]]]:
        """Yield each row's samples in row order, generating up to batch_size sequences at once.

        Sample ``j`` of row ``i`` is seeded with ``seed + i * num_samples + j``
        whatever batch it lands in.
        """
        prompts = [self._prompt_ids(self._scoring_prompt(task["prompt"])) for task in tasks]
        order = [(row, sample) for row in range(len(prompts)) for sample in range(num_samples)]
        pending: dict[int, list[dict[str, Any]]] = {}
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            seeds = [seed + row * num_samples + sample for row, sample in chunk]
            results = self._sample_batch(
                [prompts[row] for row, _ in chunk], generation_config, seeds
            )
            for (row, _), result in zip(chunk, results, strict=True):
                pending.setdefault(row, []).append(result)
                if len(pending[row]) == num_samples:
                    yield row, pending.pop(row)

    def _sample_batch(
        self, prompts: list[list[int]], generation_config: dict[str, Any], seeds: list[int]
    ) -> list[dict[str, Any]]:
        """Generate for left-padded prompts; record each row's text, finish reason, and length.

        Each step charges one pass per unfinished row. Sampling draws each row from
        its own seeded generator, so a row's text does not depend on its batch. Beam
        search keeps generate()'s own sampling and runs one row at a time.
        """
        import torch
        from transformers import LogitsProcessorList, StoppingCriteriaList

        from .sampling import (
            FinishTracker,
            PerSequenceSampler,
            effective_config,
            neutral_sampling_settings,
            stop_token_ids,
            warpers,
        )

        length = max(len(ids) for ids in prompts)
        # A negative hook position past a short prompt would select left padding.
        if any(-index > min(map(len, prompts)) for index in self._hook_positions() if index < 0):
            raise ValueError("token position is outside the module output")
        pad = _pad_token(self.tokenizer)
        device = self.device
        ids = torch.tensor([[pad] * (length - len(row)) + row for row in prompts], device=device)
        mask = torch.tensor(
            [[0] * (length - len(row)) + [1] * len(row) for row in prompts], device=device
        )
        settings = dict(generation_config)
        stops = stop_token_ids(self.model, self.tokenizer, settings)
        if stops is not None:
            settings["eos_token_id"] = stops
        # What generate() will run, including the model's own generation defaults.
        config, _ = effective_config(self.model, settings)
        sampler = _sampler_name(config)
        beams = sampler == "hf-beam-v0.1"
        tracker = FinishTracker(
            len(prompts),
            length,
            eos_token_id=config.eos_token_id,
            stop_strings=config.stop_strings,
            tokenizer=self.tokenizer,
        )
        # generate() needs the tokenizer to match stop strings.
        extra: dict[str, Any] = (
            {"tokenizer": self.tokenizer} if settings.get("stop_strings") else {}
        )
        if settings.get("pad_token_id") is None:
            extra["pad_token_id"] = pad
        if not beams:
            extra["stopping_criteria"] = StoppingCriteriaList([tracker])
        if sampler == "per-sequence-uniform-v1":
            extra["logits_processor"] = LogitsProcessorList(
                [PerSequenceSampler(seeds, warpers(config))]
            )
            # The sampler draws; generate() then takes its one-hot greedily.
            settings.update(neutral_sampling_settings(), do_sample=False)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        # manual_seed also changes MPS state when the model runs on CPU.
        mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
        try:
            with (
                torch.random.fork_rng(devices=devices, device_type="cuda"),
                self._evaluation_mode(),
                nullcontext() if beams else self._metered(lambda: tracker.unfinished),
            ):
                torch.manual_seed(seeds[0])
                generated = self.model.generate(
                    input_ids=ids, attention_mask=mask, **settings, **extra
                )
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        steps = sequences.shape[1] - length
        if beams:
            # generate() returns only the winning beam; replay its steps to find why it ended.
            for step in range(1, steps + 1):
                tracker(sequences[:, : length + step], None)
        return [
            {
                "generated": self.tokenizer.decode(
                    sequences[row, length : length + count], skip_special_tokens=True
                ),
                "finish_reason": reason,
                "generated_tokens": count,
                "sample_seed": seeds[row],
            }
            for row, (count, reason) in enumerate(tracker.results(steps))
        ]

    def evaluate(
        self,
        data: Any,
        *,
        name: str | None = None,
        held_out: bool = False,
        max_passes: int | None = None,
        mode: str = "likelihood",
        checker: Any = None,
        checker_id: str | None = None,
        generation_config: dict[str, Any] | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        num_samples: int = 1,
        pass_at: int | list[int] | None = None,
        answer_key: Any = None,
        chat_template: Any = None,
    ) -> dict[str, Any]:
        """Save per-item results and a summary; failures are saved and then re-raised.

        ``chat_template`` renders each prompt as a user turn, as in ``score``; the
        summary's ``prompt_format`` records it, and runs pair only when it matches.
        Held-out and dev identities still use the raw prompts.

        In sampled mode, ``num_samples`` draws that many seeded samples per row
        (with ``do_sample=True``): each row scores the fraction of samples correct,
        and accuracy is the mean fraction. ``pass_at`` adds unbiased pass@k.
        ``answer_key(generated, row)`` extracts the value each sample votes for; a
        tie for the most votes counts as incorrect.

        Modes are likelihood, yes_no, and sampled. Each runs up to ``batch_size``
        sequences per forward (default 8), dropping to 1 for models without position
        ids and for padding-sensitive generation settings. Runs pair only when their
        recorded scoring_config matches.
        The sampled checker receives (generated_text, expected_answer), plus a copy
        of the row when it names a ``row`` parameter; exact stripped equality is
        the default.
        A held-out dataset requires an explicit max_passes and can be used only
        once, even after edits or a failed evaluation. Dev datasets cannot later
        become held-out datasets, and held-out datasets cannot become dev data.
        Rows are tokenized and generation settings validated before any
        reservation, so those errors never consume held-out data.
        """
        if held_out and max_passes is None:
            raise ValueError("held-out evaluation requires explicit max_passes")
        PassBudget(max_passes)
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        if mode != "sampled" and (
            num_samples != 1 or pass_at is not None or answer_key is not None
        ):
            raise ValueError("num_samples, pass_at, and answer_key apply only to sampled mode")
        if answer_key is not None and not callable(answer_key):
            raise ValueError("answer_key must be callable")
        pass_ks = _pass_at_values(pass_at, num_samples)
        if checker is not None and not callable(checker):
            raise ValueError("checker must be callable")
        grade = _checker_call(checker)
        chat = self._chat_settings(chat_template)
        if checker_id is not None and (not isinstance(checker_id, str) or not checker_id):
            raise ValueError("checker_id must be a nonempty string")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if generation_config is not None and not isinstance(generation_config, dict):
            raise ValueError("generation_config must be a dictionary")
        json.dumps(generation_config or {}, allow_nan=False)
        tasks, identities = _tasks(data, mode)
        unbatchable: list[str] = []
        decoder = stops = None
        if mode == "sampled":
            from .sampling import effective_config, stop_token_ids

            generate = getattr(getattr(self, "model", None), "generate", None)
            generation_config = _generation_kwargs(generation_config, generate)
            # Decide batching from what generate() will run, model defaults included.
            decoder, _ = effective_config(getattr(self, "model", None), generation_config)
            assisted = _assisted(decoder)
            if assisted:
                raise ValueError(
                    f"assisted generation ({', '.join(assisted)}) verifies several tokens per "
                    "forward, so sampled evaluation cannot charge it per generated token"
                )
            unbatchable = _padding_sensitive(decoder)
            stops = stop_token_ids(getattr(self, "model", None), self.tokenizer, generation_config)
            if num_samples > 1 and not generation_config.get("do_sample"):
                raise ValueError(
                    "num_samples>1 needs do_sample=True; greedy samples would all be the same"
                )
        if batch_size is None:
            batch_size = 1 if unbatchable else self._default_batch_size()
        _check_batch_size(batch_size)
        if batch_size > 1 and unbatchable:
            raise ValueError(
                f"batched generation would change what {', '.join(unbatchable)} means; "
                "use batch_size=1"
            )
        self._check_batching(batch_size)
        name = _safe_name(name or f"run-{uuid.uuid4().hex[:12]}")
        with self._prompts_prepared_once(), self._chat_prompts(chat):
            self._check_inputs(tasks, mode, generation_config or {})
            dataset_hash = _digest(
                [
                    {key: task[key] for key in ("prompt", "answer", "options") if key in task}
                    for task in tasks
                ]
            )
            run_dir = Path(self.artifacts) / "evaluations" / name
            # Before the reservation, so a failure here leaves nothing behind.
            setup = self._setup()
            with _reserve(
                self.artifacts,
                self.session_path,
                identities,
                mode=mode,
                held_out=held_out,
                run=name,
            ) as (session_id, dataset_identity):
                run_dir.mkdir(exist_ok=False)
            started = time.time()
            before = self.forward_passes
            summary = {
                "name": name,
                "state": "running",
                "run_dir": str(run_dir.resolve()),
                "session_id": session_id,
                "model_id": getattr(self, "model_id", ""),
                "model_commit": getattr(self, "model_commit", None),
                "setup": setup,
                "model_revision": getattr(self, "revision", 0),
                "active_edit_ids": list(getattr(self, "active_edit_ids", [])),
                "edits": self.edits() if hasattr(self, "edits") else [],
                "attachments": self.attachments() if hasattr(self, "attachments") else [],
                "patches": self.patches() if hasattr(self, "patches") else [],
                "dataset_hash": dataset_hash,
                "dataset_identity": dataset_identity,
                "held_out": held_out,
                "mode": mode,
                "checker_custom": checker is not None,
                "num_samples": num_samples,
                "answer_key_custom": answer_key is not None,
                "prompt_format": self._prompt_format(chat),
                "checker_id": checker_id,
                "seed": seed,
                "generation_config": generation_config or {},
                "scoring_config": self._scoring_config(mode, batch_size, decoder, stops),
                "tokenizer_policy": "separate_prompt_continuation_no_special_tokens_sum",
                "max_passes": max_passes,
                "started_at": started,
                "total": len(tasks),
                "completed": 0,
                "correct": 0,
                "accuracy": None,
                "forward_passes": 0,
                # Metrics set only on success start as unknown, so a failed run says so.
                **(
                    {"truncation_share": None, "mean_generated_tokens": None}
                    if mode == "sampled"
                    else {}
                ),
                **({"pass_at": None} if pass_ks else {}),
                **({"vote_accuracy": None} if answer_key is not None else {}),
            }
            write_json(run_dir / "summary.json", summary)
            try:
                write_json(run_dir / "dataset.json", {"tasks": tasks})
                (run_dir / "items").mkdir()
                with (
                    self.limit_passes(max_passes),
                    (run_dir / "items.jsonl").open("x", encoding="utf-8") as stream,
                    nullcontext() if mode == "sampled" else self._evaluation_mode(),
                ):
                    rows = (
                        self._sample_rows(tasks, generation_config, seed, batch_size, num_samples)
                        if mode == "sampled"
                        else self._score_rows(tasks, batch_size)
                    )
                    truncated = generated_tokens = votes = 0
                    pass_totals = dict.fromkeys(pass_ks, 0.0)
                    for index, task in enumerate(tasks):
                        item = {"index": index, **task}
                        row, result = next(rows)
                        if row != index:
                            raise AssertionError("rows were scored out of order")
                        if mode == "sampled":
                            item.update(_graded(result, task, grade, answer_key, pass_ks))
                            truncated += sum(s["finish_reason"] == "max_tokens" for s in result)
                            generated_tokens += sum(s["generated_tokens"] for s in result)
                            votes += bool(item.get("vote_correct"))
                            for k in pass_ks:
                                pass_totals[k] += item["pass_at"][str(k)]
                        else:
                            scores = result
                            prediction = _prediction(scores)
                            item.update(
                                scores=scores,
                                prediction=prediction,
                                tie=prediction is None,
                                correct=prediction == task["answer"],
                                option_tokens=self._option_tokens(task["options"]),
                            )
                        stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        write_json(run_dir / "items" / f"{index:06d}.json", item)
                        summary["completed"] += 1
                        summary["correct"] += item["correct"]  # A fraction with several samples.
                summary.update(state="succeeded", accuracy=summary["correct"] / len(tasks))
                if mode == "sampled":
                    drawn = len(tasks) * num_samples
                    summary.update(
                        truncation_share=truncated / drawn,
                        mean_generated_tokens=generated_tokens / drawn,
                    )
                    if pass_ks:
                        summary["pass_at"] = {
                            str(k): total / len(tasks) for k, total in pass_totals.items()
                        }
                    if answer_key is not None:
                        summary["vote_accuracy"] = votes / len(tasks)
            except BaseException as error:
                summary.update(state="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                summary.update(
                    forward_passes=self.forward_passes - before,
                    duration_seconds=time.time() - started,
                )
                write_json(run_dir / "summary.json", summary)
            return summary
