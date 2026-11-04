"""Auditable local evaluation, continuation likelihoods, and pass limits.

Continuation policy: encode prompt and answer separately, with special tokens
disabled, then concatenate their token IDs. No whitespace is inserted. Only
answer tokens contribute log probability; an empty prompt uses BOS, or EOS if
BOS is unavailable. This explicit token boundary avoids silently scoring a
prompt suffix when tokenization merges across the text boundary. Include any
desired leading whitespace in the answer itself. Scores are sums, not averages.
"""

from __future__ import annotations

import functools
import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .budget import PassBudget, sequences_in_call
from .evaluation import EvaluationMixin
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
    counted model to ``_count_beams_once``, and owns ``_forward_passes``.
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

        Inside beam search, a prompt's beams count as one sequence.
        """
        beams = getattr(self, "_beam_width", 1)
        count = -(-sequences_in_call(args, kwargs) // beams)
        self._before_forward(count)
        self._forward_passes += count

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
        self, prompt: str, options: list[str] | dict[str, str], *, max_passes: int | None = None
    ) -> dict[str, float]:
        """Return summed continuation log probabilities using the module's token-boundary policy."""
        import torch

        prompt_ids, answers = self._encode(prompt, _choices(options))
        scores = {}
        with self.limit_passes(max_passes), self._evaluation_mode():
            for label, answer_ids in answers.items():
                ids = torch.tensor([prompt_ids + answer_ids], device=self.model.device)
                output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids))
                start = len(prompt_ids) - 1
                logits = output.logits[0, start : start + len(answer_ids)].float()
                target = torch.tensor(answer_ids, device=logits.device)
                token_scores = logits.log_softmax(-1).gather(-1, target[:, None]).squeeze(-1)
                scores[label] = float(token_scores.sum().item())
        return scores

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
        cache = getattr(self, "_prepared_prompts", None)
        if cache is not None and prompt in cache:
            return cache[prompt]
        prepare = getattr(self, "_prepare_prompt", None)
        text = prepare(prompt) if prepare is not None else prompt
        if cache is not None:
            cache[prompt] = text
        return text

    def _sample(self, prompt: str, generation_config: dict[str, Any], seed: int) -> str:
        import torch

        ids = torch.tensor(
            [self._prompt_ids(self._scoring_prompt(prompt))], device=self.model.device
        )
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        # manual_seed also changes MPS state when the model runs on CPU.
        mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
        try:
            with (
                torch.random.fork_rng(devices=devices, device_type="cuda"),
                self._evaluation_mode(),
            ):
                torch.manual_seed(seed)
                # generate() needs the tokenizer to match stop strings.
                extra = (
                    {"tokenizer": self.tokenizer} if generation_config.get("stop_strings") else {}
                )
                generated = self.model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    **generation_config,
                    **extra,
                )
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        return self.tokenizer.decode(sequences[0, ids.shape[1] :], skip_special_tokens=True)

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
    ) -> dict[str, Any]:
        """Save per-item results and a summary; failures are saved and then re-raised.

        Modes are likelihood, yes_no, and sampled. The sampled checker receives
        (generated_text, expected_answer); exact stripped equality is the default.
        A held-out dataset requires an explicit max_passes and can be used only
        once, even after edits or a failed evaluation. Dev datasets cannot later
        become held-out datasets, and held-out datasets cannot become dev data.
        Rows are tokenized and generation settings validated before any
        reservation, so those errors never consume held-out data.
        """
        if held_out and max_passes is None:
            raise ValueError("held-out evaluation requires explicit max_passes")
        PassBudget(max_passes)
        if checker is not None and not callable(checker):
            raise ValueError("checker must be callable")
        if checker_id is not None and (not isinstance(checker_id, str) or not checker_id):
            raise ValueError("checker_id must be a nonempty string")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if generation_config is not None and not isinstance(generation_config, dict):
            raise ValueError("generation_config must be a dictionary")
        json.dumps(generation_config or {}, allow_nan=False)
        tasks, identities = _tasks(data, mode)
        if mode == "sampled":
            generate = getattr(getattr(self, "model", None), "generate", None)
            generation_config = _generation_kwargs(generation_config, generate)
        name = _safe_name(name or f"run-{uuid.uuid4().hex[:12]}")
        with self._prompts_prepared_once():
            self._check_inputs(tasks, mode, generation_config or {})
            dataset_hash = _digest(
                [
                    {key: task[key] for key in ("prompt", "answer", "options") if key in task}
                    for task in tasks
                ]
            )
            run_dir = Path(self.artifacts) / "evaluations" / name
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
                "checker_id": checker_id,
                "seed": seed,
                "generation_config": generation_config or {},
                "tokenizer_policy": "separate_prompt_continuation_no_special_tokens_sum",
                "max_passes": max_passes,
                "started_at": started,
                "total": len(tasks),
                "completed": 0,
                "correct": 0,
                "accuracy": None,
                "forward_passes": 0,
            }
            write_json(run_dir / "summary.json", summary)
            try:
                write_json(run_dir / "dataset.json", {"tasks": tasks})
                (run_dir / "items").mkdir()
                with (
                    self.limit_passes(max_passes),
                    (run_dir / "items.jsonl").open("x", encoding="utf-8") as stream,
                ):
                    for index, task in enumerate(tasks):
                        item = {"index": index, **task}
                        if mode == "sampled":
                            generated = self._sample(
                                task["prompt"], generation_config, seed + index
                            )
                            correct = (
                                checker(generated, task["answer"])
                                if checker is not None
                                else generated.strip() == str(task["answer"]).strip()
                            )
                            item.update(
                                generated=generated, prediction=generated, correct=bool(correct)
                            )
                        else:
                            scores = self.score(task["prompt"], task["options"])
                            best = max(scores.values())
                            leaders = [label for label, value in scores.items() if value == best]
                            # An exact tie names no prediction, so it is never correct.
                            prediction = leaders[0] if len(leaders) == 1 else None
                            item.update(
                                scores=scores,
                                prediction=prediction,
                                tie=prediction is None,
                                correct=prediction == task["answer"],
                            )
                        stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        write_json(run_dir / "items" / f"{index:06d}.json", item)
                        summary["completed"] += 1
                        summary["correct"] += int(item["correct"])
                summary.update(state="succeeded", accuracy=summary["correct"] / len(tasks))
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
