"""Auditable local evaluation, continuation likelihoods, and pass limits.

Continuation policy: encode prompt and answer separately, with special tokens
disabled, then concatenate their token IDs. No whitespace is inserted. Only
answer tokens contribute log probability; an empty prompt uses BOS, or EOS if
BOS is unavailable. This explicit token boundary avoids silently scoring a
prompt suffix when tokenization merges across the text boundary. Include any
desired leading whitespace in the answer itself. Scores are sums, not averages.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .budget import PassBudget
from .session import read_json, session_lock, write_json


def _choices(options: list[str] | dict[str, str]) -> dict[str, str]:
    if isinstance(options, list):
        choices = {
            chr(65 + index) if index < 26 else str(index): option
            for index, option in enumerate(options)
        }
    elif isinstance(options, dict):
        choices = options.copy()
    else:
        raise ValueError("options must be a list of completions or a label-to-completion mapping")
    if not choices or any(
        not isinstance(label, str) or not label or not isinstance(value, str) or not value
        for label, value in choices.items()
    ):
        raise ValueError("options need nonempty string labels and completions")
    return choices


def _answer_label(answer: Any, choices: dict[str, str]) -> str:
    if isinstance(answer, int) and not isinstance(answer, bool):
        if 0 <= answer < len(choices):
            return list(choices)[answer]
    elif isinstance(answer, str):
        if answer in choices:
            return answer
        matches = [label for label, completion in choices.items() if completion == answer]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(
        "answer must identify an option by label, zero-based index, or unique completion"
    )


def _tasks(data: Any, mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(data, (str, Path)):
        with Path(data).open(encoding="utf-8") as stream:
            data = [json.loads(line) for line in stream if line.strip()]
    if not isinstance(data, list) or not data:
        raise ValueError("evaluation data must be a nonempty list or JSONL path")
    tasks = []
    identities = []
    for row in data:
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
            raise ValueError("every task requires a string prompt")
        if "answer" not in row:
            raise ValueError("every task requires an answer")
        task = {"prompt": row["prompt"], "answer": row["answer"]}
        if "id" in row:
            task["id"] = row["id"]
        if mode == "yes_no":
            task["options"] = _choices(row.get("options", {"yes": " yes", "no": " no"}))
            answer = row["answer"]
            if isinstance(answer, bool):
                answer = "yes" if answer else "no"
            task["answer"] = _answer_label(answer, task["options"])
        elif mode == "likelihood":
            task["options"] = _choices(row.get("options"))
            task["answer"] = _answer_label(row["answer"], task["options"])
        elif "options" in row:
            task["options"] = _choices(row["options"])
        identity = {"prompt": task["prompt"], "answer": task["answer"]}
        if "options" in task:
            identity["options"] = sorted(task["options"].values())
            try:
                identity["answer"] = task["options"][_answer_label(task["answer"], task["options"])]
            except ValueError:
                if mode != "sampled":
                    raise
        identities.append(identity)
        tasks.append(task)
    # Fail before creating a run or consuming a held-out reservation.
    json.dumps(tasks, allow_nan=False)
    return tasks, identities


def _dataset_hash(tasks: list[dict[str, Any]]) -> str:
    canonical = json.dumps(tasks, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ScoringMixin:
    """Evaluation helpers for a Lab whose forward hook calls _before_forward."""

    @property
    def budget(self) -> PassBudget:
        if not hasattr(self, "_budget"):
            self._budget = PassBudget()
        return self._budget

    def _budget_state(self) -> None:
        if not hasattr(self, "_evaluation_budgets"):
            self._evaluation_budgets: list[PassBudget] = []
            self._budget_guard = threading.RLock()

    def _before_forward(self) -> None:
        self._budget_state()
        with self._budget_guard:
            budgets = [self.budget, *self._evaluation_budgets]
            for budget in budgets:
                budget.check()
            for budget in budgets:
                budget.consume()

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

    def score(
        self, prompt: str, options: list[str] | dict[str, str], *, max_passes: int | None = None
    ) -> dict[str, float]:
        """Return summed continuation log probabilities using the module's token-boundary policy."""
        import torch

        choices = _choices(options)
        prompt_ids = self._prompt_ids(self._scoring_prompt(prompt))
        scores = {}
        with self.limit_passes(max_passes), self._evaluation_mode():
            for label, continuation in choices.items():
                answer_ids = list(self.tokenizer.encode(continuation, add_special_tokens=False))
                if not answer_ids:
                    raise ValueError(f"option {label!r} encodes to no continuation tokens")
                ids = torch.tensor([prompt_ids + answer_ids], device=self.model.device)
                output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids))
                start = len(prompt_ids) - 1
                logits = output.logits[0, start : start + len(answer_ids)].float()
                target = torch.tensor(answer_ids, device=logits.device)
                token_scores = logits.log_softmax(-1).gather(-1, target[:, None]).squeeze(-1)
                scores[label] = float(token_scores.sum().item())
        return scores

    def _scoring_prompt(self, prompt: str) -> str:
        prepare = getattr(self, "_prepare_prompt", None)
        return prepare(prompt) if prepare is not None else prompt

    def _sample(self, prompt: str, generation_config: dict[str, Any], seed: int) -> str:
        import torch

        ids = torch.tensor(
            [self._prompt_ids(self._scoring_prompt(prompt))], device=self.model.device
        )
        config = {"max_new_tokens": 32, "do_sample": False, **generation_config}
        if config.get("num_return_sequences", 1) != 1:
            raise ValueError("sampled scoring requires num_return_sequences=1")
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        # manual_seed also changes MPS state when the model runs on CPU.
        mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
        try:
            with (
                torch.random.fork_rng(devices=devices, device_type="cuda"),
                self._evaluation_mode(),
            ):
                torch.manual_seed(seed)
                generated = self.model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids), **config
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
        generation_config: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Save per-item results and a summary; failures are saved and then re-raised.

        Modes are likelihood, yes_no, and sampled. The sampled checker receives
        (generated_text, expected_answer); exact stripped equality is the default.
        A held-out dataset requires an explicit max_passes and can be used only
        once, even after edits or a failed evaluation. Dev datasets cannot later
        become held-out datasets, and held-out datasets cannot become dev data.
        """
        if mode not in {"likelihood", "yes_no", "sampled"}:
            raise ValueError("mode must be likelihood, yes_no, or sampled")
        if held_out and max_passes is None:
            raise ValueError("held-out evaluation requires explicit max_passes")
        PassBudget(max_passes)
        if checker is not None and not callable(checker):
            raise ValueError("checker must be callable")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if generation_config is not None and not isinstance(generation_config, dict):
            raise ValueError("generation_config must be a dictionary")
        json.dumps(generation_config or {}, allow_nan=False)
        if mode == "sampled":
            generation_config = {
                "max_new_tokens": 32,
                "do_sample": False,
                **(generation_config or {}),
            }
        tasks, identities = _tasks(data, mode)
        dataset_hash = _dataset_hash(
            [{key: value for key, value in task.items() if key != "id"} for task in tasks]
        )
        # The held-out guard is invariant to row order, labels, and supported
        # answer encodings. Ordered task identity remains separate for diff.
        canonical_rows = sorted(
            json.dumps(row, sort_keys=True, ensure_ascii=False) for row in identities
        )
        dataset_identity = _dataset_hash(canonical_rows)
        name = name or f"run-{uuid.uuid4().hex[:12]}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", name):
            raise ValueError("run name must be a safe identifier of at most 100 characters")
        root = Path(self.artifacts) / "evaluations"
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / name
        with session_lock(Path(self.session_path)):
            registry_path = root / "_registry.json"
            registry = (
                read_json(registry_path)
                if registry_path.exists()
                else {"session_id": uuid.uuid4().hex, "datasets": {}}
            )
            previous = registry["datasets"].get(dataset_identity)
            role = "held_out" if held_out else "dev"
            if previous is not None and (previous["role"] == "held_out" or held_out):
                raise ValueError("dataset is already reserved as dev or consumed as held-out")
            run_dir.mkdir(exist_ok=False)
            record = registry["datasets"].setdefault(dataset_identity, {"role": role, "runs": []})
            record["runs"].append(name)
            write_json(registry_path, registry)
        started = time.time()
        before = self.forward_passes
        summary = {
            "name": name,
            "state": "running",
            "run_dir": str(run_dir.resolve()),
            "session_id": registry["session_id"],
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
                            task["prompt"], generation_config or {}, seed + index
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
                        prediction = max(scores, key=scores.get)
                        item.update(
                            scores=scores,
                            prediction=prediction,
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

    def _run_summary(self, run: Any) -> tuple[dict[str, Any], Path]:
        if isinstance(run, dict):
            path = Path(run["run_dir"])
        else:
            path = Path(run)
            if not path.exists():
                path = Path(self.artifacts) / "evaluations" / path
            if path.name == "summary.json":
                path = path.parent
        return read_json(path / "summary.json"), path

    def diff(self, before: Any, after: Any) -> list[dict[str, Any]]:
        """Return changed items for successful runs of the same dataset and session."""
        left, left_dir = self._run_summary(before)
        right, right_dir = self._run_summary(after)
        if left["state"] != "succeeded" or right["state"] != "succeeded":
            raise ValueError("diff requires two successful evaluation runs")
        if any(left[key] != right[key] for key in ("session_id", "dataset_hash", "mode")):
            raise ValueError("diff requires the same session, dataset, and scoring mode")
        left_items = [
            json.loads(line) for line in (left_dir / "items.jsonl").read_text().splitlines()
        ]
        right_items = [
            json.loads(line) for line in (right_dir / "items.jsonl").read_text().splitlines()
        ]
        changes = []
        for old, new in zip(left_items, right_items, strict=True):
            if (old["prediction"], old["correct"]) == (new["prediction"], new["correct"]):
                continue
            kind = "changed"
            if old["correct"] != new["correct"]:
                kind = "improved" if new["correct"] else "regressed"
            changes.append({"index": old["index"], "change": kind, "before": old, "after": new})
        return changes


class Scorer:
    """A scoring view of an existing Lab; all state and budgets stay on that Lab."""

    def __init__(self, lab: ScoringMixin):
        self.lab = lab

    def score(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        return self.lab.score(*args, **kwargs)

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.lab.evaluate(*args, **kwargs)

    def diff(self, before: Any, after: Any) -> list[dict[str, Any]]:
        return self.lab.diff(before, after)
