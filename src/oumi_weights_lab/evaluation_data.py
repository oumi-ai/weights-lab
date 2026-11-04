"""Evaluation rows, their semantic identities, and the dev/held-out registry.

The scoring and evaluation modules both import these helpers; this module
imports neither of them.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .session import read_json, session_lock, write_json

MODES = ("likelihood", "yes_no", "sampled")
RESULT_KEYS = frozenset({"index", "prediction", "scores", "correct", "tie", "generated", "error"})


def _digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_name(name: Any) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", name):
        raise ValueError("name must be a safe identifier of at most 100 characters")
    return name


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
    # Identical completions always tie, so no answer among them could be correct.
    if len(set(choices.values())) != len(choices):
        raise ValueError("options need distinct completions")
    return choices


def _answer_label(answer: Any, choices: dict[str, str]) -> str:
    if isinstance(answer, int) and not isinstance(answer, bool):
        if 0 <= answer < len(choices):
            return list(choices)[answer]
    elif isinstance(answer, str):
        matches = [label for label, completion in choices.items() if completion == answer]
        if answer in choices:
            if matches not in ([], [answer]):
                raise ValueError(
                    f"answer {answer!r} is ambiguous: it is one option's label "
                    "and another option's completion"
                )
            return answer
        if matches:
            return matches[0]
    raise ValueError(
        "answer must identify an option by label, zero-based index, or unique completion"
    )


def _text(value: Any) -> str:
    """Text as identity compares it: NFC-normalized, without surrounding whitespace."""
    return unicodedata.normalize("NFC", str(value)).strip()


def _tasks(data: Any, mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate rows and return them with their label- and order-free identities."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
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
        if set(row) & RESULT_KEYS:
            raise ValueError("task metadata cannot use reserved evaluation result keys")
        task = row.copy()
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
        identity = {"prompt": _text(task["prompt"])}
        if "options" in task:
            identity["options"] = sorted(_text(option) for option in task["options"].values())
        # The default sampled checker compares str(answer).strip(); options are scored text.
        gold = task["answer"] if mode == "sampled" else task["options"][task["answer"]]
        identity["answer"] = _text(gold)
        identities.append(identity)
        tasks.append(task)
    # Fail before creating a run or consuming a held-out reservation.
    json.dumps(tasks, allow_nan=False)
    return tasks, identities


def _dataset_identity(identities: list[dict[str, Any]]) -> str:
    """Hash rows independent of order, labels, IDs, and metadata."""
    return _digest(
        sorted(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in identities)
    )


def _reservation_keys(
    identities: list[dict[str, Any]], mode: str
) -> tuple[set[str], set[str], set[str]]:
    """Return row hashes, every nonempty prompt's hash, and open prompts' hashes.

    The gold answer changes no score, so a row with options also reserves its
    prompt and options without the answer. In sampled mode the model answers the
    prompt alone, so those prompts are open: an open prompt conflicts with the
    same prompt in any row. Scored rows that merely share a generic prompt, such
    as "Answer:", but offer different options can sit on opposite sides.
    """
    rows = {_digest(row) for row in identities}
    rows |= {
        _digest({key: value for key, value in row.items() if key != "answer"})
        for row in identities
        if "options" in row
    }
    prompts = {_digest(row["prompt"]) for row in identities if row["prompt"]}
    return rows, prompts, prompts if mode == "sampled" else set()


def _load_registry(root: Path) -> dict[str, Any]:
    path = root / "_registry.json"
    return read_json(path) if path.exists() else {"session_id": uuid.uuid4().hex, "datasets": {}}


def _guard_dataset_roles(
    registry: dict[str, Any], keys: tuple[set[str], set[str], set[str]], *, held_out: bool
) -> None:
    """Refuse dev/held-out overlap by row or open prompt; unverifiable records fail closed."""
    rows, prompts, open_prompts = keys
    for record in registry["datasets"].values():
        prior = [
            record.get(field)
            for field in ("item_identities", "prompt_identities", "open_prompt_identities")
        ]
        if not all(isinstance(value, list) for value in prior):
            raise ValueError("registry record lacks row or prompt identities; cannot verify roles")
        prior_rows, prior_prompts, prior_open = (set(value) for value in prior)
        overlap = rows & prior_rows or open_prompts & prior_prompts or prompts & prior_open
        if (record["role"] == "held_out" or held_out) and overlap:
            raise ValueError("dataset overlaps reserved dev or held-out data")


@contextmanager
def _reserve(
    artifacts: Path,
    session_path: Path,
    identities: list[dict[str, Any]],
    *,
    mode: str,
    held_out: bool,
    run: str | None = None,
) -> Iterator[tuple[str, str]]:
    """Check roles and yield (session_id, dataset_identity) under the session lock.

    The caller creates its outputs inside the block. The reservation is saved
    when the block succeeds, before any model call can run.
    """
    root = Path(artifacts) / "evaluations"
    root.mkdir(parents=True, exist_ok=True)
    identity = _dataset_identity(identities)
    rows, prompts, open_prompts = keys = _reservation_keys(identities, mode)
    with session_lock(Path(session_path)):
        registry = _load_registry(root)
        _guard_dataset_roles(registry, keys, held_out=held_out)
        yield registry["session_id"], identity
        role = "held_out" if held_out else "dev"
        record = registry["datasets"].setdefault(identity, {"role": role, "runs": []})
        record.update(
            item_identities=sorted(rows),
            prompt_identities=sorted(prompts),
            open_prompt_identities=sorted(open_prompts),
        )
        if run is not None:
            record["runs"].append(run)
        write_json(root / "_registry.json", registry, durable=True)
