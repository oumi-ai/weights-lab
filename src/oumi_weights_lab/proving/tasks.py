"""Task families, seeds, ``build_task``, the key format and the task directory layout.

A task is a model, a question and rows, in a public directory:

    <out>/<task_id>/task.json   id, split, kind, question, rows file, options note, metric,
                                chance, claim version
    <out>/<task_id>/model/      save_pretrained of the model and tokenizer
    <out>/<task_id>/rows.jsonl  200 rows in ``lab.evaluate`` format

Task ids are opaque, ``{split}-{10 hex digits}``, so neither the id nor task.json names
the family; the family is in the key only, with the split and index. The answer key goes
to a different directory, ``<keys>/<task_id>.json``, and holds the planted components,
decoys, the seed for the scorer's own rows, and the numbers the build measured.
Nothing in the task directory reveals the seed or the planted location.

Splits. ``dev`` seeds are public (a hash of family, split and index) and recipes may be
tuned on them. ``heldout`` seeds are an HMAC of a secret that the caller keeps in a file
outside the repo; the secret is never written anywhere. ``calibration`` tasks (GPT-2
small) have no seed and are never held out. Importing this module does not import torch
or transformers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import shutil
from pathlib import Path
from typing import Any

from . import calibration, gpt2_planted, planted
from .calibration import CALIBRATION_FAMILIES
from .gpt2_planted import GPT2_FAMILIES
from .planted import PLANTED_FAMILIES, derive

SPLITS = ("dev", "heldout", "calibration")
SEEDED_FAMILIES = (*PLANTED_FAMILIES, *GPT2_FAMILIES)
ALL_FAMILIES = (*SEEDED_FAMILIES, *CALIBRATION_FAMILIES)
MIN_SECRET_BYTES = 32
ATTEMPTS = 20
TASK_ROWS = 200
VALIDATION_ROWS = 400
GPT2_VALIDATION_ROWS = 300
SCORER_ROWS = 400  # rows the scorer draws from a key's eval_seed
CLAIM_VERSION = 1
OPTIONS_NOTE = "Every row lists its options and its answer; the model's choice is scored on them."
CALIBRATION_QUESTIONS = {
    "gpt2-ioi": (
        "Each prompt tells a short story with two names. The model completes it with a name. "
        "Which parts of this model make it choose that name?"
    ),
    "gpt2-induction": (
        "Each prompt is a run of words followed by the start of the same run again. The "
        "model continues the repeat. Which parts of this model make it do that?"
    ),
}


def task_id(family: str, split: str, index: int, secret: bytes | None = None) -> str:
    """The opaque id of a task: ``{split}-{10 hex digits}``, such as ``dev-1a2b3c4d5e``.

    The digits come from SHA-256 of ``f"id:{family}:{split}:{index}"`` for ``dev`` and
    ``calibration``, and from HMAC-SHA256 of the secret over the same string for
    ``heldout``, so a held-out id cannot be matched to a family without the secret (the
    secret is required and ValueError is raised without it). Deterministic.
    """
    _check_task(family, split, index)
    text = f"id:{family}:{split}:{index}".encode()
    if split == "heldout":
        if secret is None or len(secret) < MIN_SECRET_BYTES:
            raise ValueError("a held-out task id needs the secret")
        digest = hmac.new(secret, text, hashlib.sha256).hexdigest()
    else:
        digest = hashlib.sha256(text).hexdigest()
    return f"{split}-{digest[:10]}"


def read_secret(path: str | Path) -> bytes:
    """Read the held-out secret from a file. Refuses fewer than 32 bytes after stripping."""
    secret = Path(path).read_bytes().strip()
    if len(secret) < MIN_SECRET_BYTES:
        raise ValueError(f"the secret file must hold at least {MIN_SECRET_BYTES} bytes")
    return secret


def task_seed(family: str, split: str, index: int, secret: bytes | None = None) -> int:
    """The planted-location seed of a task.

    ``dev``: the first 8 bytes of ``sha256(f"{family}:dev:{index}")``. ``heldout``: the
    first 8 bytes of ``HMAC-SHA256(secret, f"{family}:heldout:{index}")``, which needs the
    secret. Calibration tasks have no seed; asking for one raises ValueError.
    """
    _check_task(family, split, index)
    if split == "calibration":
        raise ValueError("calibration tasks have no seed")
    if split == "dev":
        digest = hashlib.sha256(f"{family}:dev:{index}".encode()).digest()
    else:
        if secret is None or len(secret) < MIN_SECRET_BYTES:
            raise ValueError("a held-out seed needs the secret")
        digest = hmac.new(secret, f"{family}:heldout:{index}".encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big")


def _check_task(family: str, split: str, index: int) -> None:
    if family not in ALL_FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {list(ALL_FAMILIES)}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {list(SPLITS)}")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative integer")
    if family in CALIBRATION_FAMILIES and split != "calibration":
        raise ValueError(f"{family} is a calibration family; use the calibration split")
    if family in SEEDED_FAMILIES and split == "calibration":
        raise ValueError(f"{family} is a planted family; use the dev or heldout split")


def make_rows(
    family: str,
    n: int,
    seed: int,
    *,
    row_spec: dict[str, Any] | None = None,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Rows for any family. Some planted families and both GPT-2 planted families need
    ``row_spec`` (from the key); calibration families need the task's ``tokenizer``."""
    if family in PLANTED_FAMILIES:
        return planted.make_rows(family, n, seed, row_spec)
    if family in GPT2_FAMILIES:
        if row_spec is None:
            raise ValueError(f"{family} rows need the key's row_spec")
        return gpt2_planted.make_rows(n, seed, row_spec)
    if tokenizer is None:
        raise ValueError(f"{family} rows need the task's tokenizer")
    return calibration.make_rows(family, tokenizer, n, seed)


def load_key(path: str | Path) -> dict[str, Any]:
    """Read a key file written by ``build_task``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_task(
    family: str,
    split: str,
    index: int,
    out_dir: str | Path,
    keys_dir: str | Path,
    *,
    secret_file: str | Path | None = None,
    model_id: str = calibration.GPT2_SMALL,
) -> dict[str, Any]:
    """Build one task directory under ``out_dir`` and its key under ``keys_dir``.

    Returns ``{"task", "task_dir", "key_path"}``. Refuses an unknown family or split, a
    family in the wrong split, a ``keys_dir`` inside the task directory, a task directory
    that already exists, a ``heldout`` build without ``secret_file`` (and a ``dev`` or
    ``calibration`` build with one), a secret under 32 bytes, and a secret file inside the
    task directory. A planted model that fails validation is redrawn from
    ``(seed, attempt + 1)``, up to 20 attempts, and then the build raises RuntimeError.
    Calibration tasks and the GPT-2 planted families download ``model_id`` (GPT-2 small)
    and save it, edited or not, into the task.
    """
    _check_task(family, split, index)
    secret = None
    if split == "heldout":
        if secret_file is None:
            raise ValueError("a held-out task needs a secret file")
        secret = read_secret(secret_file)
    elif secret_file is not None:
        raise ValueError(f"{split} seeds are public; do not pass a secret file")
    ident = task_id(family, split, index, secret)
    task_dir = Path(out_dir).resolve() / ident
    keys = Path(keys_dir).resolve()
    if _inside(keys, task_dir):
        raise ValueError("the keys directory must not be inside the task directory")
    if secret_file is not None and _inside(Path(secret_file).resolve(), task_dir):
        raise ValueError("the secret file must not be inside the task directory")
    if task_dir.exists():
        raise ValueError(f"{task_dir} already exists")
    if split == "calibration":
        built = _calibration(family, index, model_id)
    elif family in GPT2_FAMILIES:
        built = _gpt2_planted(family, task_seed(family, split, index, secret), model_id)
    else:
        built = _planted(family, task_seed(family, split, index, secret))
    model, tokenizer, rows, key, question, chance = built
    key.update(task=ident, family=family, split=split, index=index, chance=chance)
    task_dir.mkdir(parents=True)
    try:
        model.save_pretrained(task_dir / "model")
        tokenizer.save_pretrained(task_dir / "model")
        _write_rows(task_dir / "rows.jsonl", rows)
        _write_json(
            task_dir / "task.json",
            {
                "id": ident,
                "split": split,
                "kind": "claim",
                "question": question,
                "rows": "rows.jsonl",
                "options_note": OPTIONS_NOTE,
                "metric": "accuracy",
                "chance": chance,
                "claim_version": CLAIM_VERSION,
            },
        )
        keys.mkdir(parents=True, exist_ok=True)
        _write_json(keys / f"{ident}.json", key)
    except BaseException:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise
    return {"task": ident, "task_dir": str(task_dir), "key_path": str(keys / f"{ident}.json")}


def _planted(family: str, seed: int):
    spec = planted.FAMILIES[family]
    failures: list[str] = []
    for attempt in range(ATTEMPTS):
        result = planted.plant(family, seed, attempt)
        check_rows = planted.make_rows(
            family, VALIDATION_ROWS, derive(seed, "validation", attempt), result.row_spec
        )
        numbers = planted.validate_plant(result, check_rows)
        if numbers["passed"]:
            break
        failures = numbers["failures"]
    else:
        raise RuntimeError(
            f"no valid {family} model in {ATTEMPTS} attempts; the last failed with: {failures}"
        )
    numbers.update(attempt=attempt)
    del numbers["passed"], numbers["failures"]
    rows = planted.make_rows(family, TASK_ROWS, derive(seed, "rows"), result.row_spec)
    key: dict[str, Any] = {
        "components": result.components,
        "decoys": result.decoys,
        "eval_seed": derive(seed, "eval"),
        "row_spec": result.row_spec,
        "validation": numbers,
    }
    if result.neurons:
        key["neurons"] = result.neurons
    if result.directions:
        key["directions"] = result.directions
        key["decoy_directions"] = result.decoy_directions
    return result.model, result.tokenizer, rows, key, spec.question, spec.chance


def _gpt2_planted(family: str, seed: int, model_id: str):
    failures: list[str] = []
    for attempt in range(ATTEMPTS):
        try:
            result = gpt2_planted.plant(family, seed, attempt, model_id)
        except (gpt2_planted.NoWordDraw, gpt2_planted.NoHost) as error:
            failures = [str(error)]
            continue
        check_rows = gpt2_planted.make_rows(
            GPT2_VALIDATION_ROWS, derive(seed, "validation", attempt), result.row_spec
        )
        scorer_rows = gpt2_planted.make_rows(SCORER_ROWS, derive(seed, "eval"), result.row_spec)
        numbers = gpt2_planted.validate(result, check_rows, scorer_rows)
        if numbers["passed"]:
            break
        failures = numbers["failures"]
    else:
        raise RuntimeError(
            f"no valid {family} model in {ATTEMPTS} attempts; the last failed with: {failures}"
        )
    numbers.update(attempt=attempt)
    del numbers["passed"], numbers["failures"]
    rows = gpt2_planted.make_rows(TASK_ROWS, derive(seed, "rows"), result.row_spec)
    key: dict[str, Any] = {
        "components": result.components,
        "decoys": result.decoys,
        "eval_seed": derive(seed, "eval"),
        "row_spec": result.row_spec,
        "validation": numbers,
    }
    if "marker" in result.facts:
        key["marker"] = result.facts["marker"]
    return result.model, result.tokenizer, rows, key, gpt2_planted.QUESTION, gpt2_planted.CHANCE


def _calibration(family: str, index: int, model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .planted import option_accuracy, zeroed

    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager", trust_remote_code=False
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    config = model.config
    if family == "gpt2-ioi":
        components = calibration.ioi_components()
        if (config.n_layer, config.n_head) != (12, 12):
            raise ValueError("gpt2-ioi needs GPT-2 small (12 layers, 12 heads)")
        chance, extra = 0.5, {"source": "Wang et al., 2022, Interpretability in the Wild"}
    else:
        scores = calibration.induction_scores(
            model, sequences=calibration.INDUCTION_SEQUENCES, length=calibration.INDUCTION_LENGTH
        )
        components = calibration.induction_components(scores, calibration.INDUCTION_THRESHOLD)
        if not components:
            raise ValueError("no head reaches the induction threshold in this model")
        chance = 0.25
        extra = {
            "procedure": (
                f"mean attention from each second-repeat token to the token after its first "
                f"occurrence, over {calibration.INDUCTION_SEQUENCES} random sequences of "
                f"{calibration.INDUCTION_LENGTH} tokens repeated twice; heads at or above "
                f"{calibration.INDUCTION_THRESHOLD}"
            ),
            "threshold": calibration.INDUCTION_THRESHOLD,
            "scores": {
                f"{c['layer']}.{c['head']}": round(scores[c["layer"]][c["head"]], 4)
                for c in components
            },
        }
    eval_seed = derive(family, "calibration", index, "eval")
    rows = make_rows(
        family, TASK_ROWS, derive(family, "calibration", index, "rows"), tokenizer=tokenizer
    )
    check = make_rows(family, VALIDATION_ROWS, eval_seed, tokenizer=tokenizer)
    baseline = option_accuracy(model, tokenizer, check)
    with zeroed(model, components):
        ablated = option_accuracy(model, tokenizer, check)
    key = {
        "components": components,
        "decoys": [],
        "eval_seed": eval_seed,
        "row_spec": {},
        "validation": {
            "rows": VALIDATION_ROWS,
            "baseline_accuracy": baseline["accuracy"],
            "key_ablated_accuracy": ablated["accuracy"],
            **extra,
        },
    }
    return model, tokenizer, rows, key, CALIBRATION_QUESTIONS[family], chance
