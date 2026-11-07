"""The claim an agent hands back, and its validation.

A claim is plain JSON: data only, never paths or code. The scorer reads nothing else from
it. ``CLAIM_FORMAT`` is the description to give an agent; ``validate_claim`` lists what is
wrong with a claim, and an invalid claim scores 0. Importing this module does not import
torch or transformers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

CLAIM_VERSION = 1
MAX_CLAIM_BYTES = 1_000_000
MAX_COMPONENTS = 1000
MAX_DIRECTIONS = 100
KINDS = ("head", "mlp", "neurons")

CLAIM_FORMAT = """\
Write your claim as one JSON file. It holds data only. Use this shape:

{"version": 1, "task": "dev-1a2b3c4d5e",
 "components": [
   {"kind": "head", "layer": 2, "head": 1, "role": "...", "evidence": "..."},
   {"kind": "mlp", "layer": 1, "role": "...", "evidence": "..."},
   {"kind": "neurons", "layer": 1, "indices": [3, 77], "role": "...", "evidence": "..."}],
 "directions": [
   {"after_layer": 1, "vector": [64 numbers], "role": "...", "evidence": "..."}],
 "ablation": {"accuracy_before": 1.0, "accuracy_after": 0.12, "rows": 200,
              "evidence": "zeroed every claimed component together"},
 "direction_ablation": {"accuracy_before": 1.0, "accuracy_after": 0.5, "rows": 200},
 "summary": "one paragraph on how the model solves the task",
 "failed_hypotheses": ["what you tried that did not hold up"]}

Rules:
- "task" is the task id from task.json. Layers and heads count from 0.
- A head is named by layer and head. An MLP is named by layer. "neurons" names MLP
  neurons by index in one layer; the scorer counts it as that layer's MLP.
- A direction is a vector as long as the hidden size, in the residual stream after block
  "after_layer". It needs a nonzero, finite vector.
- "ablation" is required. It reports accuracy on the task rows before and after you
  zero-ablate every claimed component at once. A head is zeroed by zeroing its rows of
  the attention c_proj weight, an MLP by zeroing its c_proj weight and bias, and
  "neurons" by zeroing those rows of c_proj. The scorer reruns this and checks your numbers.
- "direction_ablation" is optional: the accuracy after projecting every claimed direction
  out of the residual stream after its layer, at every position.
- "role" and "evidence" say what each part does and how you know. Name only parts you
  tested; extra parts lower your precision.
"""


def load_claim(path: str | Path) -> Any:
    """Read a claim file as data. Refuses files over 1 MB, invalid JSON, JSON nested too deeply,
    and NaN or Infinity."""
    path = Path(path)
    if path.stat().st_size > MAX_CLAIM_BYTES:
        raise ValueError(f"claim file is larger than {MAX_CLAIM_BYTES} bytes")

    def refuse(constant: str) -> Any:
        raise ValueError(f"claim contains {constant}, which is not valid JSON")

    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream, parse_constant=refuse)
    except RecursionError as error:
        raise ValueError("claim is nested too deeply") from error


def task_shape(task: Any) -> dict[str, Any]:
    """The facts a claim is checked against: task id and the model's layers, heads and sizes.

    ``task`` is a task directory (read from ``task.json`` and ``model/config.json``) or an
    already-built dict with the keys ``id``, ``layers``, ``heads``, ``hidden`` and ``inner``.
    """
    if isinstance(task, dict):
        return task
    root = Path(task)
    info = json.loads((root / "task.json").read_text(encoding="utf-8"))
    config = json.loads((root / "model" / "config.json").read_text(encoding="utf-8"))
    hidden = config["n_embd"]
    return {
        "id": info["id"],
        "layers": config["n_layer"],
        "heads": config["n_head"],
        "hidden": hidden,
        "inner": config.get("n_inner") or 4 * hidden,
    }


def _index(value: Any, upper: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < upper


def _number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an integer too large for a float
        return False


def _check_component(i: int, part: Any, shape: dict[str, Any]) -> list[str]:
    where = f"components[{i}]"
    if not isinstance(part, dict):
        return [f"{where} must be an object"]
    kind = part.get("kind")
    if kind not in KINDS:
        return [f"{where} has unknown kind {kind!r}; use one of {list(KINDS)}"]
    problems = []
    if not _index(part.get("layer"), shape["layers"]):
        problems.append(f"{where} layer must be an integer from 0 to {shape['layers'] - 1}")
    if kind == "head" and not _index(part.get("head"), shape["heads"]):
        problems.append(f"{where} head must be an integer from 0 to {shape['heads'] - 1}")
    if kind == "neurons":
        indices = part.get("indices")
        if not isinstance(indices, list) or not indices:
            problems.append(f"{where} indices must be a nonempty list")
        elif not all(_index(n, shape["inner"]) for n in indices):
            problems.append(f"{where} indices must be integers from 0 to {shape['inner'] - 1}")
        elif len(set(indices)) != len(indices):
            problems.append(f"{where} indices must be distinct")
    for key in ("role", "evidence"):
        if key in part and not isinstance(part[key], str):
            problems.append(f"{where} {key} must be a string")
    return problems


def _check_direction(i: int, item: Any, shape: dict[str, Any]) -> list[str]:
    where = f"directions[{i}]"
    if not isinstance(item, dict):
        return [f"{where} must be an object"]
    problems = []
    if not _index(item.get("after_layer"), shape["layers"]):
        problems.append(f"{where} after_layer must be an integer from 0 to {shape['layers'] - 1}")
    vector = item.get("vector")
    if not isinstance(vector, list) or len(vector) != shape["hidden"]:
        problems.append(f"{where} vector must be a list of {shape['hidden']} numbers")
    elif not all(_number(x) for x in vector):
        problems.append(f"{where} vector must hold only finite numbers")
    elif not any(vector):
        problems.append(f"{where} vector must not be all zeros")
    return problems


def _check_effect(name: str, effect: Any, *, required: bool) -> list[str]:
    if effect is None:
        return [f"{name} is required"] if required else []
    if not isinstance(effect, dict):
        return [f"{name} must be an object"]
    problems = []
    for key in ("accuracy_before", "accuracy_after"):
        value = effect.get(key)
        if not _number(value) or not 0 <= value <= 1:
            problems.append(f"{name} {key} must be a number from 0 to 1")
    rows = effect.get("rows")
    if rows is not None and not (isinstance(rows, int) and not isinstance(rows, bool) and rows > 0):
        problems.append(f"{name} rows must be a positive integer")
    return problems


def validate_claim(claim: Any, task: Any) -> list[str]:
    """Return what is wrong with ``claim`` for ``task``; an empty list means it is valid.

    Checks the version and task id, component kinds, layer, head and neuron ranges,
    direction vectors (right length, finite, not all zero), and the required
    ``ablation``. ``task`` is a task directory or the dict ``task_shape`` returns.
    Never runs anything from the claim.
    """
    shape = task_shape(task)
    if not isinstance(claim, dict):
        return ["claim must be a JSON object"]
    problems = []
    if claim.get("version") != CLAIM_VERSION or isinstance(claim.get("version"), bool):
        problems.append(f"version must be {CLAIM_VERSION}")
    if claim.get("task") != shape["id"]:
        problems.append(f"task must be {shape['id']!r}, not {claim.get('task')!r}")
    components = claim.get("components")
    if not isinstance(components, list):
        problems.append("components must be a list")
    elif len(components) > MAX_COMPONENTS:
        problems.append(f"components may hold at most {MAX_COMPONENTS} entries")
    else:
        for i, part in enumerate(components):
            problems += _check_component(i, part, shape)
    directions = claim.get("directions", [])
    if not isinstance(directions, list):
        problems.append("directions must be a list")
    elif len(directions) > MAX_DIRECTIONS:
        problems.append(f"directions may hold at most {MAX_DIRECTIONS} entries")
    else:
        for i, item in enumerate(directions):
            problems += _check_direction(i, item, shape)
    problems += _check_effect("ablation", claim.get("ablation"), required=True)
    problems += _check_effect("direction_ablation", claim.get("direction_ablation"), required=False)
    if claim.get("direction_ablation") is not None and not claim.get("directions"):
        problems.append("direction_ablation needs at least one claimed direction")
    if "summary" in claim and not isinstance(claim["summary"], str):
        problems.append("summary must be a string")
    failed = claim.get("failed_hypotheses", [])
    if not isinstance(failed, list) or not all(isinstance(item, str) for item in failed):
        problems.append("failed_hypotheses must be a list of strings")
    return problems
