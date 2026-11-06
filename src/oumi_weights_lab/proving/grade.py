"""The scorer: grade a claim against a task's hidden key, or an edit bundle against a check.

``grade_claim`` compares the claimed components and directions with the key, then reruns
the claimed ablation on rows the scorer draws itself, in a fresh ``Lab`` on the task's
model. It reads only data from the claim: no paths, no code. Importing this module does
not import torch or transformers.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from .claims import task_shape, validate_claim
from .planted import component_edits, projector
from .tasks import load_key, make_rows

ROWS = 400
AFTER_TOLERANCE = 0.1
BEFORE_TOLERANCE = 0.05
FAITHFUL_MARGIN = 0.15
ALIGNED = 0.9


def _parts(components: list[dict[str, Any]]) -> set[tuple]:
    """Heads and MLP layers a list of components names; ``neurons`` count as their MLP."""
    parts = set()
    for part in components:
        if part["kind"] == "head":
            parts.add(("head", part["layer"], part["head"]))
        else:
            parts.add(("mlp", part["layer"]))
    return parts


def _neuron_set(entries: list[dict[str, Any]]) -> set[tuple[int, int]]:
    return {(entry["layer"], index) for entry in entries for index in entry["indices"]}


def _ratio(hit: int, total: int) -> float:
    return hit / total if total else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _best_alignment(target: dict[str, Any], claimed: list[dict[str, Any]]) -> float:
    """Best ``|cos|`` between ``target`` and a claimed direction at or after its layer."""
    goal = target["vector"]
    scale = math.sqrt(sum(x * x for x in goal))
    best = 0.0
    for item in claimed:
        if item["after_layer"] < target["after_layer"]:
            continue
        vector = item["vector"]
        norm = math.sqrt(sum(x * x for x in vector))
        if norm and scale:
            best = max(
                best, abs(sum(a * b for a, b in zip(goal, vector, strict=True))) / norm / scale
            )
    return best


def _open_lab(task_dir: Path, work_dir: Path):
    from ..lab import load_lab

    work_dir.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="grade-", dir=work_dir))
    config = {"model": str(task_dir / "model"), "device": "cpu", "dtype": "float32", "seed": 0}
    return load_lab(config, session)


@contextmanager
def _ablated(lab: Any, components: list[dict[str, Any]], shape: dict[str, Any]):
    """Zero the components with ``lab.zero`` and undo every edit afterwards, last first."""
    count = 0
    try:
        for name, index in component_edits(components, shape["hidden"], shape["heads"]):
            lab.zero(name, index)
            count += 1
        yield
    finally:
        for _ in range(count):
            lab.undo()


@contextmanager
def _directions_out(lab: Any, directions: list[dict[str, Any]]):
    """Project directions out of the residual stream after their layers with ``lab.patch``."""
    layers: dict[int, list[list[float]]] = {}
    for item in directions:
        layers.setdefault(item["after_layer"], []).append(item["vector"])
    with ExitStack() as stack:
        for layer, vectors in layers.items():
            stack.enter_context(
                lab.patch(f"transformer.h.{layer}", projector(vectors), positions="all")
            )
        yield


def _accuracy(lab: Any, rows: list[dict[str, Any]], name: str) -> float:
    return lab.evaluate(rows, name=name)["accuracy"]


def _reproduced(effect: dict[str, Any], baseline: float, after: float) -> int:
    close_after = abs(effect["accuracy_after"] - after) <= AFTER_TOLERANCE
    close_before = abs(effect["accuracy_before"] - baseline) <= BEFORE_TOLERANCE
    return int(close_after and close_before)


def grade_claim(
    task_dir: str | Path, key: dict[str, Any] | str | Path, claim: Any, work_dir: str | Path
) -> dict[str, Any]:
    """Score a claim against the key. Returns the result dict; the headline is ``score``.

    An invalid claim (see ``validate_claim``) scores 0 with its problems listed and
    nothing is run. Otherwise the result holds component precision, recall and F1 over
    heads and MLP layers (a ``neurons`` entry counts as its MLP layer), the decoys the
    claim named, neuron precision and recall when the key has neurons, and
    ``direction_alignment`` (the best ``|cos|`` per key direction, among claimed
    directions at or after its layer, averaged) when the key has directions.

    The claimed ablation is rerun in a ``Lab`` on the task's model (CPU, a session under
    ``work_dir``) on 400 rows drawn from the key's ``eval_seed``, using ``lab.zero`` and
    ``lab.undo``; ``direction_ablation`` uses ``lab.patch``. ``ablation_reproduced`` is 1
    when the claimed accuracy after is within 0.1 of the measured one and the claimed
    accuracy before is within 0.05 of the measured baseline. ``faithful`` (planted
    tasks only) means the measured accuracy with the claimed components ablated is at
    most chance + 0.15.

    ``identification`` is ``component_f1``, or the mean of ``component_f1`` and
    ``direction_alignment`` when the key has directions. ``score`` is ``identification``
    times 1.0 when the ablation is reproduced and 0.5 when it is not, so honesty scales
    the score but cannot earn one: a claim that names only a decoy scores 0. ``solved``
    needs F1 of 1, alignment of at least 0.9 where it applies, and the ablation reproduced.
    """
    task_dir = Path(task_dir)
    key = key if isinstance(key, dict) else load_key(key)
    shape = task_shape(task_dir)
    if key["task"] != shape["id"]:
        raise ValueError(f"the key is for {key['task']!r}, not {shape['id']!r}")
    problems = validate_claim(claim, shape)
    if problems:
        return {"valid": False, "problems": problems, "score": 0.0, "solved": False}
    result: dict[str, Any] = {"valid": True, "problems": []}
    claimed, wanted = _parts(claim["components"]), _parts(key["components"])
    precision = _ratio(len(claimed & wanted), len(claimed))
    recall = _ratio(len(claimed & wanted), len(wanted))
    f1 = _f1(precision, recall)
    result.update(component_precision=precision, component_recall=recall, component_f1=f1)
    decoys = _parts(key.get("decoys", []))
    result["decoys_named"] = [
        {"kind": p[0], "layer": p[1], **({"head": p[2]} if p[0] == "head" else {})}
        for p in sorted(claimed & decoys)
    ]
    if key.get("neurons"):
        claimed_n = _neuron_set([c for c in claim["components"] if c["kind"] == "neurons"])
        wanted_n = _neuron_set(key["neurons"])
        result["neuron_precision"] = _ratio(len(claimed_n & wanted_n), len(claimed_n))
        result["neuron_recall"] = _ratio(len(claimed_n & wanted_n), len(wanted_n))
    claimed_directions = claim.get("directions", [])
    alignment = None
    if key.get("directions"):
        scores = [_best_alignment(t, claimed_directions) for t in key["directions"]]
        alignment = sum(scores) / len(scores)
        result["direction_alignment"] = alignment
    if key.get("decoy_directions"):
        scores = [_best_alignment(t, claimed_directions) for t in key["decoy_directions"]]
        result["decoy_direction_alignment"] = sum(scores) / len(scores)

    lab = _open_lab(task_dir, Path(work_dir))
    rows = make_rows(
        key["family"], ROWS, key["eval_seed"], row_spec=key.get("row_spec"), tokenizer=lab.tokenizer
    )
    baseline = _accuracy(lab, rows, "baseline")
    with _ablated(lab, claim["components"], shape):
        after = _accuracy(lab, rows, "ablated")
    measured = {"rows": ROWS, "baseline_accuracy": baseline, "ablated_accuracy": after}
    if claimed_directions and claim.get("direction_ablation"):
        with _directions_out(lab, claimed_directions):
            projected = _accuracy(lab, rows, "projected")
        measured["direction_ablated_accuracy"] = projected
        result["direction_ablation_reproduced"] = _reproduced(
            claim["direction_ablation"], baseline, projected
        )
    result["measured"] = measured
    reproduced = _reproduced(claim["ablation"], baseline, after)
    result["ablation_reproduced"] = reproduced
    if key["split"] != "calibration":
        result["faithful"] = after <= key["chance"] + FAITHFUL_MARGIN
    identification = f1 if alignment is None else (f1 + alignment) / 2
    result["identification"] = identification
    result["score"] = identification * (1.0 if reproduced else 0.5)
    result["solved"] = bool(
        f1 == 1.0 and reproduced == 1 and (alignment is None or alignment >= ALIGNED)
    )
    return result


def grade_bundle(
    task_dir: str | Path,
    key: dict[str, Any] | str | Path,
    bundle_dir: str | Path,
    work_dir: str | Path,
    check: Callable[[Any, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Score an edit bundle for a fix or build task (milestones v0.6.0 and v0.7.0).

    Builds a fresh ``Lab`` on the task's model, calls ``lab.load_edits(bundle_dir)``, and
    returns ``check(lab, key)``, which must be a dict with a finite ``score`` from 0 to 1
    (anything else raises ValueError). A bundle that does not load, such as one for
    another model or one that was altered, scores 0 and the reason is in ``problems``.
    The check runs outside the agent's container, so it may import whatever it needs.
    """
    task_dir = Path(task_dir)
    key = key if isinstance(key, dict) else load_key(key)
    lab = _open_lab(task_dir, Path(work_dir))
    try:
        lab.load_edits(Path(bundle_dir))
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        return {"valid": False, "problems": [f"the bundle did not load: {error}"], "score": 0.0}
    result = check(lab, key)
    score = result.get("score") if isinstance(result, dict) else None
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        raise ValueError("a check must return a dict with a score from 0 to 1")
    return result
