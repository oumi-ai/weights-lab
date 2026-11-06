"""Comparisons between two arms of a proving-ground run, such as with and without the skill.

Each arm is a list of result records ``{"task", "split", "repeat", "score": {"score":
float, "solved": bool, ...}}``, where ``score`` is what ``grade_claim`` returns.
Importing this module does not import torch or transformers.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SAMPLES = 10_000
SEED = 0


def _per_task(records: list[dict[str, Any]], split: str | None) -> dict[str, dict[str, float]]:
    """Mean headline score and solve rate of each task over its repeats."""
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if split is None or record["split"] == split:
            runs[record["task"]].append(record["score"])
    return {
        task: {
            "score": sum(float(r["score"]) for r in scores) / len(scores),
            "solved": sum(bool(r.get("solved")) for r in scores) / len(scores),
        }
        for task, scores in runs.items()
    }


SETUP_FIELDS = ("image_id", "agent_cmd", "agent_label", "timeout")


def _setup(records: list[dict[str, Any]], side: str) -> dict[str, Any]:
    """The one setup a side's records share; ValueError when they mix setups."""
    setup = {}
    for field in SETUP_FIELDS:
        values = {json.dumps(record.get(field)) for record in records}
        if len(values) > 1:
            raise ValueError(
                f"the {side} side mixes records with different {field} values "
                f"({', '.join(sorted(values))}); compare one setup at a time"
            )
        setup[field] = records[0].get(field) if records else None
    return setup


def compare(
    left_results: list[dict[str, Any]],
    right_results: list[dict[str, Any]],
    split: str | None = None,
) -> dict[str, Any]:
    """Compare two arms on the tasks both ran; ``delta`` is right minus left.

    Averages each task's score over its repeats, pairs tasks present on both sides, and
    reports the unpaired ones. Raises ValueError when no task pairs. Uses the lab's own
    paired statistics, one observation per task with each task its own cluster, 10,000
    resamples and seed 0: the mean scores, the delta, its 95% interval and the sign-flip
    permutation p. ``split`` keeps only that split's records. Raises ValueError when a
    side mixes records with different ``image_id``, ``agent_cmd``, ``agent_label`` or
    ``timeout`` values; the report gives each side's ``left_setup`` and ``right_setup``.
    """
    from ..evaluation import _paired_report

    kept_left = [r for r in left_results if split is None or r["split"] == split]
    kept_right = [r for r in right_results if split is None or r["split"] == split]
    left_setup, right_setup = _setup(kept_left, "left"), _setup(kept_right, "right")
    left, right = _per_task(left_results, split), _per_task(right_results, split)
    tasks = sorted(left.keys() & right.keys())
    if not tasks:
        raise ValueError("no task appears on both sides, so there is nothing to pair")

    def observation(task: str, side: dict[str, dict[str, float]]) -> dict[str, Any]:
        return {"correct": side[task]["score"], "failed": False, "fractional": True}

    pairs = [(observation(t, left), observation(t, right)) for t in tasks]
    paired = _paired_report(pairs, tasks, SAMPLES, SEED, fractional=True)
    interval = paired["bootstrap"]["delta_ci"]["accuracy"]
    return {
        "split": split,
        "tasks": len(tasks),
        "unpaired_left": sorted(left.keys() - right.keys()),
        "unpaired_right": sorted(right.keys() - left.keys()),
        "left_mean": paired["before"]["accuracy"],
        "right_mean": paired["after"]["accuracy"],
        "delta": paired["delta"]["accuracy"],
        "delta_ci": interval,
        "permutation_p": paired["permutation_p"],
        "left_solve_rate": sum(left[t]["solved"] for t in tasks) / len(tasks),
        "right_solve_rate": sum(right[t]["solved"] for t in tasks) / len(tasks),
        "left_setup": left_setup,
        "right_setup": right_setup,
        "samples": SAMPLES,
        "seed": SEED,
    }


def beats(report: dict[str, Any]) -> bool:
    """True when the interval's lower bound is above zero: the right arm is better.

    This is the release gate from v0.6.0 on; for v0.5.0 it is only reported.
    """
    return report["delta_ci"][0] > 0


def record_method_result(
    methods_dir: str | Path,
    name: str,
    version: str,
    result: dict[str, Any],
    setup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a graded result to a saved method version's ``results.jsonl``.

    ``result`` holds the ``task`` id, its ``split`` and the ``score`` dict that
    ``grade_claim`` returned, and may hold the ``setup`` it ran in (``setup`` overrides it).
    A grade from ``proving-ground grade`` has none: the task id fixes the model. The record
    adds the time, and NaN is written as null. Raises ValueError for a method or version
    that is not saved, or a result without those fields. Returns the record.
    """
    from ..methods_store import append_method_result

    return append_method_result(methods_dir, name, version, result, setup)
