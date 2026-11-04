"""Offline paired evaluation reports, regression suites, and pinned dev screens.

Probabilities normalize summed continuation likelihoods over the supplied options.
They describe that candidate set, not calibrated probabilities over all answers.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import uuid
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from .evaluation_data import (
    _digest,
    _guard_dataset_roles,
    _load_registry,
    _reservation_keys,
    _reserve,
    _safe_name,
    _tasks,
)
from .session import read_json, write_json

FAILED_NLL = -math.log(1e-12)
PROTOCOL_KEYS = (
    "session_id",
    "mode",
    "seed",
    "generation_config",
    "tokenizer_policy",
    "checker_custom",
    "checker_id",
)
# Fields that name a row's source document, checked in this order for bootstrap clusters.
DOCUMENT_FIELDS = (
    "source_document",
    "source_doc",
    "document_id",
    "metadata.source_document",
    "metadata.source_doc",
    "metadata.document_id",
)


def _field(row: dict, key: str) -> Any:
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value or value[part] is None:
            raise ValueError(f"missing group/stratification field: {key}")
        value = value[part]
    return value


def _label(value: Any) -> str:
    if isinstance(value, str):
        return "@str:" + value if value.startswith("@") else value
    return "@json:" + json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def mcnemar(fixed: int, broken: int) -> float:
    """Exact two-sided binomial McNemar test, without constructing huge integers."""
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in (fixed, broken)):
        raise ValueError("discordant counts must be nonnegative integers")
    n, k = fixed + broken, min(fixed, broken)
    if n == 0 or fixed == broken:
        return 1.0
    # Start at the largest term of the lower tail, then sum relative masses.
    log_mass = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)
    relative = term = 1.0
    for i in range(k, 0, -1):
        term *= i / (n - i + 1)
        relative += term
        if term < relative * 1e-16:
            break
    return min(1.0, math.exp(log_mass + math.log(relative) + math.log(2)))


def _observation(item: dict, mode: str) -> dict:
    failed = bool(item.get("error")) or "prediction" not in item
    result = {"correct": bool(item.get("correct")) and not failed, "failed": failed}
    if mode == "sampled":
        return result
    scores = item.get("scores")
    if failed or not scores:
        return {
            **result,
            "correct": False,
            "failed": True,
            "nll": FAILED_NLL,
            "brier": 2.0,
            "confidence": 1.0,
            "gold_probability": 0.0,
        }
    if set(scores) != set(item["options"]) or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in scores.values()
    ):
        raise ValueError("scores must contain finite log likelihoods for every option")
    peak = max(scores.values())
    log_total = math.log(sum(math.exp(value - peak) for value in scores.values()))
    log_probs = {key: value - peak - log_total for key, value in scores.items()}
    probs = {key: math.exp(value) for key, value in log_probs.items()}
    return {
        **result,
        "nll": -log_probs[item["answer"]],
        "brier": sum((prob - (key == item["answer"])) ** 2 for key, prob in probs.items()),
        "confidence": max(probs.values()),
        "gold_probability": probs[item["answer"]],
    }


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    out = {
        "n": n,
        "correct": sum(r["correct"] for r in rows),
        "failures": sum(r["failed"] for r in rows),
    }
    out["accuracy"] = out["correct"] / n
    out.update(nll=None, brier=None, ece=None)
    if "nll" in rows[0]:
        out.update({key: sum(r[key] for r in rows) / n for key in ("nll", "brier")})
        bins: dict[int, list] = defaultdict(list)
        for row in rows:
            bins[min(int(row["confidence"] * 10), 9)].append(row)
        out["ece"] = sum(
            abs(sum(r["confidence"] - r["correct"] for r in cell)) / n for cell in bins.values()
        )
    return out


def _percentile(values: list[float], q: float) -> float:
    at = (len(values) - 1) * q
    lower = int(at)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (at - lower)


def _paired_report(
    pairs: list[tuple[dict, dict]], clusters: list[str], samples: int, seed: int
) -> dict:
    left, right = [p[0] for p in pairs], [p[1] for p in pairs]
    before, after = _metrics(left), _metrics(right)
    fixed = sum(not a["correct"] and b["correct"] for a, b in pairs)
    broken = sum(a["correct"] and not b["correct"] for a, b in pairs)
    changes = {
        key: after[key] - before[key] if before[key] is not None else None
        for key in ("accuracy", "nll", "brier", "ece")
    }
    groups: dict[str, list[int]] = defaultdict(list)
    for i, cluster in enumerate(clusters):
        groups[cluster].append(i)
    keys, rng = sorted(groups), random.Random(seed)
    draws: dict[str, list[float]] = {"accuracy": []}
    if "nll" in left[0]:
        draws["nll"] = []
    # Resample whole documents; related rows always travel together.
    totals = {
        key: {
            metric: sum(
                float(right[i]["correct"] if metric == "accuracy" else right[i][metric])
                - float(left[i]["correct"] if metric == "accuracy" else left[i][metric])
                for i in groups[key]
            )
            for metric in draws
        }
        for key in keys
    }
    for _ in range(samples):
        picked = rng.choices(keys, k=len(keys))
        count = sum(len(groups[key]) for key in picked)
        for metric in draws:
            draws[metric].append(sum(totals[key][metric] for key in picked) / count)
    intervals = {
        key: [_percentile(sorted(values), 0.025), _percentile(sorted(values), 0.975)]
        for key, values in draws.items()
    }
    return {
        "before": before,
        "after": after,
        "delta": changes,
        "fixed": fixed,
        "broken": broken,
        "mcnemar_p": mcnemar(fixed, broken),
        "bootstrap": {
            "seed": seed,
            "samples": samples,
            "confidence": 0.95,
            "clusters": len(keys),
            "delta_ci": intervals,
        },
    }


def _read_run(lab: Any, run: Any) -> tuple[dict, list[dict], list[dict]]:
    summary, path = lab._run_summary(run)
    tasks = read_json(path / "dataset.json")["tasks"]
    if not tasks or len(tasks) != summary["total"]:
        raise ValueError("run dataset is incomplete")
    output = {}
    items_path = path / "items.jsonl"
    for line in items_path.read_text(encoding="utf-8").splitlines() if items_path.exists() else []:
        item = json.loads(line)
        i = item["index"]
        if not isinstance(i, int) or isinstance(i, bool) or i in output or not 0 <= i < len(tasks):
            raise ValueError("duplicate or invalid evaluation item index")
        if any(item.get(key) != value for key, value in tasks[i].items()):
            raise ValueError("evaluation item does not match pinned dataset")
        output[i] = item
    if summary["state"] == "succeeded" and len(output) != len(tasks):
        raise ValueError("successful run is missing evaluation items")
    items = [
        output.get(i, {"index": i, **task, "error": "item was not completed"})
        for i, task in enumerate(tasks)
    ]
    return summary, tasks, items


class EvaluationMixin:
    """Reusable helpers; comparisons read saved runs and never call the model."""

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

    def diff(
        self, before: Any, after: Any, *, include_scores: bool = False
    ) -> list[dict[str, Any]]:
        """Return changed items between two successful runs of the same dataset.

        Rows pair by position. Both runs need the same session, mode, scoring
        protocol, and ordered prompts, options, answers, and metadata; only row
        IDs may differ. Use compare_runs for ID-checked pairing and statistics.
        """
        left, _ = self._run_summary(before)
        right, _ = self._run_summary(after)
        if left["state"] != "succeeded" or right["state"] != "succeeded":
            raise ValueError("diff requires two successful evaluation runs")
        if any(left[key] != right[key] for key in ("session_id", "dataset_hash", "mode")):
            raise ValueError("diff requires the same session, dataset, and scoring mode")
        _, left_tasks, left_items = _read_run(self, before)
        _, right_tasks, right_items = _read_run(self, after)
        if any(left.get(key) != right.get(key) for key in PROTOCOL_KEYS):
            raise ValueError("diff requires the same scoring protocol/settings")
        if left.get("checker_custom") and not left.get("checker_id"):
            raise ValueError("custom sampled checkers require explicit checker_id for comparison")
        comparable_left = [
            {key: value for key, value in task.items() if key != "id"} for task in left_tasks
        ]
        comparable_right = [
            {key: value for key, value in task.items() if key != "id"} for task in right_tasks
        ]
        if comparable_left != comparable_right:
            raise ValueError("diff requires identical ordered task content and metadata")
        changes = []
        for old, new in zip(left_items, right_items, strict=True):
            same_prediction = (old["prediction"], old["correct"]) == (
                new["prediction"],
                new["correct"],
            )
            if same_prediction and (not include_scores or old.get("scores") == new.get("scores")):
                continue
            kind = "score_changed" if same_prediction else "changed"
            if old["correct"] != new["correct"]:
                kind = "improved" if new["correct"] else "regressed"
            changes.append({"index": old["index"], "change": kind, "before": old, "after": new})
        return changes

    def compare_runs(
        self,
        before: Any,
        after: Any,
        *,
        group_by: str | None = None,
        bootstrap_group_by: str | None = None,
        top: int = 10,
        bootstrap_samples: int = 1000,
        seed: int = 0,
    ) -> dict:
        """Compare strictly paired runs, including failed rows, with document bootstrap CIs."""
        for key, value in (("group_by", group_by), ("bootstrap_group_by", bootstrap_group_by)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{key} must be a nonempty field name")
        if isinstance(top, bool) or not isinstance(top, int) or top < 0:
            raise ValueError("top must be a nonnegative integer")
        if (
            isinstance(bootstrap_samples, bool)
            or not isinstance(bootstrap_samples, int)
            or bootstrap_samples < 1
        ):
            raise ValueError("bootstrap_samples must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        a, tasks_a, items_a = _read_run(self, before)
        b, tasks_b, items_b = _read_run(self, after)
        if any(a.get(key) != b.get(key) for key in PROTOCOL_KEYS):
            raise ValueError(
                "comparison requires the same session, mode, and scoring protocol/settings"
            )
        if a.get("checker_custom") and not a.get("checker_id"):
            raise ValueError("custom sampled checkers require explicit checker_id for comparison")
        if tasks_a != tasks_b:
            raise ValueError("comparison requires identical ordered tasks, IDs, and metadata")
        ids = [_digest(task["id"]) for task in tasks_a if "id" in task]
        if len(ids) != len(set(ids)) or (ids and len(ids) != len(tasks_a)):
            raise ValueError("paired IDs must be unique and present on every row")
        pairs = [
            (_observation(x, a["mode"]), _observation(y, b["mode"]))
            for x, y in zip(items_a, items_b, strict=True)
        ]
        cluster_by = bootstrap_group_by
        if cluster_by:
            [_field(task, cluster_by) for task in tasks_a]
        for field in () if cluster_by else DOCUMENT_FIELDS:
            try:
                [_field(task, field) for task in tasks_a]
            except ValueError:
                continue
            cluster_by = field
            break
        # Explicit document metadata must never silently degrade to independent rows.
        if cluster_by is None:
            for field in DOCUMENT_FIELDS:
                if any(_has_field(task, field) for task in tasks_a):
                    raise ValueError(f"missing document group field: {field}")
        clusters = [
            _digest(_field(task, cluster_by)) if cluster_by else str(i)
            for i, task in enumerate(tasks_a)
        ]
        report = _paired_report(pairs, clusters, bootstrap_samples, seed)
        groups: dict[str, list[int]] = defaultdict(list)
        if group_by:
            for i, task in enumerate(tasks_a):
                groups[_label(_field(task, group_by))].append(i)
        grouped = {
            key: _paired_report(
                [pairs[i] for i in indices], [clusters[i] for i in indices], bootstrap_samples, seed
            )
            for key, indices in sorted(groups.items())
        }
        movers = []
        for i, ((x, y), old, new) in enumerate(zip(pairs, items_a, items_b, strict=True)):
            # A failed side has a penalty, not a measured probability or score.
            failed = x["failed"] or y["failed"]
            before_p = None if x["failed"] else x.get("gold_probability")
            after_p = None if y["failed"] else y.get("gold_probability")
            old_score, new_score = (
                old["scores"].get(old["answer"]) if old.get("scores") else None,
                new["scores"].get(new["answer"]) if new.get("scores") else None,
            )
            movers.append(
                {
                    "index": i,
                    "id": tasks_a[i].get("id", i),
                    "failed": failed,
                    "gold_probability_before": before_p,
                    "gold_probability_after": after_p,
                    "gold_probability_delta": (
                        after_p - before_p if before_p is not None and after_p is not None else None
                    ),
                    "gold_score_delta": (
                        new_score - old_score
                        if not failed and old_score is not None and new_score is not None
                        else None
                    ),
                    "correct_before": x["correct"],
                    "correct_after": y["correct"],
                }
            )
        ranked = [mover for mover in movers if not mover["failed"]]
        return {
            "before_state": a["state"],
            "after_state": b["state"],
            "aggregate": report,
            "group_by": group_by,
            "groups": grouped,
            "bootstrap_group_by": cluster_by,
            "top_probability_movers": sorted(
                ranked, key=lambda r: abs(r["gold_probability_delta"] or 0), reverse=True
            )[:top],
            "top_score_movers": sorted(
                ranked, key=lambda r: abs(r["gold_score_delta"] or 0), reverse=True
            )[:top],
            "failed_rows": [mover for mover in movers if mover["failed"]],
            "probability_scope": "candidate_set",
            "ece_bins": 10,
            "caveat": (
                "Candidate-set probabilities are not calibrated open-answer probabilities; "
                "CIs are descriptive, not sequential-testing guarantees."
            ),
        }

    def evaluate_suite(
        self, datasets: dict[str, Any], *, name: str | None = None, **defaults: Any
    ) -> dict:
        """Evaluate every named dataset, save failures, and return a suite manifest."""
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError("datasets must be a nonempty name-to-data mapping")
        name = _safe_name(name or f"suite-{uuid.uuid4().hex[:12]}")
        for key in datasets:
            _safe_name(key)
        root = Path(self.artifacts) / "suites" / name
        root.mkdir(parents=True, exist_ok=False)
        result = {
            "name": name,
            "suite_path": str((root / "suite.json").resolve()),
            "state": "running",
            "expected_datasets": list(datasets),
            "datasets": {},
        }
        write_json(root / "suite.json", result)
        for index, (key, spec) in enumerate(datasets.items()):
            options = {**defaults}
            data = spec
            if isinstance(spec, dict) and "data" in spec:
                options.update({k: v for k, v in spec.items() if k != "data"})
                data = spec["data"]
            run_name = f"suite-{uuid.uuid4().hex[:12]}-{index}"
            options["name"] = run_name
            try:
                run = self.evaluate(data, **options)
            except Exception as error:
                path = Path(self.artifacts) / "evaluations" / run_name / "summary.json"
                run = (
                    read_json(path)
                    if path.exists()
                    else {"state": "failed", "error": f"{type(error).__name__}: {error}"}
                )
            result["datasets"][key] = run
            write_json(root / "suite.json", result)
        result["state"] = (
            "succeeded"
            if all(r["state"] == "succeeded" for r in result["datasets"].values())
            else "failed"
        )
        write_json(root / "suite.json", result)
        return result

    def compare_suites(
        self,
        before: Any,
        after: Any,
        *,
        accuracy_tolerance: Any = 0.0,
        nll_tolerance: Any = 0.0,
        group_by: str | None = None,
        required_groups: list[str] | None = None,
        **comparison: Any,
    ) -> dict:
        """Offline regression gate: every dataset and group must meet both tolerances."""
        a, b = _suite(before), _suite(after)
        checks = {}
        names = sorted(
            a["datasets"].keys()
            | b["datasets"].keys()
            | set(a.get("expected_datasets", []))
            | set(b.get("expected_datasets", []))
        )
        if not names:
            raise ValueError("suites must contain datasets")
        for name in names:
            old, new = a["datasets"].get(name), b["datasets"].get(name)
            acc_tol, nll_tol = _tolerance(accuracy_tolerance, name), _tolerance(nll_tolerance, name)
            if old is None or new is None:
                checks[name] = {"pass": False, "error": "dataset is missing from one suite"}
                continue
            try:
                report = self.compare_runs(old, new, group_by=group_by, **comparison)
                surfaces = {
                    "aggregate": report["aggregate"],
                    **{f"group:{key}": value for key, value in report["groups"].items()},
                }
                missing = set(required_groups or []) - report["groups"].keys()
                passed = not missing and old["state"] == new["state"] == "succeeded"
                verdicts = {}
                # Exact: read the tolerance as its shortest decimal, so 1 of 10 rows meets 0.1.
                max_loss = Fraction(repr(acc_tol))
                for key, surface in surfaces.items():
                    delta = surface["delta"]
                    lost = surface["before"]["correct"] - surface["after"]["correct"]
                    verdicts[key] = Fraction(lost, surface["after"]["n"]) <= max_loss and (
                        delta["nll"] is None or delta["nll"] <= nll_tol
                    )
                    passed = passed and verdicts[key]
                checks[name] = {
                    "pass": bool(passed),
                    "surfaces": verdicts,
                    "missing_groups": sorted(missing),
                    "accuracy_tolerance": acc_tol,
                    "nll_tolerance": nll_tol,
                    "comparison": report,
                }
            except (ValueError, KeyError, OSError) as error:
                checks[name] = {"pass": False, "error": str(error)}
        complete = a.get("state") == b.get("state") == "succeeded"
        return {
            "pass": complete and all(check["pass"] for check in checks.values()),
            "before_state": a.get("state"),
            "after_state": b.get("state"),
            "checks": checks,
        }

    def create_screen(
        self,
        data: Any,
        *,
        name: str,
        size: int | None = None,
        stratify_by: tuple[str, ...] | list[str] = (),
        min_per_cell: int = 1,
        seed: int = 0,
        mode: str = "likelihood",
        held_out: bool = False,
    ) -> dict:
        """Pin a repeatable stratified dev subset. Screens never establish a held-out result."""
        _safe_name(name)
        if held_out:
            raise ValueError("screens are dev-only; held-out data cannot be selected")
        for key, value, minimum in (
            ("min_per_cell", min_per_cell, 1),
            ("seed", seed, None),
            ("size", size, 1),
        ):
            if value is None and key == "size":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or (minimum is not None and value < minimum)
            ):
                raise ValueError(f"invalid {key}")
        if isinstance(stratify_by, str) or any(
            not isinstance(k, str) or not k for k in stratify_by
        ):
            raise ValueError("stratify_by must be a sequence of field names")
        tasks, identities = _tasks(data, mode)
        if any(
            task.get("held_out")
            or task.get("split") in {"test", "heldout", "held_out"}
            or (
                isinstance(task.get("metadata"), dict)
                and (
                    task["metadata"].get("held_out")
                    or task["metadata"].get("split") in {"test", "heldout", "held_out"}
                )
            )
            for task in tasks
        ):
            raise ValueError("screens are dev-only; rows declare held-out data")
        row_keys = [
            _digest(task.get("id", identity))
            for task, identity in zip(tasks, identities, strict=True)
        ]
        if len(row_keys) != len(set(row_keys)):
            raise ValueError("screen rows require unique IDs or unique semantic content")
        cells: dict[str, list[int]] = defaultdict(list)
        for i, task in enumerate(tasks):
            cells[_label([_field(task, key) for key in stratify_by])].append(i)
        target = len(tasks) if size is None else min(size, len(tasks))
        minimum = sum(min(min_per_cell, len(indices)) for indices in cells.values())
        if target < minimum:
            raise ValueError("size cannot satisfy min_per_cell in every stratum")
        rng, selected = random.Random(seed), []
        for cell in sorted(cells):
            members = sorted(cells[cell], key=lambda i: row_keys[i])
            selected.extend(rng.sample(members, min(min_per_cell, len(members))))
        remaining = sorted(set(range(len(tasks))) - set(selected), key=lambda i: row_keys[i])
        selected.extend(rng.sample(remaining, target - len(selected)))
        selected.sort(key=lambda i: row_keys[i])
        rows = [tasks[i] for i in selected]
        root = Path(self.artifacts) / "screens" / name
        # A screen consumes dev access to its whole source, not just selected rows.
        with _reserve(self.artifacts, self.session_path, identities, mode=mode, held_out=False) as (
            _,
            identity,
        ):
            root.mkdir(parents=True, exist_ok=False)
            source_path = (
                str(Path(data).resolve())
                if isinstance(data, (str, Path))
                else str((root / "source.json").resolve())
            )
            if not isinstance(data, (str, Path)):
                write_json(root / "source.json", {"tasks": tasks})
            source_sha = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
            body = {
                "name": name,
                "role": "dev",
                "mode": mode,
                "seed": seed,
                "stratify_by": list(stratify_by),
                "min_per_cell": min_per_cell,
                "source_path": source_path,
                "source_sha256": source_sha,
                "source_format": "jsonl" if isinstance(data, (str, Path)) else "json",
                "source_identity": identity,
                "total": len(tasks),
                "selected": len(rows),
                "cells": {
                    cell: {"total": len(indices), "selected": len(set(indices) & set(selected))}
                    for cell, indices in sorted(cells.items())
                },
                "rows_sha256": _digest(rows),
                "row_hashes": [_digest(row) for row in rows],
            }
            manifest = {
                **body,
                "sha256": _digest(body),
                "screen_path": str((root / "screen.json").resolve()),
            }
            write_json(root / "rows.json", {"tasks": rows})
            write_json(root / "screen.json", manifest)
        return manifest

    def verify_screen(self, screen: Any) -> list[dict]:
        """Verify all pins and source bytes; return rows suitable for evaluate()."""
        path = Path(screen["screen_path"] if isinstance(screen, dict) else screen)
        if not path.exists():
            path = Path(self.artifacts) / "screens" / path / "screen.json"
        manifest = read_json(path)
        body = {k: v for k, v in manifest.items() if k not in {"sha256", "screen_path"}}
        if _digest(body) != manifest["sha256"] or manifest["role"] != "dev":
            raise ValueError("screen manifest hash mismatch or invalid role")
        source = Path(manifest["source_path"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"]:
            raise ValueError("screen source hash mismatch")
        data = source if manifest["source_format"] == "jsonl" else read_json(source)["tasks"]
        tasks, identities = _tasks(data, manifest["mode"])
        rows = read_json(path.parent / "rows.json")["tasks"]
        if (
            _digest(rows) != manifest["rows_sha256"]
            or len(rows) != manifest["selected"]
            or [_digest(row) for row in rows] != manifest["row_hashes"]
        ):
            raise ValueError("screen content hash mismatch")
        present = {_digest(row) for row in tasks}
        if any(_digest(row) not in present for row in rows):
            raise ValueError("pinned screen item is missing from source")
        registry = _load_registry(Path(self.artifacts) / "evaluations")
        keys = _reservation_keys(identities, manifest["mode"])
        _guard_dataset_roles(registry, keys, held_out=False)
        return rows


def _has_field(row: dict, key: str) -> bool:
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def _suite(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    path = Path(value)
    return read_json(path / "suite.json" if path.is_dir() else path)


def _tolerance(value: Any, name: str) -> float:
    if isinstance(value, dict):
        value = value.get(name, value.get("default", 0.0))
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("tolerances must be finite nonnegative numbers or dataset-name mappings")
    return float(value)
