"""The ``oumi-weights-lab proving-ground`` command: build, run, grade and compare.

This module lives in the proving package so that an image without the package has no
trace of the command. Handlers import the heavy modules when they run.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from ..cli import _json, _positive


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _build(args: argparse.Namespace) -> int:
    from . import tasks

    if args.split == "heldout" and args.secret_file is None:
        raise ValueError("held-out tasks need --secret-file")
    if args.split != "heldout" and args.secret_file is not None:
        raise ValueError(f"{args.split} seeds are public; do not pass --secret-file")
    if args.start < 0:
        raise ValueError("--start must not be negative")
    plan = [
        (family, index)
        for family in dict.fromkeys(args.family)
        for index in range(args.start, args.start + args.count)
    ]
    for family, index in plan:
        tasks._check_task(family, args.split, index)
    built = []
    for family, index in plan:
        info = tasks.build_task(
            family, args.split, index, args.out, args.keys, secret_file=args.secret_file
        )
        print(f"built {info['task']}", file=sys.stderr, flush=True)
        built.append({"family": family, "index": index, **info})
    _json({"tasks": built})
    return 0


def _run(args: argparse.Namespace) -> int:
    from . import runner

    records = runner.run(
        args.tasks,
        args.keys,
        arm=args.arm,
        image=args.image,
        agent_cmd=args.agent_cmd,
        agent_label=args.agent_label,
        repeats=args.repeats,
        timeout=args.timeout,
        results=args.results,
        env=args.env,
        task_ids=args.task or None,
    )
    scores = [record["score"]["score"] for record in records]
    _json(
        {
            "results": str(args.results),
            "ran": len(records),
            "timed_out": sum(record["timed_out"] for record in records),
            "invalid_claims": sum(not record["claim_valid"] for record in records),
            "mean_score": sum(scores) / len(scores) if scores else None,
        }
    )
    return 0


def _grade(args: argparse.Namespace) -> int:
    from ..methods_store import methods_dir, parse_reference
    from . import grade_claim, load_claim, load_key, record_method_result

    method = None
    if args.method is not None:
        # Refuse an unknown method before any grading work starts.
        from ..methods_store import _version_dir

        name, version = parse_reference(args.method)
        if version is None:
            raise ValueError("--method needs NAME@VERSION")
        root = methods_dir(args.methods_dir)
        _version_dir(root, name, version)
        method = (root, name, version)
    elif args.methods_dir is not None:
        raise ValueError("--methods-dir needs --method")
    key = load_key(args.key)
    try:
        claim = load_claim(args.claim)
    except (OSError, ValueError) as error:
        graded = {
            "valid": False,
            "problems": [f"the claim could not be read: {error}"],
            "score": 0.0,
            "solved": False,
        }
    else:
        with tempfile.TemporaryDirectory(prefix="proving-grade-") as work:
            graded = grade_claim(args.task, key, claim, work)
    if method is None:
        _json(graded)
        return 0
    # Print the grade first: recording can fail after the grading work is done.
    _json({**graded, "saved_method": {"name": method[1], "version": method[2]}})
    record_method_result(*method, {"task": key["task"], "split": key["split"], "score": graded})
    return 0


def _results_side(text: str) -> tuple[Path, str | None]:
    """Split ``FILE[:ARM]``; a suffix that is not an arm stays part of the file name."""
    path, _, arm = text.rpartition(":")
    if path and arm in ("skill", "control"):
        return Path(path), arm
    return Path(text), None


def _compare(args: argparse.Namespace) -> int:
    from . import beats, compare
    from .runner import read_results

    sides = []
    for text in (args.left, args.right):
        path, arm = _results_side(text)
        if not path.is_file():
            raise ValueError(f"{path} is not a results file")
        records = [r for r in read_results(path) if arm is None or r.get("arm") == arm]
        if not records:
            raise ValueError(f"{text} holds no results")
        sides.append(records)
    report = compare(sides[0], sides[1], args.split)
    _json({**report, "beats": beats(report)})
    return 0


def add_commands(commands) -> None:
    """Register ``proving-ground`` on the CLI's subcommands."""
    commands.metavar = commands.metavar.rstrip("}") + ",proving-ground}"
    proving = commands.add_parser(
        "proving-ground", help="build tasks with hidden answers, run agents on them, score claims"
    )
    steps = proving.add_subparsers(dest="proving_command", required=True)
    build = steps.add_parser("build", help="build task directories and their answer keys")
    build.add_argument("--family", action="append", required=True)
    build.add_argument("--split", required=True, choices=("dev", "heldout", "calibration"))
    build.add_argument("--count", type=_positive_int, required=True)
    build.add_argument("--start", type=int, default=0)
    build.add_argument("--out", required=True, type=Path, help="tasks directory")
    build.add_argument("--keys", required=True, type=Path, help="keys directory, kept private")
    build.add_argument("--secret-file", type=Path, help="held-out secret; held-out builds only")
    build.set_defaults(handler=_build)
    run = steps.add_parser("run", help="run an agent on tasks in containers and grade its claims")
    run.add_argument("--tasks", required=True, type=Path)
    run.add_argument("--keys", required=True, type=Path)
    run.add_argument("--arm", required=True, choices=("skill", "control"))
    run.add_argument("--image", required=True)
    run.add_argument("--agent-cmd", required=True)
    run.add_argument("--agent-label", required=True, help="the model behind the agent")
    run.add_argument("--repeats", type=_positive_int, default=3)
    run.add_argument("--timeout", type=_positive, default=3600.0, help="seconds per run")
    run.add_argument("--results", required=True, type=Path)
    run.add_argument("--env", action="append", default=[], help="environment variable name to pass")
    run.add_argument("--task", action="append", help="run only this task id")
    run.set_defaults(handler=_run)
    grade = steps.add_parser("grade", help="grade one claim against a key")
    grade.add_argument("--task", required=True, type=Path)
    grade.add_argument("--key", required=True, type=Path)
    grade.add_argument("--claim", required=True, type=Path)
    grade.add_argument(
        "--method", help="NAME@VERSION of a saved method; its results file records this grade"
    )
    grade.add_argument("--methods-dir", type=Path, help="saved methods directory (with --method)")
    grade.set_defaults(handler=_grade)
    compare = steps.add_parser("compare", help="paired comparison of two sets of results")
    compare.add_argument("left", help="RESULTS_FILE or RESULTS_FILE:ARM")
    compare.add_argument("right", help="RESULTS_FILE or RESULTS_FILE:ARM")
    compare.add_argument("--split", choices=("dev", "heldout", "calibration"))
    compare.set_defaults(handler=_compare)
