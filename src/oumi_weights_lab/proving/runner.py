"""Run an agent on proving-ground tasks in containers and grade the claims it leaves.

For each task and repeat the runner starts one container from an image, with the task
directory mounted read-only at ``/task`` and an empty ``out/`` directory at ``/out``, and
nothing else. The agent works inside and writes ``/out/claim.json``. The scorer runs on the
host, after the container has stopped, against a key the container never saw. Importing
this module does not import torch or transformers.

What the container can see is the whole isolation story, so it is built in two places that
tests check directly: ``docker_argv`` (what is mounted, what is passed in the environment,
what the prompt says) and ``check_task_dir`` (what the mounted directory holds).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .claims import CLAIM_FORMAT, load_claim
from .tasks import load_key

ARMS = ("skill", "control")
LABEL_PREFIX = "org.oumi.weights-lab."
TASK_MOUNT = "/task"
OUT_MOUNT = "/out"
WORK_DIR = "/work"
CLAIM_NAME = "claim.json"
TRANSCRIPT_NAME = "transcript.txt"
# The only entries a task directory may hold, and the only fields its task.json may have.
TASK_ENTRIES = frozenset({"task.json", "model", "rows.jsonl"})
TASK_FIELDS = frozenset(
    {"id", "split", "kind", "question", "rows", "options_note", "metric", "chance", "claim_version"}
)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")

PROMPT = """\
You are investigating a small language model.

The model is saved at {task_mount}/model in Hugging Face format. Load it offline (use
local_files_only=True); nothing can be downloaded. Python 3 with torch and transformers is
installed.

The behavior to explain: {question}

Example prompts are in {task_mount}/rows.jsonl, one JSON object per line with "prompt",
"options" and "answer". {options_note}

Work in {work_dir}. You have about {minutes} minutes in all. When the time is up the run
stops, and only a claim file that is already written will be graded.

Find which parts of the model carry this behavior, and test your explanation by changing
the model and measuring what happens. Your task id is {task}. Write your claim to
{out_mount}/{claim_name}. Write a first claim early and replace it as you learn more. The
claim will be checked by rerunning its ablation on new rows and comparing its components
with the true mechanism. Put the hypotheses you tried and dropped in "failed_hypotheses".

{claim_format}"""


@dataclass(frozen=True)
class Invocation:
    """One planned container run. ``argv`` is the complete ``docker run`` command line.

    ``mounts`` lists ``(host path, container path, "ro" or "rw")``; ``env`` the names that
    are passed through from the caller's environment (never their values).
    """

    name: str
    image: str
    argv: tuple[str, ...]
    mounts: tuple[tuple[str, str, str], ...]
    env: tuple[str, ...]
    timeout: float
    out_dir: Path
    transcript: Path


Launch = Callable[[Invocation], dict[str, Any]]
Probe = Callable[[str, str], dict[str, Any]]


def build_prompt(task_dir: str | Path, *, timeout: float) -> str:
    """The prompt for one task. It holds the task's public question and the claim format,
    and nothing from the key: no family, no location, no seed."""
    info = json.loads((Path(task_dir) / "task.json").read_text(encoding="utf-8"))
    return PROMPT.format(
        task_mount=TASK_MOUNT,
        out_mount=OUT_MOUNT,
        work_dir=WORK_DIR,
        claim_name=CLAIM_NAME,
        question=info["question"],
        options_note=info.get("options_note", ""),
        minutes=max(1, int(timeout // 60)),
        task=info["id"],
        claim_format=CLAIM_FORMAT.rstrip(),
    )


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def check_task_dir(task_dir: str | Path) -> dict[str, Any]:
    """Return the parsed ``task.json`` of a directory that is safe to mount, or raise.

    Refuses anything beyond ``task.json``, ``model/`` and ``rows.jsonl``, any symlink, and
    a ``task.json`` with fields the builder does not write (such as a family or a seed).
    What is mounted must be exactly what ``build_task`` wrote.
    """
    root = Path(task_dir)
    extra = sorted(entry.name for entry in root.iterdir() if entry.name not in TASK_ENTRIES)
    if extra:
        raise ValueError(f"{root} holds files a task must not have: {extra}")
    missing = sorted(name for name in TASK_ENTRIES if not (root / name).exists())
    if missing:
        raise ValueError(f"{root} is missing {missing}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{path} is a symlink; a task directory must not hold symlinks")
    info = json.loads((root / "task.json").read_text(encoding="utf-8"))
    unknown = sorted(set(info) - TASK_FIELDS)
    if unknown:
        raise ValueError(f"{root / 'task.json'} has fields a task must not have: {unknown}")
    if info.get("id") != root.name:
        raise ValueError(f"{root / 'task.json'} names task {info.get('id')!r}, not {root.name!r}")
    return info


def docker_argv(
    *,
    name: str,
    image: str,
    task_dir: str | Path,
    out_dir: str | Path,
    command: list[str],
    prompt: str,
    env: Iterable[str] = (),
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """The ``docker run`` command line and its mounts. This is the only place either is made.

    The task directory goes in read-only at ``/task`` and ``out_dir`` at ``/out``; nothing
    else is mounted. Each ``env`` name is passed as ``-e NAME``, so Docker reads the value
    from the caller's environment and it appears in no argument. ``command`` is the agent
    command split into words; the prompt replaces a word that is exactly ``{prompt}`` or,
    with none, is appended as the last argument. Refuses a path with a comma (it would
    break the mount syntax) and an env name that is not a plain name.
    """
    mounts = [
        (str(Path(task_dir).resolve()), TASK_MOUNT, "ro"),
        (str(Path(out_dir).resolve()), OUT_MOUNT, "rw"),
    ]
    for host, _, _ in mounts:
        if "," in host:
            raise ValueError(f"{host} has a comma, which Docker's mount syntax cannot hold")
    names = list(env)
    for item in names:
        if not _NAME.match(item):
            raise ValueError(
                f"{item!r} is not an environment variable name; pass names, not values"
            )
    if not command:
        raise ValueError("the agent command is empty")
    argv = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--workdir",
        WORK_DIR,
    ]
    for host, target, mode in mounts:
        argv += [
            "--mount",
            f"type=bind,source={host},target={target}" + (",readonly" if mode == "ro" else ""),
        ]
    for item in names:
        argv += ["-e", item]
    argv.append(image)
    if "{prompt}" in command:
        argv += [prompt if word == "{prompt}" else word for word in command]
    else:
        argv += [*command, prompt]
    return argv, mounts


def docker_launch(invocation: Invocation) -> dict[str, Any]:
    """Run the container; on timeout kill it by name. Returns ``exit_code`` and ``timed_out``.

    Output goes to the transcript file, which the caller created before the container
    started, so nothing the agent does in ``/out`` can redirect it.
    """
    with invocation.transcript.open("ab") as transcript:
        process = subprocess.Popen(
            invocation.argv,
            stdin=subprocess.DEVNULL,
            stdout=transcript,
            stderr=subprocess.STDOUT,
        )
        try:
            code = process.wait(timeout=invocation.timeout)
            return {"exit_code": code, "timed_out": False}
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "kill", invocation.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            process.kill()
            process.wait()
            return {"exit_code": None, "timed_out": True}


def docker_probe(image: str, agent_command: str) -> dict[str, Any]:
    """The agent's version, the image id and the image's labels, from Docker."""
    inspected = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspected.returncode:
        raise RuntimeError(f"docker cannot inspect {image}: {inspected.stderr.strip()}")
    info = json.loads(inspected.stdout)
    version = subprocess.run(
        ["docker", "run", "--rm", image, agent_command, "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    return {
        "agent_version": ((version.stdout or version.stderr).strip().splitlines() or [""])[-1],
        "image_id": info["Id"],
        "labels": (info.get("Config") or {}).get("Labels") or {},
    }


def read_results(path: str | Path) -> list[dict[str, Any]]:
    """The records of a results file (one JSON object per line); a missing file is empty."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} line {number} is not JSON: {error}") from error
    return records


def _discover(tasks_dir: Path, task_ids: list[str] | None) -> list[str]:
    found = sorted(p.name for p in tasks_dir.iterdir() if (p / "task.json").is_file())
    if task_ids is None:
        return found
    for ident in task_ids:
        if not _TASK_ID.match(ident) or ident not in found:
            raise ValueError(f"no task {ident!r} in {tasks_dir}")
    return list(dict.fromkeys(task_ids))


def _grade_claim(task_dir: Path, key: dict[str, Any], out_dir: Path, work_dir: Path):
    from .grade import grade_claim

    claim_path = out_dir / CLAIM_NAME
    problems = []
    if claim_path.is_symlink():
        problems.append(f"{CLAIM_NAME} is a symlink; it must be a regular file")
    elif not claim_path.exists():
        problems.append(f"no claim was written to {OUT_MOUNT}/{CLAIM_NAME}")
    elif not claim_path.is_file():
        problems.append(f"{CLAIM_NAME} is not a regular file")
    else:
        try:
            claim = load_claim(claim_path)
        except (ValueError, OSError) as error:
            problems.append(f"{CLAIM_NAME} could not be read: {error}")
        else:
            return grade_claim(task_dir, key, claim, work_dir)
    return {"valid": False, "problems": problems, "score": 0.0, "solved": False}


def run(
    tasks_dir: str | Path,
    keys_dir: str | Path,
    *,
    arm: str,
    image: str,
    agent_cmd: str,
    agent_label: str,
    repeats: int = 3,
    timeout: float = 3600,
    results: str | Path,
    env: Iterable[str] = (),
    task_ids: list[str] | None = None,
    launch: Launch | None = None,
    probe: Probe | None = None,
) -> list[dict[str, Any]]:
    """Run ``agent_cmd`` on every task ``repeats`` times in ``image`` and grade each claim.

    Returns the records it appended to ``results`` (one JSON line per run). A
    ``(task, arm, repeat)`` already in ``results`` is skipped, so a stopped run resumes.
    Each run gets a fresh directory next to ``results`` with an ``out/`` directory the
    container may write; the transcript, the claim and the scorer's working files stay
    there. The key is read on the host to grade and to record the family and split, and it
    is never mounted, nor are the secret and the results file.

    Refuses: an arm other than skill or control; a keys directory inside the tasks
    directory (or the reverse), or a results file inside the tasks directory; a task
    directory with files beyond what ``build_task`` writes; a task without a key; an
    image whose arm label differs from ``arm``; an environment name that is not set in the
    caller's environment. ``timeout`` is seconds per run; a run that overruns is killed,
    recorded with ``timed_out`` and graded on any claim written before. ``launch`` and
    ``probe`` replace Docker, for tests.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {list(ARMS)}")
    if repeats < 1 or timeout <= 0:
        raise ValueError("repeats must be at least 1 and timeout above 0")
    tasks_root, keys_root = Path(tasks_dir).resolve(), Path(keys_dir).resolve()
    results_path = Path(results).resolve()
    if not tasks_root.is_dir():
        raise ValueError(f"{tasks_root} is not a directory")
    if _inside(keys_root, tasks_root) or _inside(tasks_root, keys_root):
        raise ValueError("the keys directory and the tasks directory must not contain each other")
    if _inside(results_path, tasks_root):
        raise ValueError("the results file must not be inside the tasks directory")
    env_names = list(dict.fromkeys(env))
    for item in env_names:
        if not _NAME.match(item):
            raise ValueError(
                f"{item!r} is not an environment variable name; pass names, not values"
            )
        if item not in os.environ:
            raise ValueError(f"environment variable {item} is not set")
    ids = _discover(tasks_root, task_ids)
    keys = {}
    for ident in ids:
        check_task_dir(tasks_root / ident)
        key_path = keys_root / f"{ident}.json"
        if not key_path.is_file():
            raise ValueError(f"no key for task {ident} in {keys_root}")
        keys[ident] = load_key(key_path)
    done = {(r["task"], r["arm"], r["repeat"]) for r in read_results(results_path)}
    todo = [(i, n) for i in ids for n in range(repeats) if (i, arm, n) not in done]
    if not todo:
        return []

    facts = (probe or docker_probe)(image, shlex.split(agent_cmd)[0])
    labels = facts.get("labels") or {}
    if labels.get(LABEL_PREFIX + "arm") != arm:
        raise ValueError(
            f"{image} is labelled arm {labels.get(LABEL_PREFIX + 'arm')!r}, not {arm!r}"
        )
    runs_root = results_path.parent / f"{results_path.stem}.runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    launcher = launch or docker_launch
    records = []
    for ident, repeat in todo:
        task_dir = tasks_root / ident
        run_dir = Path(tempfile.mkdtemp(prefix=f"{ident}-{arm}-{repeat}-", dir=runs_root))
        out_dir = run_dir / "out"
        out_dir.mkdir()
        out_dir.chmod(0o777)
        transcript = out_dir / TRANSCRIPT_NAME
        transcript.touch()
        argv, mounts = docker_argv(
            name=f"wl-proving-{uuid.uuid4().hex[:12]}",
            image=image,
            task_dir=task_dir,
            out_dir=out_dir,
            command=shlex.split(agent_cmd),
            prompt=build_prompt(task_dir, timeout=timeout),
            env=env_names,
        )
        invocation = Invocation(
            name=argv[argv.index("--name") + 1],
            image=image,
            argv=tuple(argv),
            mounts=tuple(mounts),
            env=tuple(env_names),
            timeout=timeout,
            out_dir=out_dir,
            transcript=transcript,
        )
        started = datetime.now(UTC)
        clock = time.monotonic()
        outcome = launcher(invocation)
        seconds = time.monotonic() - clock
        score = _grade_claim(task_dir, keys[ident], out_dir, run_dir / "grade")
        shutil.rmtree(run_dir / "grade", ignore_errors=True)
        key = keys[ident]
        record = {
            "task": ident,
            "family": key["family"],
            "split": key["split"],
            "arm": arm,
            "repeat": repeat,
            "agent_cmd": agent_cmd,
            "agent_label": agent_label,
            "agent_version": facts.get("agent_version"),
            "image": image,
            "image_id": facts.get("image_id"),
            "lab_version": labels.get(LABEL_PREFIX + "version"),
            "revision": labels.get(LABEL_PREFIX + "revision"),
            "started": started.isoformat(),
            "seconds": round(seconds, 1),
            "exit_code": outcome.get("exit_code"),
            "timed_out": bool(outcome.get("timed_out")),
            "claim_valid": bool(score.get("valid")),
            "problems": score.get("problems", []),
            "score": score,
            "run_dir": str(run_dir),
        }
        with results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        records.append(record)
    return records
