"""On-disk sessions and atomic snippet submission."""

from __future__ import annotations

import fcntl
import json
import math
import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DIRECTORIES = ("inbox", "running", "completed", "outputs", "results", "artifacts")


@contextmanager
def session_lock(path: Path) -> Iterator[None]:
    """Serialize queue publication with terminal session transitions."""
    with (path / ".session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Replace a JSON document without exposing partial writes to readers."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def create_session(path: Path, config: dict[str, Any]) -> Path:
    path = Path(path).expanduser().resolve()
    if not config.get("model"):
        raise ValueError("a model name or checkpoint path is required")
    config = {"snippet_timeout": 300.0, "poll_interval": 0.1, **config}
    for name in ("snippet_timeout", "poll_interval"):
        value = float(config[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        config[name] = value
    path.mkdir(parents=True, exist_ok=False)
    for name in DIRECTORIES:
        (path / name).mkdir()
    write_json(path / "config.json", config)
    write_json(path / "status.json", {"state": "starting", "model": config["model"]})
    return path


def submit(path: Path, code: str) -> str:
    path = Path(path).expanduser().resolve()
    with session_lock(path):
        return _submit(path, code)


def _submit(path: Path, code: str) -> str:
    state = read_json(path / "status.json")["state"]
    if state in {"stopped", "failed", "stopping"} or (path / "STOP").exists():
        raise ValueError(f"session is not accepting snippets (state: {state})")
    if not code.strip():
        raise ValueError("snippet is empty")
    snippet_id = f"{time.time_ns():020d}-{uuid.uuid4().hex[:12]}"
    fd, temporary = tempfile.mkstemp(prefix=".submit-", dir=path / "inbox")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(code)
        os.replace(temporary, path / "inbox" / f"{snippet_id}.py")
    finally:
        Path(temporary).unlink(missing_ok=True)
    return snippet_id


def is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A dead child can remain in the process table until its parent reaps it.
    # Such a worker cannot write files and must not block shutdown or backup.
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rpartition(") ")[2].split()[0] != "Z"
    except FileNotFoundError:
        pass
    state = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def cancel_unfinished(path: Path, reason: str) -> None:
    """Finalize unexecuted or interrupted sources after the worker stops.

    The caller must hold session_lock and own the worker lock or have confirmed
    the worker is dead. Never replay snippets: they may have already edited weights.
    """
    for folder in ("running", "inbox"):
        for source in sorted((path / folder).glob("*.py")):
            result = path / "results" / f"{source.stem}.json"
            if not result.exists():
                write_json(
                    result,
                    {"id": source.stem, "state": "cancelled", "error": reason},
                )
            source.rename(path / "completed" / source.name)
