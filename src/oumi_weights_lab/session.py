"""On-disk sessions and atomic snippet submission."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .budget import PassBudget

DIRECTORIES = ("inbox", "running", "completed", "outputs", "results", "artifacts")
DTYPES = ("auto", "float32", "float16", "bfloat16")
# How --device-map spreads a model over GPUs, as Transformers names the strategies.
DEVICE_MAPS = ("balanced", "sequential")
_MEMORY_SIZE = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:[KMG]iB|[KMG]B)")
_GPU_INDEX = re.compile(r"[0-9]+")
_MAX_MEMORY_FORMAT = "--max-memory takes GPU=SIZE pairs, such as 0=70GiB,1=70GiB"
# Worker status states: starting, loading, ready, running, stopping, stopped, failed.
TERMINAL_STATES = frozenset({"stopped", "failed"})
DEFAULT_SNIPPET_TIMEOUT = 300.0
DEFAULT_POLL_INTERVAL = 0.1
DEFAULT_SHUTDOWN_GRACE = 30.0
# Saving the deadline bundle hashes every weight, which takes minutes on a model
# spread over several GPUs.
MULTI_GPU_SHUTDOWN_GRACE = 300.0


@contextmanager
def session_lock(path: Path) -> Iterator[None]:
    """Serialize queue publication with terminal session transitions."""
    with (path / ".session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: dict[str, Any], *, durable: bool = False) -> None:
    """Replace a JSON document without exposing partial writes to readers.

    With ``durable``, the file and its directory entry reach disk before returning,
    so a held-out reservation survives a power loss that follows its evaluation.
    Other writes skip the flush, which is slow on network filesystems.
    """
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            if durable:
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        if durable:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def create_session(path: Path, config: dict[str, Any]) -> Path:
    path = Path(path).expanduser().resolve()
    if not config.get("model"):
        raise ValueError("a model name or checkpoint path is required")
    PassBudget(config.get("pass_limit"))
    validate_placement(config)
    config = validate_lifecycle(config)
    config = {
        "snippet_timeout": DEFAULT_SNIPPET_TIMEOUT,
        "poll_interval": DEFAULT_POLL_INTERVAL,
        **config,
    }
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


def parse_max_memory(text: str) -> dict[str, str]:
    """Read ``0=70GiB,1=70GiB`` as GPU indices (as strings, for JSON) mapped to sizes."""
    limits: dict[str, str] = {}
    for entry in text.split(","):
        index, separator, size = (part.strip() for part in entry.partition("="))
        if not separator or not _GPU_INDEX.fullmatch(index) or str(int(index)) in limits:
            raise ValueError(_MAX_MEMORY_FORMAT)
        limits[str(int(index))] = size
    _check_max_memory(limits)
    return limits


def _check_max_memory(limits: Any) -> None:
    if (
        not isinstance(limits, dict)
        or not limits
        or not all(isinstance(index, str) and _GPU_INDEX.fullmatch(index) for index in limits)
        or not all(
            isinstance(size, str) and _MEMORY_SIZE.fullmatch(size) for size in limits.values()
        )
    ):
        raise ValueError(_MAX_MEMORY_FORMAT)


def validate_placement(config: dict[str, Any]) -> None:
    """Check ``device_map`` and ``max_memory`` before a session or cluster exists."""
    device_map, max_memory = config.get("device_map"), config.get("max_memory")
    if device_map is None:
        if max_memory is not None:
            raise ValueError("--max-memory needs --device-map")
        return
    if device_map not in DEVICE_MAPS:
        raise ValueError("--device-map must be balanced or sequential")
    if str(config.get("device", "auto")) not in ("auto", "cuda"):
        raise ValueError("--device-map spreads a model over CUDA GPUs; use --device auto or cuda")
    if max_memory is not None:
        _check_max_memory(max_memory)


def validate_lifecycle(config: dict[str, Any]) -> dict[str, Any]:
    config = config.copy()
    grace = MULTI_GPU_SHUTDOWN_GRACE if config.get("device_map") else DEFAULT_SHUTDOWN_GRACE
    for name in ("idle_timeout", "max_lifetime", "shutdown_grace"):
        value = config.get(name)
        if value is None:
            if name != "shutdown_grace":
                continue
            value = grace
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite and nonnegative")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be finite and nonnegative") from None
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        config[name] = number
    return config


def submit(path: Path, code: str) -> str:
    path = Path(path).expanduser().resolve()
    with session_lock(path):
        return _submit(path, code)


def _submit(path: Path, code: str) -> str:
    state = read_json(path / "status.json")["state"]
    if state in TERMINAL_STATES | {"stopping"} or (path / "STOP").exists():
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
