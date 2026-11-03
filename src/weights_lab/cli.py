"""Local session commands. Model dependencies are imported only by the worker."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from . import session

TERMINAL_STATES = {"stopped", "failed", "timed_out", "cancelled", "succeeded"}


def _json(value: dict) -> None:
    print(json.dumps(value), flush=True)


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def _status(path: Path) -> dict:
    status = session.read_json(path / "status.json")
    if _pid(status) is None:
        try:
            status["pid"] = int((path / "worker.pid").read_text(encoding="ascii"))
        except (OSError, ValueError):
            pass
    return status


def _pid(status: dict) -> int | None:
    pid = status.get("pid")
    return pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else None


def _alive(status: dict) -> bool:
    pid = _pid(status)
    return pid is not None and session.is_process_alive(pid)


def _stream(path: Path, offset: int) -> int:
    """Stream new bytes, including partial lines, without waiting for a newline."""
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
            offset = handle.tell()
    except FileNotFoundError:
        return offset
    if chunk:
        # Preserve UTF-8 bytes split across reads when a binary stream is available.
        if hasattr(sys.stderr, "buffer"):
            sys.stderr.flush()
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()
        else:
            sys.stderr.write(chunk.decode("utf-8", errors="replace"))
            sys.stderr.flush()
    return offset


def _wait_result(path: Path, snippet_id: str, timeout: float | None, follow: bool) -> dict:
    deadline = None if timeout is None else time.monotonic() + timeout
    offset = 0
    result_path = path / "results" / f"{snippet_id}.json"
    while True:
        if follow:
            offset = _stream(path / "outputs" / f"{snippet_id}.log", offset)
        if result_path.exists():
            result = session.read_json(result_path)
            if follow:
                _stream(path / "outputs" / f"{snippet_id}.log", offset)
            return result
        status = _status(path)
        if status.get("state") in TERMINAL_STATES or (_pid(status) and not _alive(status)):
            raise RuntimeError(f"worker stopped before snippet {snippet_id} produced a result")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for snippet {snippet_id}; it may still be running"
            )
        time.sleep(0.1)


def _start(args: argparse.Namespace) -> int:
    path = args.session.resolve()
    config = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "snippet_timeout": args.snippet_timeout,
        "poll_interval": 0.1,
    }
    if args.adapter:
        config["adapter"] = args.adapter
    session.create_session(path, config)
    try:
        with (path / "worker.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "weights_lab", "_worker", "--session", str(path)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as exc:
        with session.session_lock(path):
            session.write_json(
                path / "status.json", {"state": "failed", "model": args.model, "error": str(exc)}
            )
            session.cancel_unfinished(path, f"worker could not start: {exc}")
        raise
    (path / "worker.pid").write_text(str(process.pid), encoding="ascii")
    _json({"session": str(path), "pid": process.pid, "state": "starting"})
    return 0


def _send(args: argparse.Namespace) -> int:
    path = args.session.resolve()
    if args.file is not None:
        code = args.file.read_text(encoding="utf-8")
    elif args.code is not None:
        code = args.code
    else:
        code = sys.stdin.read()
    if not code.strip():
        raise ValueError("snippet is empty")
    snippet_id = session.submit(path, code)
    if args.wait or args.follow:
        result = _wait_result(path, snippet_id, args.timeout, args.follow)
        _json(result)
        return 0 if result.get("state") == "succeeded" else 1
    _json({"id": snippet_id, "state": "queued"})
    return 0


def _snippet_id(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in value
        )
    ):
        raise argparse.ArgumentTypeError(
            "snippet ID must contain only letters, digits, underscores, or hyphens"
        )
    return value


def _peek(args: argparse.Namespace) -> int:
    path = args.session.resolve()
    if args.id:
        result_path = path / "results" / f"{args.id}.json"
        if not result_path.exists() and not any(
            (path / folder / f"{args.id}.py").exists()
            for folder in ("inbox", "running", "completed")
        ):
            raise ValueError(f"unknown snippet ID: {args.id}")
        if args.follow:
            result = _wait_result(path, args.id, None, True)
        else:
            result = (
                session.read_json(result_path)
                if result_path.exists()
                else {"id": args.id, "state": "pending"}
            )
            _stream(path / "outputs" / f"{args.id}.log", 0)
        _json(result)
        return 0
    if not args.follow:
        _json(_status(path))
        return 0
    previous = None
    offset = 0
    while True:
        status = _status(path)
        if status != previous:
            _json(status)
            previous = status
        offset = _stream(path / "worker.log", offset)
        if status.get("state") in TERMINAL_STATES or (_pid(status) and not _alive(status)):
            return 0
        time.sleep(0.1)


def _matches_worker(pid: int, path: Path) -> bool:
    """Refuse escalation if a stale PID has been reused by another process."""
    command = subprocess.run(
        ["ps", "-p", str(pid), "-o", "args="],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return command.endswith(f"-m weights_lab _worker --session {path}")


def _stop(args: argparse.Namespace) -> int:
    path = args.session.resolve()
    status = _status(path)
    (path / "STOP").touch()
    deadline = time.monotonic() + args.timeout
    while _alive(status):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
        status = _status(path)
    forced = False
    if _alive(status):
        pid = _pid(status)
        assert pid is not None
        if not _matches_worker(pid, path):
            raise RuntimeError(
                f"refusing to signal PID {pid}: its command does not match this worker"
            )
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 2
            while session.is_process_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            if session.is_process_alive(pid) and _matches_worker(pid, path):
                os.kill(pid, signal.SIGKILL)
                forced = True
                deadline = time.monotonic() + 2
                while session.is_process_alive(pid) and time.monotonic() < deadline:
                    time.sleep(0.1)
        except ProcessLookupError:
            pass
        status = _status(path)
    if _alive(status):
        raise RuntimeError("worker is still alive after shutdown")
    with session.session_lock(path):
        if forced or status.get("state") not in TERMINAL_STATES:
            status.update(state="stopped", forced=forced, updated_at=time.time())
            status.pop("current_snippet", None)
            session.write_json(path / "status.json", status)
        session.cancel_unfinished(path, "worker stopped before completion")
    _json(status)
    return 0


def _backup(args: argparse.Namespace) -> int:
    path = args.session.resolve()
    output = args.output.resolve()
    if output == path or path in output.parents:
        raise ValueError("backup output must be outside the session directory")
    status = _status(path)
    if _alive(status) or status.get("state") not in TERMINAL_STATES:
        raise ValueError("stop the worker before creating a backup")
    names = {
        "config.json",
        "status.json",
        "worker.log",
        "inbox",
        "running",
        "completed",
        "outputs",
        "results",
        "artifacts",
    }
    sources = [child for child in path.iterdir() if child.name in names]
    for source in sources:
        if source.is_symlink() or (
            source.is_dir() and any(child.is_symlink() for child in source.rglob("*"))
        ):
            raise ValueError("session backups cannot contain symbolic links")
    with output.open("xb") as destination:
        with tarfile.open(fileobj=destination, mode="w:gz") as archive:
            for source in sorted(sources):
                archive.add(source, arcname=f"{path.name}/{source.name}")
    _json({"session": str(path), "output": str(output)})
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weights-lab",
        description="Keep a local model loaded and run Python snippets against it.",
    )
    commands = parser.add_subparsers(
        dest="command", required=True, metavar="{start,send,peek,stop,backup}"
    )
    start = commands.add_parser("start", help="launch a detached local worker")
    start.add_argument("--session", required=True, type=Path)
    start.add_argument("--model", required=True)
    start.add_argument("--adapter")
    start.add_argument("--device", default="auto")
    start.add_argument("--dtype", default="auto")
    start.add_argument("--snippet-timeout", type=_positive, default=300.0)
    start.set_defaults(handler=_start)
    send = commands.add_parser("send", help="submit Python from a file, an argument, or stdin")
    send.add_argument("--session", required=True, type=Path)
    code = send.add_mutually_exclusive_group()
    code.add_argument("--file", type=Path)
    code.add_argument("--code")
    send.add_argument("--wait", action="store_true")
    send.add_argument(
        "--follow", action="store_true", help="stream output to stderr and wait for completion"
    )
    send.add_argument("--timeout", type=_positive, default=300.0)
    send.set_defaults(handler=_send)
    peek = commands.add_parser("peek", help="inspect worker status or a snippet result")
    peek.add_argument("--session", required=True, type=Path)
    peek.add_argument("--id", type=_snippet_id)
    peek.add_argument("--follow", action="store_true")
    peek.set_defaults(handler=_peek)
    stop = commands.add_parser("stop", help="stop the worker, escalating after the timeout")
    stop.add_argument("--session", required=True, type=Path)
    stop.add_argument("--timeout", type=_positive, default=10.0)
    stop.set_defaults(handler=_stop)
    backup = commands.add_parser(
        "backup", help="archive records and artifacts from a stopped session"
    )
    backup.add_argument("--session", required=True, type=Path)
    backup.add_argument("--output", required=True, type=Path)
    backup.set_defaults(handler=_backup)
    worker = commands.add_parser("_worker")
    worker.add_argument("--session", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "_worker":
            from .worker import Worker

            return Worker(args.session.resolve()).run()
        return args.handler(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"weights-lab: {exc}", file=sys.stderr)
        return 1
