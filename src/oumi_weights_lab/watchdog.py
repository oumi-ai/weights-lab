"""Independent worker deadlines, including escalation past blocked native code."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import sys
import time
from pathlib import Path

from .session import cancel_unfinished, read_json, session_lock, write_json


def _finalize_forced(path: Path, pid: int, reason: str) -> None:
    with (path / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        with session_lock(path):
            status = read_json(path / "status.json")
            if status.get("pid") != pid:
                return
            status.update(state="stopped", reason=reason, forced=True, updated_at=time.time())
            cancel_unfinished(path, f"worker deadline exceeded: {reason}")
            write_json(path / "status.json", status)


def watch(path: Path, pid: int, started: float) -> None:
    """Watch only our direct parent; a recycled PID can never become a target."""
    config = read_json(path / "config.json")
    idle_limit = config.get("idle_timeout")
    lifetime = config.get("max_lifetime")
    grace = config.get("shutdown_grace", 30.0)
    idle_since, previous_ready = None, None
    while os.getppid() == pid:
        now = time.monotonic()
        reason = "max_lifetime" if lifetime and now - started >= lifetime else None
        try:
            status = read_json(path / "status.json")
        except (OSError, ValueError):
            status = {}
        # The worker claims a snippet (inbox -> running) before it reports "running".
        queued = any((path / "inbox").glob("*.py")) or any((path / "running").glob("*.py"))
        ready = status.get("state") == "ready" and not queued
        if ready:
            if idle_since is None or status.get("updated_at") != previous_ready:
                idle_since = now
                previous_ready = status.get("updated_at")
            if idle_limit and now - idle_since >= idle_limit:
                reason = reason or "idle_timeout"
        else:
            idle_since, previous_ready = None, None
        if reason:
            try:
                write_json(path / "shutdown.json", {"reason": reason, "pid": pid})
            except (OSError, ValueError) as error:
                # Enforcing the deadline matters more than recording why.
                print(f"watchdog could not record {reason}: {error}", file=sys.stderr, flush=True)
            if os.getppid() != pid:
                return
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + grace
            while os.getppid() == pid and time.monotonic() < deadline:
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            if os.getppid() != pid:
                return
            os.kill(pid, signal.SIGKILL)
            while os.getppid() == pid:
                time.sleep(0.01)
            _finalize_forced(path, pid, reason)
            return
        time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--started", type=float, required=True)
    args = parser.parse_args()
    watch(args.session, args.pid, args.started)
