"""One model, one Python namespace, and an ordered inbox of trusted snippets."""

from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .session import (
    DEFAULT_SHUTDOWN_GRACE,
    TERMINAL_STATES,
    cancel_unfinished,
    read_json,
    session_lock,
    write_json,
)


class _SnippetTimeout(BaseException):
    pass


class _WorkerStopped(BaseException):
    pass


@contextlib.contextmanager
def _capture(output: Any):
    """Capture Python streams and writes by native code or child processes."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(output.fileno(), 1)
        os.dup2(output.fileno(), 2)
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            yield
    finally:
        output.flush()
        for descriptor, original in zip((1, 2), saved, strict=True):
            os.dup2(original, descriptor)
            os.close(original)


class Worker:
    """Run a local session in the foreground. The CLI detaches this process."""

    def __init__(self, session_path: Path, loader: Callable | None = None):
        self.path = Path(session_path).expanduser().resolve()
        self.config = read_json(self.path / "config.json")
        self.loader = loader
        self.lab: Any = None
        self.namespace: dict[str, Any] = {}
        self._shutdown_reason: str | None = None
        self._deadline: threading.Timer | None = None

    def _status(self, state: str, **extra: Any) -> None:
        status = {
            "state": state,
            "pid": os.getpid(),
            "model": self.config["model"],
            "updated_at": time.time(),
            **extra,
        }
        with session_lock(self.path):
            if state in TERMINAL_STATES:
                cancel_unfinished(
                    self.path,
                    extra.get("error", extra.get("reason", "worker stopped before completion")),
                )
            write_json(self.path / "status.json", status)

    @staticmethod
    def _stop(signum: int, frame: Any) -> None:
        # One stop request is enough. Repeated signals must not interrupt the
        # result bookkeeping that follows the first one.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise _WorkerStopped()

    @staticmethod
    def _timeout(signum: int, frame: Any) -> None:
        raise _SnippetTimeout()

    def _execute(self, source: Path) -> None:
        snippet_id = source.stem
        running = self.path / "running" / source.name
        source.rename(running)
        started = time.time()
        clock_started = time.monotonic()
        before = getattr(self.lab, "forward_passes", 0)
        result = {"id": snippet_id, "state": "succeeded", "started_at": started}
        stopping = False
        self._status("running", current_snippet=snippet_id)
        with (
            (self.path / "outputs" / f"{snippet_id}.log").open(
                "w", encoding="utf-8", buffering=1
            ) as output,
            _capture(output),
        ):
            try:
                try:
                    signal.setitimer(signal.ITIMER_REAL, self.config["snippet_timeout"])
                    code = compile(running.read_text(encoding="utf-8"), str(running), "exec")
                    exec(code, self.namespace)
                finally:
                    # Disarm before any handler below runs. An alarm that lands after
                    # the snippet returns, or while its error is being reported, is
                    # caught here as a timeout instead of ending the worker.
                    signal.setitimer(signal.ITIMER_REAL, 0)
            except _SnippetTimeout:
                result.update(state="timed_out", error="snippet time limit exceeded")
                print(result["error"], flush=True)
            except _WorkerStopped:
                result.update(state="cancelled", error="worker stopped")
                stopping = True
            except BaseException as error:
                result.update(state="failed", error=f"{type(error).__name__}: {error}")
                traceback.print_exc()
        result.update(
            duration_seconds=time.monotonic() - clock_started,
            forward_passes=getattr(self.lab, "forward_passes", 0) - before,
        )
        write_json(self.path / "results" / f"{snippet_id}.json", result)
        running.rename(self.path / "completed" / running.name)
        if stopping:
            raise _WorkerStopped()
        self._status("ready")

    def run(self) -> int:
        """Serve until STOP or SIGTERM; return nonzero on model/startup failure.

        Snippets execute in this process with its filesystem and network access.
        Signals interrupt Python code; a blocked native operation may require the
        CLI's forced shutdown. This is a trusted workspace, not a sandbox.
        """
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Worker.run() must run in the main thread")
        with (self.path / ".worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("session already has a worker") from None
            if read_json(self.path / "status.json")["state"] != "starting":
                raise RuntimeError("session has already started; create a new session")
            previous = {
                signum: signal.getsignal(signum)
                for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)
            }
            signal.signal(signal.SIGTERM, self._stop)
            signal.signal(signal.SIGINT, self._stop)
            signal.signal(signal.SIGALRM, self._timeout)
            finished = threading.Event()
            guard = None

            def watch_stop() -> None:
                while not finished.wait(self.config["poll_interval"]):
                    if (self.path / "STOP").exists():
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
                    if guard is not None and guard.poll() is not None:
                        # Without its watchdog this worker has no idle or lifetime
                        # deadline, and a remote cluster would never autodown. Stand in
                        # for it: stop now, and force exit if native code, a snippet
                        # that swallows the stop, or the final save stalls past grace.
                        self._shutdown_reason = "watchdog_exited"
                        grace = self.config.get("shutdown_grace", DEFAULT_SHUTDOWN_GRACE)
                        self._deadline = threading.Timer(
                            grace, os.kill, (os.getpid(), signal.SIGKILL)
                        )
                        self._deadline.daemon = True
                        self._deadline.start()
                        os.kill(os.getpid(), signal.SIGTERM)
                        return

            watcher = threading.Thread(target=watch_stop, daemon=True)
            exit_code = 0
            final_error = None
            try:
                if (self.path / "STOP").exists():
                    self._status("stopped")
                    return 0
                self._status("loading")
                if self.config.get("idle_timeout") or self.config.get("max_lifetime"):
                    guard = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "oumi_weights_lab.watchdog",
                            "--session",
                            str(self.path),
                            "--pid",
                            str(os.getpid()),
                            "--started",
                            str(time.monotonic()),
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        # Inherit the worker's stderr (worker.log) so failures are visible.
                    )
                watcher.start()
                if self.loader is None:
                    from .lab import load_lab

                    self.loader = load_lab
                self.lab = self.loader(self.config, self.path)
                self.namespace = {
                    "__name__": "__oumi_weights_lab__",
                    "lab": self.lab,
                    "model": self.lab.model,
                    "tokenizer": self.lab.tokenizer,
                    "Path": Path,
                    "artifacts": self.path / "artifacts",
                }
                self._status("ready")
                while not (self.path / "STOP").exists():
                    sources = sorted((self.path / "inbox").glob("*.py"))
                    if sources:
                        self._execute(sources[0])
                    else:
                        time.sleep(self.config["poll_interval"])
            except _WorkerStopped:
                pass
            except BaseException as error:
                final_error = f"{type(error).__name__}: {error}"
                traceback.print_exc()
                exit_code = 1
            finally:
                # A watcher may already be sending a stop signal when the main
                # loop sees STOP. Suppress it before joining or saving results.
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                signal.signal(signal.SIGALRM, signal.SIG_IGN)
                finished.set()
                if watcher.ident is not None:
                    watcher.join()
                signal.setitimer(signal.ITIMER_REAL, 0)
                extra = {}
                shutdown = self.path / "shutdown.json"
                if shutdown.exists():
                    try:
                        extra = {"reason": read_json(shutdown)["reason"]}
                    except (OSError, ValueError, KeyError, TypeError):
                        extra = {"reason": "unknown"}
                elif self._shutdown_reason is not None:
                    extra = {"reason": self._shutdown_reason}
                if extra:
                    self._status("stopping", **extra)
                    save = getattr(self.lab, "save_edits", None)
                    if callable(save):
                        try:
                            extra["edit_bundle"] = str(save(f"shutdown-{time.time_ns()}"))
                        except Exception as error:
                            extra["checkpoint_error"] = f"{type(error).__name__}: {error}"
                try:
                    if final_error is None:
                        self._status("stopped", **extra)
                    else:
                        self._status("failed", error=final_error, **extra)
                finally:
                    if guard is not None:
                        guard.terminate()
                        guard.wait()
                    if self._deadline is not None:
                        self._deadline.cancel()
                    for signum, handler in previous.items():
                        signal.signal(signum, handler)
            return exit_code
