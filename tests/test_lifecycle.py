import multiprocessing
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from oumi_weights_lab.session import create_session, read_json, submit, write_json
from oumi_weights_lab.worker import Worker


def run_worker(path, slow=False, real=False):
    def load(config, session_path):
        if slow:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(60)
        if real:
            import torch

            from oumi_weights_lab import Lab

            return Lab(torch.nn.Linear(2, 1), SimpleNamespace(), session_path, model_id="tiny")
        return SimpleNamespace(model=None, tokenizer=None, forward_passes=0)

    raise SystemExit(Worker(path, loader=load).run())


def wait_for(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.01)
    raise AssertionError("lifecycle deadline did not complete")


def launch(path, *, slow=False, real=False, **config):
    create_session(path, {"model": "fake", "poll_interval": 0.01, **config})
    process = multiprocessing.get_context("spawn").Process(
        target=run_worker, args=(path, slow, real)
    )
    process.start()
    return process


def finish(process):
    if process.is_alive():
        process.kill()
    process.join(3)


def test_idle_worker_exits_with_reason(tmp_path):
    path = tmp_path / "idle"
    process = launch(path, idle_timeout=0.2, shutdown_grace=1)
    try:
        process.join(5)
        assert process.exitcode == 0
        assert read_json(path / "status.json")["reason"] == "idle_timeout"
    finally:
        finish(process)


def test_idle_shutdown_saves_an_edit_bundle(tmp_path):
    from pathlib import Path

    path = tmp_path / "saved"
    process = launch(path, real=True, idle_timeout=0.2, shutdown_grace=2)
    try:
        task = submit(path, "lab.zero('weight')")
        process.join(7)
        assert process.exitcode == 0
        assert read_json(path / "results" / f"{task}.json")["state"] == "succeeded"
        status = read_json(path / "status.json")
        bundle = Path(status["edit_bundle"])
        assert bundle.is_relative_to(path / "artifacts")
        assert read_json(bundle / "manifest.json")["active"][0]["id"] == "edit-000001"
    finally:
        finish(process)


def test_idle_timer_ignores_active_work_then_restarts(tmp_path):
    path = tmp_path / "work"
    process = launch(path, idle_timeout=0.4, shutdown_grace=1)
    try:
        task = submit(path, "import time\ntime.sleep(0.7)")
        result = path / "results" / f"{task}.json"
        wait_for(result.exists)
        assert read_json(result)["state"] == "succeeded"
        # After completing a snippet, a fresh idle interval must be available.
        assert process.is_alive()
        task2 = submit(path, "(artifacts / 'second').touch()")
        wait_for((path / "results" / f"{task2}.json").exists)
        process.join(5)
        assert process.exitcode == 0
        assert (path / "artifacts/second").exists()
    finally:
        finish(process)


def test_max_lifetime_interrupts_running_work_and_cancels_queue(tmp_path):
    path = tmp_path / "maximum"
    process = launch(path, max_lifetime=0.7, shutdown_grace=1, snippet_timeout=30)
    try:
        task = submit(path, "while True:\n    pass")
        later = submit(path, "raise AssertionError('must not run')")
        process.join(5)
        assert process.exitcode == 0
        assert read_json(path / "status.json")["reason"] == "max_lifetime"
        assert read_json(path / "results" / f"{task}.json")["state"] == "cancelled"
        assert read_json(path / "results" / f"{later}.json")["state"] == "cancelled"
    finally:
        finish(process)


def test_independent_watchdog_kills_uninterruptible_loader(tmp_path):
    path = tmp_path / "blocked"
    process = launch(path, slow=True, max_lifetime=0.2, shutdown_grace=0.15)
    task = submit(path, "print('must not run')")
    try:
        process.join(5)
        assert process.exitcode == -signal.SIGKILL
        status = wait_for(lambda: read_json(path / "status.json").get("forced"))
        assert status is True
        status = read_json(path / "status.json")
        assert status["reason"] == "max_lifetime" and status["state"] == "stopped"
        assert read_json(path / "results" / f"{task}.json")["state"] == "cancelled"
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
    finally:
        finish(process)


def test_deadline_still_stops_worker_when_reason_cannot_be_recorded(tmp_path):
    path = tmp_path / "unrecorded"
    process = launch(path, idle_timeout=0.2, shutdown_grace=1)
    # A directory in its place makes the watchdog's shutdown.json write fail.
    (path / "shutdown.json").mkdir()
    try:
        process.join(5)
        assert process.exitcode == 0
        status = read_json(path / "status.json")
        assert status["state"] == "stopped"
        assert status["reason"] == "unknown"
    finally:
        finish(process)


def test_worker_stops_when_its_watchdog_dies(tmp_path):
    path = tmp_path / "unguarded"
    process = launch(path, idle_timeout=60, shutdown_grace=1)
    try:
        wait_for(lambda: read_json(path / "status.json").get("state") == "ready")
        guard = wait_for(
            lambda: subprocess.run(
                ["pgrep", "-P", str(process.pid), "-f", "oumi_weights_lab.watchdog"],
                capture_output=True,
                text=True,
            ).stdout.split()
        )
        os.kill(int(guard[0]), signal.SIGKILL)
        process.join(5)
        assert process.exitcode == 0
        status = read_json(path / "status.json")
        assert status["state"] == "stopped"
        assert status["reason"] == "watchdog_exited"
    finally:
        finish(process)


def watched_parent(path, terminated):
    """Stand in for a worker: run the watchdog and record whether it signals us."""
    signal.signal(signal.SIGTERM, lambda *_: (terminated.touch(), os._exit(0)))
    guard = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "oumi_weights_lab.watchdog",
            *("--session", str(path), "--pid", str(os.getpid())),
            *("--started", str(time.monotonic())),
        ]
    )
    try:
        time.sleep(1.0)
    finally:
        guard.kill()
        guard.wait()


def test_snippet_claimed_before_running_status_is_not_idle(tmp_path):
    path = tmp_path / "claimed"
    create_session(path, {"model": "fake", "idle_timeout": 0.2, "shutdown_grace": 1})
    # The worker moves a snippet to running/ before it writes "running" status.
    write_json(path / "status.json", {"state": "ready", "model": "fake", "updated_at": 1.0})
    (path / "running" / "00000000000000000001-claimed.py").write_text("pass")
    terminated = tmp_path / "terminated"
    process = multiprocessing.get_context("spawn").Process(
        target=watched_parent, args=(path, terminated)
    )
    process.start()
    process.join(5)
    assert process.exitcode == 0
    assert not terminated.exists()


@pytest.mark.parametrize("key", ["idle_timeout", "max_lifetime", "shutdown_grace"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "bad"])
def test_lifecycle_validation_precedes_directory_creation(tmp_path, key, value):
    path = tmp_path / "invalid"
    with pytest.raises(ValueError, match=key):
        create_session(path, {"model": "fake", key: value})
    assert not path.exists()
