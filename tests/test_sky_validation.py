import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    "validate_sky", Path(__file__).resolve().parents[1] / "scripts/validate_sky.py"
)
validate_sky = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_sky)


def test_watchdog_waits_for_absolute_deadline_then_retries_and_marks_cleanup(tmp_path):
    clock = SimpleNamespace(value=10)
    calls = []

    def sleep(seconds):
        clock.value += seconds

    def runner(args, **kwargs):
        calls.append((clock.value, args, kwargs))
        return subprocess.CompletedProcess(args, 1 if len(calls) == 1 else 0)

    marker = tmp_path / "done"
    assert (
        validate_sky.watchdog(
            "oumi-weights-lab-val-test",
            40,
            marker,
            runner=runner,
            now=lambda: clock.value,
            sleep=sleep,
        )
        == 0
    )
    assert calls[0][0] >= 40
    assert len(calls) == 2
    assert all(call[2]["timeout"] == 120 for call in calls)
    assert marker.exists()


def test_watchdog_exits_when_main_cleanup_marker_exists(tmp_path):
    marker = tmp_path / "done"
    marker.touch()
    runner = Mock(side_effect=AssertionError("already cleaned up"))
    assert (
        validate_sky.watchdog(
            "oumi-weights-lab-val-test", 100, marker, runner=runner, now=lambda: 0
        )
        == 0
    )
    runner.assert_not_called()


def test_watchdog_reports_failure_after_three_bounded_attempts(tmp_path):
    runner = Mock(side_effect=subprocess.TimeoutExpired(["sky", "down"], 120))
    marker = tmp_path / "done"
    assert (
        validate_sky.watchdog(
            "oumi-weights-lab-val-test", 0, marker, runner=runner, sleep=lambda _: None
        )
        == 1
    )
    assert runner.call_count == 3
    assert not marker.exists()


def test_cleanup_tears_down_even_if_stop_and_first_teardown_fail(tmp_path):
    launcher = SimpleNamespace(
        stop=Mock(side_effect=RuntimeError("worker unreachable")),
        teardown=Mock(
            side_effect=[
                RuntimeError("transient failure"),
                subprocess.CompletedProcess(["sky", "down"], 0),
            ]
        ),
    )
    report = {"state": "failed"}
    validate_sky.cleanup(launcher, tmp_path, report)
    assert report["cluster_terminated"] is True
    assert "worker unreachable" in report["stop_error"]
    assert launcher.teardown.call_count == 2
    assert (tmp_path / "cleanup.done").exists()


def test_cleanup_never_marks_success_after_failed_teardown(tmp_path):
    launcher = SimpleNamespace(stop=Mock(), teardown=Mock(side_effect=RuntimeError("no network")))
    report = {"state": "succeeded"}
    validate_sky.cleanup(launcher, tmp_path, report)
    assert report["state"] == "failed"
    assert not report.get("cluster_terminated")
    assert launcher.teardown.call_count == 3
    assert not (tmp_path / "cleanup.done").exists()


def test_logged_commands_append_live_logs_but_return_only_current_output(tmp_path):
    path = tmp_path / "live.log"
    first = validate_sky.run_logged([sys.executable, "-c", "print('first')"], path)
    second = validate_sky.run_logged([sys.executable, "-c", "print('second')"], path)
    assert first.stdout == "first\n" and second.stdout == "second\n"
    assert path.read_text() == "first\nsecond\n"


def test_partial_launch_failure_still_tears_down_and_records_failure(tmp_path, monkeypatch):
    events = []

    class Launcher:
        def __init__(self, *args, **kwargs):
            pass

        def start(self, *args, **kwargs):
            events.append("launch")
            raise RuntimeError("setup failed after allocation")

        def stop(self, *args, **kwargs):
            events.append("stop")
            raise RuntimeError("worker not started")

        def teardown(self):
            events.append("teardown")
            return subprocess.CompletedProcess(["sky", "down"], 0)

    monkeypatch.setattr(validate_sky, "SkyPilotLauncher", Launcher)
    monkeypatch.setattr(
        validate_sky.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "[]", ""),
    )

    def spawn(*args, **kwargs):
        assert kwargs["start_new_session"]
        events.append("watchdog")
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(validate_sky.subprocess, "Popen", spawn)
    root = tmp_path / "validation"
    with pytest.raises(RuntimeError, match="setup failed"):
        validate_sky.validate(root, max_seconds=60)
    assert events == ["watchdog", "launch", "stop", "teardown"]
    report = json.loads((root / "report.json").read_text())
    assert report["state"] == "failed" and report["cluster_terminated"]
    assert "setup failed" in report["error"]


def test_existing_cluster_is_rejected_without_launch_or_cleanup(tmp_path, monkeypatch):
    name = "oumi-weights-lab-val-existing"
    monkeypatch.setattr(
        validate_sky.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, json.dumps([{"name": name}]), ""
        ),
    )
    factory = Mock(side_effect=AssertionError("existing cluster must remain untouched"))
    monkeypatch.setattr(validate_sky, "SkyPilotLauncher", factory)
    monkeypatch.setattr(validate_sky.subprocess, "Popen", factory)
    with pytest.raises(ValueError, match="existing cluster"):
        validate_sky.validate(tmp_path / "validation", cluster=name)
    factory.assert_not_called()


def test_launch_task_has_fixed_resources_and_worker_deadline_before_submission(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        validate_sky.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "[]", ""),
    )
    monkeypatch.setattr(
        validate_sky.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=123)
    )
    submitted = []

    def logged(args, log, **kwargs):
        if args[:2] == ["sky", "launch"]:
            submitted.append(json.loads(Path(args[-1]).read_text()))
            return subprocess.CompletedProcess(args, 1, "provisioning failed", "")
        return subprocess.CompletedProcess(args, 0, "{}", "")

    monkeypatch.setattr(validate_sky, "run_logged", logged)
    with pytest.raises(subprocess.CalledProcessError):
        validate_sky.validate(tmp_path / "validation", max_seconds=120)
    assert len(submitted) == 1
    task = submitted[0]
    assert task["resources"]["cloud"] == "nebius"
    assert task["resources"]["accelerators"] == "H200:1"
    assert task["resources"]["disk_size"] == 256
    assert task["run"].startswith("timeout --signal=TERM --kill-after=30s 120 ")
    assert (tmp_path / "validation/cleanup.done").exists()


def test_short_ssh_json_survives_stderr_warning_and_initial_report_exists(tmp_path, monkeypatch):
    root = tmp_path / "validation"

    def process(args, **kwargs):
        if args[0] == "sky":
            return subprocess.CompletedProcess(args, 0, "[]", "")
        return subprocess.CompletedProcess(args, 0, '{"state":"succeeded"}', "SSH warning\n")

    class Launcher:
        def __init__(self, *args, runner, **kwargs):
            self.runner = runner

        def start(self, *args, **kwargs):
            initial = json.loads((root / "report.json").read_text())
            assert initial["state"] == "running" and initial["watchdog_pid"] == 123
            result = self.runner(
                ["ssh", "--", "cluster", "command"], text=True, capture_output=True, check=True
            )
            assert json.loads(result.stdout)["state"] == "succeeded"
            assert result.stderr == "SSH warning\n"
            raise RuntimeError("probe finished")

        def stop(self, *args, **kwargs):
            return None

        def teardown(self):
            return subprocess.CompletedProcess(["sky", "down"], 0)

    monkeypatch.setattr(validate_sky.subprocess, "run", process)
    monkeypatch.setattr(
        validate_sky.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=123)
    )
    monkeypatch.setattr(validate_sky, "SkyPilotLauncher", Launcher)
    with pytest.raises(RuntimeError, match="probe finished"):
        validate_sky.validate(root, max_seconds=60)
    assert '"state":"succeeded"' in (root / "transport.log").read_text()
    assert "SSH warning" in (root / "transport.log").read_text()
