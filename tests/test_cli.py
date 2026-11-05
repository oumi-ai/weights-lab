import json
import os
import subprocess
import sys
import tarfile

import pytest

from oumi_weights_lab import cli, session
from oumi_weights_lab.cli import main
from oumi_weights_lab.session import create_session, read_json, submit, write_json


@pytest.fixture
def session_path(tmp_path):
    path = tmp_path / "session"
    create_session(path, {"model": "local-model"})
    return path


def test_send_file_and_peek_result(session_path, tmp_path, capsys):
    source = tmp_path / "snippet.py"
    source.write_text("print('hello')\n")
    assert main(["send", "--session", str(session_path), "--file", str(source)]) == 0
    queued = json.loads(capsys.readouterr().out)
    snippet_id = queued["id"]
    assert queued["state"] == "queued"
    assert (session_path / "inbox" / f"{snippet_id}.py").read_text() == source.read_text()

    result = {"id": snippet_id, "state": "succeeded", "duration": 0.1}
    write_json(session_path / "results" / f"{snippet_id}.json", result)
    (session_path / "outputs" / f"{snippet_id}.log").write_text("hello\n")
    assert main(["peek", "--session", str(session_path), "--id", snippet_id]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == result
    assert captured.err == "hello\n"


def test_send_rejects_empty_code(session_path, capsys):
    assert main(["send", "--session", str(session_path), "--code", "  \n"]) != 0
    assert "empty" in capsys.readouterr().err
    assert not list((session_path / "inbox").iterdir())


def test_backup_retains_source_results_and_artifacts(session_path, tmp_path, capsys):
    write_json(session_path / "status.json", {"state": "stopped"})
    (session_path / "completed" / "example.py").write_text("print(1)")
    write_json(session_path / "results" / "example.json", {"state": "succeeded"})
    (session_path / "artifacts" / "weights.bin").write_bytes(b"weights")
    destination = tmp_path / "session.tar.gz"
    assert main(["backup", "--session", str(session_path), "--output", str(destination)]) == 0
    capsys.readouterr()
    with tarfile.open(destination) as archive:
        assert archive.extractfile("session/completed/example.py").read() == b"print(1)"
        assert archive.extractfile("session/artifacts/weights.bin").read() == b"weights"
        assert "session/results/example.json" in archive.getnames()


def test_backup_refuses_output_inside_session(session_path, capsys):
    write_json(session_path / "status.json", {"state": "stopped"})
    destination = session_path / "backup.tar.gz"
    assert main(["backup", "--session", str(session_path), "--output", str(destination)]) != 0
    assert "outside" in capsys.readouterr().err
    assert not destination.exists()


def test_start_records_launch_failure(tmp_path, monkeypatch, capsys):
    path = tmp_path / "failed-session"

    def fail_to_launch(*args, **kwargs):
        raise OSError("process launch failed")

    monkeypatch.setattr("oumi_weights_lab.cli.subprocess.Popen", fail_to_launch)
    assert main(["start", "--session", str(path), "--model", "local-model"]) != 0
    assert "process launch failed" in capsys.readouterr().err
    status = read_json(path / "status.json")
    assert status["state"] == "failed"
    assert "process launch failed" in status["error"]
    assert not (path / "worker.pid").exists()


def test_stop_finalizes_claimed_and_pending_snippets_after_worker_dies(session_path, capsys):
    running_id = submit(session_path, "print('interrupted')")
    pending_id = submit(session_path, "print('pending')")
    source = session_path / "inbox" / f"{running_id}.py"
    source.rename(session_path / "running" / source.name)
    with subprocess.Popen([sys.executable, "-c", "pass"]) as process:
        process.wait(timeout=5)
        dead_pid = process.pid
    write_json(
        session_path / "status.json",
        {"state": "running", "pid": dead_pid, "current_snippet": running_id},
    )

    assert main(["stop", "--session", str(session_path), "--timeout", "0.1"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "stopped"
    assert "current_snippet" not in read_json(session_path / "status.json")
    for snippet_id in (running_id, pending_id):
        assert read_json(session_path / "results" / f"{snippet_id}.json")["state"] == "cancelled"
        assert (session_path / "completed" / f"{snippet_id}.py").exists()
    assert not list((session_path / "running").glob("*.py"))
    assert not list((session_path / "inbox").glob("*.py"))


def test_backup_refuses_live_process_even_with_stopped_status(session_path, tmp_path, capsys):
    write_json(session_path / "status.json", {"state": "stopped", "pid": os.getpid()})
    destination = tmp_path / "backup.tar.gz"
    assert main(["backup", "--session", str(session_path), "--output", str(destination)]) != 0
    assert "stop the worker" in capsys.readouterr().err
    assert not destination.exists()


def test_backup_refuses_symlink_to_external_file(session_path, tmp_path, capsys):
    write_json(session_path / "status.json", {"state": "stopped"})
    external = tmp_path / "external.txt"
    external.write_text("outside the session")
    (session_path / "artifacts" / "linked.txt").symlink_to(external)
    destination = tmp_path / "backup.tar.gz"
    assert main(["backup", "--session", str(session_path), "--output", str(destination)]) != 0
    assert "symbolic links" in capsys.readouterr().err
    assert not destination.exists()


def test_wait_result_rechecks_for_a_result_after_seeing_a_terminal_status(
    session_path, monkeypatch
):
    snippet_id = submit(session_path, "print(1)")
    result = {"id": snippet_id, "state": "cancelled", "error": "worker stopped"}

    def racing_status(path):
        # The worker finalizes this snippet and exits between the two checks.
        write_json(path / "results" / f"{snippet_id}.json", result)
        return {"state": "stopped"}

    monkeypatch.setattr(cli, "_status", racing_status)
    assert cli._wait_result(session_path, snippet_id, timeout=5, follow=False) == result


def test_stop_records_cancelled_results_before_the_terminal_status(
    session_path, monkeypatch, capsys
):
    snippet_id = submit(session_path, "print('pending')")
    write = cli.session.write_json

    def checked_write(path, value):
        if path.name == "status.json" and value.get("state") == "stopped":
            assert (session_path / "results" / f"{snippet_id}.json").exists()
        write(path, value)

    monkeypatch.setattr(cli.session, "write_json", checked_write)
    assert main(["stop", "--session", str(session_path), "--timeout", "0.1"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "stopped"


def test_stop_keeps_the_final_status_a_worker_writes_as_it_exits(session_path, monkeypatch, capsys):
    write_json(session_path / "status.json", {"state": "stopping", "pid": 4242})
    final = {"state": "stopped", "reason": "idle_timeout", "edit_bundle": "/bundles/saved"}
    checks = []

    def racing_alive(pid):
        # The worker writes its final status and exits after stop last read status.json.
        checks.append(pid)
        if len(checks) == 2:
            write_json(session_path / "status.json", final)
        return len(checks) < 2

    monkeypatch.setattr(cli.session, "is_process_alive", racing_alive)
    assert main(["stop", "--session", str(session_path), "--timeout", "5"]) == 0
    assert json.loads(capsys.readouterr().out) == final
    assert read_json(session_path / "status.json") == final


def test_terminal_states_are_the_worker_status_terminal_states():
    # status.json holds only worker states; snippet result states never appear there.
    assert cli.TERMINAL_STATES == {"stopped", "failed"}


def test_start_parser_defaults_use_shared_session_constants():
    parser = cli._parser()
    start = ["start", "--session", "s", "--model", "m"]
    args = parser.parse_args(start)
    assert args.snippet_timeout == session.DEFAULT_SNIPPET_TIMEOUT
    # Unset, so create_session can choose the default from --device-map.
    assert args.shutdown_grace is None
    for dtype in session.DTYPES:
        assert parser.parse_args([*start, "--dtype", dtype]).dtype == dtype


def test_start_puts_quantize_and_hub_kernels_in_the_config(tmp_path, monkeypatch):
    seen = []
    real = session.create_session

    def create(path, config):
        seen.append(config)
        return real(path, config)

    class Process:
        pid = 1

    monkeypatch.setattr(session, "create_session", create)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: Process())
    base = ["start", "--model", "m"]
    assert main([*base, "--session", str(tmp_path / "a")]) == 0
    flags = ["--quantize", "fp4", "--allow-hub-kernels"]
    assert main([*base, "--session", str(tmp_path / "b"), *flags]) == 0
    assert (seen[0]["quantize"], seen[0]["allow_hub_kernels"]) == (None, False)
    assert (seen[1]["quantize"], seen[1]["allow_hub_kernels"]) == ("fp4", True)
    assert read_json(tmp_path / "b" / "config.json")["quantize"] == "fp4"


def test_start_refuses_unknown_quantize_and_device_map_with_quantize(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["start", "--session", str(tmp_path / "a"), "--model", "m", "--quantize", "nf8"])
    capsys.readouterr()
    arguments = ["start", "--session", str(tmp_path / "b"), "--model", "m", "--quantize", "nf4"]
    assert main([*arguments, "--device-map", "balanced"]) != 0
    assert "cannot be combined" in capsys.readouterr().err
    assert not (tmp_path / "b").exists()
