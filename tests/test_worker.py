import multiprocessing
import os
import signal
import threading
import time
from types import SimpleNamespace

import pytest

from oumi_weights_lab.session import create_session, read_json, submit
from oumi_weights_lab.worker import Worker


def fake_loader(config, path):
    return SimpleNamespace(
        model=SimpleNamespace(name="fake-model"),
        tokenizer=SimpleNamespace(name="fake-tokenizer"),
        artifact_dir=path / "artifacts",
        forward_passes=0,
        device_memory={"cuda:0": 1.5, "cuda:1": 1.25},
    )


def run_worker(path):
    raise SystemExit(Worker(path, loader=fake_loader).run())


def run_worker_with_stop_signal_during_join(path):
    """Make the watcher deliver SIGTERM exactly when cleanup joins it."""
    import oumi_weights_lab.worker as worker_module

    signal_pending = threading.Event()
    cleanup_started = threading.Event()
    original_kill = os.kill
    original_join = threading.Thread.join

    def delayed_kill(pid, signum):
        signal_pending.set()
        assert cleanup_started.wait(5), "worker never reached watcher cleanup"
        original_kill(pid, signum)

    def join_during_cleanup(thread, *args, **kwargs):
        cleanup_started.set()
        return original_join(thread, *args, **kwargs)

    def stop_during_loading(config, session_path):
        (session_path / "STOP").touch()
        assert signal_pending.wait(5), "watcher never noticed STOP"
        return fake_loader(config, session_path)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(worker_module.os, "kill", delayed_kill)
        patch.setattr(worker_module.threading.Thread, "join", join_during_cleanup)
        result = Worker(path, loader=stop_during_loading).run()
    raise SystemExit(result)


def failing_loader(config, path):
    raise ValueError("model cannot load")


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for worker")


@pytest.fixture
def worker_session(tmp_path, request):
    path = tmp_path / "session"
    create_session(
        path,
        {"model": "fake", "snippet_timeout": getattr(request, "param", 0.2), "poll_interval": 0.01},
    )
    process = multiprocessing.get_context("spawn").Process(target=run_worker, args=(path,))
    process.start()
    try:
        wait_for(lambda: read_json(path / "status.json").get("state") == "ready")
        yield path
    finally:
        (path / "STOP").touch()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert process.exitcode == 0


def result_for(path, snippet_id):
    result_path = path / "results" / f"{snippet_id}.json"
    wait_for(result_path.exists)
    return read_json(result_path)


def test_status_reports_parameter_memory_per_device_once_loaded(worker_session):
    status = read_json(worker_session / "status.json")
    assert status["devices"] == {"cuda:0": 1.5, "cuda:1": 1.25}
    snippet = submit(worker_session, "print('hi')")
    result_for(worker_session, snippet)
    assert read_json(worker_session / "status.json")["devices"] == status["devices"]


def test_worker_keeps_namespace_and_captures_both_output_streams(worker_session):
    path = worker_session
    first = submit(path, "answer = 40\nprint(model.name, tokenizer.name)\n")
    second = submit(
        path,
        "import sys\nanswer += 2\nprint(answer)\n"
        "print('warning', file=sys.stderr)\n"
        "(artifacts / 'answer.txt').write_text(str(answer))\n",
    )

    for snippet_id in (first, second):
        result = result_for(path, snippet_id)
        assert result["state"] == "succeeded"
        assert result["id"] == snippet_id
        assert result["duration_seconds"] >= 0
        assert (path / "completed" / f"{snippet_id}.py").exists()
        assert not (path / "running" / f"{snippet_id}.py").exists()
    assert "fake-model fake-tokenizer" in (path / "outputs" / f"{first}.log").read_text()
    output = (path / "outputs" / f"{second}.log").read_text()
    assert "42" in output and "warning" in output
    assert (path / "artifacts" / "answer.txt").read_text() == "42"


def test_runtime_and_syntax_errors_do_not_kill_worker(worker_session):
    path = worker_session
    bad_runtime = submit(path, "saved = 17\nraise ValueError('deliberate failure')")
    bad_syntax = submit(path, "if !!!")
    good = submit(path, "print(saved + 1)")

    for snippet_id, expected in ((bad_runtime, "ValueError"), (bad_syntax, "SyntaxError")):
        result = result_for(path, snippet_id)
        assert result["state"] == "failed"
        assert expected in result["error"]
    assert result_for(path, good)["state"] == "succeeded"
    assert "18" in (path / "outputs" / f"{good}.log").read_text()


def test_native_and_subprocess_output_are_captured_in_each_snippet_log(worker_session):
    path = worker_session
    first = submit(
        path,
        "import os, subprocess, sys\n"
        "os.write(1, b'native-out\\n')\n"
        "os.write(2, b'native-err\\n')\n"
        "subprocess.run([sys.executable, '-c', \"print('child-out')\"], check=True)\n",
    )
    assert result_for(path, first)["state"] == "succeeded"
    second = submit(path, "print('next-snippet')\nos.write(2, b'next-native\\n')")
    assert result_for(path, second)["state"] == "succeeded"
    first_output = (path / "outputs" / f"{first}.log").read_text()
    assert "native-out" in first_output
    assert "native-err" in first_output
    assert "child-out" in first_output
    assert "next-snippet" not in first_output
    second_output = (path / "outputs" / f"{second}.log").read_text()
    assert "next-snippet" in second_output
    assert "next-native" in second_output
    assert "child-out" not in second_output


def test_timeout_interrupts_snippet_and_worker_accepts_next(worker_session):
    path = worker_session
    slow = submit(path, "saved_before_timeout = 9\nwhile True:\n    pass\n")
    good = submit(path, "print(saved_before_timeout)")
    assert result_for(path, slow)["state"] == "timed_out"
    assert result_for(path, good)["state"] == "succeeded"
    assert "9" in (path / "outputs" / f"{good}.log").read_text()


@pytest.mark.parametrize("worker_session", [0.3], indirect=True)
def test_deadline_while_reporting_an_error_does_not_end_worker(worker_session):
    path = worker_session
    # The snippet raises immediately, but rendering its error outlasts the deadline.
    # The alarm must not fire inside the worker's own error handling.
    slow_error = submit(
        path,
        "import time\n"
        "class SlowError(Exception):\n"
        "    def __str__(self):\n"
        "        time.sleep(0.6)\n"
        "        return 'slow to render'\n"
        "raise SlowError()\n",
    )
    good = submit(path, "print('still serving')")
    result = result_for(path, slow_error)
    assert result["state"] == "failed"
    assert "slow to render" in result["error"]
    assert result_for(path, good)["state"] == "succeeded"
    # The result is written just before the worker reports ready again.
    wait_for(lambda: read_json(path / "status.json")["state"] == "ready")


def test_model_load_failure_records_terminal_status(tmp_path):
    path = tmp_path / "session"
    create_session(path, {"model": "missing"})
    snippet_id = submit(path, "print('cannot run')")
    assert Worker(path, loader=failing_loader).run() == 1
    status = read_json(path / "status.json")
    assert status["state"] == "failed"
    assert "model cannot load" in status["error"]
    assert result_for(path, snippet_id)["state"] == "cancelled"
    with pytest.raises(ValueError):
        submit(path, "print('cannot run')")
    with pytest.raises(RuntimeError, match="already started"):
        Worker(path, loader=fake_loader).run()
    assert read_json(path / "status.json")["state"] == "failed"


def test_sigterm_cancels_running_snippet(worker_session):
    path = worker_session
    snippet_id = submit(path, "print('started', flush=True)\nwhile True:\n    pass\n")
    log = path / "outputs" / f"{snippet_id}.log"
    wait_for(lambda: log.exists() and "started" in log.read_text())
    os.kill(read_json(path / "status.json")["pid"], signal.SIGTERM)
    assert result_for(path, snippet_id)["state"] == "cancelled"
    wait_for(lambda: read_json(path / "status.json")["state"] == "stopped")


@pytest.mark.parametrize("worker_session", [5.0], indirect=True)
def test_stop_file_interrupts_active_snippet_and_cancels_queue(worker_session):
    path = worker_session
    running_id = submit(path, "print('started', flush=True)\nwhile True:\n    pass\n")
    log = path / "outputs" / f"{running_id}.log"
    wait_for(lambda: log.exists() and "started" in log.read_text())
    queued_id = submit(path, "(artifacts / 'should-not-exist').touch()")
    (path / "STOP").touch()

    assert result_for(path, running_id)["state"] == "cancelled"
    assert result_for(path, queued_id)["state"] == "cancelled"
    wait_for(lambda: read_json(path / "status.json")["state"] == "stopped")
    assert not (path / "artifacts" / "should-not-exist").exists()
    assert not list((path / "inbox").glob("*.py"))
    assert not list((path / "running").glob("*.py"))


def test_delayed_watcher_signal_does_not_interrupt_terminal_bookkeeping(tmp_path):
    path = tmp_path / "session"
    create_session(path, {"model": "fake", "poll_interval": 0.01})
    snippet_id = submit(path, "(artifacts / 'should-not-exist').touch()")
    process = multiprocessing.get_context("spawn").Process(
        target=run_worker_with_stop_signal_during_join, args=(path,)
    )
    process.start()
    try:
        process.join(10)
        assert not process.is_alive(), "worker hung while joining STOP watcher"
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
    assert read_json(path / "status.json")["state"] == "stopped"
    assert result_for(path, snippet_id)["state"] == "cancelled"
    assert (path / "completed" / f"{snippet_id}.py").exists()
    assert not (path / "artifacts" / "should-not-exist").exists()
    assert not list((path / "inbox").glob("*.py"))
