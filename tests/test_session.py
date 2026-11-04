from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from oumi_weights_lab.session import (
    create_session,
    is_process_alive,
    read_json,
    session_lock,
    submit,
    write_json,
)


def test_create_session_persists_config_and_queue_directories(tmp_path):
    path = tmp_path / "session"
    config = {"model": "local-model", "snippet_timeout": 1.0}
    create_session(path, config)

    persisted = read_json(path / "config.json")
    assert all(persisted[key] == value for key, value in config.items())
    for name in ("inbox", "running", "completed", "outputs", "results", "artifacts"):
        assert (path / name).is_dir()
    with pytest.raises(FileExistsError):
        create_session(path, config)


def test_concurrent_submissions_are_unique_and_complete(tmp_path):
    path = tmp_path / "session"
    create_session(path, {"model": "local-model"})
    codes = [f"value = {number}\n" for number in range(32)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = list(executor.map(lambda code: submit(path, code), codes))

    assert len(set(ids)) == len(codes)
    assert sorted(item.name for item in (path / "inbox").iterdir()) == sorted(
        f"{snippet_id}.py" for snippet_id in ids
    )
    for snippet_id, code in zip(ids, codes, strict=True):
        assert (path / "inbox" / f"{snippet_id}.py").read_text() == code


def test_submission_rejected_after_stop_requested(tmp_path):
    path = tmp_path / "session"
    create_session(path, {"model": "local-model"})
    (path / "STOP").touch()
    with pytest.raises((RuntimeError, ValueError)):
        submit(path, "print('too late')")
    assert not list((path / "inbox").iterdir())


def test_submission_waiting_on_shutdown_cannot_publish_after_terminal_state(tmp_path):
    path = tmp_path / "session"
    create_session(path, {"model": "local-model"})
    started = Event()

    def submit_during_shutdown():
        started.set()
        return submit(path, "print('too late')")

    with ThreadPoolExecutor(max_workers=1) as executor:
        with session_lock(path):
            future = executor.submit(submit_during_shutdown)
            assert started.wait(1)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
            write_json(path / "status.json", {"state": "stopped"})
        with pytest.raises(ValueError, match="not accepting"):
            future.result(timeout=1)
    assert not list((path / "inbox").glob("*.py"))


def test_json_replacement_and_process_liveness(tmp_path):
    import os

    path = tmp_path / "status.json"
    write_json(path, {"state": "loading"})
    write_json(path, {"state": "ready", "unicode": "λ"})
    assert read_json(path) == {"state": "ready", "unicode": "λ"}
    assert is_process_alive(os.getpid())
    assert not is_process_alive(-1)


def test_json_replacement_is_flushed_to_disk(tmp_path, monkeypatch):
    import os

    synced = []
    original = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), original(fd)))
    write_json(tmp_path / "_registry.json", {"datasets": {}})
    # Both the replacement file and its directory entry reach disk.
    assert len(synced) == 2
