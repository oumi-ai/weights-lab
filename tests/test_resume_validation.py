"""Controller checks use fake workers; cloud/CUDA validation is explicit opt-in."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "validate_resume", Path(__file__).resolve().parents[1] / "scripts/validate_resume.py"
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def fake_worker(monkeypatch, tmp_path, *, fail=False):
    calls = []
    monkeypatch.setattr(validation, "make_assets", lambda root: ("local-model", "local-adapter"))
    monkeypatch.setattr("oumi_weights_lab.session.is_process_alive", lambda pid: False)

    def cli(root, *args):
        calls.append(args)
        if args[0] == "start":
            return {"pid": 123}
        if args[0] == "send":
            if fail:
                raise RuntimeError("intentional snippet failure")
            phase = root.name
            artifacts = root / "session" / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / f"{phase}.json").write_text(json.dumps({"state": "succeeded"}))
            if phase == "prepare":
                (artifacts / "bundle").mkdir()
                (artifacts / "bundle" / "manifest.json").write_text("{}")
            return {"state": "succeeded"}
        assert args[0] == "stop"
        return {"state": "stopped"}

    monkeypatch.setattr(validation, "run_cli", cli)
    return calls


def test_prepare_requests_bf16_local_assets_and_stops_worker(tmp_path, monkeypatch):
    calls = fake_worker(monkeypatch, tmp_path)
    report = validation.run_phase("prepare", tmp_path, "cpu")
    args = calls[0]
    assert args[args.index("--model") + 1] == "local-model"
    assert args[args.index("--adapter") + 1] == "local-adapter"
    assert args[args.index("--dtype") + 1] == "bfloat16"
    assert args[args.index("--pass-limit") + 1] == "32"
    assert calls[-1][0] == "stop"
    assert report["bundle_hashes"] == validation.file_hashes(
        tmp_path / "prepare/session/artifacts/bundle"
    )
    assert not report["worker_alive_after_stop"]
    with pytest.raises(FileExistsError):
        validation.run_phase("prepare", tmp_path, "cpu")


def test_resume_uses_fresh_session_and_downloaded_bundle(tmp_path, monkeypatch):
    calls = fake_worker(monkeypatch, tmp_path)
    bundle = tmp_path / "verified-download"
    report = validation.run_phase("resume", tmp_path, "cpu", bundle=bundle)
    args = calls[0]
    assert args[args.index("--session") + 1] == tmp_path / "resume/session"
    assert args[args.index("--resume") + 1] == bundle.resolve()
    assert report["state"] == "succeeded"
    assert calls[-1][0] == "stop"


def test_snippet_failure_still_stops_worker_and_records_error(tmp_path, monkeypatch):
    calls = fake_worker(monkeypatch, tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="intentional snippet failure"):
        validation.run_phase("prepare", tmp_path, "cpu")
    report = json.loads((tmp_path / "prepare/report.json").read_text())
    assert calls[-1][0] == "stop"
    assert report["state"] == "failed"
    assert not report["worker_alive_after_stop"]


def test_resume_requires_bundle_before_starting_worker(tmp_path, monkeypatch):
    calls = fake_worker(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="downloaded bundle"):
        validation.run_phase("resume", tmp_path, "cpu")
    assert calls == []


def test_file_hashes_capture_exact_binary_changes(tmp_path):
    (tmp_path / "payload").write_bytes(b"\x00\xff\x10")
    first = validation.file_hashes(tmp_path)
    (tmp_path / "payload").write_bytes(b"\x00\xfe\x10")
    assert validation.file_hashes(tmp_path) != first


def test_cli_parses_only_stdout_when_transport_warns_on_stderr(tmp_path):
    import subprocess

    def runner(command, **kwargs):
        assert kwargs["capture_output"] and kwargs["text"]
        return subprocess.CompletedProcess(
            command,
            0,
            '{"state":"succeeded"}',
            "Warning: Permanently added 'test-host' to known hosts.\n",
        )

    assert validation.run_cli(tmp_path, "peek", runner=runner) == {"state": "succeeded"}
    entry = json.loads((tmp_path / "cli.jsonl").read_text())
    assert entry["stderr"].startswith("Warning:")
    assert json.loads(entry["stdout"]) == {"state": "succeeded"}
