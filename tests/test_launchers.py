import base64
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from weights_lab.launchers import (
    ModalLauncher,
    SkyPilotLauncher,
    SlurmLauncher,
    SSHLauncher,
    sanitized_workdir,
)
from weights_lab.session import read_json


class Recorder:
    def __init__(self, stdout="{}"):
        self.stdout = stdout
        self.calls = []
        self.task = None
        self.synced_files = None

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[0] == "squeue":
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ["sky", "launch"]:
            self.task = json.loads(Path(args[-1]).read_text())
            self.synced_files = [
                path.relative_to(self.task["workdir"]).as_posix()
                for path in Path(self.task["workdir"]).rglob("*")
                if path.is_file()
            ]
        return subprocess.CompletedProcess(args, 0, self.stdout, "")


def local_transport(args, **kwargs):
    """Execute the exact SSH command through a real local shell, without SSH access."""
    assert args[:3] == ["ssh", "--", "test-host"]
    return subprocess.run(["sh", "-c", args[3]], **kwargs)


def test_ssh_roundtrip_handles_hostile_code_and_paths(tmp_path):
    launcher = SSHLauncher("test-host", python=sys.executable, runner=local_transport)
    session = tmp_path / "session 'with spaces; $(false)"
    from weights_lab.session import create_session

    create_session(session, {"model": "unused"})
    code = "print('literal $(false) `false` ; $HOME')\n"
    result = launcher.send(session, code)
    assert (session / "inbox" / f"{result['id']}.py").read_text() == code
    status = json.loads(launcher.run(["peek", "--session", str(session)]).stdout)
    assert status["state"] == "starting"


def test_ssh_code_uses_stdin_and_stop_is_remote():
    runner = Recorder('{"state":"stopped"}')
    launcher = SSHLauncher("gpu-box", runner=runner)
    code = "print('sensitive snippet')"
    launcher.send("/remote/session", code)
    args, kwargs = runner.calls[-1]
    assert code not in args[-1] and kwargs["input"] == code
    launcher.stop("/remote/session")
    assert shlex.split(runner.calls[-1][0][-1]) == [
        "python3",
        "-m",
        "weights_lab",
        "stop",
        "--session",
        "/remote/session",
        "--timeout",
        "10",
    ]
    with pytest.raises(ValueError):
        SSHLauncher("-oProxyCommand=bad")


def test_ssh_start_passes_checkpoint_revisions_and_budget():
    runner = Recorder('{"state":"starting"}')
    launcher = SSHLauncher("gpu-box", runner=runner)
    launcher.start(
        "model",
        "/session",
        revision="model-rev",
        adapter="adapter",
        adapter_revision="adapter-rev",
        pass_limit=7,
    )
    args = shlex.split(runner.calls[-1][0][-1])
    for flag, value in (
        ("--revision", "model-rev"),
        ("--adapter-revision", "adapter-rev"),
        ("--pass-limit", "7"),
    ):
        assert args[args.index(flag) + 1] == value


def test_checkpoint_download_roundtrip_and_existing_destination(tmp_path):
    remote = tmp_path / "remote ' checkpoint"
    remote.mkdir()
    (remote / "config.json").write_text('{"model_type":"tiny"}')
    (remote / "nested").mkdir()
    (remote / "nested" / "weights.bin").write_bytes(bytes(range(256)) * 100)
    launcher = SSHLauncher("test-host", python=sys.executable, runner=local_transport)
    destination = launcher.download(remote, tmp_path / "downloaded")
    for original in remote.rglob("*"):
        if original.is_file():
            copied = destination / original.relative_to(remote)
            assert (
                hashlib.sha256(copied.read_bytes()).digest()
                == hashlib.sha256(original.read_bytes()).digest()
            )
    with pytest.raises(FileExistsError):
        launcher.download(remote, destination)
    file_copy = launcher.download(remote / "config.json", tmp_path / "single.json")
    assert file_copy.read_bytes() == (remote / "config.json").read_bytes()


def test_checkpoint_download_rejects_source_symlink(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "bad").symlink_to(tmp_path / "secret")
    launcher = SSHLauncher("test-host", python=sys.executable, runner=local_transport)
    with pytest.raises(subprocess.CalledProcessError):
        launcher.download(remote, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("name,digest", [("weights.bin", "0" * 64), ("../escape", "0" * 64)])
def test_checkpoint_download_rejects_corruption_and_traversal(tmp_path, name, digest):
    def corrupt_transport(args, **kwargs):
        with tarfile.open(fileobj=kwargs["stdout"], mode="w|") as archive:
            manifest = json.dumps(
                {"kind": "directory", "files": [{"path": name, "sha256": digest}]}
            ).encode()
            info = tarfile.TarInfo(".weights-lab-manifest.json")
            info.size = len(manifest)
            archive.addfile(info, io.BytesIO(manifest))
            info = tarfile.TarInfo(name)
            info.size = 3
            archive.addfile(info, io.BytesIO(b"bad"))
        return subprocess.CompletedProcess(args, 0)

    launcher = SSHLauncher("test-host", runner=corrupt_transport)
    with pytest.raises(ValueError):
        launcher.download("checkpoint", tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "escape").exists()
    assert list(tmp_path.iterdir()) == []


def test_slurm_creates_shared_session_and_cancels_job_not_pid(tmp_path):
    runner = Recorder("12345;cluster-a\n")
    launcher = SlurmLauncher(python="/env/bin/python", runner=runner)
    path = tmp_path / "shared session"
    result = launcher.start(
        "model", path, partition="gpu", gres="gpu:a100:1", time_limit="01:00:00"
    )
    assert result["job_id"] == "12345" and result["cluster"] == "cluster-a"
    assert read_json(path / "config.json")["model"] == "model"
    args = runner.calls[0][0]
    wrapped = shlex.split(args[args.index("--wrap") + 1])
    assert wrapped == ["/env/bin/python", "-m", "weights_lab", "_worker", "--session", str(path)]
    assert args[args.index("--partition") + 1] == "gpu"
    launcher.stop(path)
    assert runner.calls[-2][0] == ["scancel", "--clusters", "cluster-a", "12345"]
    assert read_json(path / "status.json")["state"] == "stopped"
    assert (path / "STOP").exists()
    with pytest.raises(ValueError, match="SlurmLauncher.stop"):
        launcher.run(["stop", "--session", str(path)])


def test_slurm_failed_submission_marks_session_failed(tmp_path):
    def failing(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr="no partition")

    launcher = SlurmLauncher(runner=failing)
    path = tmp_path / "session"
    with pytest.raises(subprocess.CalledProcessError):
        launcher.start("model", path)
    assert read_json(path / "status.json")["state"] == "failed"


def test_slurm_cancellation_finalizes_queued_sources_and_waits_for_scheduler(tmp_path):
    from weights_lab.session import submit

    states = iter(["RUNNING\n", "COMPLETING\n", ""])

    def scheduler(args, **kwargs):
        output = next(states) if args[0] == "squeue" else "42"
        return subprocess.CompletedProcess(args, 0, output, "")

    launcher = SlurmLauncher(runner=scheduler)
    path = tmp_path / "session"
    launcher.start("model", path)
    identifier = submit(path, "print('queued')")
    result = launcher.stop(path, timeout=2)
    assert result["state"] == "stopped"
    assert read_json(path / "results" / f"{identifier}.json")["state"] == "cancelled"
    assert (path / "completed" / f"{identifier}.py").exists()


def test_slurm_timeout_never_claims_worker_stopped(tmp_path):
    def scheduler(args, **kwargs):
        output = "RUNNING\n" if args[0] == "squeue" else "42"
        return subprocess.CompletedProcess(args, 0, output, "")

    launcher = SlurmLauncher(runner=scheduler)
    path = tmp_path / "session"
    launcher.start("model", path)
    with pytest.raises(TimeoutError, match="still active"):
        launcher.stop(path, timeout=0.01)
    assert read_json(path / "status.json")["state"] == "starting"


def package_source(tmp_path):
    root = tmp_path / "source"
    (root / "src" / "weights_lab").mkdir(parents=True)
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        (root / name).write_text("metadata")
    (root / "src" / "weights_lab" / "__init__.py").write_text("")
    (root / ".env").write_text("secret")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("private")
    (root / "docs").mkdir()
    (root / "docs" / "PLAN.md").write_text("private plan")
    return root


def test_sky_syncs_sanitized_package_and_keeps_foreground_worker(tmp_path):
    source = package_source(tmp_path)
    runner = Recorder()
    launcher = SkyPilotLauncher("weights-test", runner=runner)
    result = launcher.start("model", "/tmp/session", source=source, accelerators="L4:1")
    assert result["cluster"] == "weights-test"
    assert set(runner.synced_files) == {
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "src/weights_lab/__init__.py",
    }
    assert runner.task["resources"] == {"accelerators": "L4:1"}
    run = shlex.split(runner.task["run"])
    assert run[:2] == ["python3", "-c"] and "os.execv" in run[2] and "_worker" in run[2]
    assert "Popen" not in run[2]
    assert "--down" in runner.calls[0][0] and "-d" in runner.calls[0][0]
    launcher.teardown()
    assert runner.calls[-1][0] == ["sky", "down", "-y", "weights-test"]


def test_sky_stdin_snippet_survives_remote_shell_roundtrip(tmp_path):
    from weights_lab.session import create_session

    path = tmp_path / "session with spaces"
    create_session(path, {"model": "unused"})

    def local_sky(args, **kwargs):
        assert args[:3] == ["ssh", "--", "weights-test"]
        return subprocess.run(["sh", "-c", args[-1]], **kwargs)

    launcher = SkyPilotLauncher("weights-test", python=sys.executable, runner=local_sky)
    code = "print('literal shell $HOME $(false) `false`')\n"
    result = json.loads(launcher.run(["send", "--session", str(path)], input=code).stdout)
    assert (path / "inbox" / f"{result['id']}.py").read_text() == code


def test_sanitized_source_rejects_symlinked_package_files(tmp_path):
    source = package_source(tmp_path)
    (source / "src" / "weights_lab" / "secret.py").symlink_to(source / ".env")
    with pytest.raises(ValueError, match="symbolic"):
        sanitized_workdir(source, tmp_path / "destination")


def test_modal_sandbox_lifecycle_uses_stdin_and_explicit_teardown(tmp_path):
    source = package_source(tmp_path)
    state = {}

    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def add_local_dir(self, local, remote, **kwargs):
            state["files"] = [
                path.relative_to(local).as_posix()
                for path in Path(local).rglob("*")
                if path.is_file()
            ]
            return self

        def run_commands(self, command):
            state["setup"] = command
            return self

    class Stdin:
        def write(self, data):
            state["stdin"] = data

        def write_eof(self):
            state["eof"] = True

        def drain(self):
            state["drained"] = True

    class Sandbox:
        object_id = "sb-test"

        @classmethod
        def create(cls, *args, **kwargs):
            state["create"] = (args, kwargs)
            return cls()

        @classmethod
        def from_id(cls, identifier):
            assert identifier == "sb-test"
            return cls()

        def exec(self, *args):
            state["exec"] = args
            return SimpleNamespace(
                stdin=Stdin(),
                stdout=io.StringIO('{"id":"snippet"}'),
                stderr=io.StringIO(""),
                wait=lambda: 0,
            )

        def terminate(self, *, wait):
            state["terminated"] = wait

    sdk = SimpleNamespace(
        Image=Image, Sandbox=Sandbox, App=SimpleNamespace(lookup=lambda *args, **kwargs: "app")
    )
    launcher = ModalLauncher(sdk=sdk)
    result = launcher.start("model", "/session", source=source, gpu="L4", lifetime=1800)
    assert result["sandbox_id"] == "sb-test"
    assert state["create"][1]["gpu"] == "L4"
    assert state["create"][1]["timeout"] == 1800
    assert state["create"][1]["runtime"] == "gvisor"
    assert ".env" not in state["files"]
    code = "print('literal $HOME')\n"
    launcher.run(["send", "--session", "/session"], input=code)
    assert state["stdin"] == code.encode() and state["eof"] and state["drained"]
    assert code not in state["exec"]
    launcher.teardown()
    assert state["terminated"]


def test_modal_download_streams_and_verifies_real_local_checkpoint(tmp_path):
    remote = tmp_path / "checkpoint"
    remote.mkdir()
    (remote / "weights.bin").write_bytes(bytes(range(256)) * 1000)
    (remote / "config.json").write_text('{"model_type":"tiny"}')

    class Sandbox:
        def exec(self, *args, **kwargs):
            return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)

    sdk = SimpleNamespace(Sandbox=SimpleNamespace(from_id=lambda identifier: Sandbox()))
    launcher = ModalLauncher(sandbox_id="sb-local", python=sys.executable, sdk=sdk)
    destination = launcher.download(remote, tmp_path / "copied")
    assert (destination / "weights.bin").read_bytes() == (remote / "weights.bin").read_bytes()
    assert (destination / "config.json").read_bytes() == (remote / "config.json").read_bytes()
    single = launcher.download(remote / "config.json", tmp_path / "config-copy.json")
    assert single.read_bytes() == (remote / "config.json").read_bytes()
    with pytest.raises(FileExistsError):
        launcher.download(remote, destination)


def test_modal_download_rejects_corrupt_split_text_stream(tmp_path):
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        manifest = json.dumps(
            {"kind": "file", "files": [{"path": "weights.bin", "sha256": "0" * 64}]}
        ).encode()
        info = tarfile.TarInfo(".weights-lab-manifest.json")
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))
        info = tarfile.TarInfo("weights.bin")
        info.size = 3
        archive.addfile(info, io.BytesIO(b"bad"))
    encoded = base64.b64encode(archive_bytes.getvalue()).decode() + "\n"

    class Sandbox:
        def exec(self, *args, **kwargs):
            return SimpleNamespace(
                stdout=[encoded[:17], encoded[17:999], encoded[999:]],
                stderr=io.StringIO(""),
                wait=lambda: 0,
            )

    sdk = SimpleNamespace(Sandbox=SimpleNamespace(from_id=lambda identifier: Sandbox()))
    launcher = ModalLauncher(sandbox_id="sb-local", sdk=sdk)
    with pytest.raises(ValueError, match="checksum mismatch"):
        launcher.download("checkpoint", tmp_path / "copied")
    assert list(tmp_path.iterdir()) == []


def test_sky_checkpoint_copy_uses_cluster_ssh_alias(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"checkpoint")

    def local_sky(args, **kwargs):
        assert args[:3] == ["ssh", "--", "weights-test"]
        return subprocess.run(["sh", "-c", args[-1]], **kwargs)

    launcher = SkyPilotLauncher("weights-test", python=sys.executable, runner=local_sky)
    destination = launcher.download(source, tmp_path / "copied.bin")
    assert destination.read_bytes() == b"checkpoint"


def test_slurm_stop_already_finished_job_reconciles_session(tmp_path):
    def scheduler(args, **kwargs):
        if args[0] == "scancel":
            raise subprocess.CalledProcessError(1, args, stderr="Invalid job id specified")
        output = "42" if args[0] == "sbatch" else ""
        return subprocess.CompletedProcess(args, 0, output, "")

    launcher = SlurmLauncher(runner=scheduler)
    path = tmp_path / "session"
    launcher.start("model", path)
    assert launcher.stop(path)["state"] == "stopped"
