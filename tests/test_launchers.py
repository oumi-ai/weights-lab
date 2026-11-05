import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from oumi_weights_lab.launchers import (
    SSH_OPTIONS,
    SkyPilotLauncher,
    _SSHTransport,
    sanitized_workdir,
)
from oumi_weights_lab.session import read_json


class Recorder:
    def __init__(self, stdout="{}"):
        self.stdout = stdout
        self.calls = []
        self.task = None
        self.spec_text = None
        self.synced_files = None

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[:2] == ["sky", "launch"]:
            self.spec_text = Path(args[-1]).read_text(encoding="utf-8")
            self.task = json.loads(self.spec_text)
            if "workdir" in self.task:
                self.synced_files = [
                    path.relative_to(self.task["workdir"]).as_posix()
                    for path in Path(self.task["workdir"]).rglob("*")
                    if path.is_file()
                ]
        return subprocess.CompletedProcess(args, 0, self.stdout, "")


def local_transport(args, **kwargs):
    """Execute the exact SSH command through a real local shell, without SSH access."""
    assert args[:-1] == ["ssh", *SSH_OPTIONS, "--", "test-host"]
    return subprocess.run(["sh", "-c", args[-1]], **kwargs)


def run_session_bootstrap(script):
    """Run a task's session creation without replacing the child with a model worker."""
    stub = "import os\nos.dup2=lambda *args: None\nos.execv=lambda *args: None\n"
    subprocess.run([sys.executable, "-c", stub + script], check=True, capture_output=True)


def test_sky_bootstrap_serializes_path_model(tmp_path):
    recorder = Recorder()
    path = tmp_path / "session"
    model = tmp_path / "model with spaces"
    SkyPilotLauncher("weights-test", python=sys.executable, runner=recorder).start(
        model, path, source=package_source(tmp_path)
    )
    command = shlex.split(recorder.task["run"])
    assert command[:2] == [sys.executable, "-c"]
    run_session_bootstrap(command[2])
    assert read_json(path / "config.json")["model"] == str(model)


def test_ssh_roundtrip_handles_hostile_code_and_paths(tmp_path):
    launcher = _SSHTransport("test-host", python=sys.executable, runner=local_transport)
    session = tmp_path / "session 'with spaces; $(false)"
    from oumi_weights_lab.session import create_session

    create_session(session, {"model": "unused"})
    code = "print('literal $(false) `false` ; $HOME')\n"
    result = json.loads(launcher.run(["send", "--session", str(session)], input=code).stdout)
    assert (session / "inbox" / f"{result['id']}.py").read_text() == code
    status = json.loads(launcher.run(["peek", "--session", str(session)]).stdout)
    assert status["state"] == "starting"


def test_ssh_transport_uses_stdin_and_runs_commands_remotely():
    runner = Recorder('{"state":"stopped"}')
    launcher = _SSHTransport("gpu-box", runner=runner)
    code = "print('sensitive snippet')"
    launcher.run(["send", "--session", "/remote/session"], input=code)
    args, kwargs = runner.calls[-1]
    assert code not in args[-1] and kwargs["input"] == code.encode()
    launcher.run(["stop", "--session", "/remote/session", "--timeout", "10"])
    command = shlex.split(runner.calls[-1][0][-1])
    assert command[command.index("exec") + 1 :] == [
        "python3",
        "-m",
        "oumi_weights_lab",
        "stop",
        "--session",
        "/remote/session",
        "--timeout",
        "10",
    ]
    with pytest.raises(ValueError):
        _SSHTransport("-oProxyCommand=bad")


def test_checkpoint_download_roundtrip_and_existing_destination(tmp_path):
    remote = tmp_path / "remote ' checkpoint"
    remote.mkdir()
    (remote / "config.json").write_text('{"model_type":"tiny"}')
    (remote / "nested").mkdir()
    (remote / "nested" / "weights.bin").write_bytes(bytes(range(256)) * 100)
    launcher = _SSHTransport("test-host", python=sys.executable, runner=local_transport)
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
    launcher = _SSHTransport("test-host", python=sys.executable, runner=local_transport)
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
            info = tarfile.TarInfo(".oumi-weights-lab-manifest.json")
            info.size = len(manifest)
            archive.addfile(info, io.BytesIO(manifest))
            info = tarfile.TarInfo(name)
            info.size = 3
            archive.addfile(info, io.BytesIO(b"bad"))
        return subprocess.CompletedProcess(args, 0)

    launcher = _SSHTransport("test-host", runner=corrupt_transport)
    with pytest.raises(ValueError):
        launcher.download("checkpoint", tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "escape").exists()
    assert list(tmp_path.iterdir()) == []


def package_source(tmp_path):
    root = tmp_path / "source"
    (root / "src" / "oumi_weights_lab").mkdir(parents=True)
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        (root / name).write_text("metadata")
    (root / "src" / "oumi_weights_lab" / "__init__.py").write_text("")
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
        "src/oumi_weights_lab/__init__.py",
    }
    assert runner.task["resources"] == {"accelerators": "L4:1"}
    run = shlex.split(runner.task["run"])
    assert run[:2] == ["python3", "-c"] and "os.execv" in run[2] and "_worker" in run[2]
    assert "Popen" not in run[2]
    assert "--down" in runner.calls[0][0] and "-d" in runner.calls[0][0]
    assert "--infra" not in runner.calls[0][0]
    launcher.teardown()
    assert runner.calls[-1][0] == ["sky", "down", "-y", "weights-test"]


def test_sky_checkout_installs_its_source_with_adapter_support(tmp_path):
    runner = Recorder()
    result = SkyPilotLauncher("weights-test", runner=runner).start(
        "model", "/tmp/session", source=package_source(tmp_path)
    )
    assert shlex.split(runner.task["setup"]) == [
        "python3",
        "-m",
        "pip",
        "install",
        ".[adapters,datasets,multi-gpu]",
    ]
    assert result["package"] == ".[adapters,datasets,multi-gpu]"


def installed_copy(tmp_path, monkeypatch, origin):
    """Pretend this package is an installed copy whose metadata records ``origin``."""
    from importlib import metadata

    from oumi_weights_lab import launchers

    installed = tmp_path / "site-packages" / "oumi_weights_lab" / "launchers.py"
    monkeypatch.setattr(launchers, "__file__", str(installed))
    record = SimpleNamespace(read_text=lambda name: None if origin is None else json.dumps(origin))
    monkeypatch.setattr(metadata, "distribution", lambda name: record)


def test_sky_git_install_installs_the_same_commit_on_the_worker(tmp_path, monkeypatch):
    commit = "864206f11a3b6dd7cd9d7d2790efe95c3b84bf8c"
    origin = {
        "url": "https://github.com/oumi-ai/weights-lab",
        "vcs_info": {"vcs": "git", "commit_id": commit, "requested_revision": "main"},
    }
    installed_copy(tmp_path, monkeypatch, origin)
    runner = Recorder()
    result = SkyPilotLauncher("weights-test", runner=runner).start("model", "/tmp/session")
    requirement = f"oumi-weights-lab[adapters,datasets,multi-gpu] @ git+https://github.com/oumi-ai/weights-lab@{commit}"
    assert shlex.split(runner.task["setup"]) == ["python3", "-m", "pip", "install", requirement]
    assert "workdir" not in runner.task
    assert result["package"] == requirement


@pytest.mark.parametrize(
    "origin",
    [
        None,  # installed from an index or a wheel file
        {"url": "file:///home/me/weights-lab", "dir_info": {}},
        {"url": "file:///home/me/weights-lab", "vcs_info": {"vcs": "git", "commit_id": "a" * 40}},
        {"url": "https://github.com/oumi-ai/weights-lab", "vcs_info": {"vcs": "git"}},
    ],
)
def test_sky_installed_copy_without_a_fetchable_commit_asks_for_source(
    tmp_path, monkeypatch, origin
):
    installed_copy(tmp_path, monkeypatch, origin)
    runner = Recorder()
    with pytest.raises(ValueError, match="--source"):
        SkyPilotLauncher("weights-test", runner=runner).start("model", "/tmp/session")
    assert runner.calls == []


def test_sky_finds_the_checkout_it_runs_from():
    from oumi_weights_lab import launchers

    checkout = Path(launchers.__file__).resolve().parents[2]
    assert launchers._source_checkout(None) == checkout
    assert (checkout / "pyproject.toml").is_file()


@pytest.mark.parametrize("infra", ["slurm/lab/gpu", "aws/us-east-1", "kubernetes/local"])
def test_sky_routes_infrastructure_through_upstream_cli(tmp_path, infra):
    runner = Recorder()
    result = SkyPilotLauncher("weights-test", runner=runner).start(
        "model", "/tmp/session", source=package_source(tmp_path), infra=infra
    )
    command = runner.calls[0][0]
    assert command[command.index("--infra") + 1] == infra
    assert runner.task["resources"] == {"accelerators": "A100:1"}
    assert result["launcher"] == "sky"
    assert "infra" not in shlex.split(runner.task["run"])[2]


def test_sky_cpu_slurm_launch_omits_gpu_request(tmp_path):
    runner = Recorder()
    SkyPilotLauncher("weights-test", runner=runner).start(
        "model",
        "/tmp/session",
        source=package_source(tmp_path),
        infra="slurm/lab/cpu",
        accelerators=None,
        device="cpu",
    )
    assert runner.task["resources"] == {}
    assert "'device': 'cpu'" in shlex.split(runner.task["run"])[2]


@pytest.mark.parametrize(
    "infra", ["modal", "Modal", "MODAL/workspace", " modal /workspace", "", "  ", 3, False]
)
def test_sky_invalid_or_modal_infrastructure_rejected_before_provisioning(infra):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid infrastructure reached provisioning")

    with pytest.raises(ValueError, match="infra|Modal"):
        SkyPilotLauncher("weights-test", runner=forbidden).start(
            "model", "/tmp/session", source="/unavailable", infra=infra
        )


def test_sky_start_waits_for_python_and_session_files_with_quoted_home_path(tmp_path):
    source = package_source(tmp_path)
    path = tmp_path / "session ' $(false); spaces"
    remote_path = "~/" + os.path.relpath(path, Path.home())
    probes = []
    launched = []

    def bootstrap(args, **kwargs):
        if args[:2] == ["sky", "launch"]:
            launched.append(shlex.split(json.loads(Path(args[-1]).read_text())["run"])[2])
            return subprocess.CompletedProcess(args, 0, "Setup detached; Job submitted", "")
        assert args[0] == "ssh" and args[-2] == "weights-test"
        assert kwargs["check"] is False and 0 < kwargs["timeout"] <= 15
        probes.append(args)
        if len(probes) == 1:
            return subprocess.CompletedProcess(args, 127, "", "Python does not exist yet")
        if len(probes) == 3:
            run_session_bootstrap(launched[0])
        return subprocess.run(["sh", "-c", args[-1]], **kwargs)

    result = SkyPilotLauncher("weights-test", python=sys.executable, runner=bootstrap).start(
        "model", remote_path, source=source, bootstrap_timeout=2, bootstrap_poll_interval=0.001
    )
    assert len(probes) == 3
    assert result["session"] == remote_path
    assert read_json(path / "status.json")["state"] == "starting"


def test_sky_relative_start_metadata_and_probe_use_remote_workdir(tmp_path):
    runner = Recorder()
    relative = Path("sessions") / "session ' $(false); spaces"
    expected = "~/sky_workdir/" + relative.as_posix()
    launcher = SkyPilotLauncher("weights-test", runner=runner)
    result = launcher.start("model", relative, source=package_source(tmp_path))
    assert result["session"] == expected
    command = shlex.split(runner.task["run"])
    assert repr(expected) in command[2]
    probe = runner.calls[-1][0]
    assert shlex.split(probe[-1])[-2] == expected
    assert str(Path.home()) not in runner.task["run"]


def test_sky_relative_session_readiness_send_peek_and_stop_shell_roundtrip(tmp_path):
    from oumi_weights_lab.session import create_session

    path = create_session(
        tmp_path / "session ' $(false); spaces", {"model": "unused", "launch_id": "launch"}
    )
    # Exercise remote HOME expansion without changing this machine's HOME or writing there.
    relative = os.path.relpath(path, Path.home() / "sky_workdir")

    def remote_shell(args, **kwargs):
        assert args[0] == "ssh" and args[-2] == "weights-test"
        return subprocess.run(["sh", "-c", args[-1]], cwd=tmp_path, **kwargs)

    launcher = SkyPilotLauncher("weights-test", python=sys.executable, runner=remote_shell)
    launcher._wait_session(relative, timeout=2, poll_interval=0.001, launch_id="launch")
    queued = json.loads(launcher.run(["send", "--session", relative], input="print(1)").stdout)
    assert (path / "inbox" / f"{queued['id']}.py").read_text() == "print(1)"
    peeked = json.loads(launcher.run(["peek", f"--session={relative}"]).stdout)
    assert peeked["state"] == "starting"
    stopped = json.loads(launcher.stop(relative).stdout)
    assert stopped["state"] == "stopped"
    assert read_json(path / "results" / f"{queued['id']}.json")["state"] == "cancelled"


def test_sky_slurm_commands_load_cluster_home_after_noninteractive_bashrc(tmp_path):
    from oumi_weights_lab.session import create_session

    login_home = tmp_path / "login home"
    cluster_home = login_home / ".sky_clusters" / "fixture"
    relative = "sessions/session ' $(false); spaces"
    path = create_session(
        cluster_home / "sky_workdir" / relative, {"model": "unused", "launch_id": "launch"}
    )
    (login_home / ".bashrc").write_text('[ -z "$PS1" ] && return\n. "$HOME/.sky_ssh_rc"\n')
    (login_home / ".sky_ssh_rc").write_text(
        'if [ -n "$SKY_CLUSTER_NAME" ]; then\n'
        '    cd "$HOME/.sky_clusters/$SKY_CLUSTER_NAME"\n'
        '    export HOME="$PWD"\n'
        "fi\n"
    )
    environment = {**os.environ, "HOME": str(login_home), "SKY_CLUSTER_NAME": "fixture"}

    def remote_shell(args, **kwargs):
        assert args[0] == "ssh" and args[-2] == "weights-test"
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", '. "$HOME/.bashrc"; ' + args[-1]],
            env=environment,
            **kwargs,
        )

    launcher = SkyPilotLauncher("weights-test", python=sys.executable, runner=remote_shell)
    launcher._wait_session(relative, timeout=2, poll_interval=0.1, launch_id="launch")
    queued = json.loads(
        launcher.run(["send", "--session", relative], input="print('literal $(false)')").stdout
    )
    assert (path / "inbox" / f"{queued['id']}.py").read_text() == "print('literal $(false)')"
    assert json.loads(launcher.run(["peek", "--session", relative]).stdout)["state"] == "starting"
    assert json.loads(launcher.stop(relative).stdout)["state"] == "stopped"
    backup = "sessions/backup ' $(false); spaces.tar.gz"
    launcher.run(["backup", "--session", relative, "--output", backup])
    destination = launcher.download(backup, tmp_path / "downloaded.tar.gz")
    assert destination.read_bytes() == (cluster_home / "sky_workdir" / backup).read_bytes()
    assert not (login_home / "sky_workdir").exists()


def test_sky_relative_backup_and_download_use_same_remote_workdir(tmp_path, monkeypatch):
    runner = Recorder()
    launcher = SkyPilotLauncher("weights-test", runner=runner)
    launcher.run(["backup", "--session", "sessions/test", "--output=backups/test.tar.gz"])
    args = shlex.split(runner.calls[-1][0][-1])
    assert args[args.index("--session") + 1] == "~/sky_workdir/sessions/test"
    assert "--output=~/sky_workdir/backups/test.tar.gz" in args
    copied = []

    def download(self, source, destination):
        copied.append((source, destination))
        return destination

    monkeypatch.setattr(_SSHTransport, "download", download)
    target = tmp_path / "downloaded.tar.gz"
    assert launcher.download(Path("backups/test.tar.gz"), target) == target
    assert copied == [("~/sky_workdir/backups/test.tar.gz", target)]


@pytest.mark.parametrize("path", ["/tmp/session", "~/session", "~remoteuser/session"])
def test_sky_explicit_remote_paths_are_preserved(path):
    assert SkyPilotLauncher._path(path) == path


def test_sky_bootstrap_timeout_preserves_last_diagnostic(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    attempts = []

    def probe(args, **kwargs):
        attempts.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args, 75, "", "package setup still running")

    monkeypatch.setattr("oumi_weights_lab.launchers.time.monotonic", lambda: clock.now)
    monkeypatch.setattr(
        "oumi_weights_lab.launchers.time.sleep",
        lambda seconds: setattr(clock, "now", clock.now + seconds),
    )
    launcher = SkyPilotLauncher("weights-test", runner=probe)
    with pytest.raises(TimeoutError, match="package setup still running"):
        launcher._wait_session("/session", timeout=1, poll_interval=0.3, launch_id="launch")
    assert clock.now == 1
    assert attempts and all(0 < value <= 1 for value in attempts)


@pytest.mark.parametrize(
    "code,diagnostic",
    [
        (2, '{"state":"failed","error":"setup failed"}'),
        (255, "Permission denied (publickey)"),
        (255, "Host key verification failed"),
    ],
)
def test_sky_bootstrap_reports_worker_or_auth_failure_immediately(code, diagnostic):
    calls = []

    def probe(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, code, "", diagnostic)

    launcher = SkyPilotLauncher("weights-test", runner=probe)
    with pytest.raises(RuntimeError, match="bootstrap"):
        launcher._wait_session("/session", timeout=100, poll_interval=1, launch_id="launch")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "option,value",
    [
        ("bootstrap_timeout", 0),
        ("bootstrap_timeout", float("inf")),
        ("bootstrap_poll_interval", float("nan")),
        ("bootstrap_timeout", None),
    ],
)
def test_sky_bootstrap_options_rejected_before_launch(option, value, tmp_path):
    def never_launch(*args, **kwargs):
        raise AssertionError("invalid bootstrap options reached a launcher")

    with pytest.raises(ValueError, match=option):
        SkyPilotLauncher("weights-test", runner=never_launch).start(
            "model", "/session", source=tmp_path, **{option: value}
        )


def test_sky_stdin_snippet_survives_remote_shell_roundtrip(tmp_path):
    from oumi_weights_lab.session import create_session

    path = tmp_path / "session with spaces"
    create_session(path, {"model": "unused"})

    def local_sky(args, **kwargs):
        assert args[:-1] == ["ssh", *SSH_OPTIONS, "--", "weights-test"]
        return subprocess.run(["sh", "-c", args[-1]], **kwargs)

    launcher = SkyPilotLauncher("weights-test", python=sys.executable, runner=local_sky)
    code = "print('literal shell $HOME $(false) `false`')\n"
    result = json.loads(launcher.run(["send", "--session", str(path)], input=code).stdout)
    assert (path / "inbox" / f"{result['id']}.py").read_text() == code


def test_sanitized_source_rejects_symlinked_package_files(tmp_path):
    source = package_source(tmp_path)
    (source / "src" / "oumi_weights_lab" / "secret.py").symlink_to(source / ".env")
    with pytest.raises(ValueError, match="symbolic"):
        sanitized_workdir(source, tmp_path / "destination")


def test_sky_checkpoint_copy_uses_cluster_ssh_alias(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"checkpoint")

    def local_sky(args, **kwargs):
        assert args[:-1] == ["ssh", *SSH_OPTIONS, "--", "weights-test"]
        return subprocess.run(["sh", "-c", args[-1]], **kwargs)

    launcher = SkyPilotLauncher("weights-test", python=sys.executable, runner=local_sky)
    destination = launcher.download(source, tmp_path / "copied.bin")
    assert destination.read_bytes() == b"checkpoint"


def test_ssh_run_returns_raw_bytes_when_remote_output_is_not_utf8(tmp_path):
    from oumi_weights_lab.session import create_session, write_json

    path = create_session(tmp_path / "session", {"model": "unused"})
    write_json(path / "results" / "snippet.json", {"id": "snippet", "state": "succeeded"})
    (path / "completed" / "snippet.py").write_text("print(1)")
    (path / "outputs" / "snippet.log").write_bytes(b"child wrote \xff\n")
    launcher = _SSHTransport("test-host", python=sys.executable, runner=local_transport)
    result = launcher.run(["peek", "--session", str(path), "--id", "snippet"])
    assert json.loads(result.stdout)["state"] == "succeeded"
    assert result.stderr == b"child wrote \xff\n"


def test_ssh_call_sites_never_inherit_the_callers_stdin(tmp_path):
    calls = []

    def runner(args, **kwargs):
        calls.append(kwargs)
        if "stdout" in kwargs:
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 0, b"{}", b"")

    launcher = SkyPilotLauncher("weights-test", runner=runner)
    launcher.run(["peek", "--session", "/session"])
    launcher.stop("/session")
    launcher._wait_session("/session", timeout=1, poll_interval=1, launch_id="launch")
    with pytest.raises(subprocess.CalledProcessError):
        launcher.download("/session/artifacts/edited", tmp_path / "edited")
    assert len(calls) == 4
    assert all(kwargs["stdin"] is subprocess.DEVNULL for kwargs in calls)
    launcher.run(["send", "--session", "/session"], input="print(1)")
    assert calls[-1]["input"] == b"print(1)" and "stdin" not in calls[-1]


def test_sky_readiness_rejects_an_existing_session_from_another_launch(tmp_path):
    from oumi_weights_lab.session import create_session, write_json

    path = create_session(tmp_path / "session", {"model": "old-model"})
    write_json(path / "status.json", {"state": "ready", "model": "old-model"})
    probes = []

    def remote(args, **kwargs):
        if args[:2] == ["sky", "launch"]:
            # The new job cannot create a session over the old one.
            script = shlex.split(json.loads(Path(args[-1]).read_text())["run"])[2]
            with pytest.raises(subprocess.CalledProcessError):
                run_session_bootstrap(script)
            return subprocess.CompletedProcess(args, 0, "", "")
        probes.append(args)
        return subprocess.run(["sh", "-c", args[-1]], **kwargs)

    launcher = SkyPilotLauncher("weights-test", python=sys.executable, runner=remote)
    with pytest.raises(RuntimeError, match="another launch"):
        launcher.start(
            "new-model",
            path,
            source=package_source(tmp_path),
            bootstrap_timeout=30,
            bootstrap_poll_interval=0.001,
        )
    assert len(probes) == 1
    assert read_json(path / "config.json")["model"] == "old-model"


def local_sky_shell(args, **kwargs):
    assert args[0] == "ssh" and args[-2] == "weights-test"
    return subprocess.run(["sh", "-c", args[-1]], **kwargs)


@pytest.mark.parametrize(
    "state", ["starting", "loading", "ready", "running", "stopping", "stopped", "failed"]
)
def test_sky_readiness_fails_only_for_terminal_worker_states(tmp_path, state):
    from oumi_weights_lab.cli import TERMINAL_STATES
    from oumi_weights_lab.session import create_session, write_json

    path = create_session(tmp_path / "session", {"model": "m", "launch_id": "launch"})
    write_json(path / "status.json", {"state": state, "model": "m"})
    launcher = SkyPilotLauncher("weights-test", python=sys.executable, runner=local_sky_shell)
    if state in TERMINAL_STATES:
        with pytest.raises(RuntimeError, match="bootstrap"):
            launcher._wait_session(path, timeout=5, poll_interval=1, launch_id="launch")
    else:
        launcher._wait_session(path, timeout=5, poll_interval=1, launch_id="launch")


@pytest.mark.parametrize(
    "resume,expected",
    [
        ("sessions/old/artifacts/candidate", "~/sky_workdir/sessions/old/artifacts/candidate"),
        (Path("old/candidate"), "~/sky_workdir/old/candidate"),
        ("~/saved/candidate", "~/saved/candidate"),
        ("/data/candidate", "/data/candidate"),
    ],
)
def test_sky_resume_path_resolves_like_the_session_path(tmp_path, resume, expected):
    recorder = Recorder()
    path = tmp_path / "session"
    SkyPilotLauncher("weights-test", python=sys.executable, runner=recorder).start(
        "model", path, source=package_source(tmp_path), resume=resume
    )
    run_session_bootstrap(shlex.split(recorder.task["run"])[2])
    assert read_json(path / "config.json")["resume"] == expected


def test_sky_task_with_non_bmp_values_loads_with_libyaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    recorder = Recorder()
    path = tmp_path / "session 😀"
    SkyPilotLauncher("weights-test", python=sys.executable, runner=recorder).start(
        "org/model-😀", path, source=package_source(tmp_path), revision="rev-😀"
    )
    assert recorder.task["run"].isascii()
    # SkyPilot reads the task with libyaml, which rejects JSON surrogate-pair escapes.
    task = yaml.load(recorder.spec_text, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    assert task == recorder.task
    run_session_bootstrap(shlex.split(task["run"])[2])
    config = read_json(path / "config.json")
    assert (config["model"], config["revision"]) == ("org/model-😀", "rev-😀")


def test_launcher_commands_all_return_raw_bytes(tmp_path):
    runner = Recorder(stdout=b"done")
    launcher = SkyPilotLauncher("weights-test", runner=runner)
    launcher.teardown()
    # run(), stop(), and teardown() all return bytes; none request text decoding.
    assert all("text" not in kwargs for args, kwargs in runner.calls if args[:2] == ["sky", "down"])
