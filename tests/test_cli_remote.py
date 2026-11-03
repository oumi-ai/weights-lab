import json
import subprocess
import sys
from pathlib import Path

import pytest

from weights_lab import cli, launchers
from weights_lab.session import create_session, write_json


class Transport:
    def __init__(self):
        self.calls = []
        self.result = subprocess.CompletedProcess([], 0, '{"state":"ready"}\n', "")
        self.error = None
        self.active = False

    def run(self, arguments, *, input=None):
        self.calls.append(("run", arguments, input))
        if self.error is not None:
            raise self.error
        return self.result

    def start(self, model, session, **config):
        self.calls.append(("start", model, session, config))
        return {"state": "starting", "session": session}

    def stop(self, session, *, timeout):
        self.calls.append(("stop", session, timeout))
        return {"state": "stopped"}

    def download(self, source, destination):
        self.calls.append(("download", source, destination))
        with Path(destination).open("xb") as stream:
            stream.write(b"remote archive")

    def is_active(self, session):
        self.calls.append(("is_active", session))
        return self.active


class CloudTransport(Transport):
    def teardown(self):
        self.calls.append(("teardown",))


@pytest.fixture
def transports(monkeypatch):
    instances = {}
    constructors = []
    for name in ("SSHLauncher", "SlurmLauncher", "SkyPilotLauncher", "ModalLauncher"):
        instance = (
            CloudTransport() if name in {"SkyPilotLauncher", "ModalLauncher"} else Transport()
        )
        instances[name] = instance

        def construct(*args, _name=name, _instance=instance, **kwargs):
            constructors.append((_name, args, kwargs))
            return _instance

        monkeypatch.setattr(launchers, name, construct)
    return instances, constructors


@pytest.mark.parametrize(
    "target,kind",
    [
        ("--host", "SSHLauncher"),
        ("--cluster", "SkyPilotLauncher"),
        ("--sandbox-id", "ModalLauncher"),
    ],
)
def test_remote_send_reads_local_file_and_sends_contents_on_stdin(
    tmp_path, transports, capsys, target, kind
):
    instances, constructors = transports
    source = tmp_path / "local snippet.py"
    code = "print('literal $HOME $(false) `false` λ')\n"
    source.write_text(code)
    arguments = [
        "send",
        target,
        "remote",
        "--session",
        "/remote/session",
        "--file",
        str(source),
        "--wait",
        "--follow",
        "--timeout",
        "7",
        "--remote-python",
        "/env/bin/python",
    ]
    assert cli.main(arguments) == 0
    capsys.readouterr()
    assert constructors[0][0] == kind
    assert constructors[0][2]["python"] == "/env/bin/python"
    method, command, stdin = instances[kind].calls[0]
    assert method == "run"
    assert stdin == code
    assert command == [
        "send",
        "--session",
        "/remote/session",
        "--wait",
        "--follow",
        "--timeout",
        "7.0",
    ]
    assert str(source) not in command and code not in command


@pytest.mark.parametrize("failure", ["result", "text_exception", "bytes_exception"])
def test_remote_failure_preserves_stdout_stderr_and_exit_code(transports, capsys, failure):
    instances, _ = transports
    stdout = '{"id":"failed-snippet","state":"failed"}\n'
    stderr = "Traceback: expected failure\n"
    transport = instances["SSHLauncher"]
    if failure != "result":
        transport.error = subprocess.CalledProcessError(
            7,
            ["remote"],
            output=stdout.encode() if failure == "bytes_exception" else stdout,
            stderr=stderr.encode() if failure == "bytes_exception" else stderr,
        )
    else:
        transport.result = subprocess.CompletedProcess([], 7, stdout, stderr)
    assert cli.main(["peek", "--host", "gpu", "--session", "/remote/session"]) == 7
    output = capsys.readouterr()
    assert output.out == stdout
    assert output.err == stderr


def test_remote_backup_uses_remote_archive_path_and_downloads_without_overwrite(
    tmp_path, transports, capsys
):
    instances, _ = transports
    destination = tmp_path / "download.tar.gz"
    args = [
        "backup",
        "--host",
        "gpu",
        "--session",
        "/remote/session",
        "--output",
        str(destination),
        "--remote-output",
        "/remote/archive.tar.gz",
    ]
    assert cli.main(args) == 0
    assert destination.read_bytes() == b"remote archive"
    assert instances["SSHLauncher"].calls == [
        (
            "run",
            ["backup", "--session", "/remote/session", "--output", "/remote/archive.tar.gz"],
            None,
        ),
        ("download", "/remote/archive.tar.gz", destination.resolve()),
    ]
    assert json.loads(capsys.readouterr().out)["output"] == str(destination.resolve())
    calls_before = instances["SSHLauncher"].calls.copy()
    assert cli.main(args) != 0
    capsys.readouterr()
    assert instances["SSHLauncher"].calls == calls_before
    assert destination.read_bytes() == b"remote archive"


def test_slurm_start_forwards_checkpoint_pins_budget_and_scheduler_config(transports, capsys):
    instances, _ = transports
    assert (
        cli.main(
            [
                "start",
                "--launcher",
                "slurm",
                "--session",
                "/shared/session",
                "--model",
                "base",
                "--revision",
                "base-sha",
                "--adapter",
                "adapter",
                "--adapter-revision",
                "adapter-sha",
                "--pass-limit",
                "12",
                "--device",
                "cuda:0",
                "--dtype",
                "bfloat16",
                "--snippet-timeout",
                "20",
                "--partition",
                "gpu",
                "--gres",
                "gpu:a100:1",
                "--time-limit",
                "01:00:00",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert instances["SlurmLauncher"].calls == [
        (
            "start",
            "base",
            "/shared/session",
            {
                "adapter": "adapter",
                "device": "cuda:0",
                "dtype": "bfloat16",
                "snippet_timeout": 20.0,
                "pass_limit": 12,
                "revision": "base-sha",
                "adapter_revision": "adapter-sha",
                "partition": "gpu",
                "gres": "gpu:a100:1",
                "time_limit": "01:00:00",
            },
        )
    ]


@pytest.fixture
def slurm_session(tmp_path):
    path = tmp_path / "shared-session"
    create_session(path, {"model": "unused"})
    write_json(path / "status.json", {"state": "ready", "pid": 87654321})
    write_json(path / "launcher.json", {"launcher": "slurm", "job_id": "42"})
    return path


def test_slurm_stop_dispatches_scheduler_without_signaling_a_local_pid(
    slurm_session, transports, monkeypatch, capsys
):
    def forbidden(*args):
        pytest.fail("a Slurm compute-node PID must never be checked or signaled locally")

    monkeypatch.setattr(cli.os, "kill", forbidden)
    monkeypatch.setattr(cli.session, "is_process_alive", forbidden)
    assert cli.main(["stop", "--session", str(slurm_session), "--timeout", "2.5"]) == 0
    assert transports[0]["SlurmLauncher"].calls == [("stop", slurm_session.resolve(), 2.5)]
    assert json.loads(capsys.readouterr().out)["state"] == "stopped"


@pytest.mark.parametrize("command", ["peek", "send"])
def test_slurm_follow_and_wait_ignore_compute_node_pid_liveness(
    slurm_session, monkeypatch, capsys, command
):
    def forbidden(*args):
        pytest.fail("compute-node PID liveness is not local process liveness")

    def worker_progress(_):
        if command == "peek":
            write_json(slurm_session / "status.json", {"state": "stopped", "pid": 87654321})
        else:
            (source,) = (slurm_session / "inbox").glob("*.py")
            write_json(
                slurm_session / "results" / f"{source.stem}.json",
                {"id": source.stem, "state": "succeeded"},
            )

    monkeypatch.setattr(cli.session, "is_process_alive", forbidden)
    monkeypatch.setattr(cli.time, "sleep", worker_progress)
    args = [command, "--session", str(slurm_session)]
    args += ["--follow"] if command == "peek" else ["--code", "print(1)", "--wait"]
    assert cli.main(args) == 0
    states = [json.loads(line)["state"] for line in capsys.readouterr().out.splitlines()]
    assert states == (["ready", "stopped"] if command == "peek" else ["succeeded"])


@pytest.mark.parametrize(
    "targets",
    [
        ["--host", "gpu", "--cluster", "cluster"],
        ["--launcher", "sky", "--host", "gpu"],
        ["--launcher", "ssh", "--cluster", "cluster"],
        ["--sandbox-id", "existing"],
    ],
)
def test_conflicting_start_targets_are_rejected_before_resources(
    tmp_path, transports, capsys, targets
):
    path = tmp_path / "session"
    assert cli.main(["start", "--session", str(path), "--model", "base", *targets]) != 0
    capsys.readouterr()
    assert transports[1] == []
    assert not path.exists()


@pytest.mark.parametrize(
    "target",
    [
        [],
        ["--host", "gpu"],
        ["--launcher", "slurm"],
        ["--cluster", "cluster"],
        ["--launcher", "modal"],
    ],
)
@pytest.mark.parametrize("invalid", [["--pass-limit", "-1"], ["--model", ""], ["--dtype", "bogus"]])
def test_invalid_config_rejected_before_remote_or_local_resources(
    tmp_path, transports, monkeypatch, capsys, target, invalid
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid config must be rejected before starting a process")

    monkeypatch.setattr(cli.subprocess, "Popen", forbidden)
    path = tmp_path / "session"
    try:
        result = cli.main(["start", "--session", str(path), "--model", "base", *invalid, *target])
    except SystemExit as error:
        result = error.code
    assert result != 0
    capsys.readouterr()
    assert transports[1] == []
    assert not path.exists()


@pytest.mark.parametrize(
    "target,allowed,kind",
    [
        ([], False, None),
        (["--host", "gpu"], False, "SSHLauncher"),
        (["--cluster", "cluster"], True, "SkyPilotLauncher"),
        (["--sandbox-id", "sandbox"], True, "ModalLauncher"),
    ],
)
def test_teardown_only_applies_to_cloud_workers(transports, capsys, target, allowed, kind):
    result = cli.main(["stop", "--session", "/remote/session", "--teardown", *target])
    capsys.readouterr()
    assert (result == 0) is allowed
    if allowed:
        assert transports[0][kind].calls == [("stop", "/remote/session", 10.0), ("teardown",)]
    elif kind is not None:
        assert transports[0][kind].calls == []


def test_cli_import_and_help_do_not_import_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import weights_lab.cli; "
            "assert 'torch' not in sys.modules; weights_lab.cli._parser().parse_args(['--help'])",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Keep a local model loaded" in result.stdout


@pytest.mark.parametrize("command", ["send", "peek", "stop", "backup"])
def test_local_commands_expand_session_home_directory(tmp_path, monkeypatch, capsys, command):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "session"
    create_session(path, {"model": "unused"})
    write_json(path / "status.json", {"state": "ready" if command == "send" else "stopped"})
    args = [command, "--session", "~/session"]
    if command == "send":
        args += ["--code", "print('local')"]
    if command == "backup":
        args += ["--output", "~/backup.tar.gz"]
    assert cli.main(args) == 0
    capsys.readouterr()
    if command == "send":
        assert len(list((path / "inbox").glob("*.py"))) == 1
    if command == "backup":
        assert (tmp_path / "backup.tar.gz").exists()


def test_remote_session_home_is_not_expanded_on_local_machine(transports, capsys):
    assert cli.main(["peek", "--host", "gpu", "--session", "~/remote/session"]) == 0
    capsys.readouterr()
    assert transports[0]["SSHLauncher"].calls == [
        ("run", ["peek", "--session", "~/remote/session"], None)
    ]


def test_slurm_backup_consults_scheduler_even_when_status_claims_stopped(
    slurm_session, tmp_path, transports, capsys
):
    write_json(slurm_session / "status.json", {"state": "stopped", "pid": 87654321})
    transports[0]["SlurmLauncher"].active = True
    output = tmp_path / "backup.tar.gz"
    assert cli.main(["backup", "--session", str(slurm_session), "--output", str(output)]) != 0
    assert "stop" in capsys.readouterr().err
    assert transports[0]["SlurmLauncher"].calls == [("is_active", slurm_session.resolve())]
    assert not output.exists()
