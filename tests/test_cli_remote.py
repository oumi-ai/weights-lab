import io
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


class CloudTransport(Transport):
    def teardown(self):
        self.calls.append(("teardown",))


@pytest.fixture
def transports(monkeypatch):
    instances = {}
    constructors = []
    instance = CloudTransport()
    instances["SkyPilotLauncher"] = instance

    def construct(*args, **kwargs):
        constructors.append(("SkyPilotLauncher", args, kwargs))
        return instance

    monkeypatch.setattr(launchers, "SkyPilotLauncher", construct)
    return instances, constructors


@pytest.mark.parametrize("input_kind", ["file", "code", "stdin"])
def test_remote_send_delivers_snippet_contents_on_stdin(
    tmp_path, transports, capsys, monkeypatch, input_kind
):
    instances, constructors = transports
    source = tmp_path / "local snippet.py"
    code = "print('literal $HOME $(false) `false` λ')\n"
    source.write_text(code)
    arguments = [
        "send",
        "--cluster",
        "remote",
        "--session",
        "/remote/session",
        "--wait",
        "--follow",
        "--timeout",
        "7",
        "--remote-python",
        "/env/bin/python",
    ]
    if input_kind == "file":
        arguments += ["--file", str(source)]
    elif input_kind == "code":
        arguments += ["--code", code]
    else:
        monkeypatch.setattr(cli.sys, "stdin", io.StringIO(code))
    assert cli.main(arguments) == 0
    capsys.readouterr()
    assert constructors[0][0] == "SkyPilotLauncher"
    assert constructors[0][2]["python"] == "/env/bin/python"
    method, command, stdin = instances["SkyPilotLauncher"].calls[0]
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
    transport = instances["SkyPilotLauncher"]
    if failure != "result":
        transport.error = subprocess.CalledProcessError(
            7,
            ["remote"],
            output=stdout.encode() if failure == "bytes_exception" else stdout,
            stderr=stderr.encode() if failure == "bytes_exception" else stderr,
        )
    else:
        transport.result = subprocess.CompletedProcess([], 7, stdout, stderr)
    assert cli.main(["peek", "--cluster", "gpu", "--session", "/remote/session"]) == 7
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
        "--cluster",
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
    assert instances["SkyPilotLauncher"].calls == [
        (
            "run",
            ["backup", "--session", "/remote/session", "--output", "/remote/archive.tar.gz"],
            None,
        ),
        ("download", "/remote/archive.tar.gz", destination.resolve()),
    ]
    assert json.loads(capsys.readouterr().out)["output"] == str(destination.resolve())
    calls_before = instances["SkyPilotLauncher"].calls.copy()
    assert cli.main(args) != 0
    capsys.readouterr()
    assert instances["SkyPilotLauncher"].calls == calls_before
    assert destination.read_bytes() == b"remote archive"


@pytest.mark.parametrize("explicit", [False, True])
def test_sky_start_forwards_checkpoint_pins_budget_and_infra(transports, capsys, explicit):
    arguments = [
        "start",
        "--cluster",
        "lab",
        "--session",
        "~/remote/session",
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
        "--infra",
        "slurm/mycluster/gpu",
        "--accelerators",
        "A100:1",
    ]
    if explicit:
        arguments += ["--launcher", "sky"]
    assert cli.main(arguments) == 0
    capsys.readouterr()
    assert transports[0]["SkyPilotLauncher"].calls == [
        (
            "start",
            "base",
            "~/remote/session",
            {
                "adapter": "adapter",
                "device": "cuda:0",
                "dtype": "bfloat16",
                "snippet_timeout": 20.0,
                "pass_limit": 12,
                "revision": "base-sha",
                "adapter_revision": "adapter-sha",
                "source": None,
                "accelerators": "A100:1",
                "idle_minutes": 15,
                "infra": "slurm/mycluster/gpu",
            },
        )
    ]


@pytest.mark.parametrize(
    "removed",
    [
        ["--launcher", "ssh"],
        ["--launcher", "slurm"],
        ["--launcher", "modal"],
        ["--host", "gpu"],
        ["--sandbox-id", "sandbox"],
        ["--install"],
        ["--partition", "gpu"],
        ["--gres", "gpu:1"],
        ["--time-limit", "01:00:00"],
        ["--gpu", "L4"],
        ["--lifetime", "3600"],
        ["--app-name", "lab"],
    ],
)
def test_removed_flags_rejected_before_provisioning(tmp_path, transports, removed):
    path = tmp_path / "session"
    with pytest.raises(SystemExit) as error:
        cli.main(["start", "--cluster", "lab", "--model", "base", "--session", str(path), *removed])
    assert error.value.code == 2
    assert transports[1] == []
    assert not path.exists()


@pytest.mark.parametrize("target", [["--launcher", "sky"], ["--infra", "slurm/lab/gpu"]])
def test_sky_start_requires_cluster_before_resources(tmp_path, transports, capsys, target):
    path = tmp_path / "session"
    assert cli.main(["start", "--session", str(path), "--model", "base", *target]) != 0
    assert "--cluster" in capsys.readouterr().err
    assert transports[1] == []
    assert not path.exists()


@pytest.mark.parametrize(
    "target",
    [
        [],
        ["--cluster", "cluster"],
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


@pytest.mark.parametrize("target,allowed", [([], False), (["--cluster", "cluster"], True)])
def test_teardown_requires_cluster(transports, capsys, target, allowed):
    result = cli.main(["stop", "--session", "/remote/session", "--teardown", *target])
    output = capsys.readouterr()
    assert (result == 0) is allowed
    if allowed:
        assert transports[0]["SkyPilotLauncher"].calls == [
            ("stop", "/remote/session", 10.0),
            ("teardown",),
        ]
    else:
        assert "--cluster" in output.err
        assert transports[0]["SkyPilotLauncher"].calls == []


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
    assert cli.main(["peek", "--cluster", "gpu", "--session", "~/remote/session"]) == 0
    capsys.readouterr()
    assert transports[0]["SkyPilotLauncher"].calls == [
        ("run", ["peek", "--session", "~/remote/session"], None)
    ]
