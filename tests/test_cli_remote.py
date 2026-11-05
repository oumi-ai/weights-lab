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

from oumi_weights_lab import cli, launchers
from oumi_weights_lab.session import create_session, write_json


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
        return subprocess.CompletedProcess([], 0, '{"state":"stopped"}\n', "")

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


def test_sky_start_forwards_gpu_placement_only_when_given(transports, capsys):
    start = ["start", "--cluster", "lab", "--session", "/remote/s", "--model", "base"]
    placement = ["--device-map", "balanced", "--max-memory", "0=70GiB,1=70GiB"]
    assert cli.main([*start, "--accelerators", "H100:2", *placement]) == 0
    assert cli.main(start) == 0
    capsys.readouterr()
    first, second = (call[3] for call in transports[0]["SkyPilotLauncher"].calls)
    assert first["device_map"] == "balanced"
    assert first["max_memory"] == {"0": "70GiB", "1": "70GiB"}
    assert first["accelerators"] == "H100:2"
    assert "device_map" not in second and "max_memory" not in second


def test_sky_start_forwards_checkpoint_pins_budget_and_infra(transports, capsys):
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


def test_sky_resume_uses_remote_bundle_path(transports, capsys):
    assert (
        cli.main(
            [
                "start",
                "--cluster",
                "lab",
                "--session",
                "new-session",
                "--model",
                "base",
                "--resume",
                "~/saved/candidate",
            ]
        )
        == 0
    )
    capsys.readouterr()
    call = transports[0]["SkyPilotLauncher"].calls[0]
    assert call[3]["resume"] == "~/saved/candidate"


def test_sky_forwards_worker_deadlines(transports, capsys):
    assert (
        cli.main(
            [
                "start",
                "--cluster",
                "lab",
                "--session",
                "bounded",
                "--model",
                "base",
                "--idle-timeout",
                "10",
                "--max-lifetime",
                "60",
                "--shutdown-grace",
                "2",
            ]
        )
        == 0
    )
    capsys.readouterr()
    config = transports[0]["SkyPilotLauncher"].calls[0][3]
    assert (config["idle_timeout"], config["max_lifetime"], config["shutdown_grace"]) == (10, 60, 2)


@pytest.mark.parametrize("flag", ["--idle-timeout", "--max-lifetime"])
def test_invalid_deadline_does_not_construct_cloud_transport(transports, capsys, flag):
    assert (
        cli.main(
            [
                "start",
                "--cluster",
                "lab",
                "--session",
                "bounded",
                "--model",
                "base",
                flag,
                "nan",
            ]
        )
        != 0
    )
    assert "finite" in capsys.readouterr().err
    assert transports[1] == []
    assert transports[0]["SkyPilotLauncher"].calls == []


@pytest.mark.parametrize(
    "removed",
    [
        ["--launcher", "sky"],
        ["--launcher", "local"],
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


def test_sky_infra_requires_cluster_before_resources(tmp_path, transports, capsys):
    path = tmp_path / "session"
    assert (
        cli.main(["start", "--session", str(path), "--model", "base", "--infra", "slurm/lab/gpu"])
        != 0
    )
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
            "import sys; import oumi_weights_lab.cli; "
            "assert 'torch' not in sys.modules; "
            "oumi_weights_lab.cli._parser().parse_args(['--help'])",
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


@pytest.fixture
def sky(monkeypatch):
    """Drive the real SkyPilotLauncher: SSH runs in a local shell and Sky is faked.

    A successful fake `sky launch -d` runs the task's session bootstrap without starting
    a model worker; like the real detached launch, a failing job does not fail launch.
    """
    state = SimpleNamespace(calls=[], codes={}, ssh_error=None)
    real = launchers.SkyPilotLauncher

    def runner(args, **kwargs):
        state.calls.append(list(args))
        if args[0] == "ssh":
            assert args[-2] == "lab"
            if state.ssh_error is not None:
                raise state.ssh_error
            return subprocess.run(["sh", "-c", args[-1]], **kwargs)
        assert args[0] == "sky"
        code = state.codes.get(args[1], 0)
        if args[1] == "launch" and code == 0:
            task = json.loads(Path(args[-1]).read_text())
            script = shlex.split(task["run"])[2]
            stub = "import os\nos.dup2=lambda *args: None\nos.execv=lambda *args: None\n"
            subprocess.run([sys.executable, "-c", stub + script], capture_output=True)
        result = subprocess.CompletedProcess(
            args, code, f"Sky {args[1]} progress\n", f"Sky {args[1]} warning\n"
        )
        if kwargs.get("check"):
            result.check_returncode()
        return result

    def construct(cluster, python="python3"):
        return real(cluster, python=sys.executable, runner=runner)

    monkeypatch.setattr(launchers, "SkyPilotLauncher", construct)
    return state


def test_real_launcher_send_and_peek_relay_json_and_raw_snippet_output(tmp_path, sky, capsysbinary):
    path = create_session(tmp_path / "session", {"model": "unused"})
    code = "print('λ')\n"
    assert cli.main(["send", "--cluster", "lab", "--session", str(path), "--code", code]) == 0
    queued = json.loads(capsysbinary.readouterr().out)
    assert queued["state"] == "queued"
    assert (path / "inbox" / f"{queued['id']}.py").read_text(encoding="utf-8") == code
    result = {"id": queued["id"], "state": "succeeded"}
    write_json(path / "results" / f"{queued['id']}.json", result)
    # A child process may write bytes that are not UTF-8; they must not hide the result.
    (path / "outputs" / f"{queued['id']}.log").write_bytes(b"partial \xff\n")
    peek = ["peek", "--cluster", "lab", "--session", str(path), "--id", queued["id"]]
    assert cli.main(peek) == 0
    output = capsysbinary.readouterr()
    assert json.loads(output.out) == result
    assert output.err == b"partial \xff\n"


@pytest.mark.parametrize("down_code", [0, 3])
@pytest.mark.parametrize("stop_failure", [None, "remote", "ssh"])
def test_real_launcher_stop_teardown_always_tears_down_and_reports_it(
    tmp_path, sky, capsys, stop_failure, down_code
):
    path = create_session(tmp_path / "session", {"model": "unused"})
    if stop_failure == "remote":
        path = tmp_path / "missing"
    elif stop_failure == "ssh":
        sky.ssh_error = FileNotFoundError(2, "No such file or directory", "ssh")
    sky.codes["down"] = down_code
    result = cli.main(["stop", "--cluster", "lab", "--session", str(path), "--teardown"])
    output = capsys.readouterr()
    assert ["sky", "down", "-y", "lab"] in sky.calls
    # A stop failure's exit code wins; otherwise a failed teardown's code is surfaced.
    assert result == (1 if stop_failure else down_code)
    if stop_failure:
        assert output.out == ""
        assert "stop failed" in output.err
    else:
        assert json.loads(output.out)["state"] == "stopped"
    if down_code:
        assert "Sky down progress" in output.err
        assert "teardown of cluster lab failed" in output.err
    else:
        assert "cluster lab was torn down" in output.err


def test_real_launcher_backup_downloads_remote_archive(tmp_path, sky, capsys):
    path = create_session(tmp_path / "session", {"model": "unused"})
    write_json(path / "status.json", {"state": "stopped"})
    remote = tmp_path / "remote.tar.gz"
    local = tmp_path / "local.tar.gz"
    arguments = ["backup", "--cluster", "lab", "--session", str(path)]
    arguments += ["--output", str(local), "--remote-output", str(remote)]
    assert cli.main(arguments) == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported == {"session": str(path), "output": str(local), "remote_output": str(remote)}
    assert local.read_bytes() == remote.read_bytes()
    with tarfile.open(local) as archive:
        assert "session/config.json" in archive.getnames()


def test_sky_launch_failure_keeps_sky_output_off_stdout(tmp_path, sky, capsys):
    sky.codes["launch"] = 1
    arguments = ["start", "--cluster", "lab", "--session", str(tmp_path / "s"), "--model", "m"]
    assert cli.main(arguments) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Sky launch progress" in output.err and "Sky launch warning" in output.err


def test_real_launcher_start_maps_relative_resume_like_session(tmp_path, sky, capsys):
    path = tmp_path / "session"
    arguments = ["start", "--cluster", "lab", "--session", str(path), "--model", "m"]
    assert cli.main([*arguments, "--resume", "sessions/old/artifacts/candidate"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "starting"
    config = json.loads((path / "config.json").read_text())
    assert config["resume"] == "~/sky_workdir/sessions/old/artifacts/candidate"


def test_start_over_an_existing_session_fails_instead_of_reporting_the_old_worker(
    tmp_path, sky, capsys
):
    path = create_session(tmp_path / "session", {"model": "old-model"})
    write_json(path / "status.json", {"state": "ready", "model": "old-model"})
    arguments = ["start", "--cluster", "lab", "--session", str(path), "--model", "new-model"]
    assert cli.main(arguments) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "another launch" in output.err
    assert json.loads((path / "config.json").read_text())["model"] == "old-model"


def test_remote_commands_in_a_shell_loop_leave_loop_input_alone(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Like real ssh, forward this process's stdin whether or not the remote reads it.
    (bin_dir / "ssh").write_text(
        '#!/bin/sh\nwhile [ "$1" != "--" ]; do shift; done\nshift 2\ncat | sh -c "$1"\n'
    )
    (bin_dir / "ssh").chmod(0o755)
    paths = [create_session(tmp_path / f"session-{index}", {"model": "m"}) for index in (1, 2)]
    listing = tmp_path / "sessions.txt"
    listing.write_text("".join(f"{path}\n" for path in paths))
    loop = (
        'while read -r s; do "$PY" -m oumi_weights_lab peek --cluster lab '
        '--remote-python "$PY" --session "$s"; done < "$LIST"'
    )
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PY": sys.executable,
        "LIST": str(listing),
    }
    result = subprocess.run(
        ["sh", "-c", loop], env=environment, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert [json.loads(line)["state"] for line in result.stdout.splitlines()] == [
        "starting",
        "starting",
    ]
