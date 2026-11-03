import importlib.util
import subprocess
from pathlib import Path


def harness():
    path = Path(__file__).resolve().parents[1] / "scripts/validate_slurm.py"
    spec = importlib.util.spec_from_file_location("slurm_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_test_cluster_config_keeps_authentication_and_state_in_owned_namespace(tmp_path):
    config = harness().configuration(tmp_path, "validation-host", (16817, 16818))
    assert "SlurmctldHost=validation-host(127.0.0.1)" in config
    assert "NodeAddr=127.0.0.1" in config
    assert "CommunicationParameters=NoInAddrAny,NoCtldInAddrAny" in config
    assert f"AuthInfo=socket={tmp_path}/munge/socket" in config
    for line in config.splitlines():
        if "Location=" in line or "Dir=" in line or "File=" in line:
            assert str(tmp_path) in line
    assert "MaxTime=00:03:00" in config


def test_install_does_not_stop_an_existing_munge_service(tmp_path, monkeypatch):
    module = harness()
    monkeypatch.setattr(
        module.shutil, "which", lambda name: "/usr/sbin/munged" if name == "munged" else None
    )
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.install_packages(tmp_path)
    assert len(calls) == 1
    assert calls[0][0][:2] == ["apt-get", "install"]
    assert "--no-upgrade" in calls[0][0]
    assert calls[0][1]["env"]["NEEDRESTART_MODE"] == "l"


def test_install_stops_only_newly_installed_munge(tmp_path, monkeypatch):
    module = harness()
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: calls.append(args))
    assert module.install_packages(tmp_path)
    assert calls[-1] == ["systemctl", "stop", "munge"]


def test_daemons_use_private_uts_hostname():
    command = harness().isolated_daemon("slurmd", "-D")
    assert command == [
        "unshare",
        "--uts",
        "sh",
        "-c",
        'hostname localhost && exec "$@"',
        "sh",
        "slurmd",
        "-D",
    ]
