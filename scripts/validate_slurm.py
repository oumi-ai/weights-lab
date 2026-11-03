"""Run an isolated, localhost-only Slurm smoke test on a disposable Linux VM.

Requires root and the installed weights-lab environment; installs only missing
Munge/Slurm packages, never updates the Python environment. Every daemon, spool,
key, and session belongs to the new output directory. The scheduler job has a
three-minute limit and the complete validation has a ten-minute alarm.
"""

import argparse
import hashlib
import json
import os
import pwd
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def install_packages(root):
    missing_munge = shutil.which("munged") is None
    missing = any(
        shutil.which(name) is None for name in ("munged", "slurmd", "slurmctld", "sbatch")
    )
    if not missing:
        return False
    environment = {**os.environ, "DEBIAN_FRONTEND": "noninteractive", "NEEDRESTART_MODE": "l"}
    with (root / "apt.log").open("w") as stream:
        subprocess.run(
            [
                "apt-get",
                "install",
                "--no-install-recommends",
                "--no-upgrade",
                "-y",
                "munge",
                "slurm-wlm",
            ],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if missing_munge:
        # Only stop the service newly installed by this invocation.
        subprocess.run(["systemctl", "stop", "munge"], check=True)
    return True


def configuration(root, hostname, ports):
    node = f"NodeName=weightslab NodeHostname={hostname} NodeAddr=127.0.0.1"
    resources = "CPUs=2 Sockets=1 CoresPerSocket=2 ThreadsPerCore=1 RealMemory=8192 State=UNKNOWN"
    return f"""ClusterName=weights-lab-validation
SlurmctldHost={hostname}(127.0.0.1)
SlurmctldPort={ports[0]}
SlurmdPort={ports[1]}
SlurmUser=root
AuthType=auth/munge
AuthInfo=socket={root}/munge/socket
CredType=cred/munge
CommunicationParameters=NoInAddrAny,NoCtldInAddrAny
StateSaveLocation={root}/state
SlurmdSpoolDir={root}/spool
SlurmctldPidFile={root}/ctld.pid
SlurmdPidFile={root}/slurmd.pid
SlurmctldLogFile={root}/ctld.log
SlurmdLogFile={root}/slurmd.log
ProctrackType=proctrack/pgid
TaskPlugin=task/none
SelectType=select/cons_tres
SelectTypeParameters=CR_CPU
SchedulerType=sched/backfill
JobAcctGatherType=jobacct_gather/none
SlurmdParameters=config_overrides
MpiDefault=none
ReturnToService=2
KillWait=2
MessageTimeout=5
SlurmctldTimeout=10
SlurmdTimeout=10
{node} {resources}
PartitionName=validation Nodes=weightslab Default=YES MaxTime=00:03:00 State=UP
"""


def isolated_daemon(*arguments):
    return [
        "unshare",
        "--uts",
        "sh",
        "-c",
        'hostname localhost && exec "$@"',
        "sh",
        *arguments,
    ]


def free_ports():
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        return first.getsockname()[1], second.getsockname()[1]


def wait_for(predicate, processes, timeout=25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("a validation daemon exited; inspect daemon logs")
        if predicate():
            return
        time.sleep(0.2)
    raise TimeoutError("validation daemon readiness timed out")


def make_model(root):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    model = root / "model"
    backend = Tokenizer(
        models.WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    ).save_pretrained(model)
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=4,
        n_embd=8,
        n_layer=1,
        n_head=1,
        n_positions=16,
        eos_token_id=1,
        bos_token_id=1,
        pad_token_id=1,
    )
    GPT2LMHeadModel(config).save_pretrained(model)
    return model


def validate(root):
    if os.geteuid() != 0:
        raise PermissionError("run this isolated VM validation as root")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report, processes, commands = {"state": "running"}, [], []
    launcher, started = None, None
    environment_before = os.environ.copy()
    original_alarm = signal.getsignal(signal.SIGALRM)

    def deadline(signum, frame):
        raise TimeoutError("Slurm validation exceeded its ten-minute deadline")

    def run(arguments, **kwargs):
        result = subprocess.run(arguments, **kwargs)
        commands.append(
            {
                "args": list(map(str, arguments)),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        return result

    def launch(name, arguments):
        stream = (root / f"{name}.stdout.log").open("w")
        process = subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stream.close()
        processes.append(process)

    def cli(*args):
        return run(
            [sys.executable, "-m", "weights_lab", *map(str, args)],
            text=True,
            capture_output=True,
            check=True,
            timeout=90,
        )

    try:
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(600)
        report["installed_packages"] = install_packages(root)
        report["slurm_version"] = subprocess.check_output(
            ["slurmd", "--version"], text=True
        ).strip()
        munge = pwd.getpwnam("munge")
        for name in ("munge", "state", "spool"):
            (root / name).mkdir()
        os.chown(root / "munge", munge.pw_uid, munge.pw_gid)
        key = root / "munge/key"
        key.write_bytes(os.urandom(1024))
        key.chmod(0o600)
        os.chown(key, munge.pw_uid, munge.pw_gid)
        ports = free_ports()
        config = root / "slurm.conf"
        config.write_text(configuration(root, "localhost", ports))
        os.environ.update(
            SLURM_CONF=str(config),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        launch(
            "munge",
            [
                "runuser",
                "-u",
                "munge",
                "--",
                "/usr/sbin/munged",
                "-F",
                "--force",
                "--socket",
                str(root / "munge/socket"),
                "--key-file",
                str(key),
                "--pid-file",
                str(root / "munge/pid"),
                "--log-file",
                str(root / "munge/log"),
            ],
        )
        wait_for(lambda: (root / "munge/socket").exists(), processes)
        launch("slurmctld", isolated_daemon("slurmctld", "-D", "-f", str(config)))
        launch("slurmd", isolated_daemon("slurmd", "-D", "-N", "weightslab", "-f", str(config)))

        def listeners_ready():
            listeners = subprocess.check_output(["ss", "-ltnp"], text=True)
            selected = [
                line
                for line in listeners.splitlines()
                if any(f":{port} " in line for port in ports)
            ]
            if any("127.0.0.1:" not in line for line in selected):
                raise RuntimeError(f"validation daemon listening outside loopback: {selected}")
            return selected if len(selected) == 2 else None

        wait_for(listeners_ready, processes)
        selected = listeners_ready()
        report.update(ports=ports, listeners=selected, daemon_pids=[p.pid for p in processes])
        wait_for(
            lambda: (
                subprocess.run(
                    ["sinfo", "--noheader", "--format=%T"], text=True, capture_output=True
                ).stdout.strip()
                == "idle"
            ),
            processes,
        )
        (root / "setup.json").write_text(json.dumps(report, indent=2))
        from weights_lab.launchers import SlurmLauncher
        from weights_lab.session import read_json

        session = root / "session"
        launcher = SlurmLauncher(python=sys.executable, runner=run)
        started = launcher.start(
            str(make_model(root)),
            session,
            partition="validation",
            gres=None,
            time_limit="00:03:00",
            device="cpu",
            dtype="float32",
            snippet_timeout=20,
            pass_limit=8,
        )
        report["start"] = started
        code = """import json, os, torch
assert lab.device.type == 'cpu'
inputs = tokenizer('hello world', return_tensors='pt').to(lab.device)
with torch.no_grad():
    logits = model(**inputs).logits
assert logits.shape == (1, 2, 4)
(artifacts / 'job.json').write_text(json.dumps({'job_id': os.environ['SLURM_JOB_ID'],
    'device': str(lab.device), 'passes': lab.forward_passes}))
print('actual Slurm job', os.environ['SLURM_JOB_ID'])
"""
        result = launcher.run(
            ["send", "--session", str(session), "--wait", "--timeout", "90"], input=code
        )
        report["snippet"] = json.loads(result.stdout)
        report["worker"] = read_json(session / "artifacts/job.json")
        assert report["worker"]["job_id"] == started["job_id"]
        report["peek"] = json.loads(launcher.run(["peek", "--session", str(session)]).stdout)
        assert report["peek"]["state"] == "ready" and launcher.is_active(session)

        def forbidden_signal(*args):
            raise AssertionError("login process attempted to signal a compute-node PID")

        original_kill = os.kill
        os.kill = forbidden_signal
        try:
            report["stop"] = launcher.stop(session, timeout=30)
        finally:
            os.kill = original_kill
        report["active_after_stop"] = launcher.is_active(session)
        assert not report["active_after_stop"]
        assert any(item["args"][0] == "scancel" for item in commands)
        backup = root / "session.tar.gz"
        cli("backup", "--session", session, "--output", backup)
        with tarfile.open(backup) as archive:
            files = [entry for entry in archive.getmembers() if entry.isfile()]
            for entry in files:
                assert (
                    hashlib.sha256((root / entry.name).read_bytes()).digest()
                    == hashlib.sha256(archive.extractfile(entry).read()).digest()
                )
        report.update(
            state="succeeded",
            backup_files_verified=len(files),
            backup_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),
        )
    except BaseException as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        try:
            if (
                started is not None
                and launcher is not None
                and launcher.is_active(root / "session")
            ):
                launcher.stop(root / "session", timeout=15)
        except BaseException as error:
            report["job_cleanup_error"] = f"{type(error).__name__}: {error}"
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        report["daemon_cleanup"] = [{"pid": p.pid, "returncode": p.returncode} for p in processes]
        report["commands"] = commands
        report["elapsed_finished_at"] = time.time()
        for name in ("state", "spool", "munge"):
            shutil.rmtree(root / name, ignore_errors=True)
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        signal.signal(signal.SIGALRM, original_alarm)
        os.environ.clear()
        os.environ.update(environment_before)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(validate(arguments.output_dir), indent=2))
