"""Run bounded CUDA, SSH, and optional frozen Qwen checks on a dedicated Sky cluster."""

import argparse
import json
import runpy
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from oumi_weights_lab.launchers import SkyPilotLauncher, _SSHTransport

PYTHON = "/tmp/oumi-weights-lab-env/bin/python"
UV = "/tmp/oumi-weights-lab-uv/uv"
SESSION = "/tmp/oumi-weights-lab-transport"
REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
SETUP = (
    "set -euo pipefail\n"
    "curl -LsSf https://astral.sh/uv/0.10.12/install.sh | "
    "env UV_INSTALL_DIR=/tmp/oumi-weights-lab-uv UV_NO_MODIFY_PATH=1 sh\n"
    f"{UV} venv --seed --python 3.11 /tmp/oumi-weights-lab-env\n"
    f"{UV} pip install --python {PYTHON} '.[adapters,datasets]'\n"
)


def run_logged(command, log, *, source=None, timeout=300):
    """Let the OS write combined output immediately while waiting for completion."""
    log = Path(log)
    offset = log.stat().st_size if log.exists() else 0
    with log.open("ab", buffering=0) as output:
        result = subprocess.run(
            command,
            input=source,
            stdin=subprocess.DEVNULL if source is None else None,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=isinstance(source, str),
        )
    with log.open("rb") as output:
        output.seek(offset)
        captured = output.read().decode(errors="replace")
    return subprocess.CompletedProcess(command, result.returncode, captured, "")


def watchdog(cluster, deadline, marker, *, runner=subprocess.run, now=time.time, sleep=time.sleep):
    """Independent cleanup; a successful main teardown suppresses further work."""
    marker = Path(marker)
    while now() < deadline:
        if marker.exists():
            return 0
        sleep(min(15, max(0, deadline - now())))
    for attempt in range(3):
        if marker.exists():
            return 0
        try:
            result = runner(["sky", "down", "-y", cluster], timeout=120, check=False)
            if result.returncode == 0:
                marker.write_text("watchdog teardown succeeded\n")
                return 0
        except (OSError, subprocess.TimeoutExpired):
            pass
        if attempt < 2:
            sleep(10)
    return 1


def cleanup(launcher, root, report):
    """Always attempt teardown even when stop, provisioning, or validation failed."""
    try:
        launcher.stop(SESSION, timeout=15)
    except Exception as error:
        report["stop_error"] = f"{type(error).__name__}: {error}"
    errors = []
    for _ in range(3):
        try:
            result = launcher.teardown()
            if result.returncode != 0:
                raise RuntimeError(f"sky down returned {result.returncode}")
            (root / "cleanup.done").write_text("main teardown succeeded\n")
            report["cluster_terminated"] = True
            break
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    if errors:
        report["teardown_errors"] = errors
    if not report.get("cluster_terminated"):
        report["state"] = "failed"


def validate(
    output,
    *,
    source=None,
    prepared_benchmark=None,
    max_seconds=14400,
    cluster=None,
):
    if not isinstance(max_seconds, int) or max_seconds < 60 or max_seconds > 21600:
        raise ValueError("max_seconds must be between 60 and 21600")
    source = Path(source or Path(__file__).resolve().parents[2]).resolve()
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cluster = cluster or f"oumi-weights-lab-val-{datetime.now(UTC):%Y%m%d-%H%M%S-%f}"
    if not cluster.startswith("oumi-weights-lab-val-"):
        raise ValueError("validation requires a dedicated oumi-weights-lab-val-* cluster name")
    status = subprocess.run(
        ["sky", "status", "--all-users", "--output", "json"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    if any(record["name"] == cluster for record in json.loads(status.stdout)):
        raise ValueError("refusing to reuse an existing cluster")
    prepared = Path(prepared_benchmark).resolve() if prepared_benchmark else None
    if prepared:
        # Validate local frozen inputs before provisioning anything.
        runpy.run_path(str(source / "tests/live/validate_qwen_benchmark.py"))["read_prepared"](
            prepared
        )
    started, deadline = time.time(), time.time() + max_seconds
    report = {
        "state": "running",
        "cluster": cluster,
        "max_seconds": max_seconds,
        "deadline": deadline,
        "cloud": "nebius",
        "region": "us-central1",
        "gpu_request": "H200:1",
        "instance_type": "gpu-h200-sxm_1gpu-16vcpu-200gb",
    }

    def runner(command, **kwargs):
        command = list(command)
        if command[:2] == ["sky", "launch"]:
            spec = Path(command[-1])
            task = json.loads(spec.read_text())
            task["resources"].update(
                cloud="nebius",
                region="us-central1",
                disk_size=256,
                instance_type=report["instance_type"],
                accelerators="H200:1",
            )
            task["setup"] = SETUP
            task["run"] = (
                shlex.join(["timeout", "--signal=TERM", "--kill-after=30s", str(max_seconds)])
                + " "
                + task["run"]
            )
            spec.write_text(json.dumps(task))
            (root / "task.json").write_text(json.dumps(task, indent=2))
        remaining = max(1, deadline - time.time())
        timeout = (
            120
            if command[:2] == ["sky", "down"]
            else min(kwargs.get("timeout") or remaining, remaining)
        )
        if kwargs.get("stdout") is not None:
            # Checkpoint archives must reach the launcher's binary destination intact.
            return subprocess.run(
                command,
                stdin=kwargs.get("stdin"),
                stdout=kwargs["stdout"],
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=kwargs.get("check", False),
            )
        if command[0] == "ssh":
            # Short CLI calls need clean JSON stdout even when SSH warns on stderr.
            # Launcher commands exchange bytes unless the caller asks for text.
            result = subprocess.run(
                command,
                input=kwargs.get("input"),
                stdin=kwargs.get("stdin"),
                capture_output=True,
                text=kwargs.get("text", False),
                timeout=timeout,
            )
            with (root / "transport.log").open("a") as log:
                for stream in (result.stdout, result.stderr):
                    log.write(
                        stream.decode(errors="replace")
                        if isinstance(stream, bytes)
                        else stream or ""
                    )
            if kwargs.get("check"):
                result.check_returncode()
            return result
        result = run_logged(
            command,
            root / ("sky-down.log" if command[:2] == ["sky", "down"] else "transport.log"),
            source=kwargs.get("input"),
            timeout=timeout,
        )
        if kwargs.get("check"):
            result.check_returncode()
        return result

    launcher = SkyPilotLauncher(cluster, python=PYTHON, runner=runner)
    ssh = _SSHTransport(cluster, python=PYTHON, runner=runner)

    def execute(arguments, name, data=None):
        result = run_logged(
            ssh._command(arguments),
            root / name,
            source=data,
            timeout=max(1, deadline - time.time()),
        )
        result.check_returncode()
        return result

    def upload(local, remote):
        execute(
            [
                "-c",
                "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                "p.parent.mkdir(parents=True,exist_ok=True); "
                "p.write_bytes(sys.stdin.buffer.read())",
                remote,
            ],
            "uploads.log",
            Path(local).read_bytes(),
        )

    with (root / "watchdog.log").open("ab", buffering=0) as log:
        guard = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--watchdog",
                "--cluster",
                cluster,
                "--deadline",
                str(deadline),
                "--cleanup-marker",
                str(root / "cleanup.done"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    report["watchdog_pid"] = guard.pid
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        report["launch"] = launcher.start(
            "sshleifer/tiny-gpt2",
            SESSION,
            source=source,
            accelerators="H200:1",
            idle_minutes=5,
            device="cuda",
            dtype="float32",
            revision=REVISION,
            pass_limit=64,
        )
        print(json.dumps({"stage": "cluster_started", "cluster": cluster}), flush=True)
        result = json.loads(
            launcher.run(
                ["send", "--session", SESSION, "--wait", "--timeout", "180"],
                input="import json,torch\nassert lab.device.type=='cuda'\n"
                "inputs=tokenizer('hello world',return_tensors='pt').to(lab.device)\n"
                "with torch.no_grad():\n    logits=model(**inputs).logits\n"
                "assert logits.device.type=='cuda'\n"
                "(artifacts/'transport.json').write_text(json.dumps({'cuda':True,'passes':lab.forward_passes}))\n",
            ).stdout
        )
        if result["state"] != "succeeded":
            raise RuntimeError("CUDA transport snippet did not succeed")
        report["transport_snippet"] = result
        upload(source / "tests/live/validate_runtime.py", "/tmp/validate_runtime.py")
        execute(
            [
                "/tmp/validate_runtime.py",
                "--device",
                "cuda",
                "--output-dir",
                "/tmp/oumi-weights-lab-runtime",
            ],
            "cuda-runtime.log",
        )
        ssh.download("/tmp/oumi-weights-lab-runtime/report.json", root / "cuda-runtime.json")
        ssh.download("/tmp/oumi-weights-lab-runtime/session.tar.gz", root / "cuda-runtime.tar.gz")
        runtime = json.loads((root / "cuda-runtime.json").read_text())
        if runtime["state"] != "succeeded" or runtime["worker_alive_after_stop"]:
            raise RuntimeError("CUDA runtime harness did not pass and stop its worker")
        report["cuda_runtime"] = runtime
        if prepared:
            remote = "/tmp/oumi-weights-lab-benchmark"
            for name in ("manifest.json", "dataset.jsonl", "prompts.jsonl", "upstream_mathqa.py"):
                upload(prepared / name, f"{remote}/{name}")
            benchmark = source / "tests/live/validate_qwen_benchmark.py"
            upload(benchmark, "/tmp/validate_qwen_benchmark.py")
            required = runpy.run_path(str(benchmark))["REQUIRED"]
            install = [
                UV,
                "pip",
                "install",
                "--python",
                PYTHON,
                *[f"{name}=={version}" for name, version in required.items()],
            ]
            # _command prefixes Python; delegate tool execution to its subprocess module.
            execute(
                ["-c", "import subprocess,sys; subprocess.run(sys.argv[1:],check=True)", *install],
                "benchmark-install.log",
            )
            try:
                execute(
                    [
                        "-c",
                        "import sys; from pathlib import Path; sys.path.insert(0,'/tmp'); "
                        "from validate_qwen_benchmark import run_parent; "
                        f"print(run_parent(Path({remote!r})))",
                    ],
                    "benchmark.log",
                )
            finally:
                ssh.download(remote, root / "benchmark")
            report["benchmark"] = json.loads((root / "benchmark/summary.json").read_text())
            if not report["benchmark"].get("full_validated"):
                raise RuntimeError("full frozen benchmark protocol did not finish")
        report["stop"] = json.loads(launcher.stop(SESSION, timeout=30).stdout)
        launcher.run(["backup", "--session", SESSION, "--output", "/tmp/transport.tar.gz"])
        ssh.download("/tmp/transport.tar.gz", root / "transport.tar.gz")
        report["state"] = "succeeded"
    except BaseException as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup(launcher, root, report)
        report["duration_seconds"] = time.time() - started
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report.get("cluster_terminated"):
        raise RuntimeError("cluster teardown failed; independent watchdog remains armed")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prepared-benchmark", type=Path)
    parser.add_argument("--max-seconds", type=int, default=14400)
    parser.add_argument("--cluster")
    parser.add_argument("--watchdog", action="store_true")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--cleanup-marker", type=Path)
    options = parser.parse_args()
    if options.watchdog:
        raise SystemExit(watchdog(options.cluster, options.deadline, options.cleanup_marker))
    if not options.output_dir:
        parser.error("--output-dir is required")
    validate(
        options.output_dir,
        source=options.source,
        prepared_benchmark=options.prepared_benchmark,
        max_seconds=options.max_seconds,
        cluster=options.cluster,
    )
