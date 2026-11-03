"""Optional remote transports. Importing this module needs no cloud SDKs."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

from .budget import PassBudget
from .session import cancel_unfinished, create_session, read_json, session_lock, write_json

_SLURM_TERMINAL = {
    "COMPLETED",
    "CANCELLED",
    "FAILED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "SPECIAL_EXIT",
}

_SESSION_PROBE = (
    "import json,sys\nfrom pathlib import Path\n"
    "p=Path(sys.argv[1]).expanduser().resolve()\n"
    "try:\n"
    "    config=json.loads((p/'config.json').read_text())\n"
    "    status=json.loads((p/'status.json').read_text())\n"
    "except (FileNotFoundError,json.JSONDecodeError):\n"
    "    print('session files not created yet',file=sys.stderr)\n"
    "    raise SystemExit(75)\n"
    "if (not isinstance(config,dict) or not config.get('model')\n"
    "    or not isinstance(status,dict) or not status.get('state')):\n"
    "    print('invalid session configuration or status',file=sys.stderr)\n"
    "    raise SystemExit(2)\n"
    "if status.get('state') in {'failed','stopped','cancelled'}:\n"
    "    print(json.dumps(status),file=sys.stderr)\n"
    "    raise SystemExit(2)\n"
)


def _scheduler_states(runner, query):
    try:
        output = _checked(runner, query, text=True, capture_output=True).stdout
    except subprocess.CalledProcessError as error:
        message = error.stderr or ""
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        if "invalid job id" in message.lower():
            return []
        raise
    return [
        state
        for line in output.splitlines()
        if not line.strip().startswith("CLUSTER:")
        for state in line.split()
    ]


def _checked(runner, args, **kwargs):
    return runner(args, check=True, **kwargs)


def sanitized_workdir(source, destination):
    """Copy package sources and build metadata, excluding repo history and secrets."""
    source, destination = Path(source).resolve(), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        original = source / name
        if original.is_symlink() or not original.is_file():
            raise ValueError(f"package metadata must be a regular file: {name}")
        shutil.copyfile(original, destination / name)
    package = source / "src" / "weights_lab"
    if package.is_symlink() or not package.is_dir():
        raise ValueError("source must contain src/weights_lab")
    for original in package.rglob("*"):
        if original.is_symlink():
            raise ValueError("package source cannot contain symbolic links")
        if original.is_file() and original.suffix == ".py":
            target = destination / "src" / "weights_lab" / original.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    return destination


def _start_args(
    model,
    session,
    adapter=None,
    device="auto",
    dtype="auto",
    snippet_timeout=300,
    pass_limit=None,
    revision=None,
    adapter_revision=None,
):
    args = [
        "start",
        "--model",
        str(model),
        "--session",
        str(session),
        "--device",
        device,
        "--dtype",
        dtype,
        "--snippet-timeout",
        str(snippet_timeout),
    ]
    if adapter:
        args += ["--adapter", str(adapter)]
    for flag, value in (
        ("--pass-limit", pass_limit),
        ("--revision", revision),
        ("--adapter-revision", adapter_revision),
    ):
        if value is not None:
            args += [flag, str(value)]
    return args


def _configuration(model, config):
    if not isinstance(model, (str, Path)) or not str(model).strip():
        raise ValueError("model must be a nonempty checkpoint path or Hub ID")
    configuration = {
        "model": model,
        "device": "auto",
        "dtype": "auto",
        "snippet_timeout": 300,
        "poll_interval": 0.1,
        **config,
    }
    if configuration["dtype"] not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("unsupported model dtype")
    if not re.fullmatch(
        r"auto|cpu(?::[0-9]+)?|cuda(?::[0-9]+)?|mps(?::[0-9]+)?", str(configuration["device"])
    ):
        raise ValueError("unsupported model device")
    for name in ("snippet_timeout", "poll_interval"):
        number = float(configuration[name])
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be finite and positive")
    PassBudget(configuration.get("pass_limit"))
    return configuration


def _source_checkout(source):
    if source is not None:
        return Path(source)
    checkout = Path(__file__).resolve().parents[2]
    if not (checkout / "pyproject.toml").is_file():
        raise ValueError("source checkout unavailable; supply source='/path/to/weights-lab'")
    return checkout


def _json_output(result):
    return json.loads(result.stdout)


def _worker_script(session, configuration, keep_alive=False):
    script = (
        "import os,subprocess,sys,time\nfrom pathlib import Path\n"
        "from weights_lab.session import create_session\n"
        f"p=Path({str(session)!r}).expanduser().resolve()\n"
        f"create_session(p, {configuration!r})\n"
        "command=[sys.executable,'-m','weights_lab','_worker','--session',str(p)]\n"
        "log=(p/'worker.log').open('ab',buffering=0)\n"
    )
    if keep_alive:
        return script + "subprocess.run(command,stdout=log,stderr=log)\ntime.sleep(2147483647)\n"
    return (
        script
        + "os.dup2(log.fileno(),1)\nos.dup2(log.fileno(),2)\nos.execv(sys.executable,command)\n"
    )


class SSHLauncher:
    def __init__(self, host, python="python3", *, runner=subprocess.run):
        if not host or str(host).startswith("-") or any(char.isspace() for char in str(host)):
            raise ValueError("host must be an SSH host or config alias")
        self.host, self.python, self.runner = str(host), python, runner

    def _command(self, args):
        return ["ssh", "--", self.host, shlex.join([self.python, *map(str, args)])]

    def run(self, args, *, input=None):
        return _checked(
            self.runner,
            self._command(["-m", "weights_lab", *args]),
            input=input,
            text=True,
            capture_output=True,
        )

    def start(self, model, session, **config):
        return _json_output(self.run(_start_args(model, session, **config)))

    def send(self, session, code, *, wait=False, follow=False, timeout=300):
        args = ["send", "--session", str(session), "--timeout", str(timeout)]
        if wait:
            args.append("--wait")
        if follow:
            args.append("--follow")
        return _json_output(self.run(args, input=code))

    def stop(self, session, timeout=10):
        # This command runs on the remote host; remote PIDs never reach os.kill locally.
        return _json_output(
            self.run(["stop", "--session", str(session), "--timeout", str(timeout)])
        )

    def install(self, source):
        """Build an isolated source wheel locally, transfer it, then pip install remotely.

        Requires the `build` package locally and pip on the remote Python.
        """
        with tempfile.TemporaryDirectory(prefix="weights-lab-build-") as temporary:
            root = Path(temporary)
            sanitized_workdir(source, root / "source")
            _checked(
                self.runner,
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--outdir",
                    str(root / "dist"),
                    str(root / "source"),
                ],
                text=True,
                capture_output=True,
            )
            wheels = list((root / "dist").glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("wheel build must produce exactly one wheel")
            wheel = wheels[0]
            result = _checked(
                self.runner,
                self._command(["-c", _INSTALL_WHEEL, wheel.name]),
                input=wheel.read_bytes(),
                capture_output=True,
            )
            return result

    def download(self, remote_path, destination):
        """Stream a checkpoint or file and verify every file's remote SHA-256 digest."""
        destination = Path(destination).absolute()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".weights-lab-copy-", dir=destination.parent
        ) as tmp:
            root = Path(tmp)
            archive_path = root / "transfer.tar"
            with archive_path.open("wb") as output:
                _checked(
                    self.runner,
                    self._command(["-c", _EXPORT, str(remote_path)]),
                    stdout=output,
                    stderr=subprocess.PIPE,
                )
            extracted = root / "data"
            extracted.mkdir()
            manifest = _unpack_verified(archive_path, extracted)
            copied = (
                extracted / manifest["files"][0]["path"]
                if manifest["kind"] == "file"
                else extracted
            )
            os.replace(copied, destination)
        return destination


_INSTALL_WHEEL = """
import pathlib, subprocess, sys, tempfile
with tempfile.TemporaryDirectory(prefix='weights-lab-install-') as directory:
    wheel = pathlib.Path(directory) / pathlib.Path(sys.argv[1]).name
    wheel.write_bytes(sys.stdin.buffer.read())
    subprocess.run([sys.executable, '-m', 'pip', 'install', str(wheel)], check=True)
"""

_EXPORT = """
import hashlib, io, json, pathlib, sys, tarfile
source = pathlib.Path(sys.argv[1]).expanduser()
if source.is_symlink() or not source.exists():
    raise ValueError('source must exist and cannot be a symlink')
paths = sorted(source.rglob('*')) if source.is_dir() else [source]
if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in paths):
    raise ValueError('checkpoint must contain only regular files and directories')
files = []
for path in paths:
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        name = path.relative_to(source).as_posix() if source.is_dir() else path.name
        files.append({'path': name, 'sha256': digest.hexdigest()})
manifest = json.dumps({'kind': 'directory' if source.is_dir() else 'file', 'files': files}).encode()
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
    info = tarfile.TarInfo('.weights-lab-manifest.json')
    info.size = len(manifest)
    archive.addfile(info, io.BytesIO(manifest))
    for path, record in zip([path for path in paths if path.is_file()], files):
        archive.add(path, arcname=record['path'], recursive=False)
"""


def _safe_relative(name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or name != path.as_posix()
    ):
        raise ValueError(f"unsafe checkpoint archive path: {name}")
    return path


def _unpack_verified(archive_path, destination):
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        if (
            not members
            or members[0].name != ".weights-lab-manifest.json"
            or not members[0].isfile()
        ):
            raise ValueError("checkpoint transfer has no checksum manifest")
        manifest = json.load(archive.extractfile(members[0]))
        if manifest.get("kind") not in {"file", "directory"}:
            raise ValueError("invalid checkpoint kind")
        records = manifest.get("files", [])
        expected = {record["path"]: record["sha256"] for record in records}
        if len(expected) != len(records) or (manifest["kind"] == "file" and len(records) != 1):
            raise ValueError("invalid checkpoint manifest")
        for name in expected:
            _safe_relative(name)
        seen = set()
        for member in members[1:]:
            relative = _safe_relative(member.name)
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError("checkpoint archive contains unexpected files or links")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.extractfile(member) as source, target.open("xb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected[member.name]:
                raise ValueError(f"checksum mismatch for {member.name}")
            seen.add(member.name)
        if seen != set(expected):
            raise ValueError("checkpoint transfer is missing files")
        return manifest


class SlurmLauncher:
    """Submit from a login node with a filesystem shared by the compute node."""

    def __init__(self, python=sys.executable, *, runner=subprocess.run):
        self.python, self.runner = python, runner

    def start(self, model, session, *, partition=None, gres="gpu:1", time_limit=None, **config):
        path = Path(session).expanduser().resolve()
        create_session(path, _configuration(model, config))
        args = [
            "sbatch",
            "--parsable",
            "--output",
            str(path / "worker.log"),
            "--open-mode",
            "append",
        ]
        for flag, value in (("--partition", partition), ("--gres", gres), ("--time", time_limit)):
            if value is not None:
                args += [flag, str(value)]
        args += [
            "--wrap",
            shlex.join([self.python, "-m", "weights_lab", "_worker", "--session", str(path)]),
        ]
        try:
            result = _checked(self.runner, args, text=True, capture_output=True)
            identifier = result.stdout.strip().split(";", 1)
            if not identifier[0].isdigit():
                raise RuntimeError("sbatch did not return a numeric job ID")
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            with session_lock(path):
                write_json(
                    path / "status.json", {"state": "failed", "model": model, "error": str(exc)}
                )
                cancel_unfinished(path, "Slurm job submission failed")
            raise
        metadata = {"launcher": "slurm", "job_id": identifier[0], "session": str(path)}
        if len(identifier) > 1:
            metadata["cluster"] = identifier[1]
        write_json(path / "launcher.json", metadata)
        return {**metadata, "state": "queued"}

    def run(self, args, *, input=None):
        if not args or args[0] not in {"send", "peek"}:
            raise ValueError(
                "use SlurmLauncher.stop for cancellation; "
                "only send/peek run on the shared filesystem"
            )
        return _checked(
            self.runner,
            [self.python, "-m", "weights_lab", *map(str, args)],
            input=input,
            text=True,
            capture_output=True,
        )

    def is_active(self, session):
        metadata = read_json(Path(session).expanduser().resolve() / "launcher.json")
        job_id = str(metadata.get("job_id", ""))
        if metadata.get("launcher") != "slurm" or not job_id.isdigit():
            raise ValueError("session has no valid Slurm job ID")
        query = ["squeue", "--noheader", "--jobs", job_id, "--format=%T"]
        if metadata.get("cluster"):
            query += ["--clusters", metadata["cluster"]]
        return any(state not in _SLURM_TERMINAL for state in _scheduler_states(self.runner, query))

    def stop(self, session, timeout=10):
        path = Path(session).expanduser().resolve()
        if not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ValueError("timeout must be finite and positive")
        metadata = read_json(path / "launcher.json")
        job_id = metadata["job_id"]
        if metadata.get("launcher") != "slurm" or not str(job_id).isdigit():
            raise ValueError("session has no valid Slurm job ID")
        with session_lock(path):
            (path / "STOP").touch()
        args = ["scancel"]
        if metadata.get("cluster"):
            args += ["--clusters", metadata["cluster"]]
        cancellation_error = None
        try:
            _checked(self.runner, [*args, str(job_id)], text=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            # Repeated stop after the scheduler forgot the job is still successful.
            cancellation_error = exc
        query = ["squeue", "--noheader", "--jobs", str(job_id), "--format=%T"]
        if metadata.get("cluster"):
            query += ["--clusters", metadata["cluster"]]
        deadline = time.monotonic() + timeout
        while True:
            states = _scheduler_states(self.runner, query)
            if not states or all(state in _SLURM_TERMINAL for state in states):
                break
            if cancellation_error is not None:
                raise cancellation_error
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Slurm job {job_id} is still active after cancellation")
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        with session_lock(path):
            cancel_unfinished(path, "Slurm job stopped before completion")
            status = read_json(path / "status.json")
            if status.get("state") != "failed":
                status.update(state="stopped", updated_at=time.time())
            status.pop("current_snippet", None)
            write_json(path / "status.json", status)
        return {**metadata, **status}


class SkyPilotLauncher:
    def __init__(self, cluster, python="python3", *, runner=subprocess.run):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", cluster):
            raise ValueError("cluster must contain lowercase letters, digits, and hyphens")
        self.cluster, self.python, self.runner = cluster, python, runner

    @staticmethod
    def _path(path):
        """Resolve relative remote paths against Sky's workdir without using local HOME."""
        path = str(path)
        if PurePosixPath(path).is_absolute() or path.startswith("~"):
            return path
        return str(PurePosixPath("~/sky_workdir") / path)

    def start(
        self,
        model,
        session,
        *,
        source=None,
        accelerators="A100:1",
        idle_minutes=15,
        bootstrap_timeout=900,
        bootstrap_poll_interval=0.5,
        **config,
    ):
        if not isinstance(idle_minutes, int) or idle_minutes < 1:
            raise ValueError("idle_minutes must be a positive integer")
        for name, value in (
            ("bootstrap_timeout", bootstrap_timeout),
            ("bootstrap_poll_interval", bootstrap_poll_interval),
        ):
            try:
                valid = (
                    not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(f"{name} must be finite and positive")
        source = _source_checkout(source)
        configuration = _configuration(model, config)
        session = self._path(session)
        script = _worker_script(session, configuration)
        with tempfile.TemporaryDirectory(prefix="weights-lab-sky-") as temporary:
            root = Path(temporary)
            workdir = sanitized_workdir(source, root / "package")
            task = {
                "name": "weights-lab",
                "workdir": str(workdir),
                "resources": {"accelerators": accelerators},
                "setup": shlex.join([self.python, "-m", "pip", "install", "."]),
                "run": shlex.join([self.python, "-c", script]),
            }
            spec = root / "task.yaml"
            spec.write_text(json.dumps(task), encoding="utf-8")
            _checked(
                self.runner,
                [
                    "sky",
                    "launch",
                    "-c",
                    self.cluster,
                    "-d",
                    "-y",
                    "--idle-minutes-to-autostop",
                    str(idle_minutes),
                    "--down",
                    str(spec),
                ],
                text=True,
                capture_output=True,
            )
        self._wait_session(session, float(bootstrap_timeout), float(bootstrap_poll_interval))
        return {
            "launcher": "sky",
            "cluster": self.cluster,
            "session": str(session),
            "state": "starting",
        }

    def _wait_session(self, session, timeout, poll_interval):
        """Detached Sky setup may still be installing Python after job submission."""
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "--",
            self.cluster,
            shlex.join([self.python, "-c", _SESSION_PROBE, self._path(session)]),
        ]
        deadline = time.monotonic() + timeout
        diagnostic = "no readiness response"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"SkyPilot bootstrap timed out after {timeout:g}s: {diagnostic}")
            try:
                result = self.runner(
                    command, check=False, text=True, capture_output=True, timeout=min(15, remaining)
                )
                diagnostic = (
                    result.stderr or result.stdout or f"probe exited {result.returncode}"
                ).strip()[-2000:]
                if result.returncode == 0:
                    return
                if result.returncode == 2:
                    raise RuntimeError(f"SkyPilot worker failed during bootstrap: {diagnostic}")
                if result.returncode == 255 and any(
                    message in diagnostic.lower()
                    for message in (
                        "permission denied",
                        "host key verification failed",
                        "remote host identification has changed",
                    )
                ):
                    raise RuntimeError(
                        f"SkyPilot bootstrap SSH authentication failed: {diagnostic}"
                    )
            except subprocess.TimeoutExpired:
                diagnostic = "SSH readiness probe timed out"
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))

    def run(self, args, *, input=None):
        # Bypass the GPU job queue occupied by the resident worker.
        args = list(map(str, args))
        path_options = {"--session", "--output", "--file"}
        for index, argument in enumerate(args):
            option, separator, value = argument.partition("=")
            if option not in path_options:
                continue
            if separator:
                args[index] = f"{option}={self._path(value)}"
            elif index + 1 < len(args) and not args[index + 1].startswith("-"):
                args[index + 1] = self._path(args[index + 1])
        return SSHLauncher(self.cluster, self.python, runner=self.runner).run(args, input=input)

    def stop(self, session, timeout=10):
        return self.run(["stop", "--session", str(session), "--timeout", str(timeout)])

    def download(self, remote_path, destination):
        return SSHLauncher(self.cluster, self.python, runner=self.runner).download(
            self._path(remote_path), destination
        )

    def teardown(self):
        return _checked(
            self.runner, ["sky", "down", "-y", self.cluster], text=True, capture_output=True
        )


class ModalLauncher:
    """Keep session files in a finite-lived Modal sandbox until explicit teardown."""

    def __init__(self, app_name="weights-lab", sandbox_id=None, python="python", *, sdk=None):
        self.app_name, self.sandbox_id, self.python = app_name, sandbox_id, python
        self._sdk = sdk

    def _modal(self):
        if self._sdk is None:
            try:
                import modal
            except ImportError as exc:
                raise RuntimeError(
                    "install the modal package and configure Modal authentication"
                ) from exc
            self._sdk = modal
        return self._sdk

    def start(
        self,
        model,
        session,
        *,
        source=None,
        gpu="A100",
        lifetime=3600,
        bootstrap_timeout=900,
        bootstrap_poll_interval=0.5,
        **config,
    ):
        if not isinstance(lifetime, int) or lifetime < 1:
            raise ValueError("lifetime must be a positive number of seconds")
        for name, value in (
            ("bootstrap_timeout", bootstrap_timeout),
            ("bootstrap_poll_interval", bootstrap_poll_interval),
        ):
            try:
                valid = (
                    not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(f"{name} must be finite and positive")
        configuration = _configuration(model, config)
        modal = self._modal()
        source = _source_checkout(source)
        # The sandbox stays alive after the worker stops so callers can copy artifacts.
        script = _worker_script(session, configuration, keep_alive=True)
        with tempfile.TemporaryDirectory(prefix="weights-lab-modal-") as temporary:
            workdir = sanitized_workdir(source, Path(temporary) / "package")
            image = (
                modal.Image.debian_slim(python_version="3.11")
                .add_local_dir(str(workdir), "/opt/weights-lab", copy=True)
                .run_commands("python -m pip install /opt/weights-lab")
            )
            app = modal.App.lookup(self.app_name, create_if_missing=True)
            sandbox = modal.Sandbox.create(
                self.python,
                "-c",
                script,
                app=app,
                image=image,
                gpu=gpu,
                timeout=lifetime,
                runtime="gvisor",
            )
            self.sandbox_id = sandbox.object_id
        self._wait_session(session, float(bootstrap_timeout), float(bootstrap_poll_interval))
        return {
            "launcher": "modal",
            "sandbox_id": self.sandbox_id,
            "session": str(session),
            "state": "starting",
        }

    def _wait_session(self, session, timeout, poll_interval):
        sandbox = self._sandbox()
        deadline = time.monotonic() + timeout
        diagnostic = "no readiness response"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Modal sandbox {self.sandbox_id} bootstrap timed out after {timeout:g}s: "
                    f"{diagnostic}"
                )
            if sandbox.poll() is not None:
                raise RuntimeError(f"Modal sandbox {self.sandbox_id} exited during bootstrap")
            process = sandbox.exec(
                self.python,
                "-c",
                _SESSION_PROBE,
                str(session),
                timeout=max(1, math.ceil(min(15, remaining))),
            )
            code = process.wait()
            diagnostic = (
                process.stderr.read() or process.stdout.read() or f"probe exited {code}"
            ).strip()[-2000:]
            if code == 0:
                return
            if code != 75:
                raise RuntimeError(
                    f"Modal sandbox {self.sandbox_id} failed during bootstrap: {diagnostic}"
                )
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))

    def _sandbox(self):
        if not self.sandbox_id:
            raise ValueError("start a sandbox or supply its sandbox_id")
        return self._modal().Sandbox.from_id(self.sandbox_id)

    def run(self, args, *, input=None):
        command = [self.python, "-m", "weights_lab", *map(str, args)]
        process = self._sandbox().exec(*command)
        if input is not None:
            process.stdin.write(input.encode("utf-8"))
        process.stdin.write_eof()
        process.stdin.drain()
        stdout, stderr = process.stdout.read(), process.stderr.read()
        returncode = process.wait()
        result = subprocess.CompletedProcess(command, returncode, stdout, stderr)
        result.check_returncode()
        return result

    def stop(self, session, timeout=10):
        return self.run(["stop", "--session", str(session), "--timeout", str(timeout)])

    def download(self, remote_path, destination):
        """Copy ephemeral files over the text exec stream, verifying remote SHA-256."""
        destination = Path(destination).absolute()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoder = (
            "import base64,sys\n"
            "class EncodedOutput:\n"
            "    def write(self, chunk):\n"
            "        sys.stdout.write(base64.b64encode(chunk).decode('ascii')+'\\n')\n"
            "        sys.stdout.flush()\n"
            "        return len(chunk)\n"
            "encoded_output=EncodedOutput()\n"
        )
        script = encoder + _EXPORT.replace("fileobj=sys.stdout.buffer", "fileobj=encoded_output")
        command = [self.python, "-c", script, str(remote_path)]
        process = self._sandbox().exec(*command, text=True, bufsize=1)
        with tempfile.TemporaryDirectory(
            prefix=".weights-lab-copy-", dir=destination.parent
        ) as tmp:
            root = Path(tmp)
            archive_path = root / "transfer.tar"
            with archive_path.open("wb") as output:
                pending = ""
                for chunk in process.stdout:
                    pending += chunk
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        output.write(base64.b64decode(line, validate=True))
                    if len(pending) > 1024 * 1024:
                        raise ValueError("checkpoint transfer contains an oversized encoded chunk")
                if pending:
                    raise ValueError("checkpoint transfer ended with an incomplete encoded chunk")
            stderr = process.stderr.read()
            returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command, stderr=stderr)
            extracted = root / "data"
            extracted.mkdir()
            manifest = _unpack_verified(archive_path, extracted)
            copied = (
                extracted / manifest["files"][0]["path"]
                if manifest["kind"] == "file"
                else extracted
            )
            os.replace(copied, destination)
        return destination

    def teardown(self):
        self._sandbox().terminate(wait=True)
