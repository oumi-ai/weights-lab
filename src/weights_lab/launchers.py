"""Optional remote transports. Importing this module needs no cloud SDKs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

from .budget import PassBudget

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


def _configuration(model, config):
    if not isinstance(model, (str, Path)) or not str(model).strip():
        raise ValueError("model must be a nonempty checkpoint path or Hub ID")
    configuration = {
        "model": str(model),
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


def _worker_script(session, configuration):
    script = (
        "import os,sys\nfrom pathlib import Path\n"
        "from weights_lab.session import create_session\n"
        f"p=Path({str(session)!r}).expanduser().resolve()\n"
        f"create_session(p, {configuration!r})\n"
        "command=[sys.executable,'-m','weights_lab','_worker','--session',str(p)]\n"
        "log=(p/'worker.log').open('ab',buffering=0)\n"
    )
    return (
        script
        + "os.dup2(log.fileno(),1)\nos.dup2(log.fileno(),2)\nos.execv(sys.executable,command)\n"
    )


class _SSHTransport:
    def __init__(self, host, python="python3", *, runner=subprocess.run):
        if not host or str(host).startswith("-") or any(char.isspace() for char in str(host)):
            raise ValueError("host must be an SSH host or config alias")
        self.host, self.python, self.runner = str(host), python, runner

    def _command(self, args):
        # Noninteractive .bashrc can return before SkyPilot's Slurm HOME hook.
        command = (
            'if [ -r "$HOME/.sky_ssh_rc" ]; then . "$HOME/.sky_ssh_rc"; fi; exec '
            + shlex.join([self.python, *map(str, args)])
        )
        return ["ssh", "--", self.host, command]

    def run(self, args, *, input=None):
        return _checked(
            self.runner,
            self._command(["-m", "weights_lab", *args]),
            input=input,
            text=True,
            capture_output=True,
        )

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
        infra=None,
        accelerators="A100:1",
        idle_minutes=15,
        bootstrap_timeout=900,
        bootstrap_poll_interval=0.5,
        **config,
    ):
        if infra is not None:
            if not isinstance(infra, str) or not infra.strip():
                raise ValueError("infra must be a nonempty SkyPilot infrastructure selector")
            if infra.split("/", 1)[0].strip().lower() == "modal":
                raise ValueError("Modal is not supported by upstream SkyPilot")
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
                "resources": {} if accelerators is None else {"accelerators": accelerators},
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
                    *(["--infra", infra] if infra is not None else []),
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
        command = _SSHTransport(self.cluster, self.python)._command(
            ["-c", _SESSION_PROBE, self._path(session)]
        )
        command[1:1] = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
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
        return _SSHTransport(self.cluster, self.python, runner=self.runner).run(args, input=input)

    def stop(self, session, timeout=10):
        return self.run(["stop", "--session", str(session), "--timeout", str(timeout)])

    def download(self, remote_path, destination):
        return _SSHTransport(self.cluster, self.python, runner=self.runner).download(
            self._path(remote_path), destination
        )

    def teardown(self):
        return _checked(
            self.runner, ["sky", "down", "-y", self.cluster], text=True, capture_output=True
        )
