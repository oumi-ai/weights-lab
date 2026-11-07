"""Local session commands. Model dependencies are imported only by the worker."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

from . import session
from .session import TERMINAL_STATES


def _json(value: dict | list) -> None:
    print(json.dumps(value), flush=True)


def _max_memory(text: str) -> dict[str, str]:
    try:
        return session.parse_max_memory(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def _status(path: Path) -> dict:
    status = session.read_json(path / "status.json")
    if _pid(status) is None:
        try:
            status["pid"] = int((path / "worker.pid").read_text(encoding="ascii"))
        except (OSError, ValueError):
            pass
    return status


def _pid(status: dict) -> int | None:
    pid = status.get("pid")
    return pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else None


def _alive(status: dict) -> bool:
    pid = _pid(status)
    return pid is not None and session.is_process_alive(pid)


def _write(stream, data: bytes | str | None) -> None:
    """Write process output; bytes pass through undecoded when a binary stream exists."""
    if not data:
        return
    if isinstance(data, str):
        stream.write(data)
        stream.flush()
    elif hasattr(stream, "buffer"):
        stream.flush()
        stream.buffer.write(data)
        stream.buffer.flush()
    else:
        stream.write(data.decode("utf-8", errors="replace"))
        stream.flush()


def _relay(result) -> None:
    """Pass a remote CLI command's JSON stdout through; its stderr stays diagnostic."""
    _write(sys.stdout, result.stdout)
    _write(sys.stderr, result.stderr)


def _stream(path: Path, offset: int) -> int:
    """Stream new bytes, including partial lines, without waiting for a newline."""
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
            offset = handle.tell()
    except FileNotFoundError:
        return offset
    # Raw bytes keep UTF-8 sequences split across reads intact.
    _write(sys.stderr, chunk)
    return offset


def _wait_result(path: Path, snippet_id: str, timeout: float | None, follow: bool) -> dict:
    deadline = None if timeout is None else time.monotonic() + timeout
    offset = 0
    result_path = path / "results" / f"{snippet_id}.json"
    while True:
        if follow:
            offset = _stream(path / "outputs" / f"{snippet_id}.log", offset)
        if result_path.exists():
            result = session.read_json(result_path)
            if follow:
                _stream(path / "outputs" / f"{snippet_id}.log", offset)
            return result
        status = _status(path)
        if status.get("state") in TERMINAL_STATES or (_pid(status) and not _alive(status)):
            # Shutdown records cancelled results before the terminal status, so a result
            # written since the check above is now visible.
            if result_path.exists():
                continue
            raise RuntimeError(f"worker stopped before snippet {snippet_id} produced a result")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for snippet {snippet_id}; it may still be running"
            )
        time.sleep(0.1)


def _start(args: argparse.Namespace) -> int:
    if not args.model.strip():
        raise ValueError("model must be a nonempty checkpoint path or Hub ID")
    if args.cluster:
        return _remote_start(args)
    if args.infra:
        raise ValueError("--infra requires --cluster")
    if args.disk_size is not None:
        raise ValueError("--disk-size requires --cluster")
    path = args.session.expanduser().resolve()
    config = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_memory": args.max_memory,
        "quantize": args.quantize,
        "allow_hub_kernels": args.allow_hub_kernels,
        "snippet_timeout": args.snippet_timeout,
        "poll_interval": session.DEFAULT_POLL_INTERVAL,
        "pass_limit": args.pass_limit,
        "revision": args.revision,
        "adapter_revision": args.adapter_revision,
        "resume": str(args.resume.expanduser().resolve()) if args.resume else None,
        "idle_timeout": args.idle_timeout,
        "max_lifetime": args.max_lifetime,
        "shutdown_grace": args.shutdown_grace,
    }
    if args.adapter:
        config["adapter"] = args.adapter
    session.create_session(path, config)
    try:
        with (path / "worker.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "oumi_weights_lab", "_worker", "--session", str(path)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as exc:
        with session.session_lock(path):
            session.write_json(
                path / "status.json", {"state": "failed", "model": args.model, "error": str(exc)}
            )
            session.cancel_unfinished(path, f"worker could not start: {exc}")
        raise
    (path / "worker.pid").write_text(str(process.pid), encoding="ascii")
    _json({"session": str(path), "pid": process.pid, "state": "starting"})
    return 0


def _send(args: argparse.Namespace) -> int:
    path = args.session.expanduser().resolve()
    if args.file is not None:
        code = args.file.read_text(encoding="utf-8")
    elif args.code is not None:
        code = args.code
    else:
        code = sys.stdin.read()
    if not code.strip():
        raise ValueError("snippet is empty")
    snippet_id = session.submit(path, code)
    if args.wait or args.follow:
        result = _wait_result(path, snippet_id, args.timeout, args.follow)
        _json(result)
        return 0 if result.get("state") == "succeeded" else 1
    _json({"id": snippet_id, "state": "queued"})
    return 0


def _snippet_id(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in value
        )
    ):
        raise argparse.ArgumentTypeError(
            "snippet ID must contain only letters, digits, underscores, or hyphens"
        )
    return value


def _peek(args: argparse.Namespace) -> int:
    path = args.session.expanduser().resolve()
    if args.id:
        result_path = path / "results" / f"{args.id}.json"
        if not result_path.exists() and not any(
            (path / folder / f"{args.id}.py").exists()
            for folder in ("inbox", "running", "completed")
        ):
            raise ValueError(f"unknown snippet ID: {args.id}")
        if args.follow:
            result = _wait_result(path, args.id, None, True)
        else:
            result = (
                session.read_json(result_path)
                if result_path.exists()
                else {"id": args.id, "state": "pending"}
            )
            _stream(path / "outputs" / f"{args.id}.log", 0)
        _json(result)
        return 0
    if not args.follow:
        _json(_status(path))
        return 0
    previous = None
    offset = 0
    while True:
        status = _status(path)
        if status != previous:
            _json(status)
            previous = status
        offset = _stream(path / "worker.log", offset)
        if status.get("state") in TERMINAL_STATES or (_pid(status) and not _alive(status)):
            return 0
        time.sleep(0.1)


def _matches_worker(pid: int, path: Path) -> bool:
    """Refuse escalation if a stale PID has been reused by another process."""
    command = subprocess.run(
        ["ps", "-p", str(pid), "-o", "args="],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return command.endswith(f"-m oumi_weights_lab _worker --session {path}")


def _saving_until(path: Path, status: dict) -> float:
    """When a stopping worker's shutdown grace ends, or 0 when it is not stopping."""
    if status.get("state") != "stopping":
        return 0.0
    try:
        grace = float(session.read_json(path / "config.json")["shutdown_grace"])
        return float(status["updated_at"]) + grace
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0


def _stop(args: argparse.Namespace) -> int:
    path = args.session.expanduser().resolve()
    status = _status(path)
    (path / "STOP").touch()
    deadline = time.monotonic() + args.timeout
    while _alive(status):
        # A worker past its deadline saves an edit bundle within its shutdown grace;
        # killing it then would lose the bundle, and its watchdog enforces the grace.
        if time.monotonic() >= deadline and time.time() >= _saving_until(path, status):
            break
        time.sleep(0.1)
        status = _status(path)
    if not _alive(status):
        # The worker may have written its final status and exited since the last read.
        status = _status(path)
    forced = False
    if _alive(status):
        pid = _pid(status)
        assert pid is not None
        if not _matches_worker(pid, path):
            raise RuntimeError(
                f"refusing to signal PID {pid}: its command does not match this worker"
            )
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 2
            while session.is_process_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            if session.is_process_alive(pid) and _matches_worker(pid, path):
                os.kill(pid, signal.SIGKILL)
                forced = True
                deadline = time.monotonic() + 2
                while session.is_process_alive(pid) and time.monotonic() < deadline:
                    time.sleep(0.1)
        except ProcessLookupError:
            pass
        status = _status(path)
    if _alive(status):
        raise RuntimeError("worker is still alive after shutdown")
    with session.session_lock(path):
        # Like the worker, finalize snippets first so that a waiter that sees the
        # terminal status also finds its snippet's result.
        session.cancel_unfinished(path, "worker stopped before completion")
        if forced or status.get("state") not in TERMINAL_STATES:
            status.update(state="stopped", forced=forced, updated_at=time.time())
            status.pop("current_snippet", None)
            session.write_json(path / "status.json", status)
    _json(status)
    return 0


def _script(args: argparse.Namespace) -> int:
    """Run a file as __main__ in this interpreter, where ``oumi_weights_lab`` imports."""
    import runpy
    import traceback

    path = args.file.expanduser()
    if not path.is_file():
        raise ValueError(f"no such script: {path}")
    saved_argv, saved_path = sys.argv, sys.path[:]
    sys.argv = [str(path), *args.arguments]
    # Like `python FILE`, put the script's folder first so it can import its neighbours.
    sys.path.insert(0, str(path.resolve().parent))
    try:
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None or isinstance(code, int):
            return code or 0
        print(code, file=sys.stderr)
        return 1
    except Exception:
        # Keep the script's own traceback; main() would reduce it to one line.
        traceback.print_exc()
        return 1
    finally:
        sys.argv = saved_argv
        sys.path[:] = saved_path
    return 0


def _backup(args: argparse.Namespace) -> int:
    path = args.session.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == path or path in output.parents:
        raise ValueError("backup output must be outside the session directory")
    status = _status(path)
    if _alive(status) or status.get("state") not in TERMINAL_STATES:
        raise ValueError("stop the worker before creating a backup")
    names = {
        "config.json",
        "status.json",
        "worker.log",
        "inbox",
        "running",
        "completed",
        "outputs",
        "results",
        "artifacts",
    }
    sources = [child for child in path.iterdir() if child.name in names]
    for source in sources:
        if source.is_symlink() or (
            source.is_dir() and any(child.is_symlink() for child in source.rglob("*"))
        ):
            raise ValueError("session backups cannot contain symbolic links")
    with output.open("xb") as destination:
        with tarfile.open(fileobj=destination, mode="w:gz") as archive:
            for source in sorted(sources):
                archive.add(source, arcname=f"{path.name}/{source.name}")
    _json({"session": str(path), "output": str(output)})
    return 0


def _transport(args):
    from .launchers import SkyPilotLauncher

    if args.cluster:
        return SkyPilotLauncher(args.cluster, python=args.remote_python)
    return None


def _remote_start(args):
    from .budget import PassBudget
    from .launchers import SkyPilotLauncher, _configuration

    PassBudget(args.pass_limit)
    config = {
        key: getattr(args, key)
        for key in (
            "adapter",
            "device",
            "dtype",
            "snippet_timeout",
            "pass_limit",
            "revision",
            "adapter_revision",
        )
    }
    if args.resume is not None:
        # The launcher resolves a relative remote path as it does the session path.
        config["resume"] = str(args.resume)
    for key in ("device_map", "max_memory", "quantize", "idle_timeout", "max_lifetime"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.allow_hub_kernels:
        config["allow_hub_kernels"] = True
    if args.shutdown_grace is not None:
        config["shutdown_grace"] = args.shutdown_grace
    _configuration(args.model, config)  # Reject static errors before provisioning.
    transport = SkyPilotLauncher(args.cluster, python=args.remote_python)
    config.update(
        source=args.source,
        accelerators=args.accelerators,
        disk_size=args.disk_size,
        idle_minutes=args.idle_minutes,
        infra=args.infra,
    )
    _json(transport.start(args.model, str(args.session), **config))
    return 0


def _remote_command(args, transport):
    command = [args.command, "--session", str(args.session)]
    source = None
    if args.command == "send":
        source = (
            args.file.read_text(encoding="utf-8")
            if args.file is not None
            else args.code
            if args.code is not None
            else sys.stdin.read()
        )
        if args.wait:
            command.append("--wait")
        if args.follow:
            command.append("--follow")
        command += ["--timeout", str(args.timeout)]
    elif args.command == "peek":
        if args.id:
            command += ["--id", args.id]
        if args.follow:
            command.append("--follow")
    elif args.command == "stop":
        return _remote_stop(args, transport)
    elif args.command == "backup":
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(output)
        remote_output = args.remote_output or f"{args.session}.backup-{uuid.uuid4().hex}.tar.gz"
        command += ["--output", remote_output]
        try:
            transport.run(command)
        except subprocess.CalledProcessError as error:
            _relay(error)
            return error.returncode
        transport.download(remote_output, output)
        _json({"session": str(args.session), "output": str(output), "remote_output": remote_output})
        return 0
    try:
        result = transport.run(command, input=source)
    except subprocess.CalledProcessError as error:
        result = error
    _relay(result)
    return result.returncode


def _remote_stop(args, transport) -> int:
    code = None  # Stays unset when stop raises before reporting an exit code.
    teardown = 0
    try:
        try:
            result = transport.stop(str(args.session), timeout=args.timeout)
        except subprocess.CalledProcessError as error:
            result = error
        _relay(result)
        code = result.returncode
    finally:
        # A failed or interrupted stop must not leave up a cluster the caller asked to
        # destroy. Teardown reports its own failure, so the stop failure still surfaces.
        if args.teardown:
            teardown = _teardown(transport, args.cluster, stop_failed=code != 0)
    return code or teardown


def _teardown(transport, cluster: str, *, stop_failed: bool) -> int:
    """Destroy the cluster and say on stderr whether that happened. Never raises."""
    context = "stop failed; " if stop_failed else ""
    try:
        transport.teardown()
    except subprocess.CalledProcessError as error:
        # Sky's progress text is diagnostic output, never this command's JSON.
        _write(sys.stderr, error.stdout)
        _write(sys.stderr, error.stderr)
        failure, code = f"exit status {error.returncode}", error.returncode or 1
    except Exception as error:  # Reported below without masking a stop failure.
        failure, code = f"{type(error).__name__}: {error}", 1
    else:
        print(f"oumi-weights-lab: {context}cluster {cluster} was torn down", file=sys.stderr)
        return 0
    print(
        f"oumi-weights-lab: {context}teardown of cluster {cluster} failed ({failure}); "
        f"it may still be running: check `sky status` and run `sky down {cluster}`",
        file=sys.stderr,
    )
    return code


def _methods_list(args: argparse.Namespace) -> int:
    from .methods_store import list_methods

    _json(list_methods(args.methods_dir))
    return 0


def _methods_promote(args: argparse.Namespace) -> int:
    from .methods_store import promotion_draft, write_draft

    draft = promotion_draft(
        args.name,
        args.version,
        against=args.against,
        directory=args.methods_dir,
        force=args.force,
    )
    path = write_draft(draft, args.output, overwrite=args.overwrite)
    _json({"draft": str(path)})
    return 0


def _proving_available() -> bool:
    """Whether the optional proving package is installed; the agent images leave it out."""
    return importlib.util.find_spec("oumi_weights_lab.proving") is not None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oumi-weights-lab",
        description="Keep a local model loaded and run Python snippets against it.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="launch a detached local worker")
    start.add_argument("--session", required=True, type=Path)
    start.add_argument("--model", required=True)
    start.add_argument("--adapter")
    start.add_argument("--revision", help="model and tokenizer checkpoint revision")
    start.add_argument("--adapter-revision")
    start.add_argument("--resume", type=Path, help="edit bundle to resume in this new session")
    start.add_argument("--device", default="auto")
    start.add_argument("--dtype", default="auto", choices=session.DTYPES)
    start.add_argument(
        "--device-map",
        choices=session.DEVICE_MAPS,
        help="spread the model over every visible CUDA GPU; CPU and disk offload are refused",
    )
    start.add_argument(
        "--max-memory",
        type=_max_memory,
        help="with --device-map, use only these GPUs, up to this much memory on each, "
        "such as 0=70GiB,1=70GiB (default: 90%% of every GPU's free memory)",
    )
    start.add_argument(
        "--quantize",
        choices=session.QUANTIZE,
        help="load the model with bitsandbytes in this format (needs the quantized extra); "
        "quantized weights can be edited with exact undo (not double-quantized ones), and this "
        "cannot be combined with --device-map",
    )
    start.add_argument(
        "--allow-hub-kernels",
        action="store_true",
        help="let kernel code downloaded from the Hugging Face Hub run (default: blocked)",
    )
    start.add_argument("--snippet-timeout", type=_positive, default=session.DEFAULT_SNIPPET_TIMEOUT)
    start.add_argument(
        "--pass-limit",
        type=int,
        help="maximum total passes: one per sequence in each base-model forward",
    )
    start.add_argument(
        "--idle-timeout",
        type=float,
        help="worker watchdog: stop the worker after this many seconds without queued or "
        "running snippets; 0 disables (Sky default: 900)",
    )
    start.add_argument(
        "--max-lifetime",
        type=float,
        help="maximum worker seconds including load; 0 disables (Sky default: 14400)",
    )
    start.add_argument(
        "--shutdown-grace",
        type=_positive,
        help="seconds for graceful shutdown before forced kill "
        f"(default: {session.DEFAULT_SHUTDOWN_GRACE:g}, or "
        f"{session.MULTI_GPU_SHUTDOWN_GRACE:g} with --device-map)",
    )
    start.add_argument(
        "--source",
        type=Path,
        help="checkout to install on a SkyPilot worker (default: this checkout when run "
        "from one, otherwise the git commit this CLI was installed from)",
    )
    start.add_argument("--infra", help="SkyPilot infrastructure, e.g. slurm/mycluster/gpu")
    start.add_argument("--accelerators", default="A100:1")
    start.add_argument(
        "--disk-size",
        type=int,
        help="SkyPilot cluster disk in GB, for large checkpoints and exports",
    )
    start.add_argument(
        "--idle-minutes",
        type=int,
        default=15,
        help="SkyPilot autodown: minutes the cluster stays up after the worker job exits "
        "before it is torn down (default: 15). Separate from --idle-timeout, which ends "
        "the worker itself",
    )
    start.set_defaults(handler=_start)
    send = commands.add_parser("send", help="submit Python from a file, an argument, or stdin")
    send.add_argument("--session", required=True, type=Path)
    code = send.add_mutually_exclusive_group()
    code.add_argument("--file", type=Path)
    code.add_argument("--code")
    send.add_argument("--wait", action="store_true")
    send.add_argument(
        "--follow", action="store_true", help="stream output to stderr and wait for completion"
    )
    send.add_argument("--timeout", type=_positive, default=300.0)
    send.set_defaults(handler=_send)
    peek = commands.add_parser("peek", help="inspect worker status or a snippet result")
    peek.add_argument("--session", required=True, type=Path)
    peek.add_argument("--id", type=_snippet_id)
    peek.add_argument("--follow", action="store_true")
    peek.set_defaults(handler=_peek)
    stop = commands.add_parser("stop", help="stop the worker, escalating after the timeout")
    stop.add_argument("--session", required=True, type=Path)
    stop.add_argument("--timeout", type=_positive, default=10.0)
    stop.add_argument(
        "--teardown", action="store_true", help="destroy a cloud worker after stopping"
    )
    stop.set_defaults(handler=_stop)
    backup = commands.add_parser(
        "backup", help="archive records and artifacts from a stopped session"
    )
    backup.add_argument("--session", required=True, type=Path)
    backup.add_argument("--output", required=True, type=Path)
    backup.add_argument(
        "--remote-output", help="remote archive path to retain when copying a backup"
    )
    backup.set_defaults(handler=_backup)
    methods = commands.add_parser("methods", help="list saved methods and draft a recipe from one")
    steps = methods.add_subparsers(dest="methods_command", required=True)
    listing = steps.add_parser("list", help="print saved methods, versions and result summaries")
    listing.add_argument("--methods-dir", type=Path)
    listing.set_defaults(handler=_methods_list)
    promote = steps.add_parser("promote", help="write a recipe draft for a maintainer to review")
    promote.add_argument("name")
    promote.add_argument("--version", help="default: the newest version")
    promote.add_argument("--against", help="NAME[@VERSION] of the method it must beat")
    promote.add_argument("--output", required=True, type=Path, help="draft file, outside skills/")
    promote.add_argument("--methods-dir", type=Path)
    promote.add_argument(
        "--force", action="store_true", help="write the draft although --against does not win"
    )
    promote.add_argument(
        "--overwrite", action="store_true", help="replace the output file if it exists"
    )
    promote.set_defaults(handler=_methods_promote)
    script = commands.add_parser(
        "script",
        help="run a Python file in the lab's environment, for scripts that call open_lab",
    )
    script.add_argument("file", type=Path)
    script.add_argument("arguments", nargs=argparse.REMAINDER)
    script.set_defaults(handler=_script)
    worker = commands.add_parser("_worker")
    worker.add_argument("--session", required=True, type=Path)
    for command in (start, send, peek, stop, backup):
        command.add_argument("--cluster", help="SkyPilot cluster name")
        command.add_argument("--remote-python", default="python3")
    if _proving_available():
        from .proving.cli import add_commands

        add_commands(commands)
    # Name the commands in the usage line by hand so it leaves out _worker.
    commands.metavar = "{" + ",".join(name for name in commands.choices if name != "_worker") + "}"
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "_worker":
            from .worker import Worker

            return Worker(args.session.resolve()).run()
        if args.command != "start" and hasattr(args, "cluster"):
            transport = _transport(args)
            if transport is not None:
                return _remote_command(args, transport)
            if args.command == "stop" and args.teardown:
                raise ValueError("--teardown requires --cluster")
        return args.handler(args)
    except subprocess.CalledProcessError as exc:
        # Remote CLI output is relayed where it runs. Anything captured from another
        # tool, such as Sky's progress text, is diagnostic and stays off stdout.
        _write(sys.stderr, exc.stdout)
        _write(sys.stderr, exc.stderr)
        return exc.returncode or 1
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"oumi-weights-lab: {exc}", file=sys.stderr)
        return 1
