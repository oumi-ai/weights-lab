# Remote workers

The local worker also runs over SSH, on a Slurm compute node, in a SkyPilot cluster,
or in a Modal sandbox. The launcher machine needs the corresponding CLI or SDK and
its normal authentication. Model access and Hugging Face credentials must already
be configured on the machine that loads the model. No credentials are copied from
the local repository.

These transports have command-construction tests and CPU tests using a local shell
in place of SSH. Live CUDA and SSH execution was also exercised through SkyPilot
on Nebius; see [validation evidence and remaining provider limits](validation.md).
Starting a cloud launcher provisions resources and can incur charges.

## SSH

Install `weights-lab` in the remote Python environment, then use an SSH config alias:

```python
from weights_lab.launchers import SSHLauncher

remote = SSHLauncher("gpu-box", python="/opt/lab/bin/python")
remote.start("Qwen/Qwen3-0.6B", "/home/me/lab/session-1", device="cuda")
remote.send("/home/me/lab/session-1", "print(next(model.parameters()).shape)", wait=True)
status = remote.run(["peek", "--session", "/home/me/lab/session-1"])
print(status.stdout)
remote.stop("/home/me/lab/session-1")
```

`run()` returns a `subprocess.CompletedProcess` and raises `CalledProcessError` when
the remote command fails. `start`, `send`, and `stop` parse the CLI JSON result.
Arguments are shell quoted. Snippet source goes over stdin, so local shell syntax
inside the snippet is preserved. `send(..., follow=True)` waits for completion;
the Python transport captures the remote output rather than displaying a live feed.

To install a local source checkout remotely, install the Python `build` package
locally and call `remote.install("/path/to/weights-lab")`. It builds a wheel from
only `src/weights_lab/*.py`, `pyproject.toml`, `README.md`, and `LICENSE`, transfers
that wheel, and runs remote pip. Repository history, local environment files,
private planning documents, examples, and tests are excluded. Package dependencies
are downloaded by remote pip; this does not copy local virtual environments.

## Copying checkpoints

After a snippet exports a checkpoint, copy it to a new local destination:

```python
remote.download("/home/me/lab/session-1/artifacts/edited", "./edited-checkpoint")
```

This streams a tar archive and a SHA-256 manifest from the remote Python. Each
received file must match its remote digest before the destination appears. File
copies also work. An existing destination is rejected. Symbolic links, special
files, duplicate archive entries, unexpected files, missing files, and path traversal
are rejected. Empty directories inside checkpoints are omitted. Export should be
finished before copying; concurrent writes can cause checksum verification to fail.

Session `backup` archives records and artifacts, while checkpoint download copies
the chosen checkpoint itself. Keep a checkpoint outside a session backup when it
would make that backup too large.

## Slurm

Run the launcher on a login node where both the session directory and the Python
environment are visible to compute nodes:

```python
from weights_lab.launchers import SlurmLauncher

slurm = SlurmLauncher(python="/shared/env/bin/python")
job = slurm.start(
    "Qwen/Qwen3-0.6B",
    "/shared/lab/session-1",
    partition="gpu",
    gres="gpu:a100:1",
    time_limit="01:00:00",
    device="cuda",
)
print(job["job_id"])
slurm.run(["send", "--session", job["session"]], input="print(model.config)")
slurm.stop(job["session"])
```

The worker runs in the foreground of its batch job. `worker.log` lives in the shared
session directory. Submission records the scheduler job ID in `launcher.json`;
cancellation uses `scancel`, including the scheduler cluster when returned by
`sbatch --parsable`. `stop` queries `squeue` until the job disappears or reaches a
terminal state, then finalizes unfinished snippets and session status. It raises
`TimeoutError` if cancellation remains pending past `timeout`. A compute-node PID
is never signalled on the login node. Direct
`SlurmLauncher.run()` supports only `send` and `peek`.

Slurm does not transfer files or install the package on compute nodes. The shared
environment must already contain a compatible installation. See the official
[sbatch reference](https://slurm.schedmd.com/sbatch.html) for partition, GPU, and time
limit syntax.

## SkyPilot

Install and configure SkyPilot for your cloud, then pass a source checkout:

```python
from weights_lab.launchers import SkyPilotLauncher

sky = SkyPilotLauncher("weights-lab-demo")
sky.start(
    "Qwen/Qwen3-0.6B",
    "/tmp/lab/session-1",
    source="/path/to/weights-lab",
    accelerators="L4:1",
    idle_minutes=15,
    device="cuda",
)
sky.run(["send", "--session", "/tmp/lab/session-1"], input="print(model.config)")
sky.stop("/tmp/lab/session-1")
sky.download("/tmp/lab/session-1/artifacts/edited", "./edited-checkpoint")
sky.teardown()
```

Only the sanitized package directory is synced. A JSON document, which is also
valid YAML, supplies the task configuration. Setup installs the package; the job
executes the worker in the foreground so SkyPilot can track its lifetime. Startup
waits until the remote Python environment and session files exist, then returns
without waiting for model loading. `bootstrap_timeout` defaults to 900 seconds;
bootstrap failures retain readiness diagnostics. Relative paths use
`~/sky_workdir/`; absolute and explicit remote home paths are preserved. `run()` uses the SSH
alias that SkyPilot creates for the cluster. This bypasses the scheduler queue
occupied by the resident worker. Snippets supplied through `input`
are delivered to the remote CLI stdin, rather than referring to a local
`--file` path. For a local file, use `input=Path("snippet.py").read_text()`.
If the launcher itself was installed from a wheel, supply `source` explicitly;
installed package files do not include the source checkout and build metadata.

The cluster is scheduled for autodown after the specified idle period once jobs
finish. A running worker keeps the cluster active, even when its snippet inbox is
empty. `stop` stops the worker; `teardown` explicitly destroys the cluster and its
disk. Export and copy anything needed first. Provisioning or setup failures may
leave resources available for debugging, so inspect `sky status` and tear them down.
See the official [SkyPilot CLI reference](https://docs.skypilot.ai/en/latest/reference/cli.html).

## Modal

Install the `modal` SDK and authenticate it before creating a sandbox:

```python
from weights_lab.launchers import ModalLauncher

modal = ModalLauncher(app_name="weights-lab")
info = modal.start(
    "Qwen/Qwen3-0.6B",
    "/lab/session-1",
    source="/path/to/weights-lab",
    gpu="L4",
    lifetime=3600,
    device="cuda",
)
modal.run(["send", "--session", info["session"]], input="print(model.config)")
modal.stop(info["session"])
modal.download("/lab/session-1/artifacts/edited", "./edited-checkpoint")
modal.teardown()
```

Save `sandbox_id` from the start result to reconnect with
`ModalLauncher(sandbox_id=...)`. The image contains only the sanitized package and
its dependencies. `start()` waits for valid session files before returning, so an
immediate `send` can queue work while the model loads. `bootstrap_timeout` defaults
to 900 seconds and `bootstrap_poll_interval` to 0.5 seconds. On bootstrap failure,
the exception includes the sandbox ID; `modal.sandbox_id` remains available for
reconnection or `modal.teardown()`.

Snippet source is written to the exec process stdin. The sandbox
stays alive after its worker stops so files can be retrieved; its total lifetime
is bounded by `lifetime` seconds. `teardown()` terminates it explicitly and waits.
Files are ephemeral: retrieve artifacts with `download()` before terminating or
reaching the lifetime limit. `download()` streams the remote tar archive in encoded
chunks through the exec stdout stream, then verifies the same SHA-256 manifest used
by SSH downloads before publishing the local destination. It copies individual
backup archives as well as checkpoint directories. See the official
[Sandbox API](https://modal.com/docs/sdk/py/latest/Sandbox).
