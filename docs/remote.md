# Remote workers

Remote sessions use upstream SkyPilot. Install and authenticate SkyPilot for your
infrastructure, then run `sky check`. Model access and Hugging Face credentials
must be available on the worker; local credentials are not copied. Modal is not
supported.

Install the SkyPilot CLI in its own environment with the provider extras you need,
for example `uv tool install 'skypilot[nebius]'`. The model worker does not depend
on SkyPilot's Python packages.

## Start and use a worker

```sh
oumi-weights-lab start --cluster oumi-weights-lab-demo --source . \
  --accelerators L4:1 --model Qwen/Qwen3-0.6B --session /tmp/lab --device cuda
oumi-weights-lab send --cluster oumi-weights-lab-demo --session /tmp/lab --file snippet.py --follow
oumi-weights-lab peek --cluster oumi-weights-lab-demo --session /tmp/lab
oumi-weights-lab stop --cluster oumi-weights-lab-demo --session /tmp/lab
oumi-weights-lab backup --cluster oumi-weights-lab-demo --session /tmp/lab --output ./lab.tar.gz
sky down oumi-weights-lab-demo
```

`--cluster` selects SkyPilot automatically. `--launcher sky` is optional. The same
cluster name reconnects subsequent commands; there is no direct SSH launcher.
`--infra` passes an infrastructure selector to `sky launch`.

```python
from oumi_weights_lab.launchers import SkyPilotLauncher

sky = SkyPilotLauncher("oumi-weights-lab-demo")
sky.start(
    "Qwen/Qwen3-0.6B",
    "/tmp/lab",
    source="/path/to/weights-lab",
    accelerators="L4:1",
    idle_minutes=15,
    device="cuda",
)
sky.run(["send", "--session", "/tmp/lab"], input="print(model.config)")
sky.stop("/tmp/lab")
sky.download("/tmp/lab/artifacts/edited", "./edited-checkpoint")
sky.teardown()
```

For a CPU worker, pass `accelerators=None` and `device="cpu"` to `start()`.
`run()` returns a `subprocess.CompletedProcess` and raises `CalledProcessError` on
failure. Snippet source goes over stdin; arguments are shell quoted. A local file
can be sent with `input=Path("snippet.py").read_text()`. Remote output is captured
and returned when the command completes, including when `--follow` is used.

Only sanitized package source and build metadata are synced. Setup installs the
package and runs the worker in the foreground so SkyPilot tracks its lifetime.
Startup waits for the remote environment and session files, then returns while the
model loads. `bootstrap_timeout` defaults to 900 seconds; failures retain readiness
diagnostics. If installed from a wheel, supply `source` explicitly.

Relative remote session paths use `~/sky_workdir/`; absolute and explicit remote
home paths are preserved. Commands use SkyPilot's SSH alias to reach the resident
worker without queueing behind its running job. Local session paths still resolve
on the local machine.

On Slurm, transport and readiness commands load SkyPilot's `.sky_ssh_rc` hook so
they use the same per-cluster home as the worker, including with noninteractive
shells whose `.bashrc` returns early.

## Slurm through SkyPilot

Configure the Slurm connection in `~/.slurm/config` using the upstream
[Slurm getting started guide](https://docs.skypilot.ai/en/latest/reference/slurm/slurm-getting-started.html),
then check access and select the cluster and partition:

```sh
sky check
oumi-weights-lab start --cluster lab-slurm --source . \
  --infra slurm/mycluster/mypartition --accelerators A100:1 \
  --model Qwen/Qwen3-0.6B --session /tmp/lab --device cuda
oumi-weights-lab send --cluster lab-slurm --session /tmp/lab --code 'print(model.config)' --wait
oumi-weights-lab stop --cluster lab-slurm --session /tmp/lab
oumi-weights-lab backup --cluster lab-slurm --session /tmp/lab --output ./slurm-lab.tar.gz
sky down lab-slurm
```

SkyPilot handles scheduler submission, environment setup, and cluster lifecycle.
Use the same `--cluster` commands for cloud and Slurm workers.

## Preserve results and release resources

Export checkpoints inside a snippet before stopping the worker. `download()` copies
an exported checkpoint or a backup archive to a new local destination. Transfers
use a tar archive and SHA-256 manifest, verified before the destination appears.
Existing destinations, symbolic links, special files, duplicate entries, missing
or unexpected files, and path traversal are rejected. Finish exports before
copying; concurrent writes can cause verification failure. Empty directories are
omitted.

`backup` archives a stopped session's records and artifacts; it does not export live
model weights. Stop, back up or download anything needed, then tear down.
`stop --teardown` combines shutdown with cluster destruction when no more downloads
are needed. Teardown destroys the cluster and disk.

Autodown occurs after the configured idle period once jobs finish. A running worker
keeps the cluster active until it exits. Remote workers now default to
`--idle-timeout 900` (15 minutes with no queued or running snippets) and
`--max-lifetime 14400` (four hours, including model loading). Local workers have
no automatic deadline unless one is supplied. For example:

```sh
oumi-weights-lab start --cluster lab-slurm --source . --model Qwen/Qwen3-0.6B \
  --session /tmp/lab --idle-timeout 600 --max-lifetime 3600 --shutdown-grace 30
```

Set either limit to `0` to disable that limit explicitly. Python `start()` accepts
the same `idle_timeout`, `max_lifetime`, and `shutdown_grace` settings in seconds.
Idle time starts after loading and restarts after each completed snippet; queued
or running work prevents inactivity shutdown. The lifetime deadline includes
loading and active snippets.

An independent watchdog sends SIGTERM at the deadline. The worker cancels active
and queued snippets and attempts to save an [edit bundle](edit-bundles.md). Status
records `reason` (`idle_timeout` or `max_lifetime`) and `edit_bundle` when saving
succeeds. Active attachments, patches, or a long save can prevent the snapshot;
`checkpoint_error` records a graceful-save failure. After `shutdown_grace` seconds
(default 30), the watchdog kills a worker that has not exited, including one
blocked in native code; its terminal status records `forced: true`.

After the job exits, SkyPilot's separate `--idle-minutes` timer starts (default
15 minutes), then autodown releases the cluster. Results remain on the worker disk
only until teardown; retrieve them during that interval. These worker limits begin
when the worker starts, after provisioning and package setup. Provisioning or setup
failures may leave resources for debugging: inspect `sky status` and run `sky down`
when finished. See the [SkyPilot CLI reference](https://docs.skypilot.ai/en/latest/reference/cli.html)
and [validation evidence](validation.md) for measured coverage and limits.
