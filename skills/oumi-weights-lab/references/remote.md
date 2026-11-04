# Remote workers through SkyPilot

Use an agent execution environment with the `oumi-weights-lab` CLI installed separately
and upstream SkyPilot installed and authenticated for the chosen infrastructure.
The skill folder supplies instructions only. For package installation, use a real
Oumi Weights Lab checkout and its package metadata, for example `pip install /path/to/weights-lab`
in the chosen virtual environment. A copied skill or plugin cache is not that checkout.

SkyPilot startup needs a source checkout on the machine running the CLI. Pass its
path explicitly with `--source`; this checkout supplies the sanitized package
source installed on the remote worker. Model access and credentials must be
available on that worker. Use infrastructure already configured for the requested
experiment; skill installation does not configure accounts or provision resources.

```sh
sky check
oumi-weights-lab start --cluster lab-demo --source /path/to/weights-lab \
  --model Qwen/Qwen3-0.6B --session /tmp/lab --accelerators L4:1 --device cuda
oumi-weights-lab send --cluster lab-demo --session /tmp/lab --file baseline.py --follow
oumi-weights-lab peek --cluster lab-demo --session /tmp/lab
```

`--cluster` selects SkyPilot; include it on subsequent commands. `--file` is local
to the CLI machine and its contents travel over stdin. `--session` identifies a
remote path. For Slurm, configure `~/.slurm/config` using the upstream
[Slurm guide](https://docs.skypilot.ai/en/latest/reference/slurm/slurm-getting-started.html),
run `sky check`, then add `--infra slurm/mycluster/mypartition` to `start`.

Export any checkpoint under `artifacts/` while the model is alive, then stop and
back up before destroying the worker:

```sh
oumi-weights-lab stop --cluster lab-demo --session /tmp/lab
oumi-weights-lab backup --cluster lab-demo --session /tmp/lab --output ./lab-demo.tar.gz
sky down lab-demo
```

The backup includes exported checkpoints under `artifacts/`; it does not export
live model weights. Teardown destroys the remote disk. Use `stop --teardown` only
after retrieving everything needed. Remote workers default to a 900-second idle
timeout and a 14,400-second maximum lifetime, including loading. Override with
`--idle-timeout` and `--max-lifetime`; `0` explicitly disables either limit.
Local workers have no automatic deadline by default. The worker attempts an edit
bundle before deadline shutdown; `peek` reports the reason and saved path or error.
An independent watchdog forces exit after the 30-second shutdown grace if needed.
SkyPilot's separate idle/autodown timer starts after the worker job ends, so
download results during that interval. Inspect `sky status` after startup failures
and release resources when finished.

For transfer details, timeouts, and lifecycle behavior, see the canonical
[remote documentation](https://github.com/oumi-ai/weights-lab/blob/main/docs/remote.md).
