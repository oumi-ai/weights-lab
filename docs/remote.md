# Remote workers

Remote sessions run on clouds or Slurm through [SkyPilot](https://docs.skypilot.ai/).
Install it separately with your provider's extras, for example
`uv tool install 'skypilot[nebius]'`, then run `sky check`. Model credentials must be
available on the worker; local ones are not copied.

```sh
oumi-weights-lab start --cluster my-lab --accelerators L4:1 --model Qwen/Qwen3-0.6B --session /tmp/lab --device cuda
oumi-weights-lab send --cluster my-lab --session /tmp/lab --file snippet.py --follow
oumi-weights-lab stop --cluster my-lab --session /tmp/lab
oumi-weights-lab backup --cluster my-lab --session /tmp/lab --output ./lab.tar.gz
sky down my-lab
```

- For Slurm, configure `~/.slurm/config` per the
  [SkyPilot Slurm guide](https://docs.skypilot.ai/en/latest/reference/slurm/slurm-getting-started.html)
  and add `--infra slurm/CLUSTER/PARTITION`.
- The worker installs the code the CLI runs: its checkout or plugin copy, or the git
  commit it was installed from, which needs git on the worker. `--source PATH` ships a
  different checkout.
- `--device-map balanced` (or `sequential`) spreads one model over every GPU in
  `--accelerators`, such as `H100:4`; `--max-memory 0=70GiB,1=70GiB` uses only the
  GPUs it names, up to that much on each.
  A model that would need CPU or disk offload is refused before any weights load.
  With `--device-map`, the shutdown grace defaults to 300 seconds instead of 30.
- `--disk-size GB` sizes the cluster disk for a large checkpoint and its exports.
- Use a new `--session` path for each `start`. Relative session and `--resume` paths
  resolve under `~/sky_workdir/`.
- The worker exits after `--idle-timeout` seconds without work (default 900) or
  `--max-lifetime` seconds (default 14400), first trying to save an edit bundle. `0`
  disables either limit. SkyPilot then tears the cluster down after `--idle-minutes`
  (default 15).
- Teardown deletes the disk, so export checkpoints and back up or download results
  first. `stop --teardown` stops the worker and tears down in one step.

From Python, `oumi_weights_lab.launchers.SkyPilotLauncher(cluster)` provides `start`,
`run`, `stop`, `download`, and `teardown`.
