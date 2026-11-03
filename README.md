# Weights Lab

Keep an open language model loaded while a coding agent inspects it through Python.
Each local session runs snippets in one shared namespace and saves their source,
output, timing, and forward-pass counts.

The package includes exact weight undo and checkpoint export, likelihood and
sampled evaluation, pass budgets, activation recording and patching, per-row
gradients, document-split probes, and experimental fixed-weight components.
An [agent skill and plugin](docs/skill-install.md) guide experiments. Workers run
locally or through SSH, Slurm, SkyPilot, and Modal.

## Install

Python 3.11 or later on Linux or macOS is required. A GPU is optional for small models.

```sh
uv sync --python 3.12
uv run weights-lab --help
```

For use outside this checkout, run `pip install .` in a virtual environment.
To load LoRA adapters, install `pip install '.[adapters]'` or use `uv sync --extra adapters`.

## Run a session

Choose a new session directory. `start` returns immediately; loading continues in
the background. The model can be a Hugging Face model ID or a local checkpoint.

```sh
uv run weights-lab start --session /tmp/my-lab --model HuggingFaceTB/SmolLM2-135M --device cpu
uv run weights-lab peek --session /tmp/my-lab
uv run weights-lab send --session /tmp/my-lab --file examples/inspect_model.py --follow
uv run weights-lab send --session /tmp/my-lab --code 'print(lab.forward_passes)' --wait
uv run weights-lab stop --session /tmp/my-lab
uv run weights-lab backup --session /tmp/my-lab --output /tmp/my-lab.tar.gz
```

Snippets submitted during loading wait in the inbox. `--follow` streams snippet
output to stderr and waits for completion; stdout contains a JSON result. `--wait`
waits without streaming. Both return a nonzero exit code for failed or cancelled
snippets. The wait timeout controls the CLI, not the running snippet.

`send` also reads Python from stdin or `--code`. Use `peek --id ID` to read a saved
result and output, or add `--follow` to watch a pending snippet. `peek --follow`
without an ID watches worker status and the startup log.

## Inside a snippet

The namespace contains `lab`, `model`, `tokenizer`, `Path`, and `artifacts`.
Variables and model changes persist between snippets, including changes made
before a snippet raises an exception. Import any other modules you need.

```python
import torch

inputs = tokenizer("The capital of France is", return_tensors="pt").to(lab.device)
with torch.no_grad():
    logits = model(**inputs).logits
print(tokenizer.decode(logits[0, -1].argmax()))
print(lab.forward_passes)
```

The worker loads an eager model in eval mode and seeds the random generators.
Use `torch.no_grad()` for inference; gradients remain available for inspection.
The pass counter counts calls to the base language model, including each generation
step. One batched call counts once. A call that raises also counts. Each snippet's
result records its own pass count; `lab.forward_passes` is the session total.

`start` accepts `--adapter PATH_OR_ID`, `--device auto|cpu|cuda|cuda:N|mps`,
`--dtype auto|float32|float16|bfloat16`, and `--snippet-timeout SECONDS` (default 300).
Automatic device selection prefers CUDA, then MPS, then CPU. Custom Hub code is
disabled. Snippets themselves run as trusted Python with the worker's filesystem
and network access.

Use `--revision COMMIT` to pin a Hub checkpoint and `--pass-limit N` to cap the
whole session. See the [Python API](docs/api.md) for editing, evaluation, inspection,
and attachment signatures, and the [worked experiment](docs/dogfood.md) for an
actual public-model baseline, edit, undo, export, and backup.

```python
baseline = lab.evaluate("/path/dev.jsonl", name="baseline")
name = lab.parts("*mlp*weight")[0]["name"]
lab.zero(name)
changed = lab.evaluate("/path/dev.jsonl", name="zeroed")
print(lab.diff(baseline, changed))
lab.undo()
checkpoint = lab.export("restored")  # under artifacts/
```

Keep fast dev data separate from held-out data. A held-out evaluation requires
`held_out=True, max_passes=N`, and its dataset can run once per session, including
failed attempts. The registry prevents reusing held-out data as dev data. Compare
against an unedited baseline in the same live session.

## Remote workers

```sh
uv run weights-lab start --host gpu-box --model Qwen/Qwen3-0.6B --session /tmp/lab
uv run weights-lab send --host gpu-box --session /tmp/lab --code 'print(lab.parts()[0])' --wait
uv run weights-lab stop --host gpu-box --session /tmp/lab
uv run weights-lab backup --host gpu-box --session /tmp/lab --output ./lab.tar.gz
```

SSH expects the remote Python to have the package installed. `--install --source .`
builds and installs a sanitized checkout over SSH. `start --launcher slurm` submits
on a shared filesystem. `start --launcher sky --cluster NAME --source .` provisions
a SkyPilot worker; `--launcher modal --source .` returns a sandbox ID used with
`--sandbox-id ID` on later commands. Remote transports capture output and return it
at command completion. Read [remote setup and teardown](docs/remote.md) before
using a provider; cloud workers incur charges until stopped or torn down.

## Records and shutdown

Session directories contain:

- `config.json`, `status.json`, and `worker.log`: configuration, status, and startup output.
- `inbox/`, `running/`, and `completed/`: original snippet source throughout its lifecycle.
- `outputs/ID.log` and `results/ID.json`: combined stdout/stderr and execution results.
- `artifacts/`: a directory created for files written by snippets.

Snippet errors and Python timeouts leave the worker available for more work.
Touching the session's `STOP` file interrupts active Python code and stops the
worker. Native GPU operations can delay Python signal handling; `stop` escalates
to process termination after its grace period. Pending or interrupted snippets
are marked cancelled. After an abrupt process death, run `stop` to finalize records.
Create a new session to load again: Python memory and model changes are not restored.

Backups require a stopped worker and include records and files in `artifacts/`.
They do not automatically save live model weights or the Python namespace. Export
a checkpoint into `artifacts/` before stopping if those weights should be backed
up. Seeds alone do not guarantee the same results across devices or setups.

## Examples and validation

- [Published GPT-2 generation parity and HF likelihood parity](docs/parity.md).
- [Measured tiny GPT-2 worker experiment](docs/dogfood.md).
- [Pinned replay of the released Qwen weight edit](docs/qwen-replay.md).
- `uv run python examples/calendar_circuit.py --verify`: check the fixed calendar
  circuit against all 36,525 valid dates in 2000–2099.
- [Seven agent recipes](skills/weights-lab/references/) for common interventions.

CPU tests, an Apple MPS smoke test, and public-model experiments run locally.
Remote transports are tested with simulated providers and a local shell; CUDA,
paid provider deployment, and the full Qwen benchmark have not been run here.
The Qwen example replays a
released edit and does not claim to rediscover it or reproduce its benchmark.
Optional Oumi training-step integration remains outside this standalone release.

## Development

```sh
uv run --extra adapters pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Tests create tiny random Hugging Face models locally and exercise editing,
scoring, budgets, hooks, probes, gradients, circuits, adapters, worker lifecycle,
and remote routing. They do not download pretrained weights. CI runs the suite
on Python 3.11 and 3.12 with CPU PyTorch. The public-model examples download their
pinned checkpoints separately.
