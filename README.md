# Weights Lab

Load a language model once, then let a coding agent inspect and edit it through
Python snippets in a persistent worker. Compare a baseline, change a weight,
measure the effect, undo it exactly, and export the checkpoint you want to keep.
Every snippet's source, output, timing, and forward-pass count is saved.

**Version 0.1 is experimental.** Snippets run as trusted Python with filesystem and
network access. Use a fresh session for each experiment; a saved session records
what happened but cannot restore the live Python namespace.

## Start on CPU

Requires Python 3.11+ on Linux or macOS and [uv](https://docs.astral.sh/uv/).
The first run downloads a small public model; no GPU or cloud account is needed.

```sh
git clone https://github.com/oumi-ai/weights-lab.git
cd weights-lab
uv sync --python 3.12
uv run weights-lab start --session sessions/first --model HuggingFaceTB/SmolLM2-135M --device cpu
uv run weights-lab send --session sessions/first --code 'print(model.config)' --follow
uv run weights-lab send --session sessions/first --file examples/inspect_model.py --follow
```

`start` returns while the model loads. Snippets queue during loading and share
`lab`, `model`, `tokenizer`, `Path`, and `artifacts`. Variables and edits persist.
`--follow` waits and streams output to stderr; stdout contains a JSON result.
Use `peek --session sessions/first` for status, or `send --wait` without streaming.

## Measure an edit

Save this as `edit.py`, then submit it with
`uv run weights-lab send --session sessions/first --file edit.py --follow`:

```python
rows = [{"prompt": "France's capital is", "options": [" Paris", " Rome"], "answer": 0}]
baseline = lab.evaluate(rows, name="baseline")
name = lab.parts("*mlp*weight")[0]["name"]
lab.zero(name)
edited = lab.evaluate(rows, name="zeroed")
print(lab.diff(baseline, edited))
lab.undo()
checkpoint = lab.export("restored")  # saved under artifacts/
```

This is a workflow check, not evidence of a useful edit. Inspect parameter names
for your model, compare in the same session, and keep held-out data separate from
dev data. `--pass-limit N` caps session forwards; `lab.limit_passes(n)` narrows a
snippet's budget. Use `torch.no_grad()` for inference when gradients are unnecessary.

Export weights before stopping, then archive the session:

```sh
uv run weights-lab stop --session sessions/first
uv run weights-lab backup --session sessions/first --output sessions/first.tar.gz
```

The backup includes snippets, results, logs, and artifacts. It saves weights only
when you export them into `artifacts/` first. Stop interrupts active Python and
escalates after its grace period; native model operations may delay cancellation.

## Run the complete demo

```sh
uv run python -m examples.dogfood --output-dir sessions/tiny-demo --summary
```

The pinned tiny GPT-2 demo runs baseline → edit → exact undo → export → stop →
backup on CPU. In the measured run, all three continuation-score maps changed,
zero predictions changed, undo restored exact weights and scores, and the worker
used 21 forward passes. See the [terminal transcript and detailed results](docs/dogfood.md).
Choose an unused output directory each time.

## Give an agent a model workspace

The [agent skill and plugin](docs/skill-install.md) guide experiments, including
[checkpoint comparisons, activation patching, gradients, and probes](skills/weights-lab/references/).
Install as a Claude Code or Codex plugin, or as a standalone skill for Codex,
Claude Code, Cursor, Copilot CLI, and OpenCode; see the
[per-harness install commands](docs/skill-install.md).
The [Python API](docs/api.md) covers reversible edits, evaluation, pass budgets,
and experimental fixed-weight components. LoRA loading requires `uv sync --extra adapters`.

For a remote worker, use upstream SkyPilot:

```sh
uv run weights-lab start --cluster my-lab --source . --accelerators L4:1 --model Qwen/Qwen3-0.6B --session /tmp/lab
uv run weights-lab send --cluster my-lab --session /tmp/lab --code 'print(lab.forward_passes)' --wait
uv run weights-lab stop --cluster my-lab --session /tmp/lab
uv run weights-lab backup --cluster my-lab --session /tmp/lab --output ./lab.tar.gz
sky down my-lab
```

`--cluster` selects SkyPilot. It also supports Slurm through
`--infra slurm/mycluster/mypartition`. Read [remote setup and cleanup](docs/remote.md)
before provisioning; export and download results before teardown destroys the disk.

## What has been measured

CPU, Apple MPS, NVIDIA H200 CUDA, SkyPilot cloud execution, and upstream SkyPilot
through a real single-node Slurm scheduler have passed checks. See
[validation evidence and limits](docs/validation.md) and the
[machine-readable results](docs/validation-results.json).

Evaluating the base Qwen checkpoint and a **released edited checkpoint** measured
7.85% versus 51.25% accuracy on the full MATH-500 comparison; samples hitting the
32,768-token limit fell from 92.90% to 30.55%. This validates the released edit,
not autonomous discovery. See the [benchmark protocol and limits](docs/validation.md),
the separate [exact tensor replay](docs/qwen-replay.md), and
[GPT-2 generation and likelihood parity](docs/parity.md).

## Development

```sh
uv run --extra adapters pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Tests build tiny random models locally; public-model demos download pinned weights.
See [contribution instructions](CONTRIBUTING.md) and the [documentation index](docs/README.md).
