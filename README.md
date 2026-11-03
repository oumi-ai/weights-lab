# Weights Lab

Keep an open language model loaded while a coding agent inspects it through Python.
Each local session runs snippets in one shared namespace and saves their source,
output, timing, and forward-pass counts.

The first milestone is implemented: a local worker and CLI. Weight editing,
evaluation, activation tools, the agent skill, and remote launchers are still to come.

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
uv run weights-lab send --session /tmp/my-lab --file examples/inspect.py --follow
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
They do not save model weights or the Python namespace. Model checkpoint export
will come with the weight-editing milestone. Compare models within one live
session; seeds alone do not guarantee the same results across devices or setups.

## Development

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Tests create a tiny random Hugging Face model locally and exercise loading,
inference, generation, queued snippets, failures, timeouts, shutdown, and backups
on CPU. They do not download pretrained weights. CI runs the suite on Python 3.11
and 3.12 with CPU PyTorch.
