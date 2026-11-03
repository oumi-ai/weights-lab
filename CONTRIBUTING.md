# Contributing

Weights Lab 0.1.0 is experimental. Its Python API and session formats may change
as model inspection and editing workflows develop. Develop against a source checkout.

## Set up and check changes

Use Python 3.12 and install the adapter extra to exercise the full local suite:

```sh
uv sync --python 3.12 --extra adapters
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python -m build
```

The tests use small local models; the core suite needs no cloud credentials.
The scripts under `scripts/` run separate runtime and remote validation. Read
[the validation record](docs/validation.md) before running a harness that provisions
infrastructure or downloads larger models.

## Find the code

| Area | Files |
| --- | --- |
| Model loading and shared workspace | `src/weights_lab/lab.py`, `worker.py`, `session.py` |
| Weight edits, undo, and checkpoint export | `src/weights_lab/editing.py` |
| Evaluation and pass limits | `src/weights_lab/scoring.py`, `budget.py` |
| Activations, gradients, probes, and attachments | `src/weights_lab/inspection.py`, `attachments.py`, `circuits.py` |
| CLI and SkyPilot transport | `src/weights_lab/cli.py`, `launchers.py` |

Tests mirror these areas in `tests/`. Start with the
[Python API](docs/api.md) for behavior and the [documentation index](docs/README.md)
for runnable examples.

## Preserve these contracts

- Load models with eager attention so activation hooks remain observable, and
  remove temporary hooks on context exit, including after failure.
- Undo API edits from original-dtype CPU snapshots in reverse order. Restoration
  must be exact; direct tensor writes remain outside the undo history.
- Count every accepted base-model forward call, including generation steps and
  calls that raise. Reject calls beyond a pass cap before model execution.
- Keep evaluation guards persistent: dev data cannot become held-out data, and a
  held-out attempt consumes its reservation even if it fails. Preserve caller RNG
  state and module train/eval modes around evaluation.
- Keep CLI stdout machine-readable JSON. Send streamed snippet output and
  diagnostics to stderr, and preserve failure exit codes.
