# Contributing

Oumi Weights Lab 0.1.0 is experimental. Its Python API and session formats may change
as model inspection and editing workflows develop. Develop against a source checkout.

## Set up and check changes

Use Python 3.12 and install the adapter extra to exercise the full local suite:

```sh
uv sync --python 3.12 --extra adapters
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python -m build
uv run python scripts/check_repo.py
```

The tests use small local models; the core suite needs no cloud credentials.
The scripts under `scripts/` run separate runtime and remote validation. Read
[the validation record](docs/validation.md) before running a harness that provisions
infrastructure or downloads larger models.

CI runs on pull requests, pushes to `main`, and manual dispatches; feature-branch
pushes use their pull request run, and superseded runs are canceled. Package,
link, manifest, lint, build, and pure Python checks also run for docs-only changes.
Changes limited to documentation and plugin metadata skip model tests; all code,
dependency, and workflow changes run both the latest dependencies and declared
dependency floor. Model tests use tiny offline CPU models and one thread. UV's
package cache is keyed by dependency profile; virtual environments are recreated.
Docker-based [Claude Code/Codex integration tests](docs/harness-integration.md)
are opt-in and excluded from ordinary pytest runs. They do not download models;
actual agent invocation requires separately enabled API access.

## Find the code

| Area | Files |
| --- | --- |
| Model loading and shared workspace | `src/oumi_weights_lab/lab.py`, `worker.py`, `session.py` |
| Weight edits, undo, and checkpoint export | `src/oumi_weights_lab/editing.py` |
| Evaluation and pass limits | `src/oumi_weights_lab/scoring.py`, `budget.py` |
| Activations, gradients, probes, and attachments | `src/oumi_weights_lab/inspection.py`, `attachments.py`, `circuits.py` |
| CLI and SkyPilot transport | `src/oumi_weights_lab/cli.py`, `launchers.py` |

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
