# Contributing

```sh
uv sync --python 3.12 --all-extras
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run python scripts/check_repo.py
uv lock --check
```

After changing dependencies in `pyproject.toml`, run `uv lock`. The Claude Code plugin
installs exactly what `uv.lock` records, and CI fails when the lock is stale.

Tests build tiny offline models. CI skips the model tests when only docs or plugin
metadata change. `tests/live/validate_*.py` run live checks that can provision
infrastructure or download large models.

Claude Code and Codex integration tests are opt-in. Build the image with
`docker build -f tests/integration/Dockerfile -t oumi-weights-lab-harness:local .`,
then run `pytest -o addopts= -m integration tests/integration` with
`OUMI_WEIGHTS_LAB_HARNESS_IMAGE=oumi-weights-lab-harness:local`. Add
`OUMI_WEIGHTS_LAB_HARNESS_INVOKE=1` and an API key to invoke the agents.

## Contracts

- Load models with eager attention, and remove temporary hooks on exit, including
  after failure.
- Undo restores exact original-dtype snapshots in reverse order.
- Count one pass per sequence for every base-model forward, including each generation
  step and failed calls, and reject calls over a cap before they run.
- Dev data never becomes held-out data, and a failed held-out run keeps its
  reservation. Restore the caller's RNG state and train/eval modes.
- CLI stdout is JSON only; snippet output and diagnostics go to stderr.

## Release

Set the version in `pyproject.toml`, `src/oumi_weights_lab/__init__.py`, and the three
plugin manifests (`check_repo.py` checks they match), merge, then tag the merge
commit: `git tag v0.2.0 && git push origin v0.2.0`. Users pin a tag with
`uv tool install git+https://github.com/oumi-ai/weights-lab@v0.2.0`, and plugin
installs update when the manifest version changes.
