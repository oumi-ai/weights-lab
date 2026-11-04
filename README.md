<p align="center">
  <img src="https://github.com/oumi-ai/oumi/raw/main/docs/_static/logo/header_readme.svg" alt="Oumi" width="300">
</p>

# Oumi Weights Lab

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-oumi--weights--lab-blue.svg)](https://github.com/oumi-ai/weights-lab/tree/main/docs)
[![About Oumi](https://img.shields.io/badge/about-oumi.ai-blue.svg)](https://oumi.ai)

A persistent Python worker that lets a coding agent load an open model once, edit
its weights reversibly, measure the effect, and export the result. Every snippet's
source, output, and forward-pass count is saved. Version 0.1 is experimental, and
snippets run as trusted Python with your filesystem and network access.

Tested on CPU, Apple MPS, NVIDIA H200, and SkyPilot with Slurm.

## Quick start

Requires Python 3.11+ on Linux or macOS, git, and [uv](https://docs.astral.sh/uv/).

```sh
uv tool install git+https://github.com/oumi-ai/weights-lab   # append @<tag> to pin
oumi-weights-lab start --session sessions/first --model HuggingFaceTB/SmolLM2-135M --device cpu
oumi-weights-lab send --session sessions/first --file edit.py --follow
oumi-weights-lab stop --session sessions/first
oumi-weights-lab backup --session sessions/first --output sessions/first.tar.gz
```

With `edit.py`:

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

Snippets share `lab`, `model`, `tokenizer`, `Path`, and `artifacts`, and their state
persists. Stdout is JSON; `--follow` streams snippet output to stderr. For LoRA
adapters, install `'oumi-weights-lab[adapters] @ git+https://github.com/oumi-ai/weights-lab'`.

## Examples and reference checks

From a source checkout, run the first two with `uv run python -m examples.NAME --output-dir DIR`.

- `examples/dogfood.py` runs baseline, edit, exact undo, export, stop, and backup on a
  tiny public GPT-2 through the CLI.
- `examples/parity.py` checks GPT-2 generation and continuation scores against
  Hugging Face's own GPT-2 integration test.
- `examples/replay_qwen_edit.py` reapplies the released
  [weight-space programming edit](https://rikiyat.github.io/blog/weight-space-programming/)
  of Qwen3.5-0.8B with `lab.swap`; see its docstring to run it.
  `tests/live/validate_qwen_benchmark.py` evaluates it on MATH-500. On an H200
  (500 problems, 4 samples each), the edit scored 51.25% against 7.85% for the base
  model, and samples reaching the 32,768-token cap fell from 92.90% to 30.55%. This
  reproduces the released edit, not its discovery.

## Documentation

- [Python API](docs/api.md)
- [Agent skill and plugin install](docs/skill-install.md)
- [Remote workers on SkyPilot and Slurm](docs/remote.md)
- [Contributing](CONTRIBUTING.md)
