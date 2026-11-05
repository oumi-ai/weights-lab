<p align="center">
  <a href="https://oumi.ai">
    <img src="https://github.com/oumi-ai/oumi/raw/main/docs/_static/logo/header_readme.svg" alt="Oumi" width="300">
  </a>
</p>

# Oumi Weights Lab

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-oumi--weights--lab-blue.svg)](https://github.com/oumi-ai/weights-lab/tree/main/docs)
[![About Oumi](https://img.shields.io/badge/about-oumi.ai-blue.svg)](https://oumi.ai)

An agent skill and Python worker that let your coding agent experiment on an open
language model: load it once, edit its weights reversibly, measure the effect, and
export the result. It is inspired by
[weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, and Khattab), in which a general-purpose LLM reads and writes a
network's weights directly instead of relying on an optimizer such as gradient
descent. Every snippet's source, output, and forward-pass count is saved.
Version 0.3 is experimental, and snippets run as trusted Python with your filesystem
and network access.

Tested on CPU, Apple MPS, NVIDIA H200, a 72B model split across eight NVIDIA H100s,
and SkyPilot with Slurm.

![Your coding agent edits a running model's weights: GPT-2's next word changes from Paris to banana, then undo restores Paris. The same model stays loaded throughout.](docs/assets/agent-workbench.svg)

A deliberately silly weight edit, with exact undo. [Try the example](examples/banana.py).

## Install in your coding agent

Requires Linux or macOS, git, and [uv](https://docs.astral.sh/uv/).

### Claude Code

```sh
claude plugin marketplace add oumi-ai/weights-lab
claude plugin install oumi-weights-lab@oumi-weights-lab
```

The plugin includes the `oumi-weights-lab` CLI. Its first run downloads PyTorch, so
allow a few minutes.

### Codex

```sh
codex plugin marketplace add oumi-ai/weights-lab
codex plugin add oumi-weights-lab@oumi-weights-lab
uv tool install git+https://github.com/oumi-ai/weights-lab
```

### Cursor, GitHub Copilot CLI, OpenCode, and other skill-compatible agents

Copy the whole skill folder, then install the CLI:

```sh
git clone https://github.com/oumi-ai/weights-lab
mkdir -p ~/.agents/skills
cp -R weights-lab/skills/oumi-weights-lab ~/.agents/skills/
uv tool install git+https://github.com/oumi-ai/weights-lab
```

Then ask your agent, for example: "Use oumi-weights-lab to check whether zeroing
layer 3's MLP in Qwen/Qwen3-0.6B changes its answers to the questions in caps.jsonl."

To load LoRA adapters, Hugging Face datasets with `lab.hf_rows`, one model across
several GPUs (`start --device-map balanced`), or a 4-bit or 8-bit model
(`start --quantize nf4`, read-only), install the CLI as
`uv tool install 'oumi-weights-lab[adapters,datasets,multi-gpu,quantized] @ git+https://github.com/oumi-ai/weights-lab'`.
The Claude Code plugin includes all four.
Append `@<tag>` to a `git+https://` URL to pin a release.

## Use the CLI directly

```sh
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
persists. Stdout is JSON; `--follow` streams snippet output to stderr.

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
  reproduces the released edit, not its discovery. vLLM ran that benchmark;
  `tests/live/validate_truncation.py` checks that the lab's own sampling hits the cap
  as often. On 40 of the problems without the presence penalty, which transformers
  lacks, both truncated 99.4% of the base model's samples, and the lab 51.3% of the
  edited model's against 50.6% for vLLM.

## Documentation

- [Python API](docs/api.md)
- [Remote workers on SkyPilot and Slurm](docs/remote.md)
- [Contributing](CONTRIBUTING.md)
