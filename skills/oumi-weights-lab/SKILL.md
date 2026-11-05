---
name: oumi-weights-lab
description: Use when asked to ablate, zero, edit, swap, or compare an open-weights language model's parameters, including LoRA adapters; when measuring how a weight change affects a model's answers; or when recording, patching, or probing a Hugging Face causal model's activations or gradients.
compatibility: Requires the oumi-weights-lab CLI (Python 3.11+ on Linux or macOS), provided by the Claude Code plugin when uv is installed or by uv tool install git+https://github.com/oumi-ai/weights-lab, and network access to download models. Remote workers also need an authenticated SkyPilot CLI.
---

# Oumi Weights Lab

Load one model into a persistent worker, then send it Python snippets. Snippets share state, so an edit made in one snippet is measured in the next. The worker saves every snippet's source, output, and forward-pass count in the session directory. Snippets are trusted Python with the user's filesystem and network access.

## Setup

Run `oumi-weights-lab --help`. If it fails, ask the user to install the CLI with `uv tool install git+https://github.com/oumi-ai/weights-lab`, or `uv tool install 'oumi-weights-lab[adapters,datasets,multi-gpu,quantized] @ git+https://github.com/oumi-ai/weights-lab'` for LoRA adapters, Hugging Face datasets (`lab.hf_rows`), multi-GPU loading, and quantized loading. Under the Claude Code plugin, the first call builds an environment and downloads PyTorch; allow it several minutes.

## Session commands

| Step | Command |
| --- | --- |
| Start | `oumi-weights-lab start --session sessions/NAME --model MODEL_ID --pass-limit 2000 --snippet-timeout 1800` |
| Run a snippet and wait | `oumi-weights-lab send --session sessions/NAME --file /abs/path/snippet.py --follow --timeout 1800` |
| Check status | `oumi-weights-lab peek --session sessions/NAME` |
| Stop | `oumi-weights-lab stop --session sessions/NAME` |
| Archive | `oumi-weights-lab backup --session sessions/NAME --output NAME.tar.gz` |

- Use a new `--session` directory for every `start`; a stopped session cannot be restarted.
- `start` returns while the model loads, and snippets queue until it is ready. `--device` defaults to CUDA, then Apple MPS, then CPU. `--dtype` defaults to the checkpoint's own.
- For a model too big for one GPU, add `--device-map balanced` to spread it over every visible GPU (`sequential` fills GPU 0 first). `--max-memory 0=70GiB,1=70GiB` uses only the GPUs it names, up to that much on each; by default the model may use 90% of every GPU's free memory. A model that would need CPU or disk offload is refused before any weights load. `peek` reports GiB of parameters per device under `devices`.
- Add `--quantize nf4` (or `fp4`, `int8`) to load the model with bitsandbytes, on a GPU or an Apple-silicon Mac. nf4, fp4, and int8 weights edit by requantizing the blocks (4-bit) or rows (int8) an edit touches, and undo restores them byte for byte. Kernel code from the Hub is blocked unless you pass `--allow-hub-kernels`. Read [quantized models](references/quantized.md) first.
- Stdout is JSON. `--follow` streams snippet output to stderr. `--snippet-timeout` stops a snippet that runs too long; `send --timeout` only limits how long the CLI waits.
- Snippet files can live anywhere; `--file` sends their contents. Snippets run in the directory where `start` ran, so read inputs by absolute path and write outputs under `artifacts`.
- Add `--adapter PATH_OR_ID` to load a LoRA adapter. For a cloud GPU or Slurm, read [remote workers](references/remote.md).

## Snippets

A snippet sees `lab`, `model`, `tokenizer`, `Path`, and `artifacts`. This one measures an ablation and its control:

```python
import json

rows = [json.loads(line) for line in Path("/abs/path/caps.jsonl").read_text().splitlines()]
rows = [
    {
        "prompt": r["prompt"],
        "options": [" " + o for o in r["options"]],
        "answer": r["options"].index(r["answer"]),
    }
    for r in rows
]
name = "model.layers.3.mlp.down_proj.weight"  # confirm with lab.parts("*layers.3.*")
baseline = lab.evaluate(rows, name="baseline-v1", max_passes=200)
edit_id = lab.zero(name, note="ablate layer 3 MLP")
try:
    ablated = lab.evaluate(rows, name="ablated-v1", max_passes=200)
finally:
    lab.undo(edit_id)
restored = lab.evaluate(rows, name="restored-v1", max_passes=200)
print(baseline["accuracy"], ablated["accuracy"], restored["accuracy"])
print(json.dumps(lab.diff(baseline, ablated), indent=2))
```

- A row is `{"prompt": str, "options": list or {label: text}, "answer": index, label, or option text}`. Prompt and option are tokenized separately, so start each option with the space the tokenizer expects, such as `" Paris"`. An answer given as option text must match it exactly, space included; an index avoids that.
- For instruction-tuned models, pass `chat_template=True` to `evaluate` and `score` so prompts are wrapped the way the model was trained; options then follow the assistant header and usually need no leading space. Use the same setting for every run you compare.
- `evaluate` returns a summary with `accuracy`, `correct`, `total`, `forward_passes`, and `run_dir`. Each row's scores and prediction are in `run_dir/items.jsonl`. Every run needs a new name made of letters, digits, `_`, and `-`.
- Likelihood scoring costs one forward pass per option per row; `mode="sampled"` costs one per generated token (beam search: one per search step, however many beams); a batched call costs one pass per sequence. `max_passes` caps one call and `--pass-limit` caps the whole session; size both from that cost.
- Likelihood and sampled runs process up to 8 sequences per forward (`batch_size`), and runs pair only when scored the same way. Sampled items record `finish_reason` and `generated_tokens`, and the summary reports `truncation_share`. Generation stops at the tokenizer's EOS as well as the model's, so chat replies end with the turn; pass `eos_token_id` to choose the stop tokens yourself. With `num_samples=k` and `do_sample=True`, each row scores the fraction of its k samples that are correct; add `pass_at` for pass@k and `answer_key` for majority vote. While a logit attachment, or a `patch` or `record` at nonnegative positions, is active, pass `batch_size=1` to that run and to the runs you compare it with.
- `lab.parts(pattern)` lists parameter names and shapes; names differ by architecture.
- Edit through `lab.edit`, `lab.zero`, or `lab.swap`. `lab.undo()` restores the newest edit exactly in any dtype (quantized weights byte for byte), so undo several edits newest first. Writes made directly through `model` are not tracked.
- Wrap your own `model(...)` calls in `torch.no_grad()`.
- After a timeout or error, check `lab.active_edit_ids` and undo to a known state before measuring again.

## Experiment discipline

1. Write down the question and the observation you expect before running anything.
2. Measure a baseline in the same session before any edit.
3. After undoing, measure again: restored results must match the baseline.
4. If you will choose among several edits or settings, set aside held-out rows first. Evaluate only the final choice on them, once, with `held_out=True` and `max_passes`. Held-out rows can never become dev rows, even after a failed run. A single pre-planned edit needs no held-out split.
5. Report the question, model and device, parameters edited, dataset size, baseline, edited, and restored results, changed items, forward passes, and the limits of the evidence. A small hand-written set is a diagnostic, not a benchmark.
6. An attention pattern, a probe score, or a plot is a lead. Before calling something a mechanism, confirm it with an intervention, a control intervention, and new prompts. A null ablation does not show a part is unused, and an ablation that breaks the behavior does not show the part is the whole mechanism.
7. To keep edited weights, call `lab.export("NAME")` before stopping. Export fails while attachments or patches are active, and `backup` never saves weights.

## Recipes

Read only the one you need:

- [Worked example](references/worked-example.md): full baseline, ablation, undo, and report snippet.
- [Zero or swap parts](references/zero-swap.md): reversible ablations and matched checkpoint slices.
- [Remove a direction](references/direction-edits.md): project a direction out of a weight, bias, or embedding.
- [Compare checkpoints](references/compare-checkpoints.md): parameter deltas without forward passes.
- [Evaluation reports and screens](references/evaluation.md): paired statistics, regression suites, and pinned dev subsets.
- [Save and resume edits](references/edit-bundles.md): save points that carry undo history and held-out reservations.
- [Activation patching](references/activation-patching.md): replace hidden states or module inputs at chosen tokens, and capture attention weights.
- [Attention heads](references/heads.md): list heads as weight slices, ablate one, and screen for induction heads.
- [Attribution patching](references/attribution-patching.md): rank module and token sites with activation gradients, then confirm with real patches.
- [Find and test a circuit](references/circuit-workflow.md): chain the logit lens, attribution patching, patches, and ablations into one tested claim.
- [Steering vectors](references/steering.md): add a recorded direction to the residual stream.
- [Logit lens](references/logit-lens.md): read each layer's output through the final norm and head.
- [Short-row gradients](references/gradients.md): per-row gradients for selected parameters.
- [Document-split probes](references/probes.md): linear probes that keep source documents apart.
- [LoRA update diagnostics](references/lora-diagnostics.md): adapter update spectra and rank without forward passes.
- [Written-working control](references/written-working.md): test a computation in text before editing weights.
- [Hand-built parts](references/hand-built.md): temporary prompt transforms and logit adjustments.
- [Quantized models](references/quantized.md): load nf4, fp4, or int8 and know what you can edit.
- [Remote workers](references/remote.md): SkyPilot clouds and Slurm.
