---
name: weights-lab
description: Investigate an open-weights language model with a persistent Python worker, reversible parameter edits, activation probes, and controlled local evaluations. Use for experiments on model behavior or weights, including LoRA checkpoints.
---

Keep one model loaded while investigating one concrete question. Use the `weights-lab` CLI and the `lab` object in its shared Python namespace. Snippets are trusted Python with normal filesystem and network access.

This skill supplies instructions, not the Python package. Use an agent execution environment where `weights-lab` is installed separately; installing the skill or plugin does not install the CLI. A plugin cache or copied skill folder is not a source checkout.

1. Write the question, expected observation, model/adapter IDs, device/dtype, dev examples, reserved held-out examples, and pass budget in experiment notes. Separate held-out data before searching. Start a fresh session directory; an old process's Python namespace cannot be resumed from disk.
2. Start once, then submit saved snippet files. `start` returns while loading; `send --follow` waits and streams output. `peek` distinguishes loading, running, and failure. Save notes under the worker-created `artifacts` directory after startup.

   ```bash
   weights-lab start --session sessions/experiment --model HuggingFaceTB/SmolLM2-135M \
     --device cpu --dtype float32 --snippet-timeout 300 --pass-limit 1000
   weights-lab send --session sessions/experiment --file baseline.py --follow
   weights-lab peek --session sessions/experiment
   ```

3. Establish a baseline on the dev data inside this session before changing anything. Use `lab.evaluate(rows, name="baseline-v1", max_passes=...)`; compare later runs with `lab.diff(baseline, candidate)`. Use a new run name each time: letters, digits, underscores, and hyphens only, beginning with a letter or digit. Keep prompts, continuation spacing, scoring mode, and generation settings fixed.
4. Choose the cheapest informative experiment from the recipes below. Inspect names with `lab.parts()` or `model.named_modules()`; architecture-specific paths vary. Make edits through `lab.edit`, `lab.zero`, or `lab.swap` so originals are preserved. `lab.undo()` reverses the most recent edit exactly; undo multiple edits in reverse order. Direct tensor writes are possible but untracked. Repeat the restored baseline to check the control.
5. Freeze the candidate and run held-out evaluation once: `lab.evaluate(heldout, name="heldout-final", held_out=True, max_passes=...)`. Held-out data requires an explicit pass cap and cannot be rerun or repurposed as dev data, including after a failed run. Dev data cannot later become held-out data. Do not inspect held-out misses to search further.
6. Report the question, model and setup, active edit IDs/revision, dataset sizes, before/after results, changed items, forward passes, and limits of the evidence. Keep measured outcomes distinct from hypotheses. A tiny hand-built dev set is a diagnostic, not a benchmark estimate.
7. If preserving edited weights, call `lab.export("checkpoint")` while the model is alive. Relative exports go under `artifacts`; destinations must be absent or empty. Remove active attachments and activation patches first. PEFT export merges a clone into a full checkpoint, temporarily requiring a second model's memory. Save the component code separately when it cannot be represented by the checkpoint. Then stop and archive:

   ```bash
   weights-lab stop --session sessions/experiment
   weights-lab backup --session sessions/experiment --output sessions/experiment.tar.gz
   ```

The namespace includes `lab`, `model`, `tokenizer`, `Path`, and `artifacts`. The worker saves every submitted source, output, and result; evaluation saves item-level answers and summaries under `artifacts/evaluations`. Use `torch.no_grad()` for inference; eval mode alone retains gradients. Keep gradient rows short. Hooks require the uncompiled model; the loader uses eager execution and never enables custom Hub code. A timeout may leave earlier completed edits active; inspect `lab.edits()` and restore a known state before comparing. Native calls may finish before Python handles cancellation.

A pass counts one underlying model forward, including each generation step; a batch counts once. Use the session `--pass-limit` and narrower `lab.limit_passes(n)` scopes. A local evaluation cap never replaces the session cap. Compare causal conclusions within the same session: changing precision, kernels, or hardware can move near-ties.

Read only the recipe needed:

- [Compare checkpoints](references/compare-checkpoints.md): parameter deltas without extra forward passes.
- [Save and resume edits](references/edit-bundles.md): portable save points with undo history and inherited guards.
- [Evaluation reports and screens](references/evaluation.md): score changes, paired statistics, regression suites, and pinned dev subsets.
- [Zero or swap parts](references/zero-swap.md): reversible ablations and matched checkpoint slices.
- [Document-split probes](references/probes.md): representations with source documents kept apart.
- [Activation patching](references/activation-patching.md): temporary intervention on selected tokens.
- [Short-row gradients](references/gradients.md): per-row gradients without long-context memory growth.
- [Written-working control](references/written-working.md): test computation before editing weights.
- [Hand-built parts](references/hand-built.md): temporary logit modules or prompt transformations.
- [Fixed date network](references/date-network.md): frozen calendar/business-day operations and character output.
- [Worked example](references/worked-example.md): baseline, first-layer ablation, exact undo, and a saved report.

For remote sessions, read [remote workers](references/remote.md) for the SkyPilot workflow, source-checkout prerequisites, and backup before teardown. Local sessions need no SkyPilot installation.
