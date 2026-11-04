# Python API

Snippets share one namespace with `lab`, `model`, `tokenizer`, `Path`, and
`artifacts`. Models load with eager attention in eval mode; use `torch.no_grad()`
when gradients are unnecessary. `export`, `save_edits`, and `load_edits` resolve
relative paths under `artifacts/`; other paths resolve from where `start` ran.

## Weights

| Call | Effect |
| --- | --- |
| `parts(pattern="*")` | Parameter names, shapes, dtypes, and devices |
| `measure(name)` | Statistics for one parameter |
| `compare(other, pattern="*")` | Current-minus-reference statistics against a model or tensor dict |
| `edit(name, value, index=None, note="")` | Replace a parameter or slice with a tensor, or with a callable's result; shapes must match exactly |
| `zero(name, index=None)` | Zero a parameter or slice |
| `swap(name, source, source_name=None, index=None)` | Copy a parameter or slice from a tensor, model, or dict |
| `undo(edit_id=None)` | Restore the most recent edit exactly; undo is strictly last-in, first-out |
| `edits()` | Edit history |
| `export(path)` | Hugging Face checkpoint plus edit manifest; a PEFT adapter is merged into a copy |
| `save_edits(path)`, `load_edits(path)` | Save edits, undo history, pass count, and dataset reservations, then resume them in a fresh session of the same model (CLI: `start --resume PATH`) |

Only API edits can be undone or saved; writes made directly through `model` are
untracked. A resumed session can tighten the saved pass limit but not raise it.

## Evaluation

```python
rows = [
    {
        "id": "capital",
        "prompt": "France's capital is",
        "options": {"paris": " Paris", "rome": " Rome"},
        "answer": "paris",
    }
]
before = lab.evaluate(rows, name="baseline")
# ... edit ...
after = lab.evaluate(rows, name="candidate")
print(lab.diff(before, after))
report = lab.compare_runs(before, after)
```

- `score(prompt, options)` returns each option's summed log probability. Prompt and
  completion are tokenized separately without special tokens, so include a
  completion's leading space. Scores are not length-normalized. Options share
  left-padded forwards of up to `batch_size` sequences (default 8, or 1 for a
  model whose forward takes no `position_ids`);
  `batch_size=1` scores each option alone, exactly as v0.1 did.
- `evaluate(data, *, name=None, mode="likelihood", held_out=False, max_passes=None, ...)`
  accepts JSONL or rows and saves items and a summary under
  `artifacts/evaluations/NAME/`. Modes are `likelihood`, `yes_no`, and `sampled`.
  An answer is an option index, label, or completion text. An exact top-score tie
  counts as incorrect. Sampled mode decodes greedily by default and grades with
  `checker` or a stripped exact match.
- Likelihood and yes/no runs score up to `batch_size` sequences per forward
  (default 8), packing whole rows in order. The summary's `scoring_config`
  records how a run was scored, and runs pair only when it matches; runs saved
  before configs were recorded count as `legacy-v0.1`. While a logit attachment,
  or a `record` or `patch` at nonnegative positions, is active, use
  `batch_size=1` for the run and the runs you compare it with, because left
  padding shifts what those hooks see.
- Sampled mode also generates up to `batch_size` rows at once. Row `i` draws from
  its own generator seeded with `seed + i`, so seeded samples do not depend on the
  batch size. Items record `finish_reason` (`eos`, `stop_string`, or `max_tokens`),
  `generated_tokens`, and `sample_seed`; summaries report `truncation_share` and
  `mean_generated_tokens`. Beam search, `max_time`, `min_length`,
  `prefill_chunk_size`, watermarking, and `max_length` without `max_new_tokens`
  run one row at a time, and a `batch_size` above 1 with them raises.
  Items record each option's token count as `option_tokens`. Batching multiplies
  eager attention memory, so settle `batch_size` on dev rows before a held-out
  run: an out-of-memory failure still uses up the reservation.
- `diff(before, after)` and `compare_runs(before, after)` pair rows by position and
  need the same rows in the same order. `compare_runs` also checks row IDs and adds
  McNemar p-values, bootstrap intervals, and the largest movers. For sampled runs
  it also compares truncation share and mean generated length.
- `evaluate_suite` and `compare_suites` gate a candidate on per-dataset accuracy and
  NLL tolerances. `create_screen` and `verify_screen` pin a stratified dev subset.

Held-out data needs `held_out=True` and `max_passes`, runs once per session, and
stays reserved even if the run fails. Dev and held-out data can never trade roles;
overlap means an identical row, or a shared prompt when either side is sampled, after
Unicode normalization and whitespace stripping. Scored rows that share a generic prompt
but offer different options may sit on opposite sides. Probabilities are normalized over the supplied options only. These guards
keep an honest workflow disciplined; snippet code can still change budgets or
registry files directly.

## Pass budget

Every base-model forward counts one pass per sequence in the batch, including each
generation step and calls that fail. Within `generate`, a prompt's beams count as one
sequence, while each of several sampled return sequences counts on its own.
`start --pass-limit N` or `lab.budget.limit` caps the session, and
`with lab.limit_passes(n):` adds a temporary cap. Calls over a cap raise
`BudgetExceeded` before running.

## Activations and gradients

```python
inputs = tokenizer("hello world", return_tensors="pt").to(lab.device)
with lab.record(["transformer.h.0"], positions=[-1]) as captures, torch.no_grad():
    model(**inputs)
hidden = captures["transformer.h.0"][0]
with lab.patch("transformer.h.0", hidden, positions=[-1]), torch.no_grad():
    changed = model(**inputs).logits
gradients = lab.gradient("hello world", parameters=[name], max_tokens=64)
```

- `lab.device` is the device of the input embeddings; move tokenizer outputs there.
- `record` captures activations on CPU at the chosen positions, the last token by
  default. Around batched scoring, each capture is one packed batch of rows and
  options; use `batch_size=1` for one capture per option. `patch` replaces them with a tensor or a callable's result. Both remove
  their hooks on exit.
- `gradient` returns per-row mean causal-loss gradients on CPU and leaves `.grad` and
  `requires_grad` unchanged. Pass `target` to score only the answer tokens.
- `probe(features, labels, groups)` fits a ridge classifier, keeping each source
  document on one side of the split.
- `lora_diagnostics(pattern="*", adapter=None)` reports each LoRA update's singular
  values and rank without running the model.

## Attachments (experimental)

`lab.attach(component, mode="prefix")` transforms prompts for `generate`, `score`,
and `evaluate`; `mode="logits"` adds a delta to every model output and rejects calls
that pass `labels`.
