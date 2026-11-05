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
| `project_out(name, direction, *, side="output", index=None, scale=1.0, note="")` | Remove a direction from one weight or bias with a single edit; see below |
| `undo(edit_id=None)` | Restore the most recent edit exactly; undo is strictly last-in, first-out |
| `edits()` | Edit history |
| `export(path)` | Hugging Face checkpoint plus edit manifest; a LoRA adapter is merged one layer at a time, other adapters into a copy of the model |
| `save_edits(path)`, `load_edits(path)` | Save edits, undo history, pass count, and dataset reservations, then resume them in a fresh session of the same model (CLI: `start --resume PATH`) |

`project_out` writes `W - scale * d (dᵀW)` or its transpose, where `d` is the
direction scaled to unit length. Scale 1.0 removes the direction and 2.0 reflects
across it. It makes one `edit`, so undo, `edits()`, and saved bundles work as usual.
`side="output"` removes the direction from what the layer writes; `side="input"`
removes it from what the layer reads. The direction must be 1-D, finite, nonzero,
and as long as that axis.

| Parameter | Stored shape | `side="output"` (direction length) | `side="input"` (direction length) |
| --- | --- | --- | --- |
| `Linear` weight | [out, in] | out | in |
| `Conv1D` weight | [in, out] | out | in |
| `Embedding` weight | [vocab, dim] | dim | refused |
| `Linear` or `Conv1D` bias | [out] | out | refused |

`index` picks entries along the other axis and never the projected one: input
columns for a `Linear` output projection, output rows for a `Linear` input
projection, and token rows for an embedding. It takes None, an int, a slice with
step 1, or a nonempty list of distinct ints, and a bias takes none. Any other module
or attribute, such as a norm, a PEFT `lora.Linear` wrapper, or an unsupported index,
raises `ValueError`; use `edit` with your own callable. For an adapter, edit its
`base_layer` and `lora_B` weights.

Only API edits can be undone or saved; writes made directly through `model` are
untracked. A resumed session can tighten the saved pass limit but not raise it.
Edits act on each parameter where it lives, so a model split across devices edits
and undoes the same way, and a bundle saved under one device layout resumes under
another.

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
- `chat_template=True` on `score`, `evaluate`, and `generate` renders each prompt
  as a user turn with the tokenizer's chat template, ready for the assistant's
  reply; `{"system": text, "kwargs": {...}}` adds a system message and template
  variables such as `enable_thinking`. Prefix attachments transform the prompt
  first. The rendered text is tokenized without adding special tokens, because
  the template supplies them, and options follow the assistant header, so they
  usually need no leading space; `yes_no` mode's default options keep theirs, so
  pass `options=["yes", "no"]` under a template. `evaluate` records the
  template's hash, system message, variables, and a fingerprint of what the
  template renders (so templates that insert today's date change it) as
  `prompt_format`; runs pair only when it matches, while dev and held-out
  identities still use the raw prompts. A tokenizer without a chat template, or
  a template that rejects the settings, raises before anything is reserved. With
  thinking enabled, likelihood options are scored as the start of a reasoning
  trace, and sampled answers keep their `<think>` text, so grade them with a
  checker or `answer_key` that skips it.
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
  or a `record` or `patch` at positions counted from the left, is active, use
  `batch_size=1` for the run and the runs you compare it with, because left
  padding shifts what those hooks see. Items record each option's token count as
  `option_tokens`.
- Sampled mode also generates up to `batch_size` rows at once. Row `i` draws from
  its own generator seeded with `seed + i`, whatever batch it lands in; runs at
  different batch sizes still do not pair, because padding can shift logits
  slightly. Items record `finish_reason` (`eos`, `stop_string`, or `max_tokens`),
  `generated_tokens`, and `sample_seed`; summaries report `truncation_share` and
  `mean_generated_tokens`. Settings that read the padding or the whole batch run
  one row at a time, including when the model's own generation config sets them:
  beam search, `repetition_penalty`, `no_repeat_ngram_size`, `max_time`,
  `min_length`, `prefill_chunk_size`, watermarking, and `max_length` without
  `max_new_tokens`. A `batch_size` above 1 with them raises. Assisted generation
  is refused, because it cannot be charged per generated token.
- Sampled mode and `generate` stop at the tokenizer's EOS token as well as the
  model's, as vLLM does. A checkpoint without a `generation_config.json` can omit
  the token its chat turns end with: Qwen3.5 stops on `<|endoftext|>` but ends
  turns with `<|im_end|>`. Pass `eos_token_id` to choose the stop tokens yourself.
  When the tokenizer's EOS is added, `scoring_config` records `stop_token_ids`, so
  those runs do not pair with runs from v0.2.0, which ran past the end of the turn.
- `num_samples=k` (sampled mode with `do_sample=True`) draws k samples per row;
  sample `j` of row `i` is seeded with `seed + i*k + j`. Each row scores the
  fraction of its samples that are correct, and `accuracy` is the mean fraction.
  `pass_at=[1, k]` adds unbiased pass@k per row and per run. With
  `answer_key(generated, row)`, each sample votes for the value it returns (JSON
  values only); the summary reports `vote_accuracy`, and a tie counts as
  incorrect; a sample whose key is `None` abstains. With k>1, each item holds
  its `samples` (each with `generated`, `correct`, `finish_reason`,
  `generated_tokens`, `sample_seed`, and `key`), `correct_samples`, and the
  fraction as `correct`; `prediction` is the vote, or `None` without an
  `answer_key`. A checker that names a `row` parameter receives a copy of the
  row. A custom `answer_key`, like a custom checker, needs `checker_id` before
  runs can be compared.
- Batching multiplies eager attention memory, so settle `batch_size` on dev rows
  before a held-out run: an out-of-memory failure still uses up the reservation.
- `diff(before, after)` and `compare_runs(before, after)` pair rows by position and
  need the same rows in the same order. `compare_runs` also checks row IDs and adds
  McNemar p-values, bootstrap intervals, and the largest movers. With several
  samples per row it reports a paired permutation test (`permutation_p`) on the
  per-row fractions instead of McNemar, flipping whole documents; it is exact
  for up to 16 changed documents and estimated from `bootstrap_samples` random
  flips beyond that. For sampled runs
  it also compares truncation share and mean generated length.
- Runs pair only when their protocol matches: session, mode, seed, generation and
  scoring config, tokenizer policy, checker, number of samples, answer key, prompt
  format, and held-out status. `cross_session=True` on `diff` and `compare_runs`
  skips only the session check, so runs from another session (an export reloaded in
  a new session, an adapter against its merged export, one GPU against a split model)
  can pair by run directory path or summary. It must be a bool, and `compare_suites`
  passes it through to `compare_runs`. Without it, a pair
  that differs only by session is refused with a hint to pass it. Each new summary
  has a `setup` block: `devices`, `dtypes`, `torch`, `transformers`, `peft`,
  `adapter`, `attention`, `device_map`, `model_id`, and `model_commit`. With the
  opt-in, `compare_runs` adds `cross_session` with `before_session`,
  `after_session`, `before_setup`, `after_setup`, `setup_differences` (field names
  that differ, or `["setup"]` when a run has none), and a caveat. `diff` returns
  the same list of changes. Run the unchanged base model in both sessions as a
  control, because near-tie predictions can flip between setups.
- `evaluate_suite` and `compare_suites` gate a candidate on per-dataset accuracy and
  NLL tolerances. `create_screen` and `verify_screen` pin a stratified dev subset.
- `hf_rows(path, *, split, to_row, name=None, revision=None, limit=None)` loads a
  Hugging Face dataset split (with the `datasets` extra) and turns each example
  into a row with `to_row`. A Hub revision is resolved to its commit, and each
  row records its source under `metadata.hf` and gets a default `id`. A Hub
  "test" split is not treated as held-out data; pass `held_out=True` to
  `evaluate` when you mean it. A full 40-character commit hash as `revision` is
  used as given, which also works offline (`HF_HUB_OFFLINE=1`) from the cache.

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
  For a model dispatched across devices, `lab.device_map` is a copy of its
  module-to-device map (`None` for a model moved to one device), and outputs come
  back on the inputs' device. `lab.device_memory` gives GiB of parameters per
  device, which `peek` reports as `devices`.
- `record` captures activations on CPU at the chosen positions, the last token by
  default. Around batched scoring, each capture is one packed batch of rows and
  options; use `batch_size=1` for one capture per option. `patch` replaces them with a tensor or a callable's result. Both remove
  their hooks on exit.
- `at` says what to capture or replace. `record(..., at=)` takes `"output"` (the
  default), `"input"`, or `"attention"`; `patch(..., at=)` takes `"output"` or `"input"`.
  An input is the module's first tensor argument, or `hidden_states=` when the call
  passes it by keyword; a module with neither raises. Both must be `[batch, tokens, ...]`.
  Patching the input of `o_proj` (or GPT-2's `attn.c_proj`) acts on the per-head outputs.
- `positions` is `None` (the last token), an int, a list of distinct ints, `"all"`, or a
  slice. `"all"` selects every token of each call, so under cached generation the first
  call has the whole prompt and later calls have one token each. A slice follows Python
  rules on each call's length; its step must be positive, and one that selects nothing
  raises. Malformed values raise when the context opens; out-of-range ints and
  selections that pick nothing raise at the forward, because they depend on each
  call's length.
- Batched `score` and `evaluate` left-pad. A position counted from the left (an int of 0
  or more, or a slice that does not end at the end of the sequence) is refused unless
  `batch_size=1`. `"all"`, negative ints, and slices like `slice(-2, None)` (negative
  start, no stop or a negative one, step 1) are safe.
- `record(name, at="attention")` captures attention weights `[batch, heads, selected,
  keys]` from an attention block such as `model.layers.0.self_attn` or
  `transformer.h.0.attn`; positions select queries. It needs eager attention, which the
  lab loads, and a module whose output's second element is the weights, so it refuses
  layers, MLPs, and linear-attention blocks; in a hybrid model only the full-attention
  layers can be captured. Some architectures return no weights on older Transformers
  versions (OPT on 4.56, for example), and capture raises there. Attention cannot be
  patched. `lab.patches()` lists each patch's `at`.
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
