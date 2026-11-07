# Python API

Snippets share one namespace with `lab`, `model`, `tokenizer`, `Path`, and
`artifacts`. Models load with eager attention in eval mode; use `torch.no_grad()`
when gradients are unnecessary. `export`, `save_edits`, and `load_edits` resolve
relative paths under `artifacts/`; other paths resolve from where `start` ran.

## Scripts

```python
from oumi_weights_lab import open_lab

lab = open_lab("/abs/path/to/model", device="cpu", pass_limit=5000)
```

`open_lab(model, *, session=None, device="auto", dtype="auto", adapter=None, quantize=None,
seed=0, pass_limit=None, allow_hub_kernels=False, revision=None, adapter_revision=None,
device_map=None, max_memory=None, resume=None)` loads a model into a `Lab` in the calling
process. It is the same `lab` a worker gives its snippets, for a script on a model that loads
in seconds. The model is gone when the script ends, so keep a large model in a worker instead.
Every option `start` passes to the loader is accepted under the same name, and is checked
before anything is created or loaded, with the checks `start` uses. `max_memory` takes a dict
such as `{"0": "70GiB"}` or the `0=70GiB` text. `resume` is a saved-edits file to load.
The worker's own options have no counterpart in a script: the snippet timeout, idle timeout,
maximum lifetime, shutdown grace, and `--cluster`.

`session` is a new or empty directory (by default a new one under `./sessions/`), and an
existing directory with files in it is refused. A relative path is made absolute. A script
session holds only `artifacts/`, where `export`, `save_edits` and the analysis calls write.
It has no `config.json`, `status.json` or worker files, so the worker commands (`send`,
`peek`, `stop`, `backup`) do not apply to it. A session directory that `open_lab` created is
removed again if loading fails.

`oumi-weights-lab script FILE [ARGS...]` runs a Python file as `__main__` in the lab's own
environment, with `ARGS` in `sys.argv` and the script's folder first on `sys.path`, so the
import works under the Claude Code plugin and a `uv tool` install, where a plain `python` cannot import the lab. The script's exit code is the
command's exit code, and an error prints the script's own traceback.

## Weights

| Call | Effect |
| --- | --- |
| `parts(pattern="*")` | Parameter names, shapes, dtypes, and devices; quantized weights add a `quantization` block |
| `measure(name)` | Statistics for one parameter; a quantized weight is measured after dequantizing |
| `compare(other, pattern="*")` | Current-minus-reference statistics against a model or tensor dict |
| `edit(name, value, index=None, note="")` | Replace a parameter or slice with a tensor, or with a callable's result; shapes must match exactly |
| `zero(name, index=None)` | Zero a parameter or slice |
| `swap(name, source, source_name=None, index=None)` | Copy a parameter or slice from a tensor, model, or dict |
| `project_out(name, direction, *, side="output", index=None, scale=1.0, note="")` | Remove a direction from one weight or bias with a single edit; see below |
| `undo(edit_id=None)` | Restore the most recent edit exactly, packed bytes included for quantized weights; undo is strictly last-in, first-out |
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

## Quantized models

`start --quantize nf4` (or `fp4`, `int8`) loads the model with bitsandbytes on one cpu,
mps, or cuda device. It needs the `quantized` extra, which pulls in `accelerate` too, and
is refused before any weights load when bitsandbytes is missing. It cannot be combined
with `--device-map`. nf4 and fp4 are 4-bit codes with 64-weight blocks and no double
quantization; int8 is LLM.int8 with its default outlier threshold. Layers compute in the
`--dtype` (the checkpoint's float dtype under `auto`, float32 when it states none).
Embeddings, norms, biases, and `lm_head` stay float. A checkpoint that bitsandbytes
already quantized loads as it is without `--quantize`; a `--quantize` that disagrees is
refused, and so is any `--quantize` on a GPTQ, AWQ, or other checkpoint.
`lab.quantization` is None for float models, and otherwise a dict with `method`,
`source` (`"load"` or `"checkpoint"`), `compute_dtype`, `double_quant`, `blocksize`
(4-bit) or `threshold` (int8), and `bitsandbytes`. `peek` reports `method` and
`compute_dtype` under `quantization`, and each run's `setup` block records the dict as
`quantization`, so a cross-session report of nf4 against bf16 lists it.

Quantized weights read through their dequantized values, and nf4, fp4, and int8 weights edit:

- `parts` reports a quantized weight's logical `shape`, `dtype`, and `numel`, and adds
  `quantization` with `method`, `blocksize` or `threshold`, `double_quant`,
  `packed_shape`, and `packed_dtype`. Other entries are unchanged.
- `measure` and `compare` use the dequantized float32 values, report the logical shape,
  and add the same `quantization` block. Layers that are `Conv1D` in the float
  model, such as GPT-2's, are stored `[out, in]` once quantized, so `parts`, `measure`,
  and `compare` show the transposed shape, and comparing them with the float model raises
  a shape mismatch. Other layers compare against a float model of the same architecture,
  in either direction.
- `edit`, `zero`, `swap`, and `project_out` work on nf4, fp4, and int8 weights, with the index
  and shapes of the logical weight (`[out, in]`, also for GPT-2's fused layers). The edit
  dequantizes the weight, applies the change in float32, converts it to the weight's
  dtype, and requantizes only the blocks that hold a changed value, writing
  them back into the same parameter. A 4-bit block is 64 weights. An int8 weight (LLM.int8)
  has one scale per row, so its block is a row: `blocksize` is the row length, and
  editing the largest value of a row rescales the whole row, which shows in
  `collateral_l2`. int8 scales move from the weight to the layer on the first forward;
  edit and undo write wherever they live. Every other block keeps its exact bytes and scale,
  and so does every code in a touched block whose value and scale did not move. `undo`
  restores the packed bytes and scales exactly.
- An edit's record has the usual keys, with the logical `shape` and `dtype`, and a
  `quantization` block: `type`, `blocksize`, `blocks_requantized`, `requested_delta_l2`
  (how far the edit asked to move the selection), `realized_delta_l2` (how far it
  moved), `residual_l2` (distance from the request to what landed), `collateral_l2`
  (how far weights outside the selection moved because they share a block whose scale
  changed), and `changed_elements`. When a change is smaller than the quantization step
  (`requested_delta_l2` above 0 and `realized_delta_l2` 0), the edit warns, is still
  recorded, and undoes as usual.
- Edits refuse with `ValueError`, and change nothing, on double-quantized weights
  (`bnb_4bit_use_double_quant`, whose scales are shared across the whole tensor; reload
  the float checkpoint with `--quantize`), on weights bitsandbytes has repacked for CPU
  kernels (undo refuses those with `RuntimeError`), and on int8 layers that keep float
  weights (`has_fp16_weights`). Float
  parameters, including the bias of a quantized layer, edit and undo exactly. On a GPTQ
  or AWQ checkpoint, an edit of a non-float parameter names the method.
- `swap` accepts a quantized weight as its source and copies its dequantized float values.
- `save_edits` and `start --resume` carry quantized edits; see
  [Saving and exporting quantized edits](#saving-and-exporting-quantized-edits).
- `gradient` skips quantized weights when `parameters` is omitted, raises `ValueError`
  when you name one, and still takes activation gradients (`modules=`) through
  quantized layers.
- `heads` and `head_weights` read through the dequantized weights. transformers stores
  GPT-2's fused layers quantized as `[out, in]`, so a head's `index` selects rows of
  `c_attn` and columns of `c_proj` there, the transpose of the float model; the weights
  `head_weights` returns have the same shapes either way.
- `export` writes a quantized model through `save_pretrained`, and refuses to merge a
  PEFT adapter into a quantized base.
- An adapter (`--adapter`) loads on a quantized base. On an int8 base it needs peft 0.14.0
  or newer, and `start` refuses before loading weights when the installed peft is older;
  nf4 and fp4 work on every supported peft.

### Saving and exporting quantized edits

A bundle with a quantized edit has `format_version` 2; a bundle with none stays at 1, and
`load_edits` reads both and refuses any other version. For each active quantized edit the
payload holds the packed parts before and after it, named `{id}-before.<part>` and
`{id}-after.<part>` (`data` and `absmax` for nf4 and fp4, `data` and `scales` for int8), all CPU copies. The
manifest's entry for the edit gains a `quantization` block with `type`, `blocksize`,
the logical `shape` and `dtype`, the `parts`, and the edit's `quantization` report. The
manifest also records the lab's `quantization` settings: `method` and `blocksize` or
`threshold`, or null for a float model.

Resume replays each edit in order. The packed parts the model holds must equal the edit's
`before` parts, or `load_edits` raises `ValueError` and changes nothing. It then writes the
`after` parts, and keeps the `before` parts as the edit's undo snapshot, so undo after
resume is exact and still last in, first out.

A bundle with quantized edits resumes only into a lab with the same method and the same
`blocksize` or `threshold`. `load_edits` checks that before the fingerprint, and the
message names both settings, including when the lab is float. Only a float bundle into
a quantized lab goes through the fingerprint check, and its message says that
quantization differs. The fingerprint
covers the packed bytes and scales and the model config's `quantization_config`; it is the
same across loads and across the first forward, and a saved bundle's base fingerprint is
taken over the pre-edit bytes.

`export` on a quantized model calls `save_pretrained`, so the packed bytes and scales
go out as they are in memory. `from_pretrained(path)` with no other settings then gives a
model of the same method with identical bytes, scales, and logits. The export manifest
has a `quantization` block with the same settings. Merging a PEFT adapter into a
quantized base is refused with `ValueError`, before anything is copied.

`--allow-hub-kernels` lets kernel code downloaded from the Hugging Face Hub run, as
bitsandbytes does on macOS 26 and on CPUs with AVX512-BF16 when the optional `kernels`
package is installed. Without it every session blocks that package, and bitsandbytes
falls back to its built-in kernels. If `kernels` was already imported, the session fails
to start and names the flag. `lab.allow_hub_kernels` and the `setup` block record it.

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
  `adapter`, `attention`, `device_map`, `quantization`, `allow_hub_kernels`, `model_id`,
  and `model_commit`. With the
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
- `gradient(..., modules=, positions="all", at="output")` also returns gradients of that
  loss with respect to module activations. `modules` is a name or a nonempty list of
  distinct names; `at` is `"output"` or `"input"` as for `patch` (`"attention"` raises).
  Each entry is `[rows, selected tokens, ...]` on CPU in the activation's dtype. With
  `modules`, `parameters=None` returns no parameter gradients; pass both for both.
  `positions` takes the same forms as `record` but defaults to `"all"`, not the last
  token: the last position predicts nothing scored, so its output gradient is zero.
  `positions` or `at` without `modules` raises, and so does a module that did not run
  or ran twice in one forward. An input gradient is through that module's use of its
  input only, not through other uses of the same tensor such as a residual connection,
  so it is the same on a frozen and a trainable model. When the activation has no
  gradient path (every parameter frozen, or a frozen PEFT base), the gradient is taken
  at a detached copy that replaces the activation; forward values do not change.
- `probe(features, labels, groups)` fits a ridge classifier, keeping each source
  document on one side of the split.
- `contrast_directions(pairs, *, layers=None, method="mean_diff", positions=-1,
  at="output", held_out=0.25, seed=0, batch_size=8)` computes a direction at each
  decoder layer from `(positive, negative)` prompt pairs, or `{"positive", "negative"}`
  dicts, and checks it on pairs it was not computed from. Each prompt runs once, so the
  call costs two passes per pair, counted against any pass cap before it starts.
  - A prompt is represented by its activation at `positions`, which must be a negative
    int (default -1, the last token) because prompts are left-padded in batches. The site
    is the residual stream after each decoder block (`at="output"`) or before it
    (`at="input"`). Blocks are found by name, as `layers.N` or `h.N`; `layers` lists
    indexes, and None means all.
  - `method="mean_diff"` averages positive minus negative over the training pairs.
    `method="pca"` takes the top right singular vector of the uncentered training
    differences, the first principal component after randomly flipping each pair's sign. It
    is signed so the mean difference projects positively. `explained_variance` is the share
    of the differences' second moment it carries, sigma1^2 over the sum of sigma^2; if every
    difference is zero it is None and the direction is all zeros.
  - Pairs that share a prompt, directly or through a chain (a common neutral negative, a
    reused positive), form a group, and a group goes whole to one side of the split, so no
    prompt appears on both sides. Groups are picked from `seed` until the held-out side
    reaches `held_out` of the pairs, 0 to 0.5, rounded up, but never more than half the
    pairs, so at least half train. With `held_out` above 0 there must be at least two
    groups; `held_out=0` keeps every pair for training. Rows report `pair_groups`.
  - Rows are sorted by layer and hold `layer`, `module`, `at`, `method`, `direction` (a
    unit float32 CPU tensor), `norm` (the raw mean-difference norm, to scale steering by),
    `explained_variance` (PCA only, else None), `train_pairs` and `heldout_pairs` (pair
    indexes), and the held-out checks. `heldout_pair_accuracy` is the share of pairs whose
    positive projects above its negative. `heldout_auc` separates positive from negative
    prompts by projection. `heldout_effect_size` is the difference of mean projections over
    the pooled standard deviation, and is None with fewer than two held-out pairs. All
    three are None when `held_out=0`. A zero `norm` means the contrast leaves no trace at
    that site, and the direction is all zeros.
  - It refuses fewer than 4 pairs, empty or duplicate pairs, a pair that also appears
    reversed, a single group with `held_out` above 0, a pair whose two prompts are equal, a position left padding could move, a prompt too short for the position,
    unknown layers, and a model without numbered decoder layers. The direction goes
    straight to `project_out` or a steering `patch`. A strong held-out result says the
    contrast is linearly readable, not that the model uses it; test that with steering or
    ablation.
- `lora_diagnostics(pattern="*", adapter=None)` reports each LoRA update's singular
  values and rank without running the model.

## Building blocks for new methods

```python
result = lab.run_with(
    "The capital of France is",
    activations=[{"module": "model.layers.3", "positions": -1, "value": replacement}],
    record=["model.layers.5"],
)
jac = lab.jacobian("The capital of France is", "model.layers.3", "model.layers.5")
rows = lab.sweep("The capital of France is", [None, [intervention_a], [intervention_b]])
```

An intervention is `{"module": name, "at": "output" | "input", "positions": None, "value":
tensor or callable}`, as for `patch`: positions default to the last token, and a callable
receives the selected tensor and returns its replacement.

- `run_with(inputs, *, activations=(), weights=None, record=(), logits="last", grad=False)`
  runs the model once. `inputs` is a string, a list of strings (left-padded with an
  attention mask and position ids, as scoring batches are), or a dict of tensors, which
  passes through unchanged for full control. Text is prepared the way `score` prepares a
  prompt: prefix attachments and any chat template apply, and special tokens follow
  scoring's rule, so numbers line up with `score` and `evaluate` on the same text. Active
  logit attachments are model hooks, so they do apply to the returned logits, and a
  padded batch refuses them as scoring does. `activations` are interventions that apply
  for this call only. `record` takes module names or `{"module", "at", "positions"}`
  dicts, each module once, and a record sees the replaced value. `logits` is `"last"`,
  `"all"`, or `None`.
- `weights` maps parameter names to whole replacement tensors through
  `torch.func.functional_call`; nothing is written into the model. Shapes must match.
  A tied parameter has one value for all its names: replacing `lm_head.weight` on a tied
  model also replaces the input embedding for that call. Two names for one parameter with
  different tensors raise, and so does a quantized weight; use `edit` and `undo` there.
- Results are `{"logits", "records", "forward_passes"}`. By default they are detached
  float32 CPU tensors. With `grad=True` the call keeps its graph and logits and records
  stay on their device, attached, so `torch.autograd.grad` reaches parameters, replaced
  tensors, and recorded values. `requires_grad` flags and train/eval modes are not changed.
- Hooks are removed, parameters are unchanged, and `lab.patches()` is untouched after
  any error, including `KeyboardInterrupt`. Absolute (nonnegative) positions raise on a
  padded batch with the message scoring gives; negative positions and `"all"` do not.
- `jacobian(inputs, source, target, *, source_at="output", target_at="output",
  source_positions=None, target_positions=None)` returns the derivative of the target
  selection with respect to the source selection for one sequence, shaped
  `[*target_shape, *source_shape]` without the batch dimension, as a float32 CPU tensor.
  `jvp(..., tangent)` returns `(target_value, derivative)` and `vjp(..., cotangent)`
  returns `(target_value, gradient)`, with the same point arguments.
- The model runs with the source selection replaced through a hook, so the derivative is
  the effect of changing that selection with everything else held fixed. The source and
  the target must each run once in the forward; otherwise the call raises and names the
  module. A target that runs before the source has a zero derivative. Parameters gain no
  `.grad`. Each of these three calls costs one forward pass.
- `sweep(inputs, interventions, *, record=(), logits="last", batch_size=8)` repeats one
  prompt (a string, or a dict with batch size 1) once per entry of `interventions`, each
  entry None or a list of interventions. Row i gets only entry i's interventions, and rows
  may set the same module to different values. It runs in chunks of `batch_size`, one pass
  per sequence, and returns one `{"logits", "records"}` per entry, shaped as `run_with`
  returns for that entry alone. The budget is checked before each chunk: a sweep over the
  cap fails before the chunk that would cross it, and earlier chunks have run and counted.
  Weight replacements are not supported; call `run_with` in a loop.

## Attention heads

```python
heads = lab.heads(layers=[0])
lab.zero(**heads[1]["o"][0])
weights = lab.head_weights(0, 1)  # q, k, v, o as float32 CPU tensors, plus biases
```

- `heads(layers=None)` lists one dict per (layer, head), ordered by layer then head.
  It has `layer`, `head`, `kv_head`, `group` (heads sharing the key/value head),
  `module`, and `q`, `k`, `v`, `o` lists of parts, plus `gate` for gated queries. A part
  is `{"name", "index"}`, so `lab.zero(**part)` and `lab.edit(**part, value=...)` work,
  and the indexes survive edit bundles. `q`, `k`, `v` and `gate` add a bias part where
  the model has one; `o` has none. `o_input` is `{"module", "at": "input", "features":
  [start, stop]}`, the slice of the output projection's input that this head writes.
  Heads in one group share `k` and `v` parts, so zeroing them hits the whole group.
- `head_weights(layer, head)` returns float32 CPU tensors in `nn.Linear` orientation:
  `q`, `k`, `v` as `[head_dim, hidden]`, `o` as `[hidden, head_dim]`, `gate` for gated
  queries, and `q_bias`, `k_bias`, `v_bias`, `gate_bias` as `[head_dim]` when present.
- Layouts: separate `q_proj`/`k_proj`/`v_proj` with `o_proj` or `out_proj` (Llama,
  Mistral, Qwen2/3, Gemma, OLMo, OPT); gated queries, where `q_proj` holds each head's
  query then gate (Qwen3.5, Qwen3-Next); and GPT-2's fused `c_attn`. Linear-attention
  blocks are skipped. A model with two attention stacks that reuse layer indexes (a
  vision tower and a text decoder, or an encoder and decoder) raises
  `NotImplementedError`: `heads` supports one stack of numbered layers.
- Raises `ValueError` for layers or heads the model lacks, or sizes that are not whole
  heads. Raises `NotImplementedError` for fused `qkv_proj` (Phi-3), `query_key_value`
  (GPT-NeoX, Falcon, Bloom), latent attention, a model with no attention block, and
  LoRA models: export the merged model and start a session on it.
- On a quantized model the weights are dequantized for the read; see
  [Quantized models](#quantized-models).

### Ablation scan

```python
rows = [{"prompt": "...", "options": [" a", " b"], "answer": 0}, ...]
scan = lab.ablation_scan(rows)  # every head and every MLP
scan["baseline"], scan["parts"][0]  # {"part": ("head", 3, 7), "value": ..., "delta": ...}
```

- `ablation_scan(rows, parts=None, *, metric="accuracy", groups=None,
  batch_size=None, max_passes=None)` zeroes each part in turn and scores `rows` with it
  removed. It is the sweep an agent otherwise writes as `lab.zero`, `lab.evaluate` and `lab.undo` in a
  loop, without a run directory per part.
- `rows` are in the format of `evaluate`'s likelihood mode and run through the same
  batched scoring, so left padding, position ids and prefix attachments behave the same.
  `batch_size` defaults as in `evaluate`: 8, or 1 for a model that takes no position ids.
  The baseline is the model as it stands, so active edits are part of it.
- `parts` is None (every head and every MLP) or a list of `("head", layer, head)`,
  `("mlp", layer)`, or a list of those, which is ablated together as one group.
  `groups="layers"` makes one group per layer from the given heads. `groups="pairs"`
  ablates every pair of the given entries.
- A head is ablated by zeroing its slice of the output projection's input through a
  temporary hook, which gives the same logits as `lab.zero` on its `o` part. An MLP is
  ablated by zeroing its output. Nothing is written into the weights and nothing stays
  attached, even after an error.
- `metric` is `"accuracy"` (a tie is incorrect), `"margin"` (the mean answer log
  probability minus the best other option's), or a callable that takes one
  `{label: log probability}` dict per row and returns a finite number, higher is better.
- Passes are rows x options x (1 + parts), one per sequence. The whole scan is checked
  against the pass budget and `max_passes` before it starts, and a scan that does not fit
  raises `BudgetExceeded` with the passes it needs. `groups="pairs"` counts its pairs
  first and refuses before building any.
- Returns `{"baseline", "parts": [{"part", "value", "delta"}, ...], "passes"}`, sorted
  from the largest drop. A single part is a tuple and a group is a list of tuples.
- A part that changes nothing may have a backup, so scan groups as well. A part that
  breaks the behavior may not be the whole mechanism. `find_circuit` and `path_patch`
  are the next step; see [Circuit analyses](#circuit-analyses).

## Circuit analyses

```python
margin = ("yes", "no")  # or any callable: metric(last_position_logits) -> scalar tensor
attribution = lab.logit_attribution(
    "hello world yes no hello", margin, by=("component", "position")
)
patched = lab.path_patch(clean, corrupt, [(0, 1)], [(1, 0, "v"), "logits"], metric=margin)
edges = lab.edge_attribution(clean, corrupt, metric=margin, steps=5)
circuit = lab.find_circuit(clean, corrupt, metric=margin, top=50)
```

These work on the layouts `heads` supports, where each layer adds an attention output and
an MLP output to the residual stream unchanged. A layer whose attention is not split into
heads (Qwen3.5's linear attention) is one node, `("attn", layer)`. A **metric** is a
callable over the batch's last-position logits (`[batch, vocab]`, float32) that returns a
scalar tensor, or an `(answer, counterfactual)` pair of strings: the mean logit difference
between the first token of each. The identities below (attributions sum to the logit, the
full circuit equals the clean metric) hold to the precision of the model's dtype: about
1e-4 in float32, and a few percent of the value in bf16 and fp16.

- `logit_attribution(prompt, target, *, by=("component",), positions=-1)` splits one logit
  (`target` an answer string) or a logit difference (`target` an `(answer,
  counterfactual)` pair; no callable here) at one token position into the embeddings, each
  head, each MLP, and the output-projection biases. The final norm's scale is frozen at
  this pass's value (LayerNorm also subtracts each node's mean), so the contributions plus
  the norm and head biases sum to the model's own value. The result carries both as
  `total` and `actual`. `by=("component", "position")` also splits each head over source
  tokens by attention weight times value (times the query gate for gated heads). Returns
  `{"total", "actual", "tokens", "embeddings", "heads": [{"layer", "head", "value",
  "positions"}], "mlps": [{"layer", "value"}], "mixers": [...], "biases":
  {"attention_output", "final_norm", "output_head"}}`. One forward pass. An MLP's value
  includes the bias of its own output layer.
- `path_patch(clean, corrupt, senders, receivers, *, metric, freeze_mlps=False)` follows
  Wang et al. (2022). `clean` and `corrupt` are paired lists of prompts with equal token
  counts per pair. Senders are `(layer, head)`, `("mlp", layer)`, `("attn", layer)`, and
  `"embed"`. Receivers are `(layer, head, "q"|"k"|"v")`, `("mlp", layer)`, `("attn",
  layer)`, and `"logits"` (the residual entering the final norm). Senders go to their
  corrupt output at every position, every other head is frozen at its clean output (every
  MLP too with `freeze_mlps=True`), and only the receivers' inputs are then replaced in a
  clean run. A `k` or `v` receiver is the key/value head that `head` reads. Returns
  `{"clean", "corrupt", "patched", "fraction"}`, with `fraction = (patched - clean) /
  (corrupt - clean)`, or None when those two are equal. With every head, MLP and `"embed"`
  as senders and `"logits"` as the receiver, `patched` equals `corrupt`; with no senders
  it equals `clean`. Four forward passes per prompt pair.
- `edge_attribution(clean, corrupt, *, metric, steps=5)` scores every edge from an
  upstream node (`"embed"`, a head, an MLP, an attention block) to a later reader: a
  head's `(layer, head, "q")`, a key/value head's `(layer, kv_head, "k")` or `"v"`,
  `("mlp", layer)`, `("attn", layer)`, or `"logits"`. The score is the sender's
  corrupt-minus-clean output dotted with the metric's gradient at the residual the reader
  sees, summed over positions and averaged over `steps` points that move the input
  embeddings from clean to corrupt. One backward pass per step gives every edge's
  gradient, through each reader's own norm and projection slice. With `steps=1` it is
  plain attribution patching. Returns a list of `{"sender", "receiver", "score"}` sorted
  by absolute score. `2 + steps` forward passes per prompt pair.
- `find_circuit(clean, corrupt, *, metric, edges=None, top=50, steps=5)` keeps the `top`
  edges by absolute score (or tests the `edges` you give, as `edge_attribution` returns
  them; `[]` is the empty circuit), drops edges that are not on a path from the embeddings
  to the logits, and confirms the circuit by patching: a clean run where every edge
  outside the circuit carries its corrupt value. Returns `{"clean", "corrupt", "circuit",
  "faithfulness", "edges", "total_edges"}`, with `faithfulness = (circuit - corrupt) /
  (clean - corrupt)`. Keeping every edge gives the clean metric and keeping none gives the
  corrupt one. `3 + steps` forward passes per prompt pair, or 3 with `edges`.
- Prompt pairs with equal lengths run together as one batch; the results average over
  pairs. A call that would pass the pass limit is rejected before anything runs. Hooks are
  removed on exit, and train/eval modes are restored.
- Raises `ValueError` for unequal token counts, unknown senders or receivers, and a bad
  metric. Raises `NotImplementedError`, naming the module, for norms on a sublayer's
  output (Gemma 2, OLMo 2), parallel attention and MLP blocks, LoRA models, bitsandbytes
  models, and any model where the recomputation does not match the model's own residual
  stream, attention output, or logits (a residual multiplier, a logit softcap). The call
  refuses rather than return approximate numbers.

## Sparse autoencoders

Needs the `saes` extra (SAELens), which the plugin and remote workers include. `lab.load_sae` reads a public sparse autoencoder or
transcoder; the returned `LabSAE` reads, patches, and steers its features through
`lab.record` and `lab.patch`, so pass counting, positions, and cleanup on error behave
as they do there.

```python
sae = lab.load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
result = sae.features(["The capital of France is"], positions="all")
print(result["fvu"], result["l0"])  # check these before reading features
with sae.patch({1234: 0.0}, positions="all"):
    ablated = lab.score("The capital of France is", [" Paris", " Rome"])
```

- `load_sae(release=None, sae_id=None, *, path=None, device=None,
  allow_model_mismatch=False, model_kwargs=None, hook_name=None)` loads by SAELens release and id, or from
  a local directory in SAELens's format, onto `device` (default the session's). It
  refuses, before any forward pass: a hook name that `hook_site` does not map; a site
  whose width is not `d_in` (a transcoder's output not `d_out`); a model mismatch where
  the SAE's model name is known (a small table covers `gpt2`, `gpt2-small`, and
  `gemma-2-*`; a name it does not know loads with a warning and `model_check` saying it
  was not checked; `allow_model_mismatch=True` accepts a fine-tune); an architecture
  without an affine decoder (standard, gated, JumpReLU, top-k, and the transcoder
  variants are supported); `normalize_activations` other than `none`, `layer_norm`, and
  `constant_norm_rescale`; and an adapter (PEFT) session. Quantized sessions work, since
  only activations are read. Raises `ImportError` naming the extra when SAELens is
  missing.
- `hook_name` replaces the input hook the SAE records, for a release whose recorded hook is
  not where it reads (Gemma Scope 2 transcoders record `blocks.L.hook_mlp_in` but read the
  MLP's own input: pass `blocks.L.mlp.hook_in`). The override gets the same refusals as any
  hook name: unmapped, a layer the model lacks, or a width that is not `d_in`. A
  transcoder's output site still comes from its recorded `hook_name_out`. `LabSAE.hook_name`
  is the name used and `recorded_hook_name` the one in the config; the repr shows both when
  they differ.
- `hook_site(lab, hook_name, model_kwargs=None)` maps a TransformerLens hook name to a
  `Site` with `module`, `at`, `width`, and the transform that makes a Hugging Face
  activation match the dictionary's training data. Mapped: `blocks.L.` followed by
  `hook_resid_pre` (block input), `hook_resid_post` (block output), `hook_resid_mid`,
  `hook_mlp_in`, `hook_attn_out`, `hook_mlp_out`, `attn.hook_z` (the output projection's
  input), `ln2.hook_normalized`, and `mlp.hook_in`. Gemma 2's attention and MLP outputs
  are read after their post-norms. Anything else raises `ValueError`.
- SAELens loads a model without TransformerLens's weight processing, then applies the
  SAE's recorded `model_from_pretrained_kwargs`. `hook_site` applies the same: with
  `center_writing_weights` (`gpt2-small-res-jb` records it) the residual stream,
  attention output, and MLP output of a LayerNorm model lose their mean over the hidden
  dimension; with `fold_value_biases` (and `fold_ln`) `hook_z` loses the value bias and
  the norm's bias. Where a release records nothing, pass `model_kwargs` to say what it
  was trained under, and check `fvu`. Normalized sites and `hook_resid_mid` can be read
  but not edited: `patch` and `steer` raise `ValueError`.
- `LabSAE` has `module`, `at`, `hook_name`, `d_in`, `d_sae`, `kind` (`"sae"` or
  `"transcoder"`), `recorded_hook_name`, `output_module` and `output_at` (transcoders), `model_check`,
  `neuronpedia_id`, `prepend_bos`, and the SAELens object as `sae`.
- `encode(activations)` takes an activation as the lab records it and applies the site
  transform; `decode(features)` returns the reconstruction in the dictionary's
  coordinates. SAELens subtracts and adds `b_dec` inside these, so the lab does not.
  `transform(x)` is the site transform alone. `decoder_rows(indices)` returns the vectors
  a unit rise in each feature adds. They are not unit norm, except for a top-k SAE that
  rescales by decoder norm.
- `features(inputs, positions=None)` runs the model once per sequence and returns
  `features` (`[rows, d_sae]`, CPU float32), `sequence`, `position`, `rows`, and `mse`,
  `fvu` (fraction of variance unexplained across the rows), and `l0`. `inputs` is text, a
  list of texts, or a dict with `input_ids`. Text gets the BOS token when the SAE was
  trained with one. Positions are as for `record`; padding in `attention_mask` is left
  out. A transcoder is scored on the output it predicts.
- `patch(edits, positions=None)` is a context manager. `edits` maps a feature index to a
  value or to a callable of its current value. The site changes by
  `(f_new - f_old)[changed] @ W_dec[changed]`, so the reconstruction error is kept, and
  an edit that changes nothing leaves the activation unchanged to the bit. A transcoder
  changes its output from features read at its input.
- `steer(features, positions=None)` adds `scale * decoder row` for each `{index:
  scale}`. For `layer_norm` and `constant_norm_rescale` SAEs the push is scaled by each
  token's own normalization.
- `labels(indices, timeout=5.0)` returns Neuronpedia's explanation for each index, or
  `None` when there is no id, no label, or the network does not answer in time. Found
  labels are cached under `artifacts/saes`, and a feature with no label is not asked about
  again by the same object. A label is a lead, not evidence.

## Attribution graphs

Needs the `saes` extra. `lab.attribution_graph` builds a feature-level graph of how the model
produced its answer on one prompt, from one transcoder per MLP layer (see
[Sparse autoencoders](#sparse-autoencoders)). The graph is JSON an agent can read, and
`measure` tests a node in the real model through `LabSAE.patch`.

```python
transcoders = [lab.load_sae(release, sae_id) for sae_id in ids]  # one per layer
graph = lab.attribution_graph("The capital of France is", transcoders)
print(graph.metadata["replacement_score"], len(graph.nodes), len(graph.edges))
graph.to_json("graph.json")
top = next(n["id"] for n in graph.nodes if n["kind"] == "logit")
print(graph.top_paths(top, 3))
feature = "feat.12.4.317"  # a node id from graph.nodes
result = graph.measure([feature])  # set the feature to 0 in the real model, on graph.inputs
print(result["effect"][top])  # the change in the top logit node's value
```

- `attribution_graph(prompt, transcoders, *, max_logits=10, logit_probability=0.95,
  max_features=10_000, node_threshold=0.8, edge_threshold=0.98, batch_size=16,
  max_memory_gb=8, labels=False)` returns an `AttributionGraph`. `transcoders` is a list of transcoders or a dict from layer to
  transcoder. It follows Ameisen et al., 2025, "Circuit Tracing".
- The graph describes a replacement model: the session's model with each MLP's output
  written as its transcoder's features times their decoder rows plus an error term,
  attention patterns held at their real values, and each norm's denominator held fixed.
  Every forward value is the real model's; only the gradients change. A skip transcoder's
  skip connection is a linear path from the MLP's input, so it carries edges like any other;
  the error term is what remains after the features and the skip. Only the `max_features`
  features with the largest activations are nodes (`None` keeps all); the rest are absorbed
  into the error terms.
- Costs 3 forward passes, whatever the prompt or the feature count: one to record, one to run
  the replacement model, and one with every leaf at zero to find each target's constant part.
  Backward passes do not count. The budget is checked for all 3 first.
- The adjacency work holds two dense float32 matrices of targets by sources, so memory grows
  with the square of the node count. After the recording pass the peak is estimated, and a
  graph over `max_memory_gb` (default 8: room on a 16 GB machine for the model and buffers,
  about 30,000 nodes) is refused with a message naming `max_features`, which defaults to
  10,000, the cap circuit-tracer uses. That refusal costs 1 pass.
- `labels=True` asks Neuronpedia for each kept feature's label, one request per feature
  without a cached label; it is off by default and nothing uses the network without it.
- The call checks and raises `ValueError` rather than return a graph that fails: the
  replacement model's logits equal the model's, each feature's encoder pre-activation equals
  its activation, and each target's incoming edges plus its `bias` equal its `activation`,
  each to 1e-4 of the size of the terms added. The numbers are in `metadata["checks"]`.
- Refuses: a model that is not float32; a quantized or adapter session; an architecture
  `heads` does not support, or one with an MLP layer that has no softmax attention (a hybrid
  with linear attention); transcoders that are not one per MLP layer, are plain SAEs, belong
  to another session, or do not read the MLP's input and write its output; a transcoder with
  run-time input normalization; a norm other than LayerNorm or RMSNorm; an encoder that is not
  ReLU, JumpReLU or top-k; and settings out of range. Hooks, the attention setting, train and
  eval modes, and RNG state are restored on exit, even after an error.
- Nodes are dicts with `id`, `kind`, `layer`, `position`, `token`, `token_id`, `feature`,
  `activation`, and `influence`, plus `label` when the transcoder's `labels` give one, and
  `bias` on feature and logit nodes. The ids are `emb.P`, `err.L.P`, `feat.L.P.F` and
  `logit.TOKEN_ID`. `activation` is the feature's value, the L2 norm of an embedding or error
  vector, or a logit minus the mean logit. `influence` is the node's share of the unpruned
  graph's total influence on the logit nodes. Logit nodes also have `probability`.
- Edges are `{source, target, weight}`. A target is a feature's pre-activation or a logit
  minus the mean logit. The weight is the source's value times the gradient of the target
  with respect to it. Pruning follows the paper and circuit-tracer's `prune_graph`: nodes are kept
  up to `node_threshold` of the total influence, edges are rescored on the pruned graph
  (renormalized, influence recomputed) and kept up to `edge_threshold`, and features and
  errors with no outgoing edge, and features with no incoming one, are dropped until none
  remain. Tokens and logit nodes are never pruned. A threshold of 1.0 for both keeps every
  node and every nonzero edge. It differs from circuit-tracer in three ways: each logit also
  keeps its strongest parent and that edge, so no logit is left without a cause; `influence`
  is not recomputed after pruning; and the edge cutoff is found by bisection, so ties at the
  cutoff are all kept. Pruned edges are missing from the sum above, so check the sum on a
  graph built with both thresholds at 1.0.
- `metadata` holds `prompt`, `tokens`, `input_ids` (including a BOS token the transcoders
  asked for), `model_id`, `transcoders`, `settings`, `replacement_score` (the share of
  backward paths from the logits to the tokens that never cross an error node),
  `forward_passes`, and `checks`.
- `AttributionGraph` has `nodes`, `edges`, `metadata`, `to_dict()`, `to_json(path)`,
  `from_dict(data)`, `from_json(path)`, `node(id)`, `top_paths(node, k=5)`, and
  `inputs`, `measure(node_ids, value=0.0)`, and `intervene(node_ids, value=0.0)`. A graph
  read from a file cannot `measure` or `intervene`.
- `inputs` are the exact model inputs the graph ran on (`input_ids` and, if the tokenizer
  made one, `attention_mask`), including a BOS token the transcoders asked for. Graph
  positions index these tokens. `lab.score` and `lab.evaluate` tokenize without that BOS
  token, so their positions are off by one and must not be paired with `intervene`.
- `measure(node_ids, value=0.0)` runs the real model on `graph.inputs` with those feature
  nodes set to `value` through `LabSAE.patch`, and returns `baseline`, `intervened`, and
  `effect`: each logit node's id mapped to its logit minus the mean logit, before and after,
  and the difference. It costs 2 forward passes, checked against the budget first.
- `top_paths` returns the `k` strongest paths into a node, each with `nodes` (source first),
  `weights`, and `score`: the product of each edge's share of its target's incoming absolute
  weight.
- `intervene` is a context manager that sets feature nodes to `value` at their positions
  through `LabSAE.patch`, keeping the reconstruction error. Run the model on `graph.inputs`
  inside it, for example `lab.model(**graph.inputs)`. A forward pass in the context on any
  other tokens, which includes `lab.score` and `lab.evaluate`, raises `ValueError` naming
  `graph.measure`. It raises `ValueError`, before adding a hook, for an unknown node and for
  a node that is not a feature.
- Edges describe the replacement model. They cannot show an effect that moves through
  attention patterns, and a large error node marks what the transcoders do not explain.

## Proving ground

```python
from oumi_weights_lab import proving

info = proving.build_task("copy-head", "dev", 0, "tasks", "keys")
result = proving.grade_claim(info["task_dir"], proving.load_key(info["key_path"]), claim, "work")
report = proving.compare(control_records, skill_records, split="heldout")  # delta = skill - control
```

Tasks with a planted mechanism and a hidden answer key, a scorer, and a paired comparison. They measure whether an agent that has the skill understands models better than one that does not. See [Proving ground](proving-ground.md) for the families, splits, images and how to run them. Importing `proving` does not import torch.

- `build_task(family, split, index, out_dir, keys_dir, *, secret_file=None, model_id="openai-community/gpt2")` writes `out_dir/<task id>/` (`task.json`, `model/`, `rows.jsonl`) and `keys_dir/<task id>.json`, and returns `{"task", "task_dir", "key_path"}`. Task ids are opaque, `{split}-{10 hex digits}`. A held-out build needs a secret file of at least 32 bytes, and a dev or calibration build refuses one. The calibration families and the families planted inside GPT-2 (`gpt2-copy-redundant`, `gpt2-copy-selfrepair`) load `model_id`; the GPT-2 planted families refuse a model that is not GPT-2. Raises `ValueError` for a keys directory inside the task directory, an existing task directory, or a family in the wrong split, and `RuntimeError` when no valid model comes out of 20 draws.
- `grade_claim(task_dir, key, claim, work_dir)` returns the score dict: component precision, recall and F1, `direction_recall`, `direction_precision` and `direction_score` where the key has directions, `identification`, `ablation_reproduced`, `faithful`, `score` and `solved`. `score` is `identification` halved once if the ablation did not reproduce and halved again if the claim is not faithful. It reruns the claim's ablation on 400 rows of its own in a fresh lab. An invalid claim scores 0 with `problems`; `validate_claim(claim, task_dir)` lists them. `CLAIM_FORMAT` describes the claim to give an agent.
- `compare(left, right, split=None)` pairs two lists of result records by task, averages repeats, and returns both means, the difference (right minus left), its 95% interval, the permutation p, the solve rates and each side's setup. Put the baseline on the left. It raises `ValueError` when no task pairs or a side mixes image ids, agent commands, agent labels or timeouts. `beats(report)` is true when the interval is above zero, meaning the right side won.
- The command line wraps these as `oumi-weights-lab proving-ground build|run|grade|compare`. `run` needs Docker and refuses a keys directory inside the tasks directory.
- `record_method_result(methods_dir, name, version, result, setup=None)` appends a graded result (`task`, `split`, the `score` dict, the time and the setup if one is given) to a saved method version's `results.jsonl`, writing NaN as null. It raises `ValueError` for a method or version that is not saved. `proving-ground grade ... --method NAME@VERSION [--methods-dir DIR]` prints the grade, then records it, and refuses an unknown method or version before it grades. See [Saved methods](#saved-methods).

## Saved methods

```python
def rank_heads(lab, prompt, answer):
    return {"ranking": []}


meta = lab.save_method(rank_heads, note="zeroes one head at a time")
result = lab.run_method("rank_heads", prompt, answer)
# result["saved_method"] == {"name": "rank_heads", "version": meta["version"]}
```

A method an agent writes outlives the session. A method is a plain function whose first parameter is `lab`. It is saved with its source, its proving-ground results and the setup it ran on, and a later session loads it by name.

- `lab.save_method(function, *, name=None, note="")` saves the function and returns its metadata: `name`, `version`, `function`, `signature`, `docstring`, `note`, `created_at`, `lab_version`, `setup` (the block that run summaries record, from the first save) and `setups` (every setup the same version was saved in; saving identical source again in a different setup adds to this list). `name` defaults to the function's name and holds lowercase letters, digits, `_` and `-`. The version is the first 12 hex digits of the SHA-256 of the saved source, so saving the same source again returns the existing entry. The source must run on its own: a module global becomes an import line (`import torch`, `import numpy as np`), and anything else raises `ValueError` naming it: a non-module global (also one used by a default, decorator or nested function), a closure variable, a first parameter not named `lab`, a lambda, or a source `inspect.getsource` cannot read. Pass values as arguments or import inside the function. A function defined in a snippet can be saved by a later snippet.
- Files go to `OUMI_WEIGHTS_LAB_METHODS` if set, else `methods/` in the directory where the session started: `methods/<name>/<version>/method.py`, `meta.json` and `results.jsonl`.
- `lab.methods()` lists each saved method with its versions (newest first), their notes and a summary of their proving-ground results: `results`, `tasks`, `mean_score`, `solve_rate` and `splits`. Each task counts once, averaged over its repeats. A grade whose score was NaN is stored as null and left out of the summary.
- `lab.record_method_result(name, version, result)` records a graded result (`task`, `split` and the `score` dict) in that version's `results.jsonl` together with this session's setup. `proving-ground grade --method` records without a setup: the task id fixes the model. Records written either way are compared only when their setups match, or when one side has none and the task is the same.
- `lab.load_method(name, version=None)` returns a `Method` with `name`, `version`, `source` and `meta`: the newest version by creation time unless one is named. It runs the saved Python with your permissions, as snippets do, and refuses a file edited since it was saved. Calling the method runs `function(lab, *args, **kwargs)`. A dict result comes back as a copy with `"saved_method": {"name", "version"}` added, and the call raises `ValueError` if the dict already has a `saved_method` key. Other results come back unchanged. `lab.run_method(name, *args, version=None, **kwargs)` loads and calls in one step.
- `oumi-weights-lab methods list [--methods-dir DIR]` prints the same list as JSON.
- `oumi-weights-lab methods promote NAME [--version V] [--against NAME[@VERSION]] --output FILE.md [--methods-dir DIR] [--force] [--overwrite]` writes a recipe draft for a maintainer: the docstring, the source, every setup the method was saved or graded in, a table of results per task and, with `--against`, the paired comparison on the tasks both were graded on. The comparison uses only results from matching setups, and the draft says what it left out. It refuses a method without scored results, and with `--against` a method that does not beat the other by the paired interval, unless `--force`. It refuses an output path under a `skills` directory, and an existing file unless `--overwrite`. Promotion into the skill is a reviewed change.

## Attachments (experimental)

`lab.attach(component, mode="prefix")` transforms prompts for `generate`, `score`,
and `evaluate`; `mode="logits"` adds a delta to every model output and rejects calls
that pass `labels`.
