# Python API

Snippets share one `Lab`, its model and tokenizer, and a writable `artifacts` path.
Import other dependencies within snippets. Models are eager and in eval mode;
use `torch.no_grad()` when gradients are unnecessary. The model's native tokenization
and chat template remain available through `lab.tokenizer`.

## Weights

```python
lab.parts("*mlp*")  # names, shapes, dtypes, devices, sizes
lab.measure("transformer.h.0.mlp.c_proj.weight")
```

`compare(other, pattern="*")` accepts another torch model or a dictionary of named
tensors. It reports current-minus-reference statistics. Statistics exclude nonfinite
values and report their count separately; large tensors are measured in chunks.

```python
name = "transformer.h.0.mlp.c_proj.weight"  # GPT2; inspect parts() for other models
edit_id = lab.zero(name)
print(lab.edits())
lab.undo(edit_id)
edit_id = lab.edit(name, lambda weight: weight * 0.5, index=(slice(None), 0))
lab.undo()  # always the most recent active edit
```

`edit(name, value, index=None, note="")` accepts a tensor or a callable. Callables
receive an isolated copy, and can return a new tensor or mutate the copy in place.
The replacement must have exactly the selected shape. Failed edits leave the
parameter unchanged. Snapshots keep the original dtype on CPU; undo restores exact
values, including tied parameters. Undo is strictly LIFO, even for disjoint slices.

`swap(name, source, source_name=None, index=None, note="")` copies a parameter from
a tensor, model, or dictionary. Source tensors must have the full target shape;
when an index is provided the same index is selected in both. Direct writes through
`model` are possible but are outside the undo history.

`export(path)` writes a standard Hugging Face checkpoint, tokenizer, and edit
manifest. Relative paths are under `artifacts/`. Existing nonempty output directories
are rejected. PEFT exports copy and merge the adapter into a full model, preserving
base-weight edits and leaving the live model untouched; allow memory for that copy.
Remove experimental attachments and activation patches before export: they cannot be serialized as ordinary
checkpoint weights. `backup` archives session records, including exported checkpoints
under `artifacts/`; it does not export the live model itself.

`save_edits(path)` saves a portable bundle of exact edited slices, undo history,
pass accounting, and dataset reservations. `load_edits(path)` resumes that save
point into a fresh identical model; CLI `start --resume PATH` performs the same
step during loading. The inherited pass limit cannot be raised. It does not
restore arbitrary Python variables. See [edit bundles](edit-bundles.md).

## Scoring

JSONL rows have `prompt`, `options`, and `answer`. An optional `id` identifies a row.
Options can be a list (labels A, B, ...) or a dictionary from labels to completions.
An answer can be a zero-based option index, label, or exact completion text.

```json
{"id":"capital","prompt":"France's capital is","options":{"paris":" Paris","rome":" Rome"},"answer":"paris"}
```

```python
baseline = lab.evaluate("/path/dev.jsonl", name="baseline")
lab.zero(name)
edited = lab.evaluate("/path/dev.jsonl", name="zeroed")
print(lab.diff(baseline, edited))
lab.undo()
```

`score(prompt, options, max_passes=None)` returns each option's summed conditional
log probability. Prompt and completion are encoded **separately**, without special
tokens, then concatenated. Only continuation tokens are scored. Include leading
spaces in completions when required by the tokenizer. This explicit boundary policy
can differ from evaluation harnesses that retokenize concatenated text or normalize
by answer length. One option costs one forward call; unequal-length options use
summed likelihood, not a length-normalized score. An empty prompt uses BOS or EOS.

`evaluate(data, name=None, held_out=False, max_passes=None, mode="likelihood", ...)`
accepts JSONL or a list of rows. It returns a summary dictionary and saves
`artifacts/evaluations/NAME/{summary.json,dataset.json,items.jsonl,items/000000.json,...}`.
Items retain answers, predictions, scores, and correctness. Summaries record dataset
hash, session identity, checkpoint revision, edit history, attachments, patches, and passes.
Failed runs preserve partial results and raise the original error; their accuracy
is null. Run names cannot overwrite existing runs. `diff` compares successful runs
of the same data, mode, and session, returning improved, regressed, or changed items.
Use `diff(before, after, include_scores=True)` to include changes to scores even
when the prediction is unchanged. `compare_runs` reports grouped metrics, largest
score/probability movers, exact paired McNemar tests, and deterministic bootstrap
intervals. Named `evaluate_suite` / `compare_suites` runs support regression
checks, and `create_screen` / `verify_screen` pins repeatable dev subsets. See
[evaluation workflows and probability limitations](evaluation.md).

For yes/no rows, use `mode="yes_no"` and a boolean or yes/no answer. For free answers:

```python
run = lab.evaluate(
    [{"prompt": "What is two plus two?", "answer": "4"}],
    mode="sampled",
    name="generation",
    max_passes=32,
    checker=lambda generated, expected: generated.strip() == expected,
    generation_config={"max_new_tokens": 32, "do_sample": False},
)
```

Sampled evaluation defaults to greedy decoding. Generation uses the supplied seed
and restores the caller's CPU, CUDA, and Apple MPS random states, including after
failure. Stochastic comparisons need matching decoding settings as well as a
matching seed. A Python checker can grade a sampled answer; the default is stripped
exact equality. Scoring preserves each module's train/eval mode.

Keep a held-out set separate from dev data. It requires `held_out=True` and an
explicit pass cap, and can run once per session. The persistent registry identifies
datasets by semantic content, not filenames or run names. Dev data cannot become
held-out data; consumed held-out data cannot be reused as dev data. A failed held-out
attempt also consumes its reservation. This supports disciplined experiments; it
does not sandbox an agent with unrestricted model and filesystem access.

`lab.budget.limit` caps the whole session, including raw `model(...)` calls and each
generation step. `with lab.limit_passes(10): ...` adds a temporary cap. Limits reject
a call before execution. Accepted calls that fail still count. `BudgetExceeded`
signals exhaustion. CLI `start --pass-limit N` sets the session cap.

## Activations, gradients, and probes

```python
inputs = tokenizer("hello world", return_tensors="pt").to(lab.device)
with lab.record(["transformer.h.0"], positions=[-1]) as captures:
    with torch.no_grad():
        model(**inputs)
hidden = captures["transformer.h.0"][0]  # CPU [batch, chosen positions, hidden]

with lab.patch("transformer.h.0", hidden, positions=[-1]):
    with torch.no_grad():
        changed = model(**inputs).logits

gradients = lab.gradient("hello world", parameters=[name], max_tokens=64)
```

`record` stores one detached CPU tensor per call and removes hooks on context exit.
The default position is the last token. `patch` accepts a tensor or a callable over
selected activations, preserving other tuple outputs. Token positions are explicit:
a fixed position can be absent during cached generation, so use last-token patches
or a callable suited to that run. Inspection rejects compiled models.

`gradient(prompt_or_inputs, target=None, parameters=None, max_tokens=512)` returns
per-row mean causal-loss gradients, shaped `[rows, *parameter_shape]`, on CPU.
Provide `target` to score only answer tokens, or pass tokenized inputs with labels
and `-100` for masked tokens. Existing `.grad` values and parameter flags survive.
Request selected parameters and short rows: gradients retain activations even in
eval mode, and returning all gradients can be expensive.

`lora_diagnostics(pattern="*", adapter=None, top_k=8)` measures effective scaled
LoRA updates through small QR/SVD calculations, with no forwards or dense `B @ A`
allocation. It reports singular values, update/base norms and numerical/stable
rank for each selected adapter separately. See [supported variants and scaling
semantics](lora-diagnostics.md).

`probe(features, labels, groups, train_groups=None, seed=0, test_fraction=.25, ridge=.001)`
fits a linear ridge classifier. Groups are source document IDs, and a document can
appear on only one side of the split. Training must contain every class. Results
include held-out accuracy, train accuracy, coefficients, intercept, classes, and the
document groups used on each side. Split original documents before collecting
related rows; row-level random splits would leak information.

## Experimental attachments

```python
from oumi_weights_lab.circuits import DateFacts

with lab.attach(DateFacts(), mode="prefix", name="calendar-facts"):
    text = lab.generate("What weekday was 2026-11-03?", max_new_tokens=24)
```

Prefix transforms apply to `lab.generate`, `score`, and `evaluate`. Raw tokenizer or
model calls do not apply them. Logit attachments receive `(model, args, kwargs,
output)` and return a finite delta broadcastable to `output.logits`:

```python
with lab.attach(lambda model, args, kwargs, output: bias, mode="logits"):
    run = lab.evaluate(dev, name="with-bias")
```

Handles support `remove()` and context cleanup. `attachments()` lists active
components, and attachment changes advance the experiment revision. The fixed
tensor building blocks in `oumi_weights_lab.circuits` provide token matching, nearest
matches, lookup tables, digit addition with carries, integer comparison, and a
calendar circuit exact over valid dates from 2000 through 2099. They use fixed
weights, indexing, and integer operations; they are not trained neural arithmetic.

The separate [fixed-weight date network](date-network.md) adds calendar/business
offsets, weekend rolling, date comparisons, and ISO character logits. Its frozen
tables and linear layers run offline; the character vocabulary is independent
of any language model's tokenizer.

## Remote sessions

`oumi_weights_lab.launchers.SkyPilotLauncher(cluster, python="python3")` manages remote
sessions through upstream SkyPilot. `start(model, session, source=..., infra=None,
accelerators="A100:1", idle_minutes=15, **config)` accepts the same model settings
as the local CLI. Set `infra="slurm/mycluster/mypartition"` to select Slurm, or
`accelerators=None` for a CPU resource request. Use `run(arguments, input=code)`,
`stop(session, timeout=10)`, `download(source, destination)`, and `teardown()` for
subsequent operations. See [remote workers](remote.md) for setup and lifecycle.
