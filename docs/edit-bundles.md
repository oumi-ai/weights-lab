# Save edits and resume an experiment

An edit bundle saves API edits and their exact undo snapshots without exporting
the entire model. It also records edit history, revision, forward passes, the
session pass limit, and dev/held-out reservations at the time of saving.

In a running worker:

```python
lab.zero("transformer.h.0.mlp.c_fc.weight", note="first MLP ablation")
bundle = lab.save_edits("candidate-v1")
print(bundle)  # absolute path under this session's artifacts/
```

Stop the worker and keep the bundle, or download it before tearing down a remote
machine. Start a **new session directory**, using the same original model,
adapter, precision, and tokenizer:

```sh
weights-lab start --session sessions/resumed --model /path/to/original-checkpoint \
  --device cpu --dtype float32 \
  --resume /path/to/old-session/artifacts/candidate-v1
weights-lab send --session sessions/resumed --code 'print(lab.edits()); lab.undo()' --follow
```

Alternatively, call `lab.load_edits("/absolute/path/to/bundle")` in a fresh
model workspace before any forwards, edits, or evaluation. Relative paths for
both methods resolve under `artifacts/`. Existing destinations are refused.
Loading is permitted only once into a fresh experiment; it cannot rewind a live
experiment or reset its budget.

The default resume limit inherits the saved cap. An explicit `--pass-limit` may
tighten it, but cannot raise a finite inherited cap or fall below passes already
used. Held-out attempts remain consumed, including failed attempts. Dev data
stays dev data, and the evaluation session identity continues across the resume.

Bundles contain `manifest.json` and `edits.safetensors`. They use replacement
snapshots rather than floating point subtraction/addition, preserving BF16 bits,
overlapping slices, and normal tied parameters. The history includes undone edits
so later edit IDs remain unique. Safetensors carries no executable callbacks;
loading never replays snippets or unpickles Python objects.

Before applying edits, loading checks manifest and payload checksums and a SHA-256
fingerprint of every original model tensor and buffer, tensor aliases, model
configuration, tokenizer, and PEFT configuration/scaling/selection. A matching
model ID alone is insufficient. Wrong checkpoints or dtypes fail before weights
change. For manually selected adapters, select the same adapters before loading.
Saving hashes all model tensors and uses CPU copies of edited parameters, so
large models can take time and host memory even though no forward calls run.

The bundle represents an **explicit save point**. It does not automatically
capture later work, arbitrary Python variables, unsaved direct tensor writes,
or the live namespace. Keep the source session's logs and evaluation artifacts
for the full trace; the bundle carries reservations, not copies of prior run
reports. Remove active activation patches and attachments before saving. Save
outside evaluation and temporary pass-limit contexts.

On SkyPilot, `--resume` names a path already available on the remote worker;
it does not upload a local bundle. Source code is still supplied separately
through `--source`. Use [remote backup and transfer](remote.md) to preserve the
bundle and its source session before teardown. Full portable model export
remains `lab.export(...)`.
