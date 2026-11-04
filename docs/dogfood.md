# Public-model edit, inspect, undo, and backup

Run a complete baseline → edit → undo → export → stop → backup experiment on CPU:

```sh
uv run python -m examples.dogfood --output-dir sessions/tiny-demo --summary
```

Use a new output directory each time. The first run downloads the public
[`sshleifer/tiny-gpt2` checkpoint at `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`](https://huggingface.co/sshleifer/tiny-gpt2/tree/5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be).
`--summary` prints the measured result; omit it for the full JSON report. Both forms
save the full evidence and CLI transcript.

This terminal transcript came from a local CPU run on 2026-11-03, using the cached
public checkpoint. The final two output lines containing checkpoint and backup
paths are omitted here:

```text
$ .venv/bin/python -m examples.dogfood --output-dir sessions/presentation-demo-20261103-1845 --summary
Model: sshleifer/tiny-gpt2 (cpu, float32)
Baseline: 2/3 dev answers correct
After edit: 2/3 dev answers correct
Edit: zero transformer.h.0.mlp.c_fc.weight
Score maps changed: 3/3
Predictions changed: 0
Undo: weights exact=True, scores exact=True
Export: all reloaded weights exact=True
Forward passes: 21/32
Worker: stopped
```

The edit changed likelihoods without changing any predictions. This demonstrates
measurement, persistence, and exact restoration on three dev examples; it does not
show a model-quality improvement or measure benchmark accuracy.

## What the demo does

`examples/dogfood.py` sends three saved Python snippets through the detached CLI
worker. The question is whether zeroing the first MLP input projection changes
continuation scores, and whether undo restores the original weights and scores.

The script starts one CPU float32 worker with a 32-pass session budget. It keeps
the baseline data and model in that worker across the three snippets, captures
the first MLP's last-token activation, evaluates three hand-authored dev tasks,
zeros `transformer.h.0.mlp.c_fc.weight`, evaluates again, undoes the edit, and
asserts exact weight and score restoration. It exports and reloads the restored
checkpoint, checks every saved model tensor, stops the worker, and archives the
stopped session.

## Measured details

Measured on 2026-11-03, macOS arm64, PyTorch 2.15.0, Transformers 5.22.1:

| Prompt | Completion | Baseline summed log probability | After zeroing |
| --- | --- | ---: | ---: |
| `The dog` | ` was` | -10.825937271118164 | -10.825936317443848 |
| `The dog` | ` is` | -10.833880424499512 | -10.833880424499512 |
| `Hello` | ` world` | -10.841591835021973 | -10.841602325439453 |
| `Hello` | ` there` | -10.795316696166992 | -10.795298576354980 |
| `The capital of France is` | ` Paris` | -10.829025268554688 | -10.829298019409180 |
| `The capital of France is` | ` London` | -10.848685264587402 | -10.850220680236816 |

All three per-item score maps changed. Predictions stayed `was`, `there`, and
`Paris`, so `lab.diff(baseline, edited)` returned an empty list: that helper reports
changed predictions or correctness. The full per-option likelihoods remain in
each run's item files. The three illustrative answers were `was`, `world`, and
`Paris`; the 2/3 result in all runs describes this small dev trace.

| Evidence | Observed result |
| --- | --- |
| Model revisions | Baseline 0, edited 1, restored 2 |
| Active edit | `edit-000001` during edited evaluation; none after undo |
| Captured MLP activation shape | `[1, 1, 2]` |
| Baseline MLP activation norm | 0.0011785099050030112 |
| Edited MLP activation norm | 0 |
| Restored MLP activation norm | Exact baseline value |
| Restored scores and projection weights | Exact equality with baseline |
| Reloaded exported checkpoint | Every model tensor exactly matches the restored model |
| Evaluation forwards | Six per run, 18 total |
| Inspection forwards | One per model revision, three total |
| Total counted forwards | 21 of the 32-pass budget |
| Final worker state | `stopped` |

All runs share dataset hash
`f1f3af7c2808559de6514f90f2fe7e9b601d9f4fe0dbb0b0f3930c760a0cbc0f`.
They are dev runs; this example does not touch held-out data or measure benchmark
quality. The tiny checkpoint is useful for exercising the workflow cheaply.

## Saved evidence

The measured run is in the ignored local directory
`sessions/presentation-demo-20261103-1845/`. The output directory contains
`transcript.json`, all submitted Python snippets,
`dogfood.json`, the session's per-run summaries and item JSON files, the restored
HF checkpoint at `session/artifacts/restored-checkpoint`, and `session.tar.gz`.
The checkpoint includes the tokenizer and `oumi_weights_lab_manifest.json` with the
undo history. The backup includes the source snippets, outputs, results, and
artifacts needed to inspect the experiment without keeping a worker alive.

The script stops the worker in a `finally` block, including after a failed snippet.
On success it verifies the backup contains the checkpoint, evidence, and completed
snippet sources. The default `run(output_dir)` Python API still returns the full
evidence dictionary; `--summary` changes terminal formatting only.
