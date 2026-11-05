# Compare checkpoints

Use parameter deltas to locate differences before spending forwards. Loading a second full model consumes additional memory; keep it on CPU and restrict the pattern. Check that model architecture, vocabulary, and named shapes match before interpreting deltas.

```python
from transformers import AutoModelForCausalLM

other = AutoModelForCausalLM.from_pretrained(
    "path/to/reference-checkpoint", dtype="float32", trust_remote_code=False
).eval()
print(lab.parts("*down_proj.weight"))
print(lab.measure("model.layers.0.mlp.down_proj.weight"))
deltas = lab.compare(other, pattern="*down_proj.weight")
import json

(artifacts / "parameter-deltas.json").write_text(json.dumps(deltas, indent=2))
del other
```

`compare` also accepts a parameter-name-to-tensor mapping. Deltas are current minus reference; each result contains finite-value mean, population standard deviation, min, max, L2 norm, and nonfinite count. Statistics use bounded chunks; shape mismatches fail. `parts` lists parameter metadata and aliases, including tied weights. Values describe weights, not behavioral significance; follow a candidate difference with a controlled dev experiment in the current session.

## Compare with a run from another session

`compare_runs` and `diff` refuse runs from different sessions unless you pass
`cross_session=True`. Use it for a bf16 model against its export reloaded elsewhere, a
PEFT adapter against its merged export, or one GPU against a split model.

1. Evaluate the rows in the current session.
2. `lab.export("merged")`, then start a new session on the export.
3. Evaluate the same rows there, with the same mode, seed, `batch_size`, and held-out status.
4. Compare, passing the other run's directory path:

```python
report = lab.compare_runs(before, "/path/to/other-session/evaluations/run-name", cross_session=True)
print(report["aggregate"]["fixed"], report["aggregate"]["broken"])
print(report["cross_session"]["setup_differences"])
```

Only the session check is skipped. Every other protocol setting still has to match, and a
mismatch is refused. The report's `cross_session` entry lists the setup fields that
differ (devices, dtypes, library versions, adapter, attention, device map). Near-tie
predictions can flip between setups, so also run the unchanged base model in both
sessions and compare those two runs as a control. A run saved before setup was recorded
reports `["setup"]`.

A run from a remote worker has to be fetched first. Call
`SkyPilotLauncher.download(remote_run_dir, local_path)` on the run's directory in the
worker's artifacts, where `local_path` must not exist yet, then pass `local_path` as the run.
