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
