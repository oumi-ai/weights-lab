# LoRA update diagnostics

For a loaded LoRA model, `lab.lora_diagnostics(pattern="*", top_k=8)` reports
singular values, numerical/stable rank, and magnitude of each active adapter's
effective scaled update. It uses factor QR decompositions and a small SVD;
it does not form a dense `B @ A` or spend model forwards. An explicit
`adapter="default"` selects that adapter even when inactive. Multiple adapters
are reported independently, never as a combined spectrum. Unsupported DoRA,
embedding, convolutional and runtime variants are rejected. A low rank or small
norm is a diagnostic, not evidence that deleting the adapter improves behavior.

```python
import json

rows = lab.lora_diagnostics("*q_proj", top_k=4)  # Inspect module names first.
for row in rows:
    print(row["module"], row["adapter"], row["spectral_norm"], row["stable_rank"])
(artifacts / "lora-diagnostics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
```

Each row describes `scaling[adapter] * B @ A` with the layer's actual scaling.
`relative_frobenius_norm` compares the update with an unmerged dense base weight
and is `null` when that base is unavailable, quantized, merged, or zero.
