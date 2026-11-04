# LoRA update diagnostics

`lab.lora_diagnostics(pattern="*", adapter=None, top_k=8)` measures the weight
update represented by ordinary linear PEFT LoRA factors, without running the model.
It returns JSON-compatible dictionaries, one per module and adapter.

```python
rows = lab.lora_diagnostics("*q_proj", top_k=4)
for row in rows:
    print(row["module"], row["adapter"], row["top_singular_values"])

# Inspect a loaded adapter even when it is inactive.
rows = lab.lora_diagnostics(adapter="default")
```

In an existing session with a loaded LoRA adapter:

```sh
uv run oumi-weights-lab send --session sessions/first --code 'import json; print(json.dumps(lab.lora_diagnostics("*q_proj"), indent=2, allow_nan=False))' --follow
```

`pattern` is a case-sensitive glob over module names, as returned by
`lab.model.named_modules()`. An unmatched pattern returns an empty list. With
`adapter=None`, each module's active adapters are selected; disabled adapter
layers are skipped. A string or nonempty list of distinct strings selects named
adapters explicitly, including inactive or disabled adapters. Explicit names
missing from all matching modules raise `ValueError`. Multiple selected adapters
produce separate rows: these spectra do not describe the sum of their updates.

Each row describes `scaling[adapter] * B @ A`, using the layer's actual scaling
(including rank-stabilized LoRA scaling). `top_k` must be a positive integer.
The returned singular-value list has at most `min(top_k, factor_rank, in, out)`
entries; implicit zero singular values outside this small spectrum are omitted.

- `module`, `adapter`, `active`, `disabled`, and `merged` identify the factors and
  their current state. Explicitly selected inactive factors still describe a
  potential update. Merged factors describe their stored update, already included
  in the base weight, rather than an additional forward contribution.
- `shape` is `[out, in]`; `factor_rank` is the factor inner dimension.
- `top_singular_values`, `spectral_norm`, and `frobenius_norm` describe the scaled
  update. Norms use the complete small spectrum, regardless of `top_k`.
- `numerical_rank` counts singular values greater than `rank_tolerance`, defined
  as `max(out, in) * float64_epsilon * spectral_norm`. This is the rank of the
  stored factors computed in float64, not a measure of task usefulness.
- `stable_rank` is `frobenius_norm**2 / spectral_norm**2`, or zero for a zero
  update. It measures how broadly the update's squared magnitude is distributed.
- `a_frobenius_norm` and `b_frobenius_norm` describe the unscaled factors.
- `base_frobenius_norm` measures an ordinary dense, floating point, unmerged base
  weight. `relative_frobenius_norm` divides the update norm by that norm. The base
  norm is `null` for unavailable, quantized, or merged base weights; the ratio is
  also `null` for a zero base norm. Quantized weights are never dequantized.

The calculation detaches factors, copies them to CPU float64, computes reduced QR
R factors of `B` and `A.T`, and takes singular values of `Rb @ Ra.T`. It never
materializes the full `B @ A` update or runs a forward pass. Storage and work
scale with the factor sizes and the small rank matrix; very large adapter ranks
can still be expensive. Base norms are read in bounded chunks. Model weights,
training flags, gradients, and adapter selections are unchanged.

Supported inputs are ordinary two-dimensional floating point linear LoRA factors
with explicit per-adapter scaling. DoRA, embedding LoRA, convolutional factors,
PEFT runtime LoRA variants, and factor biases raise `ValueError` when selected.
This is a deterministic weight diagnostic: it does not account for training-time
dropout or measure activation effects. Nonfinite factors, scaling, base weights,
or overflowing results also raise `ValueError` rather than returning NaN/Infinity.
