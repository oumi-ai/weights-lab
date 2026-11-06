# Zero or swap parts

Read the model's parameter names first. Choose a specific layer branch or slice and record why it tests the hypothesis. A residual branch's output projection is often a useful ablation point; zeroing arbitrary normalization weights has a different meaning.

```python
name = "model.layers.0.mlp.down_proj.weight"  # Verify on this model.
print(lab.measure(name))
baseline = lab.evaluate(dev, name="baseline-v1", max_passes=100)
edit_id = lab.zero(name, index=(slice(None), slice(0, 16)), note="ablate selected columns")
try:
    changed = lab.evaluate(dev, name="zero-v1", max_passes=100)
    print(lab.diff(baseline, changed))
finally:
    lab.undo(edit_id)
```

To zero every head and MLP in turn and score the same rows, don't write this loop. `lab.ablation_scan(rows)` runs the sweep in batched passes with nothing written to the weights and no run per part; it also takes groups of parts.

`edit(name, value_or_callable, index=None, note="")` requires the exact selected shape; scalar broadcasting is rejected. A callable receives an isolated copy and may return a replacement or mutate its copy and return `None`. Failed callables do not change weights or history.

`swap(name, source, source_name=None, index=None)` accepts a tensor, model, or name-to-tensor mapping. The source tensor must match the full target shape; an index selects the same region on both. It copies values rather than replacing parameters. Stack multiple edits and undo in strict reverse order, even when they affect disjoint parts. `lab.edits()` includes active and undone history; `lab.active_edit_ids` names the currently active stack. Undo snapshots retain the original dtype on CPU and cover only the edited region.
