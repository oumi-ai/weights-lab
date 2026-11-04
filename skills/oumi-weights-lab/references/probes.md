# Probe representations with document splits

Keep whole source documents together. Randomly splitting rows from the same document measures leakage as well as representation quality. Reserve a test group before choosing a layer or token position; do not tune repeatedly on probe test accuracy.

For an architecture whose hidden module is `model.layers.0`:

```python
import torch

module = "model.layers.0"  # Inspect model.named_modules() first.
inputs = tokenizer(texts, return_tensors="pt", padding=True).to(lab.device)
with torch.no_grad(), lab.record(module, positions=[0]) as captures:
    model(**inputs)
features = captures[module][0][:, 0, :]
result = lab.probe(
    features,
    labels,
    groups=document_ids,
    train_groups=training_document_ids,
    ridge=1e-3,
)
print(
    {
        key: result[key]
        for key in (
            "accuracy",
            "train_accuracy",
            "train_size",
            "test_size",
            "train_groups",
            "test_groups",
        )
    }
)
torch.save(result, artifacts / "probe.pt")
```

The example uses position zero, which must be a meaningful unpadded token in your batch. For last-content-token features with variable-length padded rows, record each row separately at its actual content position and concatenate the captures. `record` stores detached CPU tensors shaped `[batch, selected_positions, ...]`, as a list for every module call. Generation produces multiple entries. `probe` fits a ridge linear classifier with an intercept and keeps document groups disjoint; every class must appear in training. Save group IDs and feature-selection choices with the result. This measures linearly recoverable information; it does not establish causal use.
