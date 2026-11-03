# Per-row gradients on short inputs

Eval mode does not disable autograd. Select a small parameter set, cap token length, and use one short row at a time when diagnosing memory use.

```python
import torch

name = "model.layers.0.mlp.down_proj.weight"
gradients = lab.gradient(
    "The capital of France is",
    target=" Paris",
    parameters=[name],
    max_tokens=64,
)
print(gradients[name].shape)  # [rows, *parameter.shape]
torch.save(gradients, artifacts / "answer-gradient.pt")
```

With a string target, only continuation tokens contribute causal loss; include intended leading whitespace in the target. Dictionary inputs support batched `input_ids`, `attention_mask`, and `labels`; label `-100` masks unscored tokens. Padding is masked through `attention_mask`. `gradient` computes each row's mean causal-loss gradient separately and preserves existing parameter gradients, train/eval flags, and `requires_grad` flags. Do not run it inside `torch.inference_mode()`.

A gradient direction is diagnostic evidence, not a calibrated edit size. If testing a step, use `lab.edit` with an isolated callable, a small preregistered scale sweep on dev data, and exact undo after each candidate. Check a random-direction control and report regressions. Keep the held-out set out of the sweep.
