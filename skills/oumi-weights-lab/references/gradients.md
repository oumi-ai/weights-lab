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

With a string target, only continuation tokens contribute causal loss; include intended leading whitespace in the target. Dictionary inputs support batched `input_ids`, `attention_mask`, `labels`, and `position_ids`; label `-100` masks unscored tokens. Left or right padding marked by `attention_mask` is excluded: pad tokens are not scored and do not predict the following token. Unless you pass `position_ids`, padded rows count positions from their first unpadded token when the model accepts `position_ids`, so absolute-position models such as GPT-2 match unpadded rows. `parameters` accepts any name `lab.parts()` lists, including tied aliases such as `lm_head.weight`. A tied tensor's gradient covers all of its uses; aliases requested together map to one shared result tensor. With `parameters=None`, each tensor appears once, under its first name. `gradient` computes each row's mean causal-loss gradient separately and preserves existing parameter gradients, train/eval flags, and `requires_grad` flags. Do not run it inside `torch.inference_mode()`.

A gradient direction is diagnostic evidence, not a calibrated edit size. If testing a step, use `lab.edit` with an isolated callable, a small preregistered scale sweep on dev data, and exact undo after each candidate. Check a random-direction control and report regressions. Keep the held-out set out of the sweep.

## Activation gradients

`modules` asks for the gradient of the same loss with respect to a module's activation. Each entry is `[rows, tokens, ...]` on CPU.

```python
grads = lab.gradient("The capital of France is", target=" Paris", modules=["model.layers.3"])
print(grads["model.layers.3"].shape)  # [1, tokens, hidden]
last = lab.gradient(
    "The capital of France is", target=" Paris", modules="model.layers.3", positions=[-2]
)
```

`at="input"` takes the gradient at what the module is called with. `positions` takes an int, a list, `"all"`, or a slice, and defaults to `"all"`: the last token predicts nothing that is scored, so its output gradient is zero. Without `parameters`, no parameter gradients are returned. It works on a frozen model, since the gradient is taken at a detached copy of the activation. A module that runs twice in one forward raises, because its gradient would mix the calls.
