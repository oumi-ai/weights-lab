# Patch selected activations

Compare clean and corrupted prompts with matched token positions. First measure the unpatched corruption on dev examples. A restored activation should be the only change in the intervention.

```python
import torch

module = "model.layers.0"
clean = tokenizer("The capital of France is", return_tensors="pt").to(lab.device)
corrupt = tokenizer("The capital of Germany is", return_tensors="pt").to(lab.device)
with torch.no_grad(), lab.record(module, positions=[-1]) as recorded:
    clean_logits = model(**clean).logits.detach().cpu()
saved = recorded[module][0]
with torch.no_grad():
    corrupt_logits = model(**corrupt).logits.detach().cpu()
    with lab.patch(module, saved, positions=[-1]):
        patched_logits = model(**corrupt).logits.detach().cpu()
torch.save(
    {"clean": clean_logits, "corrupt": corrupt_logits, "patched": patched_logits},
    artifacts / "patch-logits.pt",
)
```

`patch` is a context manager; its hook is removed on exit, including exceptions. It accepts a tensor or a callable that returns replacement activations; tensor values may broadcast to the selected shape. Module output must be `[batch, tokens, ...]` or a tuple starting with such a tensor. Negative positions count from the end and stay correct in batched scoring, and one deeper than a sequence raises as it would unbatched; while a nonnegative position is active, `lab.score` and `lab.evaluate` need `batch_size=1`, because left padding shifts tokens; positions that resolve to the same token, such as `[2, -1]` on three tokens, raise `ValueError`. Check shapes explicitly so accidental broadcasting cannot masquerade as evidence.

Patch a single-token boundary before trying full generation: generation changes token windows and position meanings. Repeat with a same-prompt patch and an unrelated donor control. The operation changes activations, not stored weights; name the intervention in notes, since it does not appear as a permanent EditBook edit.
