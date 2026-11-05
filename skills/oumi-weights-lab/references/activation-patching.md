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

## Positions and the input side

`positions` also takes `"all"` (every token of each call) and a slice such as `slice(-2, None)`. Under cached generation `"all"` covers the prompt on the first call and one token on each later call. A slice needs a positive step and must select at least one token; `lab.score` and `lab.evaluate` with `batch_size>1` accept it only when it is anchored to the end, like `slice(-2, None)`.

`at="input"` records or replaces what a module is called with, before it runs. For an attention block's output projection (`o_proj`, or `attn.c_proj` in GPT-2) the input is the per-head outputs, so a head is a slice of the features. Zero one head:

```python
projection = "model.layers.0.self_attn.o_proj"
head, width = 1, 4  # width = the head dimension: config.head_dim when set, else hidden size / heads


def zero_head(value):
    value = value.clone()
    value[..., head * width : (head + 1) * width] = 0
    return value


with torch.no_grad(), lab.patch(projection, zero_head, positions="all", at="input"):
    ablated = model(**inputs).logits.detach().cpu()
```

This matches `lab.zero(f"{projection}.weight", index=(slice(None), slice(head * width, (head + 1) * width)))` for a `Linear`; GPT-2's `Conv1D` stores `[in, out]`, so there the index is `(slice(head * width, (head + 1) * width), slice(None))`. Check the two agree on a prompt before trusting either.

`record(module, at="attention")` captures attention weights `[batch, heads, queries, keys]` from an attention block (`model.layers.0.self_attn`, `transformer.h.0.attn`), with positions selecting queries. It needs eager attention and refuses other modules. Each row sums to one and is zero above the diagonal.
