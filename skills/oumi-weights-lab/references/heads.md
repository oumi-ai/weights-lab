# Attention heads

`lab.heads()` lists every attention head as slices of the model's own weights. `lab.head_weights(layer, head)` returns one head's matrices for offline analysis. Both read base weights only.

Supported layouts: separate `q_proj`, `k_proj`, `v_proj` with `o_proj` or `out_proj` (Llama, Mistral, Qwen, Gemma, OLMo, OPT), gated queries (Qwen3.5, Qwen3-Next full attention), and GPT-2's fused `c_attn`. Linear-attention blocks are skipped. Fused `qkv_proj`, `query_key_value`, and latent attention raise `NotImplementedError`, and so does a LoRA model: export the merged model and start a session on it. A model with two attention stacks that reuse layer indexes, such as a vision tower plus a text decoder, also raises, because a layer number would be ambiguous.

## List heads

```python
heads = lab.heads()  # or lab.heads(layers=[0, 3])
head = heads[5]
head["layer"], head["head"], head["kv_head"], head["group"]
head["o"]  # [{"name": ..., "index": ...}]
head["o_input"]  # {"module": ..., "at": "input", "features": [start, stop]}
```

Each of `q`, `k`, `v`, `o` (and `gate` for gated queries) is a list of parts. A part is `{"name", "index"}`, so `lab.zero(**part)` and `lab.edit(**part, value=...)` take it as is. Weight parts come first; `q`, `k`, `v` and `gate` add a bias part when the model has one. `o` has no bias part, because the output bias is shared across heads.

## Attention patterns

```python
module = head["module"]
inputs = tokenizer("hello world yes no hello", return_tensors="pt").to(lab.device)
with lab.record(module, positions="all", at="attention") as captured, torch.no_grad():
    model(**inputs)
pattern = captured[module][0][0, head["head"]]  # [queries, keys]
```

## Ablate one head

Zero its slice of the output projection. This is a reversible weight edit:

```python
ids = [lab.zero(**part) for part in head["o"]]
...
for _ in ids:
    lab.undo()
```

Or patch the output projection's input, which edits nothing:

```python
start, stop = head["o_input"]["features"]


def drop(value):
    value = value.clone()
    value[..., start:stop] = 0
    return value


with lab.patch(head["o_input"]["module"], drop, positions="all", at="input"), torch.no_grad():
    logits = model(**inputs).logits
```

The two give the same logits. To ablate every head and MLP in turn and score a set of rows, don't loop over these: `lab.ablation_scan(rows)` does it in batched passes, with the patch form and nothing written to the weights. It is a first sweep, so scan groups too (`groups="layers"`) and follow it with `find_circuit`. With grouped-query attention, zeroing `k` or `v` parts changes every head in `head["group"]`, because they share those rows. Zero `o` to ablate one head.

## Search for induction heads

`head_weights` returns `q`, `k`, `v` as `[head_dim, hidden]` and `o` as `[hidden, head_dim]`, all float32 on CPU. For a head `h1` in an earlier layer than `h2`, the K-composition score is `‖W_K^{h2} W_O^{h1} W_V^{h1}‖_F / (‖W_K^{h2}‖_F ‖W_O^{h1} W_V^{h1}‖_F)`:

```python
def k_composition(lab, first, second):
    a = lab.head_weights(*first)
    b = lab.head_weights(*second)
    ov = a["o"] @ a["v"]  # [hidden, hidden]
    through = b["k"] @ ov  # [head_dim, hidden]
    return (through.norm() / (b["k"].norm() * ov.norm())).item()
```

Score every earlier-layer head against every later-layer head, then take the top few. Confirm each with captured attention (an induction head attends to the token after an earlier copy of the current token on repeated random text) and with an ablation.

## Caveats

- Rotary embeddings and query/key norms (Qwen3, OLMo 2, Gemma 3) act between the projections and the scores, so products of weights alone do not give attention patterns.
- Composition scores are a screening heuristic. Rank candidates with them; do not report a head as a circuit member on that evidence.
- Gated models multiply each head's output by a sigmoid of the gate, so zeroing `o` is the clean ablation there.
