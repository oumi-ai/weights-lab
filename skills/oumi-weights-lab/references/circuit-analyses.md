# Circuit analyses

Three calls answer the questions the [circuit workflow](circuit-workflow.md) asks by hand. Each runs on the layouts [Attention heads](heads.md) supports, where every layer adds its attention and MLP outputs to the residual stream unchanged. A model with norms on a sublayer's output (Gemma 2, OLMo 2), parallel attention and MLP blocks, a LoRA adapter, or bitsandbytes layers raises `NotImplementedError` that names the module. The same happens when the recomputation does not match the model, as with a residual multiplier or a logit softcap. Export a merged model for adapters. The identities below hold to the precision of the model's dtype: about 1e-4 in float32, and a few percent of the value in bf16 and fp16.

A **metric** is a callable over the batch's last-position logits, `metric(logits) -> scalar tensor`, or an `(answer, counterfactual)` pair of strings, meaning the mean logit difference between their first tokens. Use the same metric for every call you compare. For path and edge analyses pass `clean` and `corrupt` as paired lists of prompts, each pair with the same token count.

| Question | Call | Passes |
| --- | --- | --- |
| Which parts wrote the answer logit? | `logit_attribution` | 1 |
| Does the effect of sender A reach receiver B? | `path_patch` | 4 per pair |
| Which edges carry the behavior? | `edge_attribution` | 2 + steps per pair |
| Is a proposed circuit enough? | `find_circuit` | 3 + steps per pair |

## Direct logit attribution

```python
result = lab.logit_attribution(
    "The capital of France is", (" Paris", " Berlin"), by=("component", "position")
)
assert abs(result["total"] - result["actual"]) < 1e-4
top = sorted(result["heads"], key=lambda h: -abs(h["value"]))[:5]
```

It splits the logit difference at the last token into the embeddings, each head, each MLP, and the output biases. The final norm's scale is frozen at this pass's value, so the parts add up to the model's own number. Use it to find heads that write the answer directly.

It is a linear estimate. Ablating a head also changes the norm's scale and everything downstream, so a head with a small value can still matter through later layers. It shows only direct effects on the logits. With `"position"`, each head's `positions` shows which source tokens it read from; the parts sum to the head's value.

## Path patching

```python
senders = [(3, 7), (3, 9)]
receivers = [(8, 1, "v"), (8, 2, "v")]
result = lab.path_patch(
    clean_prompts, corrupt_prompts, senders, receivers, metric=(" Paris", " Berlin")
)
print(result["fraction"])  # 0 is no effect, 1 is the whole clean-to-corrupt change
```

Path patching puts the senders' corrupt outputs into the receivers only, with every other head held at its clean output. So it measures one route, not the sender's total effect. What is frozen decides the answer. Every head other than the senders is always frozen, so no path runs through another head. With the default, MLPs between sender and receiver respond, so paths through them count. With `freeze_mlps=True` only the direct path from sender to receiver counts. Add `"embed"` as a sender when the prompts differ at tokens the receiver reads directly. Test a path against a control path of the same kind before reading anything into a number.

## Edge attribution patching

```python
edges = lab.edge_attribution(clean_prompts, corrupt_prompts, metric=(" Paris", " Berlin"), steps=5)
circuit = lab.find_circuit(clean_prompts, corrupt_prompts, metric=(" Paris", " Berlin"), top=100)
print(circuit["faithfulness"], len(circuit["edges"]), circuit["total_edges"])
```

`edge_attribution` scores every edge from the embeddings, a head, or an MLP to a later head input, MLP, or the logits, using integrated gradients over the input embeddings. It is an estimate from gradients, so it proposes edges. `find_circuit` takes the top edges, drops those not on a path from the embeddings to the logits, and then confirms by patching for real: a clean run in which every edge outside the circuit carries its corrupt value. Faithfulness is 1 when the circuit alone reproduces the clean metric and 0 when it does no better than the corrupt run. Keeping every edge gives 1 and keeping none gives 0.

Faithfulness is measured on the prompts that found the circuit. Check it on new prompts, and compare with a circuit of the same size built from random edges, before calling the circuit the mechanism. To test edges you chose yourself, pass `edges=[{"sender": ..., "receiver": ...}]`, in the form `edge_attribution` returns.

## Which to use

Start with `logit_attribution` to see who writes the answer. Use `path_patch` to test a specific story, such as "head A feeds head B's values". Use `edge_attribution` and `find_circuit` when you have no story and want a candidate circuit. In every case, confirm with the interventions in the [circuit workflow](circuit-workflow.md): an ablation, a control, and new prompts.
