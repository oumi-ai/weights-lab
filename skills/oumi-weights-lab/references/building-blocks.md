# Building blocks for new methods

When no existing call answers your question, write the method as one short function from three pieces. `run_with` runs the model once with activations or parameters replaced. `sweep` tries many interventions in one batched pass. `jacobian`, `jvp` and `vjp` differentiate between two points. Each removes its hooks on exit, leaves parameters unchanged, and counts one pass per sequence.

An intervention is `{"module", "at", "positions", "value"}`, as for `patch`: `at` is `"output"` or `"input"`, positions default to the last token, and a callable value receives the selected tensor.

## Rank sites, then confirm the best

This method scores each head of one layer by how much zeroing its slice of the `o_proj` input lowers the answer's logit.

```python
def head_effects(prompt, answer_id, layer):
    heads = lab.heads(layers=[layer])

    def zeroing(part):
        start, stop = part["features"]

        def zero(selected):
            selected = selected.clone()
            selected[..., start:stop] = 0
            return selected

        return {"module": part["module"], "at": "input", "value": zero}

    rows = lab.sweep(prompt, [None] + [[zeroing(h["o_input"])] for h in heads])
    clean = float(rows[0]["logits"][0, answer_id])
    return sorted(
        (float(row["logits"][0, answer_id]) - clean, h["head"]) for row, h in zip(rows[1:], heads)
    )  # most negative first: zeroing these heads hurts the answer most


lead = head_effects("The capital of France is", answer_id, layer=3)[0]
```

Confirm the top head with a separate run that changes nothing else:

```python
best = lab.heads(layers=[3])[lead[1]]["o_input"]
check = lab.run_with(
    "The capital of France is",
    activations=[
        {
            "module": best["module"],
            "at": "input",
            "positions": None,
            "value": lambda x: x.index_fill(-1, torch.arange(*best["features"]), 0),
        }
    ],
)
```

`sweep` runs `batch_size` rows per forward (8 by default), and the pass budget counts every row. To try parameters instead, call `run_with(..., weights={name: tensor})` in a loop; nothing is written into the model.

## Ask how a direction moves a later layer

`jvp` gives the change at a target for a small step along a direction at a source, from one forward pass.

```python
direction = steering_vector / steering_vector.norm()  # shape [hidden]
value, change = lab.jvp(
    "The capital of France is",
    "model.layers.3",
    "model.layers.8",
    direction[None],  # the source selection without its batch dimension: [1, hidden]
)
print(change.norm())
```

Both points must run once in the forward, and a target that runs before the source has a zero derivative. `jacobian` returns the whole matrix and `vjp` pulls a target direction back to the source.

## Keep the results honest

A method's output is a lead. A head that ranks high, or a direction with a large derivative, shows where to look. Confirm it with a causal test: a patch or ablation scored with `score` or `evaluate` on held-out rows, compared against a matched control.
