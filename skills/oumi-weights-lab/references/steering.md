# Steer with a direction from contrasting prompts

Add a direction to a layer's output and measure what it changes. First score the unsteered model on dev examples; a steering effect is only a difference from that baseline.

Compute the direction from pairs of prompts that differ in one thing, such as the same sentence written positively and negatively. `contrast_directions` records the last token of each prompt at every decoder layer, holds some pairs back, and checks the direction on them.

```python
pairs = [
    ("The movie was wonderful", "The movie was terrible"),
    ("I loved the food", "I hated the food"),
    # ... at least 8 pairs; the minimum is 4
]
rows = lab.contrast_directions(pairs, held_out=0.25, seed=0)
for row in rows:
    print(
        row["layer"], row["heldout_pair_accuracy"], row["heldout_auc"], row["heldout_effect_size"]
    )
```

Read the held-out columns before you steer. A layer where held-out accuracy or AUC is near chance has no direction worth adding. Pairs that share a prompt stay on one side of the split, so pairs built around one neutral prompt count as one group; the call refuses a split it cannot make without leaking.

Picking the best layer on those numbers uses them up, so the best score is optimistic. Choose the layer on the held-out check, then confirm it on fresh pairs that played no part in the choice, and report the second number.

```python
best = max(rows, key=lambda row: row["heldout_auc"])  # the choice; do not report this score
confirm = lab.contrast_directions(new_pairs, layers=[best["layer"]], held_out=0.5, seed=1)[0]
print(confirm["heldout_pair_accuracy"], confirm["heldout_auc"])  # report these
```

The confirming call computes its own direction from `new_pairs`; use the direction from the first call (`best`) to steer, and use the second call only to judge the layer. The check shows the contrast is readable along the direction; the sweep below shows whether the model responds to it.

```python
row = best
layer, direction = row["module"], row["direction"]  # direction is a unit vector
```

Add it with `patch`, scaled. `row["norm"]` is the size of the raw mean difference, so a scale of 1.0 times the norm adds one typical contrast. A tensor of shape `[hidden]` broadcasts over every selected token. Use `positions="all"` so the prompt and every generated token are steered. Pass the row's `at` so a direction computed at `at="input"` is added at the input.

```python
scale = 4.0 * row["norm"]
with lab.patch(
    layer,
    lambda value: value + scale * direction.to(value.device),
    positions="all",
    at=row["at"],
):
    steered_text = lab.generate("The movie was", max_new_tokens=20)
    steered_scores = lab.score("The movie was", [" good", " bad"])
baseline_scores = lab.score("The movie was", [" good", " bad"])
```

`"all"` means the patch applies on every forward call. With a KV cache, the first call covers the whole prompt and each later call covers one new token, so generated tokens are steered too. Greedy generation gives the same tokens with and without the cache. A patch at `positions=[-1]` would instead steer only the last token of each call: the last prompt token on the first cached call, then every new token, but not the earlier prompt tokens.

Check the direction before trusting it:

- sweep a few multiples of `row["norm"]` (including 0 and a negative one) on dev data and report each;
- repeat with a random direction of the same norm;
- keep held-out data out of the sweep.

The contrast pairs, including the held-out ones, are not evaluation data. Use dev examples for the sweep, and keep the held-out evaluation set untouched.

`"all"` is allowed under `batch_size>1` in `lab.score` and `lab.evaluate`, but it also hooks the pad positions. A `record` at `"all"` captures them in dimension 1, and a tensor replacement should be shaped `[hidden]` so it broadcasts over every token, padded or not. The patch is not an edit: it does not appear in `lab.edits()`, and `lab.patches()` lists it while active.
