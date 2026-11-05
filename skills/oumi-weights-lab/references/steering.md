# Steer with a recorded direction

Add a direction to a layer's output and measure what it changes. First score the unsteered model on dev examples; a steering effect is only a difference from that baseline.

Capture the direction as the mean difference between residual-stream outputs on two sets of prompts. Record at the last position of each prompt.

```python
import torch

layer = "model.layers.12"  # GPT-2: "transformer.h.12"


def last_output(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(lab.device)
    with torch.no_grad(), lab.record(layer, positions=[-1]) as recorded:
        model(**inputs)
    return recorded[layer][0][0, 0]  # [hidden]


positive = torch.stack([last_output(prompt) for prompt in positive_prompts])
negative = torch.stack([last_output(prompt) for prompt in negative_prompts])
direction = positive.mean(0) - negative.mean(0)
```

Add it with `patch`, scaled. A tensor of shape `[hidden]` broadcasts over every selected token. Use `positions="all"` so the prompt and every generated token are steered.

```python
scale = 4.0
with lab.patch(layer, lambda value: value + scale * direction.to(value.device), positions="all"):
    steered_text = lab.generate("The movie was", max_new_tokens=20)
    steered_scores = lab.score("The movie was", [" good", " bad"])
baseline_scores = lab.score("The movie was", [" good", " bad"])
```

`"all"` means the patch applies on every forward call. With a KV cache, the first call covers the whole prompt and each later call covers one new token, so generated tokens are steered too. Greedy generation gives the same tokens with and without the cache. A patch at `positions=[-1]` would instead steer only the last token of each call: the last prompt token on the first cached call, then every new token, but not the earlier prompt tokens.

Check the direction before trusting it:

- sweep a few scales (including 0 and a negative one) on dev data and report each;
- repeat with a random direction of the same norm;
- keep held-out data out of the sweep.

`"all"` is allowed under `batch_size>1` in `lab.score` and `lab.evaluate`, but it also hooks the pad positions. A `record` at `"all"` captures them in dimension 1, and a tensor replacement should be shaped `[hidden]` so it broadcasts over every token, padded or not. The patch is not an edit: it does not appear in `lab.edits()`, and `lab.patches()` lists it while active.
