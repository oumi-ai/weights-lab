# Rank patch sites with activation gradients

Patching every module at every token costs one pass per site. Attribution patching estimates all of them from one clean recording and one gradient pass, then spends real patches on the best few. The estimate is first-order, so it misjudges large effects. Treat it as a ranking to confirm, not a measurement.

1. Pick a clean and a corrupt prompt with equal token counts, and name the clean answer.
2. Record the clean activations at every token for the modules you want to rank.
3. Take activation gradients of the loss on the clean answer, run on the corrupt prompt.
4. Estimate each site as (clean - corrupt) times the gradient, summed over features.
5. Rank the sites by the size of the estimate.
6. Confirm the top few with `patch` runs that put the clean activation at that position, scored with `score` or `evaluate`.

```python
import torch

modules = [f"model.layers.{i}" for i in range(4)]
clean_text, corrupt_text, answer = "The capital of France is", "The capital of Germany is", " Paris"


def build(text):
    """Prompt plus answer as one row, with only the answer scored."""
    prefix = tokenizer(text, add_special_tokens=False)["input_ids"]
    reply = tokenizer(answer, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prefix + reply])
    labels = torch.tensor([[-100] * len(prefix) + reply])
    return {"input_ids": ids.to(lab.device)}, labels, len(prefix)


clean, _, size = build(clean_text)
corrupt, labels, corrupt_size = build(corrupt_text)
assert size == corrupt_size  # equal token counts

with torch.no_grad(), lab.record(modules, positions="all") as clean_acts:
    model(**clean)
with torch.no_grad(), lab.record(modules, positions="all") as corrupt_acts:
    model(**corrupt)
grads = lab.gradient({**corrupt, "labels": labels}, modules=modules)

sites = []
for name in modules:
    delta = clean_acts[name][0] - corrupt_acts[name][0]  # [1, tokens, hidden]
    estimate = (delta * grads[name]).sum(-1)[0]  # [tokens]
    sites += [(float(value), name, token) for token, value in enumerate(estimate[:size])]
sites.sort(key=lambda site: -abs(site[0]))
print(sites[:5])
```

Only prompt tokens are ranked, because later positions hold the answer itself.

The loss falls when the gradient step moves toward the clean answer, so a negative estimate means restoring that site lowers the corrupt run's loss. Rank by the sign you care about and say which in your notes.

Confirm before reporting. For each top site, patch the clean activation into the corrupt run at that token and score the answer:

```python
name, token = sites[0][1], sites[0][2]
saved = clean_acts[name][0][:, token : token + 1]
with lab.patch(name, saved, positions=[token]):
    patched = lab.score(corrupt_text, [answer])
```

Compare with the unpatched corrupt score and a control patch at an unrelated token. Check that `score` tokenizes the corrupt prompt to the same positions as the ranked row. Keep held-out data out of the ranking.
