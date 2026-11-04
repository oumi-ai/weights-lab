# Hand-built components

Use temporary components to test a narrow computation hypothesis. `attach` returns a removable handle and supports a context manager. Save the component source and its controls alongside results; standalone checkpoint export rejects active attachments.

A checked fact prefix is the smallest experiment. It explicitly supplies information to the model:

```python
def weekday_fact(prompt):
    return "Fact: 2026-11-03 is a Tuesday.\n" + prompt


with lab.attach(weekday_fact, mode="prefix", name="weekday-fact"):
    print(lab.score("The weekday of 2026-11-03 is", {"Tue": " Tuesday", "Wed": " Wednesday"}))
```

Prefix components transform text passed through `lab.score`, `lab.evaluate`, and `lab.generate`, before any chat template wraps it. Raw `model(...)` and `model.generate(...)` bypass prefix transformations.

A logits component receives `(model, args, kwargs, output)` and returns a finite delta broadcastable to the original logits shape. The delta changes returned logits only, so a model call that passes `labels` raises `ValueError` rather than returning a loss without the delta; compute any loss from the returned logits. Score with `batch_size=1` while a logits component is attached, and use the same batch size for its baseline:

```python
def favor_token(model, args, kwargs, output):
    delta = output.logits.new_zeros(output.logits.shape[-1])
    delta[token_id] = 0.1  # Select token_id from tokenizer; establish a baseline first.
    return delta


with lab.attach(favor_token, mode="logits", name="token-bias-control"):
    result = lab.evaluate(dev, name="token-bias-v1", max_passes=100, batch_size=1)
```

Test the component independently on its full input domain when feasible, then test model-plus-component on untouched examples. Controls should separate extra information, extra tokens, and the intended computation.
