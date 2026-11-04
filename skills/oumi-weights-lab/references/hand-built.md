# Hand-built components

Use temporary components to test a narrow computation hypothesis. `attach` returns a removable handle and supports a context manager. Save the component source and its controls alongside results; standalone checkpoint export rejects active attachments.

A checked fact prefix is the smallest date experiment:

```python
from oumi_weights_lab.circuits import DateFacts, DateCalculator
import torch

calculator = DateCalculator()
print(calculator(torch.tensor([[2026, 11, 3]])).item())  # Monday=0.
with lab.attach(DateFacts(), mode="prefix", name="date-facts"):
    print(lab.score("The weekday of 2026-11-03 is", {"Tue": " Tuesday", "Wed": " Wednesday"}))
```

Prefix components transform text passed through `lab.score`, `lab.evaluate`, and `lab.generate`. Raw `model(...)` and `model.generate(...)` bypass prefix transformations. `DateCalculator` accepts valid `[year, month, day]` integer tensors in 2000–2099. `DateFacts` prepends checked weekday facts for valid ISO dates in that range; this explicitly supplies information to the model.

A logits component receives `(model, args, kwargs, output)` and returns a finite delta broadcastable to the original logits shape:

```python
def favor_token(model, args, kwargs, output):
    delta = output.logits.new_zeros(output.logits.shape[-1])
    delta[token_id] = 0.1  # Select token_id from tokenizer; establish a baseline first.
    return delta


with lab.attach(favor_token, mode="logits", name="token-bias-control"):
    result = lab.evaluate(dev, name="token-bias-v1", max_passes=100)
```

The package also provides frozen `TokenMatcher`, `NearestMatch`, `LookupTable`, `DigitAdder`, and `Compare` tensor modules. They use explicit indexing and arithmetic as well as fixed weights; describe them as constructed computation, not learned neural arithmetic. Test the component independently on its full input domain when feasible, then test model-plus-component on untouched examples. Controls should separate extra information, extra tokens, and the intended computation.
