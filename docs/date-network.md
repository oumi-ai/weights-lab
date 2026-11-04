# Fixed-weight date network

`FixedDateNetwork` is an optional, offline example of constructing a small neural
module for exact calendar operations. It uses frozen PyTorch embeddings and linear
layers, plus ReLU for ordering. No model download, training, API, or accelerator is
required:

```bash
uv run python examples/date_network.py
uv run python examples/date_network.py 2026-10-31 --calendar-days -1 --business-days 1
uv run pytest -q tests/test_date_network.py
```

```python
from weights_lab.date_network import FixedDateNetwork

network = FixedDateNetwork()
start = network.encode("2028-02-28")
answer = network(start, 2)
print(network.render(answer))  # 2028-03-01, crossing leap day
print(network.render(network(start, 5, business=True)))  # 2028-03-06
print(network.gap(start, answer).item())  # 2
print(network.compare(start, answer).item())  # -1: start is earlier

saturday = network.encode("2026-10-31")
print(network.render(network.roll_weekend(saturday)))  # 2026-11-02
print(network.render(network.roll_weekend(saturday, direction="backward")))  # 2026-10-30

logits = network.logits(answer)  # [10 positions, 11 characters]
assert network.decode(logits.argmax(-1)) == "2028-03-01"
```

The default domain is **2000-01-01 through 2099-12-31, inclusive**. Construct a
different contiguous domain with `FixedDateNetwork(start="2099-01-01",
end="2100-12-31")`; domains may contain at most 36,600 days. Table construction
uses Python's Gregorian calendar, including century leap-year rules. Invalid
dates, invalid integer inputs, and results outside the domain raise `ValueError`.
Operations never wrap or silently clip an answer. Keep weights in **float32**;
reduced precision cannot represent every day ID exactly and is rejected.

Inputs to the network are integer day IDs relative to the domain's start.
`encode` handles an ISO string, a `datetime.date`, or a flat sequence of either.
The arithmetic methods broadcast tensor inputs, so an array shaped `[N, 1]` of
dates and `[M]` offsets produces `[N, M]` results. `render` returns a string for a
scalar ID and nested Python lists for batches.

Business days are **Monday through Friday only. No public holidays are observed:
a holiday on a weekday still counts.** The start date is never counted: positive offsets
count visited weekdays moving forward, and negative offsets count them moving
backward. Saturday plus one business day is Monday; Saturday minus one is Friday.
Zero preserves Saturday. Weekend roll keeps weekdays unchanged. Calendar gap is
`end - start`; comparison returns the sign of `left - right`.

Construction precomputes date features, ISO character IDs, and an inverse table
from weekday rank to date ID. Inference adds calendar offsets with a frozen linear
map. Business offsets select a direction-specific rank, add with the same linear
map, then use the inverse embedding. Gap uses a frozen subtraction map; comparison
uses differences of ReLUs. Weekend roll and ISO rendering use frozen embeddings.
Python still performs validation, tensor indexing, sign/zero control selection,
and token decoding. `datetime` is used only for construction and input encoding,
never for an inference answer.

The default network contains 610,614 frozen float32 parameters, about 2.44 MB of
weight storage. It does not allocate a dense date-by-date matrix or a language
model vocabulary projection. Its 11 output symbols are `0123456789-`, and the
winning character logit is 1 while other logits are 0. These are **standalone
character logits**, not calibrated probabilities or tokenizer IDs suitable for
adding to a language model's logits. The example demonstrates constructed
computation over a finite domain; it does not demonstrate learned date arithmetic
or that a pretrained model has acquired this computation.

To supply a computed fact in a lab experiment, attach it as an optional prefix:

```python
question = "What date is two calendar days after 2028-02-28?"
computed = network.render(network(network.encode("2028-02-28"), 2))
fact = f"Calendar facts: 2028-02-28 + 2 calendar days = {computed}.\n"

with lab.attach(lambda prompt: fact + prompt, mode="prefix", name="fixed-date-network"):
    result = lab.score(question, {"correct": " 2028-03-01", "other": " 2028-02-29"})
```

`lab` must already be loaded for that optional snippet. The prefix explicitly
gives the model extra information; it does not train or modify the base model.
Compare an unmodified baseline, a matched token-length prefix control, and the
computed facts on separate dev and held-out questions. For multiple questions,
build facts from each question's explicit operands instead of reusing a fixed
answer. Prefix attachments apply to `lab.score`, `lab.evaluate`, and `lab.generate`;
raw model calls bypass them. Remove the attachment before checkpoint export.
