# Optional fixed-weight date network

Use `weights_lab.date_network.FixedDateNetwork` when an experiment needs bounded
calendar-day arithmetic, Monday–Friday business-day arithmetic, weekend roll,
calendar gaps/comparisons, or exact ISO character output. Run the offline demo:

```bash
uv run python examples/date_network.py 2026-10-31 --business-days 1
uv run pytest -q tests/test_date_network.py
```

```python
from weights_lab.date_network import FixedDateNetwork

network = FixedDateNetwork()
start = network.encode("2026-10-31")
result = network(start, 1, business=True)
assert network.render(result) == "2026-11-02"
assert network.decode(network.logits(result).argmax(-1)) == "2026-11-02"
```

Default domain: 2000-01-01 through 2099-12-31 inclusive. Custom contiguous domains
may contain up to 36,600 days. Host parsing/table construction uses `datetime`;
inference uses frozen embeddings, linear addition/subtraction, ReLU comparison,
and explicit tensor validation/control selection. Weights stay float32. Invalid
inputs and results outside the domain raise `ValueError`.

Business offsets exclude the start and count visited weekdays in either
direction; zero preserves the start, including weekends. No public holidays,
month offsets, year offsets, natural-language date parsing, or inclusive-start
counting are implemented. `gap(start, end)` returns signed calendar days;
`compare(left, right)` returns -1/0/1. Batched tensor shapes broadcast.

The network has about 2.44 MB of frozen weights. Its ten output positions use an
11-character vocabulary, `0123456789-`. These logits are unrelated to the loaded
model's tokenizer. Do not attach them as model-logit deltas. For an optional lab
attachment, compute a checked fact and supply it using `mode="prefix"`; preserve
the question and label the experiment as supplying extra information. Establish
baseline and prefix-length controls and test on untouched questions. Remove the
attachment before exporting a checkpoint.

This is constructed finite computation, not trained arithmetic or a trainable
date adapter. See `docs/date-network.md` for the architecture, full API examples,
and prefix recipe; `references/hand-built.md` covers attachment controls.
