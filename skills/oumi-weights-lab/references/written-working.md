# Written working as a computation control

Before changing weights, ask whether providing correct intermediate work fixes the answer. Compare the original prompt, correct work, irrelevant matched-length text, and plausible incorrect work. Hold the answer options and answer-scoring policy fixed. Never include the correct answer verbatim in the supplied work.

```python
prompt = "A shop has 17 apples and receives 8 more. How many apples? Answer:"
options = {"correct": " 25", "wrong": " 24"}
plain = lab.score(prompt, options, max_passes=2)
working = lab.score("Working: 17 + 8 = 17 + 3 + 5.\n" + prompt, options, max_passes=2)
control = lab.score("Note: apples may be red, green, or yellow.\n" + prompt, options, max_passes=2)
print({"plain": plain, "working": working, "control": control})
```

For a collection, build separate dev rows and save each condition with `evaluate` and distinct run names. Different prompts have different dataset hashes, so `lab.diff` will reject those conditions: compare item-aligned saved answers explicitly. Summed continuation log likelihood is used; prompts and options are tokenized separately with no special tokens and no inserted whitespace. Shorter options may receive higher summed likelihood simply because of length.

An improvement supports the hypothesis that added computation or information helps. It does not by itself show which internal circuit is missing, nor justify claiming that weights have been repaired. Reserve any held-out final condition before examining its outcomes.
