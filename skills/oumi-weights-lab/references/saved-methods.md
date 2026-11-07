# Save and reuse a method

Use this when you invent a way to inspect or edit a model and want it to outlive the session. A saved method keeps its source, its known-answer results and the setup it ran on. A later session loads it by name, and every result it returns says which version produced it.

## 1. Write it as a method

A method is a plain function whose first parameter is `lab`.

```python
def rank_heads(lab, prompt, answer):
    """Rank heads by how much zeroing each one lowers the answer's probability."""
    import torch

    ...
    return {"ranking": ranking}
```

It must stand on its own. A module it uses becomes an import line when it is saved. Any other global name, a closure variable, or a first parameter not named `lab` is refused, and the message names them. Pass values as arguments, or import inside the function. It may return anything. A dict result comes back as a copy with a `saved_method` field holding the name and version, so don't use that key yourself.

## 2. Save it

```python
meta = lab.save_method(rank_heads, note="zeroes one head at a time")
```

The name defaults to the function's name (lowercase letters, digits, `_` and `-`). The version is a hash of the saved source, so saving the same code again returns the same entry and changing one character makes a new version. Saving the same code in a different setup keeps the version and adds that setup to its list. Files go to `methods/<name>/<version>/` in the directory where the session started, or to the directory in `OUMI_WEIGHTS_LAB_METHODS`. Keep that directory somewhere that outlives the session. `lab.methods()` lists what is saved, with notes and result summaries.

## 3. Test it on known answers

Run the method on dev tasks, write a claim for each, and grade it with the method's name and version. Each grade is added to that version's results. In a session, `lab.record_method_result(name, version, graded)` records a grade with the session's setup.

```sh
oumi-weights-lab proving-ground grade --task tasks/<id> --key keys/<id>.json \
  --claim claim.json --method rank_heads@<version>
```

See [Test a method on known answers](proving-ground.md) for the tasks and the claim format. Edit the method and save it again before the next round, so each version's results come from exactly its own code. Never grade a saved method on held-out tasks yourself.

## 4. Compare with an existing method

Grade both methods on the same dev tasks, at least five. The next step pairs them by task and refuses a method that does not beat the other by the paired interval.

## 5. Load it later

```python
result = lab.run_method("rank_heads", prompt, answer)  # the newest version
result = lab.run_method("rank_heads", prompt, answer, version="3f9a1c0b2d4e")
method = lab.load_method("rank_heads")  # .source, .meta, call it
```

Loading runs the saved Python with your permissions, so load only methods you trust. A saved file that was edited after saving is refused. Report the version with every result you quote.

## 6. Promote a winner

```sh
oumi-weights-lab methods list
oumi-weights-lab methods promote rank_heads --against baseline@<version> --output draft.md
```

The draft has the docstring, the source, every setup the method ran in, a table of results per task and the comparison. The comparison uses only results from matching setups and says what it left out. It is written for a maintainer to review and edit into the skill, so it never goes into the skills directory. Promotion is refused without results, and with `--against` when the method does not win, unless you pass `--force`. An existing draft file is replaced only with `--overwrite`.
