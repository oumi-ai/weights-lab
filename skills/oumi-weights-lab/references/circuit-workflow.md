# Find and test a circuit

A circuit claim says a small set of components carries a behavior. This recipe chains the other recipes in the order that tests such a claim. Each step names the recipe that does the work.

## 1. Pick a behavior and a contrast pair

Choose a clean prompt and a corrupt prompt that differ in one token and have different answer tokens, such as "The capital of France is" with " Paris" and "The capital of Germany is" with " Berlin". Check that both tokenize to the same length. Score both prompts with `score` and keep the baseline. A useful measure is the clean answer's score minus the corrupt answer's score:

```python
def margin(prompt):
    s = lab.score(prompt, [" Paris", " Berlin"], batch_size=1)
    return s[" Paris"] - s[" Berlin"]


baseline = {"clean": margin(clean_text), "corrupt": margin(corrupt_text)}
```

If the two baselines are close, the pair does not isolate the behavior. Pick another pair.

## 2. Locate candidates

- [Logit lens](logit-lens.md): find the layer where the answer first becomes the top token.
- [Attribution patching](attribution-patching.md): rank module and token sites. The ranking is an estimate.
- [Attention heads](heads.md): list heads and capture their attention with `record(..., at="attention")`.

Take a short list. These steps produce leads, not results.

## 3. Name a hypothesis before intervening

Write down which components you think carry the behavior. For each intervention below, write what you expect if you are right and what would show you are wrong. Do this before running anything, so the result cannot choose its own story.

## 4. Intervene

- Sufficient: [patch](activation-patching.md) the clean activations into the corrupt run at the named sites. If the margin moves toward the clean one, and further than it does under the control patch, those sites are enough to produce the behavior in this pair.
- Necessary: in the clean run, ablate the same components with [zero or swap](zero-swap.md), `patch`, or [project_out](direction-edits.md). For a head, see [Attention heads](heads.md). If the clean margin falls well below its baseline, and falls more than it does under the control ablation, they are needed. The answer rarely flips, so look at the margin, not the winning token.
- Control: repeat each intervention at an unrelated site of the same kind, such as another head in the same layer or another token position. Compare the effect sizes.
- Undo: remove every edit newest first, then score again. The restored scores must equal the baseline.

## 5. Validate beyond the pair

Run the same interventions on new prompts and controlled variants, such as other names or other templates. A mechanism that works on one pair may only fit that pair. Report how many variants moved the way the hypothesis predicts and how many did not.

If you chose among many sites or settings, set aside held-out rows first and run one final evaluation on them, as [Evaluation reports and screens](evaluation.md) describes. A single pre-planned set of sites needs no split.

## 6. Caveats

- Representations can be distributed and overlapping. An ablation that changes nothing does not show a part is unused, because other parts can compensate. Ablate groups, such as all heads in a layer.
- An ablation that breaks the behavior does not show the part is the whole mechanism. Patch only the named components into the corrupt run and check that they restore the behavior by themselves.
- An attention pattern, a probe score ([Document-split probes](probes.md)) or a plot is a lead, not evidence. Confirm it with an intervention.
- Report the pair, the sites, the baseline, each intervention and its control, the restored result, the new prompts, forward passes, and what the evidence does not show. See the report item in [Worked example](worked-example.md).
