# Find and test a circuit

A circuit claim says a set of components carries a behavior. This recipe is the protocol for making one: locate candidates, state a hypothesis, test it causally, measure how much of the behavior it explains, try to prove it wrong, and report what failed along with what held. Each step names the recipe that does the work; [Choose a method](method-map.md) says what each method can and cannot show. [What a finished circuit looks like](worked-circuits.md) shows the result on public models.

## 1. Pick a behavior and a contrast pair

Choose a clean prompt and a corrupt prompt that differ in one token and have different answer tokens, such as "The capital of France is" with " Paris" and "The capital of Germany is" with " Berlin". Check that both tokenize to the same length. Score both prompts with `score` and keep the baseline. A useful measure is the clean answer's score minus the corrupt answer's score:

```python
def margin(prompt):
    s = lab.score(prompt, [" Paris", " Berlin"], batch_size=1)
    return s[" Paris"] - s[" Berlin"]


baseline = {"clean": margin(clean_text), "corrupt": margin(corrupt_text)}
```

If the two baselines are close, the pair does not isolate the behavior. Pick another pair. Build a set of pairs from templates as well, because step 6 needs prompts the search did not use.

## 2. Locate candidates

- [Logit lens](logit-lens.md): find the layer where the answer first becomes the top token.
- `logit_attribution` ([Circuit analyses](circuit-analyses.md)): find the heads and MLPs that write the answer directly.
- [Attribution patching](attribution-patching.md) and `edge_attribution`: rank sites and edges. The ranking is an estimate.
- [Attention heads](heads.md): capture attention with `record(..., at="attention")`.
- [Sparse autoencoder features](saes.md) and [Attribution graphs](attribution-graphs.md): find features and how they connect on one prompt.

These steps produce leads, not results. Keep going past the first strong lead: list every candidate that could matter, not just the one that fits your first idea.

## 3. State the hypothesis and its rivals

Write down which components you think carry the behavior and what each one does. Then write at least one rival hypothesis that fits the leads as well, such as "a different head with the same attention pattern does the copying", or "the direction the probe found is decodable but unused". For each intervention below, write what you expect under each hypothesis. Do this before running anything, so the result cannot choose its own story.

## 4. Test it causally

- **Sufficient:** [patch](activation-patching.md) the clean activations into the corrupt run at the named sites. If the margin moves toward the clean one, and further than under a control patch, those sites are enough in this pair.
- **Necessary:** in the clean run, ablate the same components with [zero or swap](zero-swap.md), `patch`, or [project_out](direction-edits.md). If the clean margin falls well below its baseline, and further than under a control ablation, they are needed. Look at the margin, not the winning token.
- **Routes:** where the hypothesis says A feeds B, test that route with `path_patch` ([Circuit analyses](circuit-analyses.md)).
- **Controls:** repeat each intervention at an unrelated site of the same kind, such as another head in the same layer.
- **Undo:** remove every edit newest first, then score again. The restored scores must equal the baseline.

## 5. Measure how faithful and complete it is

- **Faithfulness:** run the model with everything outside your components set to its corrupt value, and see how much of the clean-minus-corrupt margin remains. `find_circuit` reports this for a set of edges. A claim that carries 30% of the behavior is a partial claim; say so.
- **Completeness:** for a few subsets of your components, remove them from the circuit-only run and from the full model. If the full model barely moves while the circuit collapses, something outside your set compensates.
- **Minimality:** remove each component from the circuit-only run. One whose removal changes nothing does not belong in the claim.

## 6. Try to prove it wrong

- Run the interventions that separate your hypothesis from its rivals.
- **Look-alikes:** check every component that matches your lead, such as all heads with the same attention pattern and every direction a probe finds. Ablate each alone. Only the ones whose ablation matters belong in the claim.
- **Redundancy:** when one ablation changes little, ablate pairs and groups. Backup components hide each other.
- **Opposition:** look for parts that push against the answer; they change what faithfulness means.
- **New prompts:** rerun the tests on prompts and templates the search did not use, and count how many moved as predicted.
- If you chose among many sites or settings, run one final evaluation on held-out rows, as [Evaluation reports and screens](evaluation.md) describes.

## 7. Report what held and what failed

Report the behavior and metric, the prompt sets, each component's role with its causal test, control and effect size, faithfulness and completeness, the redundancy search, the rival hypotheses and the tests that ruled them out, the restored result, forward passes, and what the evidence does not show. A failed hypothesis is a result; list it. See the report item in [Worked example](worked-example.md).

If no method answers a step's question, don't drop the step: see [When current methods fall short](new-methods.md).
