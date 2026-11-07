# When current methods fall short

Sometimes no method in [Choose a method](method-map.md) answers the question, or the ones that apply disagree. Don't narrow the question to fit the tools, and don't report that it can't be answered. Design a method, test it where the answer is known, and only then use it on the real question.

## 1. Say what the current methods cannot show

Write down the question, the methods you tried or ruled out, and the specific thing each cannot show for this question. For example: "Activation patching shows layer 6 matters for this pair, but not which of the two features at layer 6 the later heads read." The gap is the method's specification.

## 2. Design it as one function

Write the method as a plain function whose first argument is `lab`, built from [Building blocks for new methods](building-blocks.md) and the existing calls: `run_with`, `sweep`, `jacobian`, `jvp`, `vjp`, `record`, `patch`, and `gradient`. Keep it small enough to read in one screen. Put its assumptions in the docstring, such as "assumes the feature is linear at the site" or "needs prompts of equal length".

```python
def rank_by_head_jvp(lab, prompt, direction, layer):
    """How much each later head's output moves along `direction` added at `layer`. Linear, local."""
    ...
```

## 3. Test it on known answers

Build dev tasks with the [proving ground](proving-ground.md), whose answers are planted and known. Run your method on them, turn what it finds into a claim, and grade it. Run an existing method on the same tasks the same way.

- Use several tasks and more than one family. A method that wins on one task has not shown much.
- Never use held-out tasks. They measure the skill, and tuning on them spoils that.
- If your method needs a property the planted tasks lack, say so and build a small check that has it, with a known answer.

## 4. Report where it wins and where it loses

Give the scores of both methods per task, and the paired comparison. Say which tasks it fails and why you think it fails. A method that loses on known answers is not evidence on the real question; change it or drop it.

## 5. Use it, then confirm

Run the method on the real question. Treat its output as you would any ranking: a lead. Confirm the top results with the interventions and controls in the [circuit workflow](circuit-workflow.md).

To use the method again in a later session, save it with its proving-ground results. See [Save and reuse a method](saved-methods.md).
