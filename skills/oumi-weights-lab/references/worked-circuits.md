# What a finished circuit looks like

Two published results on GPT-2 small (`openai-community/gpt2`) show the depth a circuit claim should reach. Read them for the shape of the evidence, not to reuse their answers: a different model, task, or prompt set needs its own tests.

## Induction heads

**Behavior.** On a sequence of random tokens repeated twice, the model predicts the second copy almost perfectly, though it cannot predict the first.

**Mechanism.** Elhage et al. (2021) and Olsson et al. (2022) described it in small attention-only models. A previous-token head writes each token's predecessor into its position. An induction head uses that information in its keys (K-composition). From each token in the second copy, the induction head attends to the token after the earlier occurrence and copies it to the output. In GPT-2 small, later analyses name 4.11 as a previous-token head, and 5.1, 5.5, 6.9, 7.2, and 7.10 among the induction heads. The set depends on the score and threshold used; the IOI circuit below lists 5.5, 5.8, 5.9, and 6.9 as its induction heads.

**Evidence a claim like this needs:**
- **Lead:** attention from each second-copy token to the token after its first occurrence. Score every head this way ([Attention heads](heads.md)).
- **Necessity:** ablating the induction heads together raises the loss on the second copy, while a control ablation of other heads in the same layers does not.
- **Composition:** the effect runs from the previous-token head into the induction heads' keys, not their queries or values. Test it with path patching ([Circuit analyses](circuit-analyses.md)) or read it from the weights as a K-composition score.
- **Weights:** the induction heads' OV circuits copy tokens, and their QK circuits match a token against the previous-token information.

## Indirect-object identification

**Behavior.** "When Mary and John went to the store, John gave a drink to" is completed with " Mary", the name that appears once, not the repeated subject. The metric is the logit difference between the two names.

**Mechanism** (Wang et al., 2022). 26 heads in seven classes:
- **Duplicate-token heads** (0.1, 0.10, 3.0) and **induction heads** (5.5, 5.8, 5.9, 6.9), supported by **previous-token heads** (2.2, 4.11), mark the second mention of the subject as a repeat.
- **S-inhibition heads** (7.3, 7.9, 8.6, 8.10) move that signal to the last position, where it steers the next class away from the subject.
- **Name-mover heads** (9.6, 9.9, 10.0) attend to the remaining name and copy it to the logits.
- **Negative name movers** (10.7, 11.10) write against the answer.
- **Backup name movers** (9.0, 9.7, 10.1, 10.2, 10.6, 10.10, 11.2, 11.9) take over the name movers' job when those are ablated.

**Evidence that made it a claim:**
- **Path patching, working backward from the logits:** first the heads that affect the logits directly, then the heads that affect those heads' queries, keys or values, and so on.
- **A role for every class,** with attention patterns as leads and interventions as tests. For example, the name movers' OV circuits copy names, and patching the S-inhibition heads changes which name the name movers attend to.
- **Tests of the whole circuit.** Faithfulness: the circuit alone recovers most of the model's logit difference. Completeness: removing parts from the circuit and from the full model has similar effects. Minimality: every head contributes. The paper reports where these tests fall short.
- **Redundancy, found by ablating the name movers and watching other heads take over.** One-at-a-time ablations would have missed the backups, and the negative heads show that not every component pushes in the answer's direction.

## What a finished claim contains

1. The behavior, the metric, the prompt set, and the model.
2. Components grouped by role. Each role has a causal test, a control, and the size of its effect.
3. Faithfulness: how much of the behavior the components carry alone. Completeness: what happens when parts are removed from the circuit and from the full model.
4. A search for redundancy and opposition: backup components, and parts that push against the answer.
5. The hypotheses that failed, with the tests that ruled them out.
6. Checks on prompts that were not used to find the circuit.
7. What the evidence does not show.
