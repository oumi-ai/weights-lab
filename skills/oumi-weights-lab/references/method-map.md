# Choose a method

Pick a method from the question you are asking, not from the tools you know best. Each method answers one kind of question and is silent or misleading on others. A row's last column is what the method cannot tell you, so plan the next test before you trust its answer.

| Question | Method | What it shows | What it cannot show |
| --- | --- | --- | --- |
| At which layer does the answer appear? | [Logit lens](logit-lens.md) | When the residual stream, read through the final norm and head, favors the answer | Information present in a form the output head does not read; early layers read this way are unreliable |
| Which sites matter for this pair of prompts? | [Activation patching](activation-patching.md) | Whether one site's clean activation is enough to move the metric, or needed for it | Why it matters; effects that need several sites together; anything outside the contrast the pair isolates |
| Which of many sites should I patch first? | [Attribution patching](attribution-patching.md) | A first-order estimate of every site's patch effect, from one gradient pass | Large or saturating effects; it is a ranking to confirm with real patches |
| Which heads and MLPs write the answer directly? | `logit_attribution` in [Circuit analyses](circuit-analyses.md) | Each component's direct, linear contribution to the logit or logit difference | Effects through later layers; how the norm's scale changes when a part is removed |
| Does A's effect reach B through a given route? | `path_patch` in [Circuit analyses](circuit-analyses.md) | The effect along one route, with other heads frozen | A component's total effect; the answer depends on what is frozen |
| Which edges carry the behavior, and is a set of them enough? | `edge_attribution` and `find_circuit` in [Circuit analyses](circuit-analyses.md) | Ranked edges, then a real patching test of the chosen set with a faithfulness score | That the circuit is unique, or that it holds on prompts outside the pairs |
| What does this head read and write? | [Attention heads](heads.md) | Its attention pattern, its QK and OV weights, and what ablating it does | Use: a pattern is a lead until an intervention confirms it |
| Is a property linearly present at a site? | [Probes](probes.md) and `contrast_directions` in [Steering vectors](steering.md) | Whether the property is decodable along a direction, checked on held-out data | Whether the model uses that direction; a decodable direction can be ignored downstream |
| What does adding or removing a direction do? | [Steering vectors](steering.md) and [Remove a direction](direction-edits.md) | The causal effect of the direction on outputs | That the direction is the model's own variable; side effects you did not measure |
| Which sparse features are active, and what do they do? | [Sparse autoencoder features](saes.md) | A sparse decomposition of one site, with feature patches and steering | What the reconstruction error carries; labels are guesses until tested |
| How do features connect on one prompt? | [Attribution graphs](attribution-graphs.md) | Direct effects between transcoder features, tokens and logits in a replacement model | Computation in attention patterns, which are frozen; what error nodes carry; other prompts |
| How does one point respond to another? | `jacobian`, `jvp` in [Building blocks for new methods](building-blocks.md) | The local linear map between two recorded points | Behavior far from this input |
| None of these fits the question | [When current methods fall short](new-methods.md) | A method you design, tested on known answers first | Anything, until it has been tested |

## Combine them

A finished claim uses several rows: a cheap method to find candidates, an intervention to test them, and a control for each intervention. The [circuit workflow](circuit-workflow.md) gives the order and the protocol for a claim. [What a finished circuit looks like](worked-circuits.md) shows the result on public models.

## Common mistakes

- Stopping at the first component that matters. Measure how much of the behavior your components account for before you look away.
- Treating a probe hit, an attention pattern, or a feature label as the mechanism. Each is a lead.
- Ablating one part at a time and calling the rest unused. Redundant parts hide each other; ablate groups.
- Deciding a question cannot be answered because no row above answers it. Design the method instead.
