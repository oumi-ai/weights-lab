# Attribution graphs

`lab.attribution_graph` shows how the model produced its answer on one prompt, as a graph of features. You read it as JSON, pick a path that looks important, and test that path with a real intervention. It needs the `saes` extra and one transcoder for every MLP layer; see [Sparse autoencoder features](saes.md) for loading them.

## What the graph shows

The nodes are the tokens of the prompt, the transcoder features that fire at each layer and position, one error node per layer and position, and the top next tokens. An edge is the direct effect of one node on another: the source's value times how much the target changes per unit of the source. A target is a feature's input (its pre-activation) or a logit minus the mean logit.

The graph describes a replacement model, not the model itself. In that model each MLP's output is its transcoder's features times their decoder rows, plus an error node that holds whatever the transcoder misses. Attention patterns and norm denominators are held at the values the real model had on this prompt. Every forward value is the same as in the real model. The call checks this before it returns a graph, and raises if the replacement model's logits differ from the model's, or if a target's edges plus its constant part do not add up to its value.

What it cannot show:

- **Attention.** The patterns are frozen, so an effect that works by changing what a head attends to does not appear. Edges show what flows through the heads, not why they attend there.
- **What the transcoders miss.** A large error node, or a low `replacement_score`, means much of the computation runs through the part the transcoders did not explain. Do not describe a mechanism from the features alone then. A skip transcoder's skip connection is not part of the error: it is a path from the MLP's input to its output, with edges like any other.
- **Features beyond `max_features`.** If you cap them, the rest are folded into the error nodes.
- **Causation in the real model.** An edge is a direct effect in the replacement model. Only an intervention tests the real one.

## Build one

```python
tcs = [lab.load_sae(release, sae_id) for sae_id in sae_ids]  # one transcoder per layer
graph = lab.attribution_graph("The capital of France is", tcs)
print(graph.metadata["replacement_score"], graph.metadata["checks"])
graph.to_json(lab.artifacts / "graph.json")
```

Needs a float32 session, and one transcoder per MLP layer that reads the MLP's input and writes its output. A list or a dict from layer to transcoder both work. A missing layer, a plain SAE, a transcoder that normalizes its input at run time, an architecture `heads` cannot list, and a model with a layer that has no softmax attention are refused with a message that says which.

It takes 3 forward passes however long the prompt is. Only the 10,000 features with the largest activations become nodes by default (`max_features`); the rest fold into the error nodes, so read the replacement score with that in mind. Memory grows with the square of the node count: the call estimates it after the first pass and refuses a graph over `max_memory_gb` (8 by default), naming `max_features`. Lower it for a smaller graph.

Check each transcoder with `sae.features(...)["fvu"]` first, as in the SAE recipe. A transcoder that reads the wrong site gives a graph full of error. Gemma Scope 2 transcoders record their input as `blocks.L.hook_mlp_in` but read the MLP's own input; `load_sae` maps that name to the raw residual stream, and `fvu` shows it as a huge number. Load them with `hook_name="blocks.L.mlp.hook_in"`, which overrides the recorded hook, then check `fvu` again:

```python
tc = lab.load_sae(release, sae_id, hook_name=f"blocks.{layer}.mlp.hook_in")
```

## Read the JSON

Nodes:

- `id`: `emb.P` for the token at position P, `err.L.P` for layer L's error at P, `feat.L.P.F` for feature F at layer L and position P, `logit.T` for token id T.
- `kind`, `layer`, `position`, `token`, `token_id`, `feature`; logit nodes also have `probability`.
- `activation`: a feature's value, the size of an embedding or error vector, or a logit minus the mean logit.
- `influence`: the node's share of everything that reaches the logit nodes. Sort by it to see what matters.
- `bias` (features and logits): the part of the node's value no source accounts for. Its incoming edges plus `bias` equal `activation`, unless pruning removed some edges.
- `label`, only with `labels=True`: Neuronpedia's explanation, when the transcoder has one. This asks the network, one request per feature without a cached label, so it is off by default. It is a machine-written lead. Read the text a feature fires on before you trust it.

Edges are `{source, target, weight}`. A positive weight pushes the target up on this prompt, a negative one pushes it down. `metadata` has the tokens, the settings, the checks, and `replacement_score`.

```python
data = graph.to_dict()
logit = next(n for n in data["nodes"] if n["kind"] == "logit")
into = [e for e in data["edges"] if e["target"] == logit["id"]]
for edge in sorted(into, key=lambda e: -abs(e["weight"]))[:5]:
    print(edge, graph.node(edge["source"]))
for path in graph.top_paths(logit["id"], 3):
    print(path["score"], path["nodes"])
```

By default the graph is pruned to the nodes that carry 80% of the influence and the edges that carry 98%, scored on the pruned graph. Tokens and logit nodes always stay, with each logit's strongest parent, and a feature or error with no edge left is dropped. For the checks on edge sums, build once with `node_threshold=1.0, edge_threshold=1.0`, which keeps everything.

## Test a path

An edge is a hypothesis. Test it by setting a feature in the real model and measuring the effect, with a control.

```python
result = graph.measure(["feat.12.4.317"])  # sets the feature to 0 at its position
control = graph.measure(["feat.12.4.902"])  # a control: same layer and position, weak edge
top = next(n["id"] for n in graph.nodes if n["kind"] == "logit")
print(result["effect"][top], control["effect"][top])
```

`measure` runs the model on `graph.inputs` twice, once as it is and once with the feature set, and returns each logit node's value (logit minus the mean logit) before and after. Pick the control from the graph: a feature at the same layer and position with a much weaker edge to the logit. If the target feature moves the logit and the control does not, the path matters on this prompt. If both move it, or neither, the graph's suggestion did not hold up in the real model.

`measure` keeps the transcoder's error, so the change is exactly the feature's decoder row times the change in value. The graph's edge is the replacement model's prediction of that effect; the real effect also passes through later layers, which re-read the changed stream, so expect the two to differ and report the measured one. Setting the feature to its own value (`value=node["activation"]`) is a check that changes nothing.

Do not pair `graph.intervene` with `lab.score` or `lab.evaluate`. The graph's positions index `graph.inputs`, which include a BOS token that the transcoders ask for and that `score` does not add, so a score run would patch the wrong tokens. `intervene` raises when the tokens differ. Use it only to run your own forward pass on the graph's tokens:

```python
with graph.intervene(["feat.12.4.317"]):
    logits = lab.model(**graph.inputs).logits  # any other tokens raise
```

A graph written with `to_json` and read back with `AttributionGraph.from_json` holds no model and cannot `measure` or `intervene`. Rebuild it from the lab.
