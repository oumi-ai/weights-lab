# Remove a direction from a weight

`project_out` removes one direction from a weight or bias with a single `edit`. Undo, `edits()`, and saved bundles work as they do for any edit. Take the direction from a probe, a difference of means, or a recorded activation, and check it has the length of the axis you project: for `Linear` and `Conv1D` weights that is the output size on the output side and the input size on the input side, and for an embedding it is the hidden size.

## Remove it from chosen neurons of `down_proj`

`down_proj.weight` is `[hidden, intermediate]`. The output side removes the direction from what each selected neuron writes. `index` picks the neurons, which are the input columns.

```python
name = "model.layers.12.mlp.down_proj.weight"  # Verify on this model.
neurons = sorted(chosen_neurons)[:64]
edit_id = lab.project_out(name, direction, index=neurons, note="drop refusal direction")
```

Leave `index` out to change every column. A slice with step 1 or an int also works.

## Remove it from the module output

A bias also writes into the output. Project it too, or the module output keeps a constant component along the direction.

```python
module = "transformer.h.5.mlp.c_proj"
first = lab.project_out(f"{module}.weight", direction)
second = lab.project_out(f"{module}.bias", direction)
```

Models whose projections have no bias need only the weight.

## A rank-one edit with plain `edit`

For anything `project_out` refuses, build the change yourself.

```python
def rank_one(block):
    update = torch.outer(u.to(block.device, block.dtype), v.to(block.device, block.dtype))
    return block - update


lab.edit(name, rank_one, note="rank-one update")
```

## Tied embeddings

Editing the input embedding changes a tied output head too, because both names point at one tensor. `lab.parts()` lists every alias but does not mark them. Check whether two names share storage with `lab.model.get_parameter(a) is lab.model.get_parameter(b)`, and decide whether you want both changed.

```python
rows = lab.project_out("model.embed_tokens.weight", direction, index=[2, 3])
```

## Adapters

A PEFT `lora.Linear` wrapper is not a `Linear`, so `project_out` refuses it. Edit its children instead: `...base_layer.weight`, `...lora_A.default.weight`, or `...lora_B.default.weight`.

## Check, then undo

Record the module output before and after, and compare the component along the direction. The module has a bias, so project both the weight and the bias, then undo both.

```python
unit = direction.float() / direction.float().norm()
with lab.record(module) as before, torch.no_grad():
    lab.model(**inputs)
lab.project_out(f"{module}.weight", direction)
lab.project_out(f"{module}.bias", direction)
with lab.record(module) as after, torch.no_grad():
    lab.model(**inputs)
print((before[module][0] @ unit).abs().max(), (after[module][0] @ unit).abs().max())
lab.undo()
lab.undo()
```

The component falls to about zero only because both were projected. Each undo restores the original dtype exactly.
