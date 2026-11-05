# Logit lens

Read what each layer would predict by applying the model's final norm and output head to the layer's output.

Record every decoder layer's output at the last position, then project with `torch.no_grad()`.

```python
import torch

inputs = tokenizer("The capital of France is", return_tensors="pt").to(lab.device)
# Llama and Qwen: model.layers.N. GPT-2: transformer.h.N.
layers = [
    name
    for name, _ in model.named_modules()
    if name.count(".") == 2 and name.split(".")[-2] in ("layers", "h")
]
final_norm = model.get_submodule("model.norm" if hasattr(model, "model") else "transformer.ln_f")
head = model.get_output_embeddings()

with torch.no_grad(), lab.record(layers, positions=[-1]) as recorded:
    logits = model(**inputs).logits[0, -1].detach().cpu()
with torch.no_grad():
    lens = {
        name: head(final_norm(recorded[name][0].to(lab.device))).detach().cpu()[0, 0]
        for name in layers
    }
top = {name: tokenizer.decode(values.argmax().item()) for name, values in lens.items()}
```

Self-check before reading anything into the lens: the last layer's lens must reproduce the model's logits.

```python
assert torch.allclose(lens[layers[-1]], logits, atol=1e-4)
```

If it fails, the architecture does not end with the norm and head alone. Some put the last norm inside the last layer, or add a scale or softcap after the head. Fix the projection until the check passes.

Earlier layers are read off a space they were not trained to be read in, so a lens at early layers is a hint, not a measurement. Compare the same prompt against a control prompt, and report the layer where the answer first becomes the top token.
