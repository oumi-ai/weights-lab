# Sparse autoencoder features

`lab.load_sae` loads a public sparse autoencoder (SAE) or transcoder for one hook point of the session's model. You can then read features from activations, and patch or steer chosen features. It reads through SAELens, which needs the `saes` extra:

```sh
uv tool install 'oumi-weights-lab[saes] @ git+https://github.com/oumi-ai/weights-lab'
```

The Claude Code plugin and remote workers include the extra, so it is already there. In a CLI installed without it, `load_sae` raises an `ImportError` that names it.

## Load one and check it first

```python
sae = lab.load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre")  # release, SAE id
print(sae, sae.module, sae.at, sae.model_check)
```

The release and id come from SAELens's list of public SAEs. A local directory in SAELens's format loads with `lab.load_sae(path="/abs/dir")`. The call refuses a hook name the lab cannot map, a width that is not the model's, and an SAE trained on a different model. It does this before any forward pass.

A mismatch between how the dictionary was trained and what the lab records does not raise. It shows up as a large reconstruction error. Measure that error on ordinary text before you trust any feature:

```python
check = sae.features(prompts, positions=slice(1, None))  # skip position 0, the BOS token
print(check["fvu"], check["l0"], check["rows"])
```

- `fvu` is the fraction of variance the SAE leaves unexplained. Compare `1 - fvu` with the variance explained the release reports, which SAELens lists beside each SAE. Near 1 or above means the SAE explains no more than the mean does; suspect the site or its transform first, since some releases are also just poor at some layers.
- `l0` is the mean number of active features per token. It should be near the release's reported L0.
- The BOS token's activations are far larger than other tokens', so an `fvu` over all positions can look good while the real text is poorly explained. Check without position 0.
- Text gets a BOS token first when the SAE was trained with one. Compare `sae.prepend_bos`.
- If `fvu` is far off, the SAE may record nothing about the model processing it was trained under. Pass what you know, for example `lab.load_sae(..., model_kwargs={"center_writing_weights": True})` for a GPT-2 SAE trained on mean-centered residuals, or `{"fold_ln": True, "fold_value_biases": True}` for an attention-output SAE, and measure again.
- A release can also record the wrong input hook. Gemma Scope 2 transcoders record `blocks.L.hook_mlp_in` but read the MLP's own input, and `fvu` comes out in the thousands. Load with `hook_name="blocks.L.mlp.hook_in"` to override the recorded hook (`sae.recorded_hook_name` keeps it), and measure again.

## Read the top features

```python
import torch

result = sae.features("The Eiffel Tower is in", positions="all")
strength = result["features"].max(dim=0).values  # one number per feature
top = strength.topk(10)
print(top.indices.tolist(), top.values.tolist())
print(sae.labels(top.indices.tolist()))
```

`labels` asks Neuronpedia for each feature's explanation, and returns `None` for a feature with no label, an SAE with no Neuronpedia id, or a network that does not answer. Found labels are saved in the session's `artifacts/saes`. A label is a machine-written guess. Read the text a feature fires on before you describe what it does.

`sae.encode(activations)` and `sae.decode(features)` work on tensors you record yourself. `encode` takes an activation as `lab.record(sae.module, at=sae.at)` returns it and applies the site's transform.

## Patch and steer

`patch` sets features and keeps everything else, including the reconstruction error. It changes the activation by the difference in the chosen features times their decoder rows, and by nothing else. A patch that changes no feature leaves the model's outputs unchanged to the bit.

```python
prompt, options = "The Eiffel Tower is in", [" Paris", " Rome"]
baseline = lab.score(prompt, options)
with sae.patch({1234: 0.0}, positions="all"):  # set feature 1234 to zero everywhere
    ablated = lab.score(prompt, options)
with sae.patch({1234: lambda value: value * 2}, positions="all"):  # or a function of its value
    doubled = lab.score(prompt, options)
with sae.steer({1234: 4.0}, positions="all"):  # add 4 x its decoder row
    steered = lab.score(prompt, options)
```

- Decoder rows are not unit norm unless the SAE rescales by decoder norm. Check `sae.decoder_rows([1234]).norm(dim=-1)` before comparing scales across features.
- A transcoder reads the MLP's input and writes the MLP's output, so patching it changes what the MLP adds to the residual stream.
- `hook_resid_mid` and normalized sites such as `ln2.hook_normalized` can be read but not patched, because no Hugging Face module output carries them exactly. `patch` says so.
- Positions work as in `lab.patch`. The default is the last token of each call.

Treat a feature effect like any other intervention: measure the baseline, run a control that patches random features of similar activation, undo, and measure again. A feature that fires on a concept does not show the model uses it for that concept; the patch is the test.

## Which models

Hook names that map are `blocks.L.` followed by `hook_resid_pre`, `hook_resid_post`, `hook_resid_mid`, `hook_mlp_in`, `hook_attn_out`, `hook_mlp_out`, `attn.hook_z`, `ln2.hook_normalized`, or `mlp.hook_in`. These work for models whose attention `heads` can list: GPT-2, and Llama, Qwen, and Gemma 2 style blocks. Gemma 2's attention and MLP outputs are read after its extra post-norms, as its residual stream receives them. An SAE trained on a fine-tune of the session's model needs `allow_model_mismatch=True`. A session with a LoRA adapter is refused.
