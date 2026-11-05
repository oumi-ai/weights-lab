# Quantized models

`--quantize` loads the model with bitsandbytes so a large checkpoint fits in less memory. You can read, score, record, patch, and take activation gradients on the packed weights. nf4 and fp4 weights also edit, zero, swap, and project out, and undo restores them byte for byte. int8 weights are read-only.

## Choose a format

- `nf4` is the usual choice: a 4-bit code shaped for normally distributed weights, about a quarter of the 16-bit size.
- `fp4` is the other 4-bit code. Use it only to compare against nf4.
- `int8` is LLM.int8, about half the 16-bit size.

All three leave embeddings, norms, biases, and `lm_head` in float. Expect small score differences from the float model, and measure the unquantized model in a second session as a control.

## Load one

```sh
oumi-weights-lab start --session sessions/q --model MODEL_ID --quantize nf4 --dtype bfloat16
```

- On a GPU (`--device cuda`) or an Apple-silicon Mac on macOS 14 or later with torch 2.5 or later (`--device mps`), loading works as for a float model. `--device cpu` works too and is slow.
- `--quantize` cannot be combined with `--device-map`: the model must fit on one device.
- It needs the `quantized` extra, which the Claude Code plugin includes. The plugin leaves it out on Intel Macs and on macOS before 14, where bitsandbytes has no wheel, and `--quantize` there fails with that explanation. Without bitsandbytes the start fails before any weights load, with the install command.
- Layers compute in `--dtype`. With the default, that is the checkpoint's float dtype.

`lab.quantization` shows what loaded, and `peek` reports its `method` and `compute_dtype`:

```python
print(lab.quantization)
print([p for p in lab.parts("*layers.0.self_attn.q_proj*")])
```

## What works

- `lab.parts` and `lab.measure` report the logical shape and dtype, and add a `quantization` block with the packed layout. `measure` and `lab.compare` use the dequantized values, so a quantized model compares against its float original. Layers that are `Conv1D` in the float model, such as GPT-2's, are stored `[out, in]` once quantized, so those show the transposed shape and comparing them with the float model raises a shape mismatch.
- `lab.evaluate`, `lab.score`, generation, `lab.record`, `lab.patch`, `lab.heads`, and `lab.head_weights` work. `lab.gradient` skips quantized weights and returns the float ones; asking for a quantized weight by name is refused. Activation gradients with `modules=` work through quantized layers.
- Edits to the float parameters (embeddings, norms, biases, `lm_head`) work and undo exactly.
- `lab.edit`, `lab.zero`, `lab.swap`, and `lab.project_out` work on nf4 and fp4 weights; see Editing 4-bit weights below. On int8 weights they raise `ValueError` and change nothing.
- `lab.export` is refused; export from a float session.
- `lab.save_edits` and `--resume` work for float-parameter edits, and `save_edits` refuses while an edit to a packed weight is active. Resume with the same `--quantize`, because the saved fingerprint covers the packed weights.

## Editing 4-bit weights

An edit on an nf4 or fp4 weight takes the same arguments as on a float weight, with the logical index (`[out, in]`):

```python
edit_id = lab.zero("model.layers.3.mlp.down_proj.weight", index=(slice(None), [17, 40]))
print(lab.edits()[-1]["quantization"])
lab.undo(edit_id)
```

- The weight is stored in blocks of 64 consecutive weights in row-major order, each with its own scale. An edit changes only the blocks that hold a changed value, so a block is the most it can disturb. A column edit touches one block per row.
- Inside a touched block, the values you did not select can still move. If your edit changes the block's largest value, the scale changes and every other value in the block is rounded again. `collateral_l2` is how far those moved, and `realized_delta_l2` is how far the selection itself moved. Compare both with `requested_delta_l2`. If `collateral_l2` is a large share of `realized_delta_l2`, make smaller changes or edit whole blocks.
- A 4-bit block has 16 levels, so a small change can round back to the stored value. The edit then warns that the change was smaller than the quantization step, and `realized_delta_l2` is 0. Use a larger change, and do not read a null result from such an edit.
- Zeroing is exact. A zeroed block dequantizes to 0 and stays finite.
- Head ablation works as on a float model: `lab.zero(**part)` for each part of `lab.heads()[i]["o"]` zeroes the head's `o_proj` columns.
- `lab.undo()` restores the packed bytes and scales exactly, newest edit first, so logits return bit for bit.

Edits are refused, and nothing changes, for a checkpoint that was saved with double quantization (`bnb_4bit_use_double_quant`): its scales are shared across the whole tensor, so no edit can stay inside a block. Reload the float checkpoint with `--quantize nf4` instead. A weight that bitsandbytes repacked for CPU kernels is refused too; sessions turn that repack off when they load a model.

## Prequantized checkpoints

A checkpoint that bitsandbytes already quantized loads as it is, without `--quantize`. `lab.quantization["source"]` is then `"checkpoint"`. Passing a different `--quantize` is refused before loading. A GPTQ or AWQ checkpoint loads as it does today, and its packed weights refuse edits. A prequantized nf4 checkpoint saved with double quantization refuses edits too.

## Hub kernels

bitsandbytes can download kernel code from the Hugging Face Hub when the optional `kernels` package is installed. Sessions block that by default and bitsandbytes uses its built-in kernels. Pass `--allow-hub-kernels` to let it run, and the run's `setup` block records `allow_hub_kernels`.

## Compare with the float model

Start one session per setup, run the same rows in both, and pair the runs with `cross_session=True`. The setup difference lists `quantization`:

```python
float_run = float_lab.evaluate(rows, name="float-v1")
quant_run = quant_lab.evaluate(rows, name="nf4-v1")
report = quant_lab.compare_runs(float_run, quant_run, cross_session=True)
print(report["cross_session"]["setup_differences"])
```

Near-tie predictions can flip between setups, so read accuracy changes against that control.
