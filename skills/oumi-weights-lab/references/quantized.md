# Quantized models

`--quantize` loads the model with bitsandbytes so a large checkpoint fits in less memory. Quantized weights are read-only here: you can read, score, record, patch, and take activation gradients, but you cannot edit the packed weights.

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
- It needs the `quantized` extra, which the Claude Code plugin includes. Without bitsandbytes the start fails before any weights load, with the install command.
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
- `lab.edit`, `lab.zero`, `lab.swap`, and `lab.project_out` on a quantized weight raise `ValueError` and change nothing. `lab.export` is refused too; export from a float session.
- `lab.save_edits` and `--resume` work for float-parameter edits. Resume with the same `--quantize`, because the saved fingerprint covers the packed weights.

## Prequantized checkpoints

A checkpoint that bitsandbytes already quantized loads as it is, without `--quantize`. `lab.quantization["source"]` is then `"checkpoint"`. Passing a different `--quantize` is refused before loading. A GPTQ or AWQ checkpoint loads as it does today, and its packed weights refuse edits.

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
