# Replay the released Qwen weight edit

The [replay script](../examples/replay_qwen_edit.py) replays released tensor values using `lab.swap`, verifies their hashes, and undoes them. It does not rediscover a direction or reproduce the research benchmark. Attribution: Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), Experiment 8. The [released checkpoint card](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming/blob/d1671a178705bcab7644806d23c535517f35723b/README.md) links the rebuild and evaluation provenance.

The pinned [upstream edit manifest](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/manifest.json) identifies two BF16 `[1024, 3584]` tensors. Both have 64 changed columns; 65,260 entries differ in layer 15 and 64,928 in layer 19. Thus 130,188 entries differ bitwise, while the research describes 131,072 touched positions. BF16 rounding accounts for the distinction noted in the manifest.

| Source | Pinned revision |
| --- | --- |
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` |
| `RikiyaT/Qwen3.5-0.8B-weight-space-programming` | `d1671a178705bcab7644806d23c535517f35723b` |
| `RikiyaT/lm_understand-logs` | `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` |

| Parameter | SHA256 of released raw BF16 bytes |
| --- | --- |
| `model.language_model.layers.15.mlp.down_proj.weight` | `82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53` |
| `model.language_model.layers.19.mlp.down_proj.weight` | `8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2` |

The script independently checks these facts when run. It hashes the safetensors' original dtype bytes, not float32 conversions. The default audit compares every tensor name, shape, dtype, and raw hash across both pinned checkpoints; it rejects any changed tensor outside the two named above. This reads both checkpoint files, about 1.75 GB each, through the Hugging Face cache. The base download is shared with normal model loading. Audit uses CPU tensor views and never instantiates a second model. Two replacement matrices and small edited-column undo snapshots remain in memory.

From the repository checkout, in the installed package environment:

```bash
weights-lab start --session sessions/qwen-replay \
  --model Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --device cpu --dtype bfloat16 --pass-limit 0 --snippet-timeout 3600
weights-lab send --session sessions/qwen-replay \
  --file examples/replay_qwen_edit.py --follow --timeout 3600
weights-lab peek --session sessions/qwen-replay
weights-lab stop --session sessions/qwen-replay
weights-lab backup --session sessions/qwen-replay --output sessions/qwen-replay.tar.gz
```

The loader needs a Transformers version that supports `Qwen3_5ForConditionalGeneration`. The model is the multimodal release; this recipe uses its language-model parameters and supplies no images. Custom Hub code remains disabled. Model loading needs sufficient RAM even though the default replay performs no forwards.

Successful execution produces `artifacts/qwen-replay/verification.json` and `report.json`, including actual hashes, changed columns, edit IDs, and `restored_exactly`. The script refuses active edits or attachments, verifies the live target tensors before changing them, swaps the selected columns, verifies the resulting full tensors, then undoes in reverse order. The final live tensors must match the pinned base hashes. These files report the actual run; no full checkpoint audit or benchmark result is bundled here.

For a faster target-only check, submit `qwen_replay = {"audit_all": False}` in an earlier snippet. The report then says `tensor_count_checked: 2`; it does not establish that every other checkpoint tensor is unchanged. Use a new `run_prefix` for another replay in the same session.

## Optional sampled diagnostic

Create a new session with the same pinned model, a suitable GPU/device, a positive session pass limit, and enough snippet time for generation. Provide a dev JSONL file with `prompt` and `answer` fields and define a checker callable `check_answer(generated, expected)` in a prior snippet. Grading must match the task; exact string equality is usually insufficient for mathematical answers. Then configure:

```python
qwen_replay = {
    "data": "/absolute/path/my-dev.jsonl",
    "checker": check_answer,
    "max_passes": 512,
    "run_prefix": "qwen-diagnostic-v1",
    "generation_config": {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": 128,
    },
    "seed": 0,
}
```

Submit the same replay file. It applies the pinned tokenizer's thinking-enabled chat template and evaluates baseline, edited, and restored conditions with the same seed and configuration. Each condition gets the stated pass cap; size it for the number of rows and maximum generation length, and reserve three times that amount in the session cap. Answers and summaries remain under `artifacts/evaluations`; changed items are included in the replay report.

This short Transformers diagnostic omits the upstream presence penalty and uses a much shorter generation limit and a different execution backend. It is not the published evaluation protocol. To reproduce that protocol, follow the authors' pinned [evaluation source](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/evaluate.py), dataset versions, grading, sample counts, and generation settings. Local score changes support only the diagnostic actually run; replaying known edits does not test autonomous discovery.
