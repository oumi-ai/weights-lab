# Worked example: remove the first layer's contributions

Question: does removing the first transformer layer's attention and MLP contributions change short continuation preferences? This is a diagnostic on four hand-written dev items. It is not an estimate of a published benchmark.

Use the public [SmolLM2-135M base checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M). Its [configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json) identifies a Llama causal model; the paths below are architecture-specific. CPU/float32 keeps the example accessible. The first download needs network access and disk space for the checkpoint. Record a Hub commit with `--revision` when exact checkpoint reproducibility matters.

Save this as `first-layer.py` in your working directory:

```python
import json
from pathlib import Path

notes = {
    "question": "Do first-layer attention and MLP contributions change continuation preferences?",
    "intervention": "Zero both output projections in layer 0, then restore exactly.",
    "control": "Repeat the baseline after undo in the same loaded session.",
    "final_model": "Original weights; reserved held-out examples run only after restoration.",
    "scope": "Four hand-written dev items and two separate held-out items; no benchmark claim.",
}
(artifacts / "notes.json").write_text(json.dumps(notes, indent=2))
dev = [
    {
        "prompt": "The capital of France is",
        "options": {"A": " Paris", "B": " Berlin"},
        "answer": "A",
    },
    {
        "prompt": "The capital of Germany is",
        "options": {"A": " Paris", "B": " Berlin"},
        "answer": "B",
    },
    {"prompt": "A dog is an", "options": {"A": " animal", "B": " ocean"}, "answer": "A"},
    {"prompt": "Two plus two equals", "options": {"A": " four", "B": " nine"}, "answer": "A"},
]
heldout = [
    {"prompt": "The opposite of hot is", "options": {"A": " cold", "B": " tall"}, "answer": "A"},
    {"prompt": "A cat is an", "options": {"A": " ocean", "B": " animal"}, "answer": "B"},
]
(artifacts / "dev.json").write_text(json.dumps(dev, indent=2))
(artifacts / "heldout.json").write_text(json.dumps(heldout, indent=2))
names = [
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
]
print([lab.measure(name) for name in names])
baseline = lab.evaluate(dev, name="baseline-v1", max_passes=8)
ids = []
try:
    for name in names:
        ids.append(lab.zero(name, note="Remove first-layer residual contribution"))
    ablated = lab.evaluate(dev, name="ablated-v1", max_passes=8)
finally:
    for edit_id in reversed(ids):
        lab.undo(edit_id)
restored = lab.evaluate(dev, name="restored-v1", max_passes=8)
final = lab.evaluate(heldout, name="heldout-final", held_out=True, max_passes=4)


def items(run):
    return [
        json.loads(line) for line in (Path(run["run_dir"]) / "items.jsonl").read_text().splitlines()
    ]


before, changed, after = items(baseline), items(ablated), items(restored)
report = {
    **notes,
    "model_id": lab.model_id,
    "device": str(lab.device),
    "parameter_names": names,
    "baseline_accuracy": baseline["accuracy"],
    "ablated_accuracy": ablated["accuracy"],
    "restored_accuracy": restored["accuracy"],
    "changed_items": lab.diff(baseline, ablated),
    "maximum_option_score_delta": max(
        abs(old["scores"][key] - new["scores"][key])
        for old, new in zip(before, changed)
        for key in old["scores"]
    ),
    "restored_scores_identical": all(
        old["scores"] == new["scores"] for old, new in zip(before, after)
    ),
    "heldout_final_accuracy": final["accuracy"],
    "heldout_size": len(heldout),
    "active_edit_ids": lab.active_edit_ids,
    "edit_history": lab.edits(),
    "forward_passes": lab.forward_passes,
}
(artifacts / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
```

Run from the installed package environment, choosing a fresh session path:

```bash
weights-lab start --session sessions/first-layer --model HuggingFaceTB/SmolLM2-135M \
  --device cpu --dtype float32 --snippet-timeout 300 --pass-limit 64
weights-lab send --session sessions/first-layer --file first-layer.py --follow --timeout 600
weights-lab peek --session sessions/first-layer
weights-lab stop --session sessions/first-layer
weights-lab backup --session sessions/first-layer --output first-layer.tar.gz
```

Inspect `sessions/first-layer/artifacts/report.json` and the per-item scores. The report measures score movement even if no winning labels change. It records the undo control and exposes any failure to restore the original scores. Expected execution uses 28 forwards; this is a pass-count expectation, not a predicted accuracy. This document contains no measured outcome. The saved worker records and report supply the evidence for the actual run.
