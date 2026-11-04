"""Exercise a public tiny GPT-2 model through the detached local worker on CPU.

Run: uv run python -m examples.dogfood --output-dir /tmp/oumi-weights-lab-dogfood
The first run downloads a small public checkpoint. All experiment files, the
exported checkpoint, CLI transcript, and stopped-session backup stay in output-dir.
"""

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

MODEL_ID = "sshleifer/tiny-gpt2"
MODEL_REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"

BASELINE = """
import json
import torch
import transformers
tasks = [
    {"prompt": "The dog", "options": {"was": " was", "is": " is"}, "answer": "was"},
    {"prompt": "Hello", "options": {"world": " world", "there": " there"}, "answer": "world"},
    {"prompt": "The capital of France is",
     "options": {"Paris": " Paris", "London": " London"}, "answer": "Paris"},
]
parameter = "transformer.h.0.mlp.c_fc.weight"
original_weight = model.get_parameter(parameter).detach().cpu().clone()
baseline = lab.evaluate(tasks, name="baseline", max_passes=6)
def inspect_mlp():
    with lab.record("transformer.h.0.mlp", positions=[-1]) as values:
        with torch.no_grad():
            model(**tokenizer("The dog", return_tensors="pt").to(lab.device))
    value = values["transformer.h.0.mlp"][0]
    return {"shape": list(value.shape), "norm": float(value.norm())}
baseline_activation = inspect_mlp()
print(json.dumps({"baseline": baseline, "activation": baseline_activation}))
"""

EDIT = """
edit_id = lab.zero(parameter, note="dogfood: zero first MLP input projection")
edited = lab.evaluate(tasks, name="zero-first-mlp", max_passes=6)
edited_activation = inspect_mlp()
prediction_changes = lab.diff(baseline, edited)
print(json.dumps({"edited": edited, "prediction_changes": prediction_changes,
                  "activation": edited_activation}))
"""

UNDO = """
lab.undo(edit_id)
restored = lab.evaluate(tasks, name="restored", max_passes=6)
restored_activation = inspect_mlp()
def items(summary):
    text = (Path(summary["run_dir"]) / "items.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines()]
before_items, edited_items, restored_items = map(items, [baseline, edited, restored])
score_changes = [
    {"index": before["index"], "baseline": before["scores"], "edited": after["scores"]}
    for before, after in zip(before_items, edited_items, strict=True)
    if before["scores"] != after["scores"]
]
assert score_changes, "the chosen edit did not change any measured continuation scores"
assert all(before["scores"] == after["scores"]
           for before, after in zip(before_items, restored_items, strict=True))
assert torch.equal(original_weight, model.get_parameter(parameter).detach().cpu())
assert baseline_activation == restored_activation
export_path = lab.export("restored-checkpoint")
from transformers import AutoModelForCausalLM
exported = AutoModelForCausalLM.from_pretrained(export_path).eval()
assert all(torch.equal(value.cpu(), exported.state_dict()[name].cpu())
           for name, value in model.state_dict().items())
evidence = {
    "model": lab.model_id, "model_commit": getattr(lab, "model_commit", None),
    "torch_version": torch.__version__, "transformers_version": transformers.__version__,
    "device": str(lab.device), "parameter": parameter, "edit_id": edit_id,
    "baseline": baseline, "edited": edited, "restored": restored,
    "score_changes": score_changes, "prediction_changes": prediction_changes,
    "baseline_items": before_items, "edited_items": edited_items, "restored_items": restored_items,
    "activations": {"baseline": baseline_activation, "edited": edited_activation,
                    "restored": restored_activation},
    "undo_scores_exact": True, "undo_weights_exact": True,
    "exported_weights_exact": True,
    "forward_passes": lab.forward_passes, "export": str(export_path),
}
(artifacts / "dogfood.json").write_text(json.dumps(evidence, indent=2) + "\\n")
print(json.dumps(evidence))
"""


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    session = output_dir / "session"
    transcript = []

    def cli(*arguments):
        result = subprocess.run(
            [sys.executable, "-m", "oumi_weights_lab", *map(str, arguments)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        transcript.append(
            {
                "arguments": list(map(str, arguments)),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        (output_dir / "transcript.json").write_text(json.dumps(transcript, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"CLI failed: {result.stderr}\n{result.stdout}")
        return json.loads(result.stdout)

    cli(
        "start",
        "--session",
        session,
        "--model",
        MODEL_ID,
        "--revision",
        MODEL_REVISION,
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--snippet-timeout",
        60,
        "--pass-limit",
        32,
    )
    try:
        for name, source in (("baseline", BASELINE), ("edit", EDIT), ("undo", UNDO)):
            snippet = output_dir / f"{name}.py"
            snippet.write_text(source)
            cli("send", "--session", session, "--file", snippet, "--wait", "--timeout", 120)
        evidence = json.loads((session / "artifacts/dogfood.json").read_text())
    finally:
        cli("stop", "--session", session, "--timeout", 10)
    backup = output_dir / "session.tar.gz"
    cli("backup", "--session", session, "--output", backup)
    with tarfile.open(backup) as archive:
        names = set(archive.getnames())
        assert "session/artifacts/restored-checkpoint/config.json" in names
        assert "session/artifacts/dogfood.json" in names
        assert any(name.startswith("session/completed/") and name.endswith(".py") for name in names)
    evidence["backup"] = str(backup)
    evidence["session_state"] = json.loads((session / "status.json").read_text())["state"]
    (output_dir / "dogfood.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def summary(evidence: dict) -> str:
    """Format the measured workflow for a terminal; retain JSON as the default."""
    lines = [f"Model: {evidence['model']} ({evidence['device']}, float32)"]
    for label, key in (("Baseline", "baseline"), ("After edit", "edited")):
        result = evidence[key]
        lines.append(f"{label}: {result['correct']}/{result['total']} dev answers correct")
    lines += [
        f"Edit: zero {evidence['parameter']}",
        f"Score maps changed: {len(evidence['score_changes'])}/{len(evidence['baseline_items'])}",
        f"Predictions changed: {len(evidence['prediction_changes'])}",
        f"Undo: weights exact={evidence['undo_weights_exact']}, "
        f"scores exact={evidence['undo_scores_exact']}",
        f"Export: all reloaded weights exact={evidence['exported_weights_exact']}",
        f"Forward passes: {evidence['forward_passes']}/32",
        f"Worker: {evidence['session_state']}",
        f"Checkpoint: {evidence['export']}",
        f"Backup: {evidence['backup']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", action="store_true", help="print a concise measured report")
    args = parser.parse_args()
    evidence = run(args.output_dir.resolve())
    print(summary(evidence) if args.summary else json.dumps(evidence, indent=2))
