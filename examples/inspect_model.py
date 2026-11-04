"""Submit with oumi-weights-lab send --session DIR --file examples/inspect_model.py --follow."""

import json

import torch

print(f"Model: {lab.model_id}")
print(f"Device: {lab.device}")
parameters = sum(parameter.numel() for parameter in model.parameters())
print(f"Parameters: {parameters:,}")

inputs = tokenizer("The capital of France is", return_tensors="pt").to(lab.device)
with torch.no_grad():
    logits = model(**inputs).logits
next_token = tokenizer.decode(logits[0, -1].argmax())
print(f"Next token: {next_token!r}")

report = {"model": lab.model_id, "parameters": parameters, "next_token": next_token}
(artifacts / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
