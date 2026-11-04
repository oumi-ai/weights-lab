"""Force one prompt's next token from Paris to banana, then undo the weight edit.

Run: uv run python -m examples.banana
This deliberately changes one answer; it does not demonstrate a general knowledge edit.
GPT-2 is loaded once on CPU. The first run downloads its public checkpoint.
"""

import argparse
import json
from pathlib import Path

import torch

from oumi_weights_lab.lab import load_lab

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
PROMPT = "The capital of Italy is Rome. The capital of France is"


def run(output_dir: Path, model_path: Path | None = None) -> dict:
    lab = load_lab(
        {
            "model": str(model_path) if model_path else MODEL_ID,
            "revision": MODEL_REVISION,
            "device": "cpu",
            "dtype": "float32",
        },
        output_dir,
    )
    inputs = lab.tokenizer(PROMPT, return_tensors="pt")
    target_ids = lab.tokenizer.encode(" banana", add_special_tokens=False)
    assert len(target_ids) == 1, "The target must be one token"
    target = target_ids[0]
    assert target not in inputs.input_ids[0].tolist()
    weight = lab.model.get_input_embeddings().weight
    assert weight is lab.model.get_output_embeddings().weight
    original_row = weight[target].detach().clone()

    def next_token() -> str:
        with torch.no_grad():
            generated = lab.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1,
                pad_token_id=lab.tokenizer.eos_token_id,
            )
        return lab.tokenizer.decode(generated[0, inputs.input_ids.shape[1] :])

    baseline = next_token()
    assert baseline == " Paris", repr(baseline)
    with torch.no_grad():
        observed = lab.model(**inputs, output_hidden_states=True)
    original_logits = observed.logits[0, -1].clone()
    hidden = observed.hidden_states[-1][0, -1]
    margin = 5.0
    # Minimum-norm row correction for this hidden vector and desired logit margin.
    correction = (
        (original_logits.max() + margin - original_logits[target]) * hidden / hidden.dot(hidden)
    )
    edit_id = lab.edit(
        "transformer.wte.weight",
        original_row + correction,
        index=target,
        note="Force banana as the next token for one fixed prompt",
    )
    edited = next_token()
    assert edited == " banana", repr(edited)
    lab.undo(edit_id)
    restored = next_token()
    with torch.no_grad():
        restored_logits = lab.model(**inputs).logits[0, -1]
    row_exact = torch.equal(
        weight[target].detach().view(torch.uint8), original_row.view(torch.uint8)
    )
    logits_exact = torch.equal(original_logits, restored_logits)
    assert row_exact and logits_exact and restored == " Paris"
    result = {
        "model": MODEL_ID,
        "model_commit": MODEL_REVISION,
        "device": str(lab.device),
        "dtype": str(lab.model.dtype),
        "prompt": PROMPT,
        "baseline_next_token": baseline,
        "edited_next_token": edited,
        "restored_next_token": restored,
        "edited_parameter": "transformer.wte.weight",
        "edited_row": target,
        "edited_values": original_row.numel(),
        "logit_margin": margin,
        "undo_row_bytes_exact": row_exact,
        "undo_logits_exact": logits_exact,
        "forward_passes": lab.forward_passes,
        "scope": "Forced next-token change on one prompt; no generalization claim or benchmark",
    }
    (output_dir / "banana-demo.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("sessions/banana-demo"))
    parser.add_argument("--model-path", type=Path, help="Use a local GPT-2 snapshot directory")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir.resolve(), args.model_path), indent=2))
