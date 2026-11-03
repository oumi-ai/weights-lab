"""Check GPT-2 against upstream golden tokens and HF continuation loss on CPU.

Run: uv run python -m examples.parity --output-dir /tmp/weights-lab-parity
The first run downloads the public GPT-2 124M checkpoint (roughly 500 MB).
"""

import argparse
import json
from pathlib import Path

import torch
import transformers

from weights_lab.lab import load_lab

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SOURCE = (
    "https://github.com/huggingface/transformers/blob/v4.56.2/"
    "tests/models/gpt2/test_modeling_gpt2.py#L763-L788"
)
# Published by Hugging Face in GPT2ModelLanguageGenerationTest.
EXPECTED_IDS = [
    464,
    3290,
    373,
    1043,
    287,
    257,
    2214,
    1474,
    262,
    16246,
    286,
    2688,
    290,
    2688,
    27262,
    13,
    198,
    198,
    464,
    3290,
]


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    lab = load_lab(
        {"model": MODEL_ID, "revision": MODEL_REVISION, "device": "cpu", "dtype": "float32"},
        output_dir,
    )
    ids = torch.tensor([EXPECTED_IDS[:2]], device=lab.device)
    with torch.no_grad():
        generated = lab.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=18,
            pad_token_id=lab.tokenizer.eos_token_id,
        )[0].tolist()
    if generated != EXPECTED_IDS:
        raise AssertionError(f"upstream generation mismatch: {generated}")
    prompt = lab.tokenizer.decode(EXPECTED_IDS[:2])
    continuation = lab.tokenizer.decode(EXPECTED_IDS[2:])
    prompt_ids = lab.tokenizer.encode(prompt, add_special_tokens=False)
    continuation_ids = lab.tokenizer.encode(continuation, add_special_tokens=False)
    assert prompt_ids + continuation_ids == EXPECTED_IDS
    score = lab.score(prompt, {"published_continuation": continuation})["published_continuation"]
    input_ids = torch.tensor([prompt_ids + continuation_ids], device=lab.device)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    with torch.no_grad():
        mean_loss = float(lab.model(input_ids=input_ids, labels=labels).loss)
    reference_score = -mean_loss * len(continuation_ids)
    difference = abs(score - reference_score)
    if difference > 1e-4:
        raise AssertionError(f"continuation likelihood differs from HF loss by {difference}")
    result = {
        "model": MODEL_ID,
        "model_commit": MODEL_REVISION,
        "source": SOURCE,
        "device": str(lab.device),
        "dtype": str(lab.model.dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "published_generation_ids": EXPECTED_IDS,
        "observed_generation_ids": generated,
        "published_generation_matches": True,
        "prompt": prompt,
        "continuation": continuation,
        "continuation_tokens": len(continuation_ids),
        "continuation_score": score,
        "hf_mean_continuation_loss": mean_loss,
        "hf_reference_score": reference_score,
        "absolute_score_difference": difference,
        "forward_passes": lab.forward_passes,
        "scope": "published generation tokens and implementation likelihood parity; no benchmark",
    }
    (output_dir / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir.resolve()), indent=2))
