"""Check that a public sparse autoencoder reconstructs the activations the lab records.

A site transform that does not match how the dictionary was trained fails silently, as a
large reconstruction error. This loads a real model and its public SAE, measures the
fraction of variance unexplained (fvu) on 32 short prompts, and repeats the measurement
with the site transform removed as a control. The check passes when fvu without the BOS
token is at most 0.30 (the release reports about 0.10 on its own data, and short prompts
are harder) and the control is at least twice as bad.

GPT-2 small runs on the CPU and downloads about 500 MB. It also compares each mapped hook
with TransformerLens's own value when `transformer-lens` is installed, which the `saes`
extra provides:

    python tests/live/validate_saes.py gpt2

Gemma 2 2B needs one GPU and a Hugging Face token that has accepted the model's license
(`huggingface-cli login`, or set HF_TOKEN). It reads the Gemma Scope residual SAE for
layer 12:

    python tests/live/validate_saes.py gemma2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROMPTS = [
    "The capital of France is Paris, and the river Seine runs through it.",
    "She opened the old wooden door and stepped into the quiet kitchen.",
    "In 1969, astronauts landed on the Moon for the first time.",
    "To install the package, run the command in your terminal.",
    "The committee will meet on Tuesday to discuss the budget.",
    "He plays the guitar every evening after dinner.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The new phone has a faster processor and a better camera.",
    "Many birds migrate south for the winter months.",
    "Please send me the report before the end of the day.",
    "The soup needs a little more salt and some fresh basil.",
    "Scientists found a new species of frog in the rainforest.",
    "My brother lives in Toronto and works as a nurse.",
    "The train to Boston leaves from platform four.",
    "Photosynthesis turns sunlight, water and carbon dioxide into sugar.",
    "They argued about the movie all the way home.",
    "The museum is free on the first Sunday of every month.",
    "A good night of sleep makes the next day much easier.",
    "The court ruled that the law was unconstitutional.",
    "Fresh snow covered the hills and the small village below.",
    "I forgot my umbrella, so I got completely soaked.",
    "The function returns a list of integers sorted in ascending order.",
    "Her first novel was published when she was only nineteen.",
    "The company announced record profits in the third quarter.",
    "We watched the sun set over the ocean from the balcony.",
    "The teacher asked the students to open their books to page ten.",
    "Honey never spoils if it is stored in a sealed jar.",
    "The football match ended in a draw after extra time.",
    "He painted the fence white and planted roses along it.",
    "The library closes at nine, so we should leave soon.",
    "Mount Everest is the highest mountain above sea level.",
    "A small bakery on the corner sells warm bread every morning.",
]

CRITERIA = {
    "statistic": "fvu over the rows of 32 prompts, BOS excluded, mean-centered as trained",
    "passes_when": "fvu <= 0.30 and the control without the site transform is at least 2x worse",
}


def reference(release, sae_id):
    """The l0 and variance explained a release lists for one SAE, when it lists them."""
    import sae_lens
    import yaml

    text = (Path(sae_lens.__file__).parent / "pretrained_saes.yaml").read_text()
    for entry in yaml.safe_load(text).get(release, {}).get("saes", []):
        if entry["id"] == sae_id:
            return {key: entry.get(key) for key in ("l0", "variance_explained")}
    return {}


def measure(sae, label):
    everything = sae.features(PROMPTS, positions="all")
    without_bos = sae.features(PROMPTS, positions=slice(1, None))
    row = {
        "label": label,
        "fvu_all_positions": everything["fvu"],
        "fvu_without_bos": without_bos["fvu"],
        "mse_without_bos": without_bos["mse"],
        "l0_without_bos": without_bos["l0"],
        "rows_without_bos": without_bos["rows"],
    }
    print(json.dumps(row), file=sys.stderr)
    return row


def compare_with_transformer_lens(lab, model_id, layers):
    """Max absolute difference between the lab's transformed activations and TransformerLens's."""
    try:
        import torch
        from transformer_lens.model_bridge import TransformerBridge

        from oumi_weights_lab import saes
    except ImportError as error:
        return {"skipped": str(error)}
    bridge = TransformerBridge.boot_transformers(model_id, device="cpu")
    bridge.enable_compatibility_mode(disable_warnings=True)
    kwargs = {"center_writing_weights": True, "fold_ln": True, "fold_value_biases": True}
    hooks = [
        "hook_resid_pre",
        "hook_resid_mid",
        "hook_resid_post",
        "hook_attn_out",
        "hook_mlp_out",
        "attn.hook_z",
        "ln2.hook_normalized",
        "mlp.hook_in",
    ]
    tokens = bridge.to_tokens(PROMPTS[0])
    names = {f"blocks.{layer}.{hook}" for layer in layers for hook in hooks}
    with torch.no_grad():
        _, cache = bridge.run_with_cache(tokens, names_filter=lambda name: name in names)
    differences = {}
    for name in sorted(names):
        site = saes.hook_site(lab, name, kwargs)
        with lab.record(site.module, positions="all", at=site.at) as captured:
            with torch.no_grad():
                lab.model(input_ids=tokens.to(lab.device))
        ours = site.apply(captured[site.module][0])
        theirs = cache[name].reshape(ours.shape).float().cpu()
        scale = float(theirs.abs().mean())
        differences[name] = {
            "max_abs_diff": float((ours - theirs).abs().max()),
            "mean_abs_activation": scale,
        }
        print(f"{name}: {differences[name]}", file=sys.stderr)
    return differences


def run_gpt2(args, session):
    from oumi_weights_lab.lab import load_lab

    lab = load_lab({"model": "gpt2", "device": args.device, "dtype": "float32", "seed": 0}, session)
    release, sae_id = "gpt2-small-res-jb", "blocks.8.hook_resid_pre"
    sae = lab.load_sae(release, sae_id, device=args.device)
    result = {
        "model": "gpt2",
        "release": release,
        "sae_id": sae_id,
        "model_check": sae.model_check,
        "site": [sae.module, sae.at, "center" if sae.site.center else "none"],
        "reference": reference(release, sae_id),
        "matched": measure(sae, "as trained (mean-centered)"),
    }
    sae.site.center = False
    result["control"] = measure(sae, "control (no centering)")
    sae.site.center = True
    result["transformer_lens"] = (
        {"skipped": "--skip-transformer-lens"}
        if args.skip_transformer_lens
        else compare_with_transformer_lens(lab, "gpt2", [3, 8])
    )
    return result


def run_gemma2(args, session):
    import torch

    from oumi_weights_lab.lab import load_lab

    if not torch.cuda.is_available():
        raise SystemExit("gemma2 needs one CUDA GPU")
    token = os.environ.get("HF_TOKEN")
    if token is None:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except ImportError:
            token = None
    if not token:
        raise SystemExit(
            "google/gemma-2-2b is gated: accept its license on the Hub, then run "
            "`huggingface-cli login` or set HF_TOKEN"
        )
    lab = load_lab(
        {"model": "google/gemma-2-2b", "device": "cuda", "dtype": "bfloat16", "seed": 0}, session
    )
    release, sae_id = "gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical"
    sae = lab.load_sae(release, sae_id, device="cuda")
    result = {
        "model": "google/gemma-2-2b",
        "release": release,
        "sae_id": sae_id,
        "model_check": sae.model_check,
        "site": [sae.module, sae.at],
        "reference": reference(release, sae_id),
        "matched": measure(sae, "as trained"),
    }
    # Gemma Scope reads the residual stream with no transform; the control shifts it.
    original = sae.site.apply
    sae.site.apply = lambda x: original(x) - original(x).mean(-1, keepdim=True)
    result["control"] = measure(sae, "control (mean-centered, which Gemma Scope was not)")
    sae.site.apply = original
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model", choices=["gpt2", "gemma2"])
    parser.add_argument("--device", default="cpu", help="gpt2 only; gemma2 uses CUDA")
    parser.add_argument("--skip-transformer-lens", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="session directory, default a temp folder")
    args = parser.parse_args(argv)
    session = args.output_dir or Path(tempfile.mkdtemp(prefix="validate-saes-")) / "session"
    session = session.expanduser().resolve()
    session.mkdir(parents=True, exist_ok=True)
    result = (run_gpt2 if args.model == "gpt2" else run_gemma2)(args, session)
    matched = result["matched"]["fvu_without_bos"]
    control = result["control"]["fvu_without_bos"]
    result["criteria"] = CRITERIA
    result["state"] = "passed" if matched <= 0.30 and control >= 2 * matched else "failed"
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if main()["state"] != "passed":
        raise SystemExit(1)
