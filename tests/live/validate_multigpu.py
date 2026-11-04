"""Validate one model spread over two or more CUDA GPUs in a real worker.

Run it on a node with at least two GPUs, such as a SkyPilot ``H100:2`` cluster:
``python tests/live/validate_multigpu.py --output-dir /tmp/multigpu``. By default it
builds a small local GPT-2 with a LoRA adapter, so nothing downloads. ``--model``
names a checkpoint instead, to time hashing, bundles, and export on a large model;
``--no-adapter`` then skips building an adapter for it.
"""

import argparse
import importlib.util
import json
from pathlib import Path

CHECKS = """
import json, re, shutil, time
from pathlib import Path
import accelerate, torch, transformers
from transformers import AutoModelForCausalLM
from oumi_weights_lab.lab import load_lab
from oumi_weights_lab.persistence import _fingerprint

session_dir = artifacts.parent
config = json.loads((session_dir / "config.json").read_text())
status = json.loads((session_dir / "status.json").read_text())
report = {"torch": torch.__version__, "transformers": transformers.__version__,
          "accelerate": accelerate.__version__, "devices": status.get("devices"),
          "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
          "parameters": sum(value.numel() for value in model.parameters()),
          "adapter": hasattr(model, "peft_config")}

# The model spans at least two GPUs, with nothing on CPU or disk.
gpus = sorted({value.device for value in model.parameters()}, key=lambda device: device.index)
assert all(device.type == "cuda" for device in gpus), gpus
assert len(gpus) >= 2, f"the model sits on {gpus}; it must span at least two GPUs"
assert lab.device_map and len(set(lab.device_map.values())) >= 2
assert len([key for key in status.get("devices", {}) if key.startswith("cuda")]) >= 2
first, last = gpus[0], gpus[-1]
placed = list(lab.device_map.values())
report["device_map_modules"] = {str(device): placed.count(device) for device in set(placed)}
inputs = tokenizer("hello world yes no", return_tensors="pt").to(lab.device)
assert lab.device == first

def logits(target=model):
    with torch.no_grad():
        return target(**inputs).logits.detach().cpu()

baseline = logits()
before = lab.forward_passes
logits()
assert lab.forward_passes - before == 1  # one pass per sequence

# Record blocks on the first and the last GPU, and patch the last.
block = re.compile(r".*\\.(h|layers)\\.[0-9]+")
blocks = [name for name, _ in model.named_modules() if block.fullmatch(name)]
def block_on(device):
    return next(name for name in blocks
                if next(model.get_submodule(name).parameters()).device == device)
early, late = block_on(first), block_on(last)
with lab.record([early, late]) as captured:
    logits()
for name in (early, late):
    assert captured[name][0].device.type == "cpu" and torch.isfinite(captured[name][0]).all()
with lab.patch(late, lambda value: torch.zeros_like(value)):
    assert not torch.equal(logits(), baseline)
assert torch.equal(logits(), baseline)

# A gradient for a weight on the last GPU.
def weight_on(device, block):
    return next(name for name, value in model.named_parameters()
                if name.startswith(block + ".") and value.device == device and value.ndim == 2
                and "lora_" not in name)
early_weight, late_weight = weight_on(first, early), weight_on(last, late)
gradient = lab.gradient("hello world", target=" yes", parameters=[late_weight])[late_weight]
assert gradient.device.type == "cpu" and torch.isfinite(gradient).all()
assert gradient.abs().sum() > 0
report["gradient_parameter"] = late_weight

# Batched scoring matches scoring one sequence at a time.
def items(run):
    lines = (Path(run["run_dir"]) / "items.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]
rows = [{"prompt": "hello", "options": [" world", " yes", " no"], "answer": 0},
        {"prompt": "world yes", "options": [" no", " hello"], "answer": 1}]
one = items(lab.evaluate(rows, name="batch-one", batch_size=1))
many = items(lab.evaluate(rows, name="batch-many", batch_size=8))
assert [item["prediction"] for item in one] == [item["prediction"] for item in many]
report["batch_score_difference"] = max(
    abs(a["scores"][key] - b["scores"][key]) for a, b in zip(one, many) for key in a["scores"])
assert report["batch_score_difference"] < 1e-3
# Chat models often ship a repetition penalty, which batched generation refuses.
sampled = lab.evaluate([{"prompt": "hello", "answer": "world"}, {"prompt": "no", "answer": "yes"}],
                       name="sampled", mode="sampled", batch_size=2,
                       generation_config={"max_new_tokens": 4, "repetition_penalty": 1.0})
assert sampled["forward_passes"] == sum(item["generated_tokens"] for item in items(sampled))

# Exact edits and undo on both GPUs; time the hashing and the bundle.
names = [early_weight, late_weight]
originals = {name: model.get_parameter(name).detach().clone() for name in names}
for name in names:
    lab.edit(name, lambda value: value * 1.5, note="multi-GPU check")
edited = logits()
assert not torch.equal(edited, baseline)
started = time.perf_counter()
_fingerprint(lab)
report["fingerprint_seconds"] = time.perf_counter() - started
started = time.perf_counter()
bundle = lab.save_edits("multigpu-bundle")
report["save_edits_seconds"] = time.perf_counter() - started
for _ in names:
    lab.undo()
assert all(torch.equal(model.get_parameter(name), value) for name, value in originals.items())
assert torch.equal(logits(), baseline)

# A second copy fits only when the GPUs have room for it.
size = sum(value.numel() * value.element_size() for value in model.parameters())
free = sum(torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count()))
room = free > 1.3 * size
report["second_copy_fits"] = room
assert room or ALLOW_PARTIAL, (
    "no room for a second copy, so resume and the export reload cannot be checked; "
    "use more GPUs or pass --allow-partial"
)

# Resume the bundle into a fresh split model.
if room:
    started = time.perf_counter()
    resumed = load_lab({**config, "resume": str(bundle)}, session_dir.parent / "resumed-session")
    report["resume_seconds"] = time.perf_counter() - started
    assert len({value.device for value in resumed.model.parameters()}) >= 2
    assert torch.equal(logits(resumed.model), edited)
    del resumed
    torch.cuda.empty_cache()

# Export (merging any adapter layer by layer) and reload across the same GPUs.
started = time.perf_counter()
path = lab.export("multigpu-checkpoint")
report["export_seconds"] = time.perf_counter() - started
report["export_bytes"] = sum(file.stat().st_size for file in Path(path).glob("*.safetensors"))
if room:
    dtype = next(model.parameters()).dtype
    reloaded = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, attn_implementation="eager", device_map=lab.device_map).eval()
    difference = float((logits(reloaded) - baseline).abs().max())
    report["export_max_logit_difference"] = difference
    if not report["adapter"]:
        # The same weights, dtype, attention, and placement give the same logits.
        assert difference == 0, difference
    elif dtype == torch.float32:
        assert difference < 1e-4, difference
    else:
        # Merging rounds each adapted weight once in the lower precision.
        assert difference < 0.02 * float(baseline.abs().max()), difference
    del reloaded
    torch.cuda.empty_cache()
# The session backup that follows would otherwise archive the whole checkpoint.
shutil.rmtree(path)
report["checks_passed"] = True
(artifacts / "runtime.json").write_text(json.dumps(report, indent=2) + "\\n")
"""


def runtime():
    path = Path(__file__).with_name("validate_runtime.py")
    spec = importlib.util.spec_from_file_location("validate_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_assets(root, model=None, *, adapter=True):
    """A four-block GPT-2 and LoRA adapter, or a given checkpoint with an optional adapter."""
    if model is None:
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        model = root / "model"
        vocabulary = {"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3, "yes": 4, "no": 5}
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
        ).save_pretrained(model)
        torch.manual_seed(0)
        GPT2LMHeadModel(
            GPT2Config(
                vocab_size=6,
                n_embd=64,
                n_layer=4,
                n_head=4,
                n_positions=32,
                eos_token_id=1,
                bos_token_id=1,
                pad_token_id=1,
            )
        ).save_pretrained(model)
    if not adapter:
        return str(model), None
    return runtime().make_assets(root, model)


def validate(
    output_dir, model=None, *, adapter=True, dtype="float32", wait=1800, allow_partial=False
):
    """Run the checks; ``allow_partial`` lets a model with no room for a second copy
    skip resuming and reloading its export instead of failing."""
    harness = runtime()
    return harness.validate(
        "cuda",
        output_dir,
        model,
        checks=f"ALLOW_PARTIAL = {bool(allow_partial)}\n" + CHECKS,
        assets=lambda root, model: make_assets(root, model, adapter=adapter),
        options={
            "device-map": "balanced",
            "dtype": dtype,
            "pass-limit": None,
            "snippet-timeout": wait,
        },
        wait=wait,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", help="checkpoint to spread over the GPUs instead")
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--wait", type=int, default=1800, help="seconds for the checks")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="skip resume and the export reload when a second copy does not fit",
    )
    args = parser.parse_args()
    report = validate(
        args.output_dir,
        args.model,
        adapter=not args.no_adapter,
        dtype=args.dtype,
        wait=args.wait,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(report, indent=2))
