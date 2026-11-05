"""Validate quantized loading, editing, bundles and export in a real CUDA or MPS worker.

Run it on a machine with the ``quantized`` extra installed:
``python tests/live/validate_quantized.py --device cuda --output-dir /tmp/quantized``.
It loads a public instruct model with bitsandbytes (``--model``, ``--quantize``), then
checks edits, undo, a saved bundle, and an export reloaded with plain transformers.
A small model such as ``Qwen/Qwen2.5-0.5B-Instruct`` keeps a run quick.
"""

import argparse
import importlib.util
import json
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

CHECKS = """
import json, shutil, warnings
from pathlib import Path
import torch, transformers
import bitsandbytes
from transformers import AutoModelForCausalLM
from oumi_weights_lab.quantized import lookup

session_dir = artifacts.parent
config = json.loads((session_dir / "config.json").read_text())
assert lab.quantization and lab.quantization["method"] == METHOD, lab.quantization
report = {"torch": torch.__version__, "transformers": transformers.__version__,
          "bitsandbytes": bitsandbytes.__version__, "device": str(lab.device),
          "quantization": lab.quantization}

# Every quantized layer reports its logical shape, not its packed one.
layers = [(name, module) for name, module in model.named_modules()
          if hasattr(module, "in_features") and lookup(model, name + ".weight") is not None]
assert layers, "no quantized layers were loaded"
parts = {part["name"]: part for part in lab.parts()}
for name, module in layers:
    part = parts[name + ".weight"]
    assert part["quantization"]["method"] == METHOD, part
    assert part["shape"] == [module.out_features, module.in_features], (name, part["shape"])
    assert part["numel"] == module.out_features * module.in_features
report["quantized_layers"] = len(layers)

inputs = tokenizer("hello world yes no", return_tensors="pt").to(lab.device)

def logits():
    with torch.no_grad():
        return model(**inputs).logits.detach().cpu().clone()

baseline = logits()
assert torch.isfinite(baseline).all()
scored = lab.evaluate([{"prompt": "hello", "options": [" world", " yes"], "answer": 0}],
                      name="quantized-baseline")
assert scored["forward_passes"] >= 1

def snapshot(weight_name):
    return {key: value.clone() for key, value in lookup(model, weight_name).snapshot().items()}

def same(first, second):
    return first.keys() == second.keys() and all(torch.equal(first[k], second[k]) for k in first)

weight = next(name for name in parts if name.endswith("layers.0.self_attn.q_proj.weight"))
other = next(name for name in parts if name.endswith("layers.0.self_attn.o_proj.weight"))
start, other_start = snapshot(weight), snapshot(other)

# Edit one block, then undo: bytes, scales and logits match exactly.
edit_id = lab.edit(weight, lambda value: value * 2 + 0.05, index=(0, slice(0, 8)), note="block")
record = lab.edits()[-1]["quantization"]
assert record["blocks_requantized"] == 1 and record["changed_elements"] > 0, record
assert not same(start, snapshot(weight))
assert not torch.equal(logits(), baseline)
lab.undo(edit_id)
assert same(start, snapshot(weight))
assert torch.equal(logits(), baseline)

# A change below the quantization step warns, is recorded, and undoes.
eps = torch.finfo(lookup(model, weight).dtype).eps
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    tiny = lab.edit(weight, lambda value: value * (1 + 4 * eps), index=(1, slice(2, 3)))
messages = [str(item.message) for item in caught]
assert any("quantization step" in message for message in messages), messages
lab.undo(tiny)
assert same(start, snapshot(weight))

# A head ablation changes the logits, and undo restores them.
head = lab.heads(layers=[0])[1]
ablated = [lab.zero(**part) for part in head["o"]]
assert not torch.equal(logits(), baseline)
for _ in ablated:
    lab.undo()
assert same(other_start, snapshot(other)) and torch.equal(logits(), baseline)

# Save the edits, stop, resume them in a fresh worker, and undo exactly.
lab.edit(weight, lambda value: value * 2 + 0.05, index=(0, slice(0, 8)), note="saved")
lab.edit(other, lambda value: value * 0.5, index=(slice(0, 4), slice(None)), note="saved too")
edited_state, edited_other, edited = snapshot(weight), snapshot(other), logits()
bundle = lab.save_edits("quantized-bundle")
assert json.loads((bundle / "manifest.json").read_text())["format_version"] == 2
assert same(edited_state, snapshot(weight)) and torch.equal(logits(), edited)
# Resume in a fresh worker (a new process), where no forward has run yet.
import subprocess, sys, tempfile

wait = float(config["snippet_timeout"])

def cli(*args, timeout=wait):
    done = subprocess.run([sys.executable, "-m", "oumi_weights_lab", *map(str, args)],
                          capture_output=True, text=True, timeout=timeout + 30)
    assert done.returncode == 0, done.stderr + done.stdout
    return json.loads(done.stdout)

scratch = Path(tempfile.mkdtemp(prefix="resumed-", dir=session_dir.parent))
fresh, snippet = scratch / "worker", scratch / "snippet.py"
snippet.write_text(RESUMED.replace("__NAMES__", json.dumps([weight, other])))
try:
    cli("start", "--session", fresh, "--model", config["model"], "--device", config["device"],
        "--dtype", config["dtype"], "--quantize", METHOD, "--resume", bundle)
    cli("send", "--session", fresh, "--file", snippet, "--wait", "--timeout", wait)
    seen = torch.load(fresh / "artifacts" / "resumed.pt")
finally:
    # Stop a worker that started, whatever went wrong, then drop its 8MB of files.
    if (fresh / "status.json").exists():
        cli("stop", "--session", fresh, "--timeout", 30, timeout=30)
    shutil.rmtree(scratch, ignore_errors=True)
assert seen["active"] == lab.active_edit_ids
assert same(edited_state, seen["edited"][weight]) and same(edited_other, seen["edited"][other])
assert torch.equal(seen["edited_logits"], edited)
assert same(start, seen["undone"][weight]) and same(other_start, seen["undone"][other])
assert torch.equal(seen["undone_logits"], baseline)
if lab.device.type == "cuda":
    torch.cuda.empty_cache()

# Export, reload with plain transformers, and compare with the edited live model.
path = lab.export("quantized-checkpoint")
# The lab loads with eager attention, so the reload must too for the logits to match.
reloaded = AutoModelForCausalLM.from_pretrained(
    path, attn_implementation="eager", device_map={"": str(lab.device)}
).eval()
reloaded_layers = [name for name, _ in layers if lookup(reloaded, name + ".weight") is not None]
assert len(reloaded_layers) == len(layers), "the export reloaded with float layers"
for name in (weight, other):
    reloaded_parts = lookup(reloaded, name).snapshot()
    assert same(snapshot(name), {k: v.clone() for k, v in reloaded_parts.items()})
with torch.no_grad():
    reloaded_logits = reloaded(**inputs).logits.detach().cpu()
report["export_max_logit_difference"] = float((reloaded_logits - edited).abs().max())
assert torch.equal(reloaded_logits, edited), report["export_max_logit_difference"]
del reloaded
# The session backup that follows would otherwise archive the checkpoint and the bundle.
shutil.rmtree(path)
shutil.rmtree(bundle)
report["checks_passed"] = True
(artifacts / "runtime.json").write_text(json.dumps(report, indent=2) + "\\n")
"""


# Runs in the resumed worker: read the packed parts, score, undo both edits, read and score again.
RESUMED = """
import torch
from oumi_weights_lab.quantized import lookup
NAMES = __NAMES__
inputs = tokenizer("hello world yes no", return_tensors="pt").to(lab.device)
def parts():
    return {name: {key: value.clone() for key, value in lookup(model, name).snapshot().items()}
            for name in NAMES}
def scores():
    with torch.no_grad():
        return model(**inputs).logits.detach().cpu().clone()
seen = {"active": lab.active_edit_ids}
seen["edited"] = parts()
seen["edited_logits"] = scores()
lab.undo()
lab.undo()
seen["undone"] = parts()
seen["undone_logits"] = scores()
torch.save(seen, artifacts / "resumed.pt")
"""


def prefix(quantize):
    """The constants the checks start from."""
    return f"METHOD = {quantize!r}\nRESUMED = {RESUMED!r}\n"


def runtime():
    path = Path(__file__).with_name("validate_runtime.py")
    spec = importlib.util.spec_from_file_location("validate_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(
    device,
    output_dir,
    model=DEFAULT_MODEL,
    *,
    quantize="nf4",
    dtype="float32",
    wait=3600,
):
    """Run the checks on ``model`` quantized as ``quantize``, with no adapter and no pass cap."""
    harness = runtime()
    return harness.validate(
        device,
        output_dir,
        model,
        checks=prefix(quantize) + CHECKS,
        assets=lambda root, model: (model, None),
        options={
            "quantize": quantize,
            "dtype": dtype,
            "pass-limit": None,
            "snippet-timeout": wait,
        },
        wait=wait,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=["cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quantize", choices=["nf4", "fp4", "int8"], default="nf4")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="public checkpoint to quantize")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--wait", type=int, default=3600, help="seconds for the checks")
    args = parser.parse_args()
    report = validate(
        args.device,
        args.output_dir,
        args.model,
        quantize=args.quantize,
        dtype=args.dtype,
        wait=args.wait,
    )
    print(json.dumps(report, indent=2))
