"""Validate one real CPU/CUDA/MPS worker; create local assets unless --model is supplied."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

CHECKS = """
import json
import torch
import transformers
import weights_lab
from transformers import AutoModelForCausalLM
from weights_lab import BudgetExceeded
expected_device = json.loads((artifacts.parent.parent / "device.json").read_text())
assert lab.device.type == expected_device
assert all(value.device.type == expected_device for value in model.parameters())
assert hasattr(model, "peft_config")
assert any("lora_B" in name and value.abs().sum() > 0
           for name, value in model.named_parameters())
seen_devices = []
def check_device(module, args, kwargs):
    seen_devices.append(kwargs["input_ids"].device.type)
handle = model.register_forward_pre_hook(check_device, with_kwargs=True)
inputs = tokenizer("hello world", return_tensors="pt").to(lab.device)
module_name = next(name for name, module in model.named_modules() if name.endswith(".mlp"))
parameter = next(name for name, value in model.named_parameters()
                 if ".mlp." in name and value.ndim == 2)
target = model.get_submodule(module_name)
hooks_before = len(target._forward_hooks)
with lab.record(module_name) as recorded:
    with torch.no_grad():
        original_logits = model(**inputs).logits.detach().clone()
assert recorded[module_name][0].device.type == "cpu"
assert torch.isfinite(recorded[module_name][0]).all()
assert len(target._forward_hooks) == hooks_before
with lab.patch(module_name, lambda value: torch.zeros_like(value)):
    with torch.no_grad():
        patched_logits = model(**inputs).logits.detach().clone()
assert not torch.equal(original_logits, patched_logits)
assert len(target._forward_hooks) == hooks_before
flags = [value.requires_grad for value in model.parameters()]
gradient = lab.gradient("hello", target=" world", parameters=[parameter])[parameter]
assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
assert flags == [value.requires_grad for value in model.parameters()]
before = lab.forward_passes
with torch.no_grad():
    generated = model.generate(**inputs, min_new_tokens=2, max_new_tokens=2, do_sample=False)
assert generated.device.type == expected_device
assert lab.forward_passes - before == 2
tasks = [{"prompt": "hello", "options": [" world", " yes"], "answer": 0}]
baseline = lab.evaluate(tasks, name="baseline", max_passes=2)
heldout = [{"prompt": "world", "options": [" yes", " no"], "answer": 0}]
final = lab.evaluate(heldout, name="held-out", held_out=True, max_passes=2)
before = lab.forward_passes
try:
    lab.evaluate(heldout, name="repeat", held_out=True, max_passes=2)
except ValueError:
    pass
else:
    raise AssertionError("held-out data was reused")
assert lab.forward_passes == before
try:
    lab.evaluate([{"prompt": "hello world", "options": [" yes", " no"], "answer": 0}],
                 name="exhausted", held_out=True, max_passes=1)
except BudgetExceeded:
    pass
else:
    raise AssertionError("evaluation budget did not stop the second forward")
failed = json.loads((artifacts / "evaluations/exhausted/summary.json").read_text())
assert failed["state"] == "failed" and failed["accuracy"] is None
before = lab.forward_passes
try:
    with lab.limit_passes(0):
        model(**inputs)
except BudgetExceeded:
    pass
else:
    raise AssertionError("direct forward bypassed budget")
assert lab.forward_passes == before
original_weight = model.get_parameter(parameter).detach().clone()
edit_id = lab.zero(parameter, note="runtime validation")
edited = lab.evaluate(tasks, name="edited", max_passes=2)
lab.undo(edit_id)
assert torch.equal(original_weight, model.get_parameter(parameter))
restored = lab.evaluate(tasks, name="restored", max_passes=2)
def item(run):
    return json.loads((Path(run["run_dir"]) / "items/000000.json").read_text())
assert item(baseline)["scores"] != item(edited)["scores"]
assert item(baseline)["scores"] == item(restored)["scores"]
with torch.no_grad():
    restored_logits = model(**inputs).logits
torch.testing.assert_close(original_logits, restored_logits, rtol=0, atol=0)
handle.remove()
export_path = lab.export("checkpoint")
exported = AutoModelForCausalLM.from_pretrained(
    export_path, dtype=torch.float32, attn_implementation="eager").to(lab.device).eval()
with torch.no_grad():
    exported_logits = exported(**inputs).logits
torch.testing.assert_close(restored_logits, exported_logits, rtol=1e-4, atol=1e-5)
assert seen_devices and set(seen_devices) == {expected_device}
report = {"device": str(lab.device), "torch": torch.__version__,
          "cuda_device_name": torch.cuda.get_device_name() if expected_device == "cuda" else None,
          "transformers": transformers.__version__, "forward_passes": lab.forward_passes,
          "budget_used": lab.budget.used, "observed_input_devices": sorted(set(seen_devices)),
          "generation_forward_passes": 2, "gradient_norm": float(gradient.norm()),
          "activation_shape": list(recorded[module_name][0].shape),
          "hook_cleanup": True, "held_out_reuse_rejected": True,
          "budget_rejection": True, "edit_undo_exact": True, "export_reload_matches": True,
          "export_max_logit_difference": float((exported_logits-restored_logits).abs().max()),
          "baseline": baseline, "edited": edited, "restored": restored, "held_out": final}
import hashlib
package = Path(weights_lab.__file__).parent
report["package_source_sha256"] = {
    str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(package.rglob("*.py"))
}
(artifacts / "runtime.json").write_text(json.dumps(report, indent=2) + "\\n")
"""


def make_assets(root, model=None):
    import torch
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        AutoModelForCausalLM,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
    )

    if model is None:
        model = root / "model"
        vocabulary = {"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3, "yes": 4, "no": 5}
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
        ).save_pretrained(model)
        torch.manual_seed(0)
        dimensions = dict(vocab_size=6, n_embd=8, n_layer=1, n_head=1, n_positions=32)
        tokens = dict(eos_token_id=1, bos_token_id=1, pad_token_id=1)
        GPT2LMHeadModel(GPT2Config(**dimensions, **tokens)).save_pretrained(model)
    base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
    target = "c_attn" if base.config.model_type == "gpt2" else "q_proj"
    lora = LoraConfig(task_type="CAUSAL_LM", target_modules=[target], r=2, lora_alpha=4)
    lora.fan_in_fan_out = target == "c_attn"
    adapted = get_peft_model(base, lora)
    with torch.no_grad():
        for name, value in adapted.named_parameters():
            if "lora_B" in name:
                value.fill_(0.125)
    adapter = root / "adapter"
    adapted.save_pretrained(adapter)
    return str(model), str(adapter)


def validate(device, output_dir, model=None, *, checks=CHECKS):
    from weights_lab.cli import _matches_worker
    from weights_lab.session import is_process_alive

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"state": "running", "requested_device": device}
    session, pid = root / "session", None

    def cli(*args):
        result = subprocess.run(
            [sys.executable, "-m", "weights_lab", *map(str, args)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        with (root / "cli.jsonl").open("a") as stream:
            entry = dict(
                args=list(map(str, args)),
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            stream.write(json.dumps(entry) + "\n")
        if result.returncode:
            raise RuntimeError(f"CLI failed: {result.stderr}\n{result.stdout}")
        return json.loads(result.stdout)

    try:
        model, adapter = make_assets(root, model)
        (root / "device.json").write_text(json.dumps(device))
        options = {
            "model": model,
            "adapter": adapter,
            "device": device,
            "dtype": "float32",
            "snippet-timeout": 120,
            "pass-limit": 64,
        }
        start = ["start", "--session", session]
        for flag, value in options.items():
            start += [f"--{flag}", value]
        pid = cli(*start)["pid"]
        source = root / "checks.py"
        source.write_text(checks)
        result = cli("send", "--session", session, "--file", source, "--wait", "--timeout", 150)
        report.update(json.loads((session / "artifacts/runtime.json").read_text()))
        report["snippet_result"] = result
        report["state"] = "succeeded"
    except BaseException as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if pid is not None:
                try:
                    report["stop"] = cli("stop", "--session", session, "--timeout", 5)
                finally:
                    if is_process_alive(pid) and _matches_worker(pid, session):
                        os.kill(pid, signal.SIGKILL)
                        report["stop"] = cli("stop", "--session", session, "--timeout", 5)
                report["worker_alive_after_stop"] = is_process_alive(pid)
                if report["worker_alive_after_stop"]:
                    raise RuntimeError("worker survived cleanup")
                backup = root / "session.tar.gz"
                cli("backup", "--session", session, "--output", backup)
                with tarfile.open(backup) as archive:
                    files = [member for member in archive.getmembers() if member.isfile()]
                    for member in files:
                        original = root / member.name
                        assert (
                            hashlib.sha256(original.read_bytes()).digest()
                            == hashlib.sha256(archive.extractfile(member).read()).digest()
                        )
                report.update(backup=str(backup), backup_files_verified=len(files))
        except BaseException as error:
            report.update(state="failed", cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", help="optional local or public GPT-2/Llama-style small model")
    args = parser.parse_args()
    print(json.dumps(validate(args.device, args.output_dir, args.model), indent=2))
