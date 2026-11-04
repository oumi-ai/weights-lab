"""Bounded two-worker BF16/PEFT edit-bundle validation, without model downloads.

Run ``prepare`` on a worker host, download its bundle with a verified transport,
then upload that copy and run ``resume``. Cloud provisioning is deliberately
outside this harness; CPU-only tests never launch cloud resources.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

COMMON = """
import hashlib,json,os,torch
from oumi_weights_lab import BudgetExceeded
assert lab.device.type == EXPECTED_DEVICE
assert all(p.device.type == EXPECTED_DEVICE for p in model.parameters())
base_name = next(n for n,p in model.named_parameters() if '.mlp.' in n and p.ndim == 2)
adapter_name = next(n for n,p in model.named_parameters() if 'lora_B' in n)
assert model.get_parameter(base_name).dtype == torch.bfloat16
assert hasattr(model, 'peft_config')
def digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((str(tensor.dtype),tuple(tensor.shape))).encode()
        + tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def weights():
    return {n:digest(p) for n,p in model.named_parameters()}
def logits():
    inputs = tokenizer('hello world',return_tensors='pt').to(lab.device)
    with torch.no_grad():
        return model(**inputs).logits.detach().cpu()
heldout = [{'prompt':'world','options':[' yes',' no'],'answer':0}]
"""

PREPARE = (
    COMMON
    + """
assert model.get_parameter(adapter_name).abs().sum() > 0
baseline_weights = weights()
baseline_logits = digest(logits())
try:
    lab.evaluate(heldout,name='failed-heldout',held_out=True,max_passes=1,batch_size=1)
except BudgetExceeded:
    pass
else:
    raise AssertionError('held-out evaluation must spend one pass then fail')
failed = json.loads((artifacts/'evaluations/failed-heldout/summary.json').read_text())
assert failed['state']=='failed'
base_edit = lab.zero(base_name,note='BF16 CUDA resume validation')
base_only_logits = digest(logits())
assert base_only_logits != baseline_logits
adapter_edit = lab.zero(adapter_name,note='PEFT CUDA resume validation')
edited_weights = weights()
edited_logits = digest(logits())
assert edited_logits != base_only_logits
before_save = lab.forward_passes
bundle = lab.save_edits('bundle')
assert lab.forward_passes == before_save
report = dict(state='succeeded',device=str(lab.device),worker_pid=os.getpid(),
    torch=torch.__version__,
    cuda_device_name=torch.cuda.get_device_name() if EXPECTED_DEVICE=='cuda' else None,
    base_name=base_name,adapter_name=adapter_name,
    base_dtype=str(model.get_parameter(base_name).dtype),
    adapter_dtype=str(model.get_parameter(adapter_name).dtype),
    base_edit=base_edit,adapter_edit=adapter_edit,baseline_weights=baseline_weights,
    baseline_logits=baseline_logits,base_only_logits=base_only_logits,
    edited_weights=edited_weights,edited_logits=edited_logits,
    forward_passes=before_save,pass_limit=lab.budget.limit,
    registry=json.loads((artifacts/'evaluations/_registry.json').read_text()),
    failed_heldout_summary=failed,save_consumed_zero_passes=True)
(artifacts/'prepare.json').write_text(json.dumps(report,indent=2)+'\\n')
"""
)

RESUME = (
    COMMON
    + """
expected_path = artifacts.parents[2]/'prepare/session/artifacts/prepare.json'
expected = json.loads(expected_path.read_text())
assert os.getpid() != expected['worker_pid']
assert lab.forward_passes == expected['forward_passes']
assert lab.budget.limit == expected['pass_limit']
assert weights() == expected['edited_weights']
assert json.loads((artifacts/'evaluations/_registry.json').read_text()) == expected['registry']
spent_on_load = lab.forward_passes
assert digest(logits()) == expected['edited_logits']
before_retry = lab.forward_passes
try:
    lab.evaluate(heldout,name='retry-failed-heldout',held_out=True,max_passes=2)
except ValueError as error:
    assert 'reserved' in str(error)
else:
    raise AssertionError('failed held-out attempt was reusable after resume')
assert lab.forward_passes == before_retry
lab.undo(expected['adapter_edit'])
assert digest(logits()) == expected['base_only_logits']
lab.undo(expected['base_edit'])
assert weights() == expected['baseline_weights']
assert digest(logits()) == expected['baseline_logits']
report = dict(state='succeeded',device=str(lab.device),worker_pid=os.getpid(),
    source_worker_pid=expected['worker_pid'],base_dtype=str(model.get_parameter(base_name).dtype),
    adapter_dtype=str(model.get_parameter(adapter_name).dtype),
    load_consumed_zero_passes=True,inherited_forward_passes=spent_on_load,
    inherited_pass_limit=lab.budget.limit,forward_passes=lab.forward_passes,
    all_parameter_bytes_match=True,edited_logits_match=True,
    adapter_undo_output_exact=True,base_undo_output_exact=True,all_undo_bytes_exact=True,
    failed_heldout_reservation_preserved=True,rejected_retry_consumed_zero_passes=True)
(artifacts/'resume.json').write_text(json.dumps(report,indent=2)+'\\n')
"""
)


def file_hashes(root):
    """Give the controller exact file-level evidence for the transport roundtrip."""
    root = Path(root)
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def run_cli(root, *args, runner=subprocess.run):
    result = runner(
        [sys.executable, "-m", "oumi_weights_lab", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    with (root / "cli.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                dict(
                    args=list(map(str, args)),
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            )
            + "\n"
        )
    if result.returncode:
        raise RuntimeError(f"CLI failed: {result.stderr}\n{result.stdout}")
    return json.loads(result.stdout)


def make_assets(root):
    spec = importlib.util.spec_from_file_location(
        "runtime_assets", Path(__file__).with_name("validate_runtime.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_assets(root)


def run_phase(phase, output, device, *, bundle=None):
    from oumi_weights_lab.cli import _matches_worker
    from oumi_weights_lab.session import is_process_alive

    if phase not in {"prepare", "resume"} or device not in {"cpu", "cuda"}:
        raise ValueError("phase must be prepare/resume and device cpu/cuda")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    phase_root = root / phase
    phase_root.mkdir(exist_ok=False)
    session = phase_root / "session"
    pid = None
    report = {"state": "running", "phase": phase, "device": device}
    try:
        if phase == "prepare":
            model, adapter = make_assets(root)
        else:
            if bundle is None:
                raise ValueError("resume requires the downloaded bundle path")
            model, adapter = str(root / "model"), str(root / "adapter")
        args = [
            "start",
            "--session",
            session,
            "--model",
            model,
            "--adapter",
            adapter,
            "--device",
            device,
            "--dtype",
            "bfloat16",
            "--pass-limit",
            "32",
            "--snippet-timeout",
            "120",
        ]
        if bundle is not None:
            args.extend(["--resume", Path(bundle).resolve()])
        pid = run_cli(phase_root, *args)["pid"]
        snippet = phase_root / "checks.py"
        snippet.write_text(
            f"EXPECTED_DEVICE = {device!r}\n" + (PREPARE if phase == "prepare" else RESUME)
        )
        result = run_cli(
            phase_root,
            "send",
            "--session",
            session,
            "--file",
            snippet,
            "--wait",
            "--timeout",
            "150",
        )
        if result["state"] != "succeeded":
            raise RuntimeError(f"{phase} snippet failed: {result}")
        report.update(json.loads((session / f"artifacts/{phase}.json").read_text()))
        report["snippet_result"] = result
        if phase == "prepare":
            report["bundle_hashes"] = file_hashes(session / "artifacts/bundle")
    except BaseException as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if pid is not None:
                try:
                    report["stop"] = run_cli(
                        phase_root, "stop", "--session", session, "--timeout", "5"
                    )
                finally:
                    if is_process_alive(pid) and _matches_worker(pid, session):
                        os.kill(pid, signal.SIGKILL)
                        report["stop"] = run_cli(
                            phase_root, "stop", "--session", session, "--timeout", "5"
                        )
                report["worker_alive_after_stop"] = is_process_alive(pid)
                if report["worker_alive_after_stop"]:
                    raise RuntimeError("worker survived validation cleanup")
        except BaseException as error:
            report.update(state="failed", cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            (phase_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "resume"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run_phase(args.phase, args.output_dir, args.device, bundle=args.bundle), indent=2
        )
    )
