"""Run CUDA and transport checks on a finite-lived Modal sandbox, then destroy it."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from weights_lab.launchers import ModalLauncher


def execute(sandbox, *command, source=None, timeout=300):
    process = sandbox.exec(*command, timeout=timeout)
    if source is not None:
        process.stdin.write(source)
    process.stdin.write_eof()
    process.stdin.drain()
    stdout, stderr = process.stdout.read(), process.stderr.read()
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command, stdout, stderr)
    return stdout


def validate(output, *, gpu="L4", lifetime=1200, source=None):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(source or Path(__file__).resolve().parents[1]).resolve()
    launcher = ModalLauncher(app_name="weights-lab-validation")
    report = {"state": "running", "gpu_request": gpu, "lifetime_seconds": lifetime}
    started = time.time()

    def checkpoint():
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        info = launcher.start(
            "sshleifer/tiny-gpt2",
            "/lab/transport",
            source=source,
            gpu=gpu,
            lifetime=lifetime,
            device="cuda",
            dtype="float32",
            pass_limit=32,
            revision="5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be",
        )
        report.update(info)
        report["state"] = "running"
        checkpoint()
        print(json.dumps({"stage": "sandbox_created", **info}), flush=True)
        sandbox = launcher._sandbox()
        hardware = execute(
            sandbox, "nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"
        )
        report["hardware"] = hardware.strip()
        source_code = (
            "import json,torch\n"
            "assert lab.device.type == 'cuda'\n"
            "inputs=tokenizer('hello world',return_tensors='pt').to(lab.device)\n"
            "with torch.no_grad():\n    logits=model(**inputs).logits\n"
            "assert logits.device.type == 'cuda'\n"
            "(artifacts/'transport.json').write_text(json.dumps({'cuda':True,'passes':lab.forward_passes}))\n"
        )
        result = launcher.run(
            ["send", "--session", "/lab/transport", "--wait", "--timeout", "180"], input=source_code
        )
        report["transport_snippet"] = json.loads(result.stdout)
        assert report["transport_snippet"]["state"] == "succeeded"
        print(json.dumps({"stage": "cuda_transport_passed"}), flush=True)
        execute(sandbox, "python", "-m", "pip", "install", "peft>=0.13")
        execute(
            sandbox,
            "python",
            "-c",
            "import pathlib,sys; "
            "pathlib.Path('/tmp/validate_runtime.py').write_bytes(sys.stdin.buffer.read())",
            source=(source / "scripts/validate_runtime.py").read_bytes(),
        )
        execute(
            sandbox,
            "python",
            "/tmp/validate_runtime.py",
            "--device",
            "cuda",
            "--output-dir",
            "/lab/runtime",
        )
        launcher.download("/lab/runtime/report.json", root / "cuda-runtime.json")
        launcher.download("/lab/runtime/session.tar.gz", root / "cuda-runtime.tar.gz")
        runtime = json.loads((root / "cuda-runtime.json").read_text())
        assert runtime["state"] == "succeeded" and not runtime["worker_alive_after_stop"]
        report["cuda_runtime"] = runtime
        report["stop"] = json.loads(launcher.stop("/lab/transport").stdout)
        assert report["stop"]["state"] == "stopped"
        launcher.run(["backup", "--session", "/lab/transport", "--output", "/lab/transport.tar.gz"])
        launcher.download("/lab/transport.tar.gz", root / "transport.tar.gz")
        report["state"] = "succeeded"
        print(json.dumps({"stage": "checks_and_downloads_passed"}), flush=True)
    except BaseException as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        if isinstance(error, subprocess.CalledProcessError):
            (root / "failure-stdout.log").write_text(str(error.stdout or ""))
            (root / "failure-stderr.log").write_text(str(error.stderr or ""))
        raise
    finally:
        try:
            if launcher.sandbox_id:
                sandbox = launcher._sandbox()
                launcher.teardown()
                report["sandbox_returncode_after_teardown"] = sandbox.poll()
                report["sandbox_terminated"] = (
                    report["sandbox_returncode_after_teardown"] is not None
                )
                if not report["sandbox_terminated"]:
                    raise RuntimeError("sandbox is still running after teardown")
        finally:
            report["duration_seconds"] = time.time() - started
            checkpoint()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--lifetime", type=int, default=1200)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    validate(args.output_dir, gpu=args.gpu, lifetime=args.lifetime, source=args.source)
