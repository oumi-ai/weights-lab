"""The quantized validator's wiring, with a stand-in harness; CUDA and MPS run it for real."""

import importlib.util
from pathlib import Path


def load(name):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checks_are_valid_python():
    validator = load("validate_quantized")
    compile(validator.prefix("nf4") + validator.CHECKS, "checks.py", "exec")
    compile(validator.RESUMED.replace("__NAMES__", "[]"), "resumed.py", "exec")


def test_checks_cover_every_stage():
    checks = load("validate_quantized").CHECKS
    for stage in (
        "lab.parts()",
        "lab.evaluate",
        "lab.undo",
        "quantization step",
        "lab.heads",
        "lab.save_edits",
        "--resume",
        "lab.export",
        "from_pretrained",
    ):
        assert stage in checks


def test_validator_starts_a_quantized_worker_without_an_adapter_or_pass_cap(tmp_path, monkeypatch):
    validator = load("validate_quantized")
    calls = []

    class Harness:
        def validate(self, device, output_dir, model, **options):
            calls.append((device, output_dir, model, options))
            return {"state": "succeeded"}

    monkeypatch.setattr(validator, "runtime", Harness)
    assert validator.validate("mps", tmp_path / "out", "org/small", quantize="fp4", wait=600) == {
        "state": "succeeded"
    }
    ((device, output_dir, model, options),) = calls
    assert (device, output_dir, model) == ("mps", tmp_path / "out", "org/small")
    assert options["checks"] == validator.prefix("fp4") + validator.CHECKS
    assert options["options"] == {
        "quantize": "fp4",
        "dtype": "float32",
        "pass-limit": None,
        "snippet-timeout": 600,
    }
    assert options["wait"] == 600
    assert options["assets"](tmp_path, "org/small") == ("org/small", None)


def test_defaults_are_a_public_seven_billion_instruct_model_and_nf4(tmp_path, monkeypatch):
    validator = load("validate_quantized")
    assert validator.DEFAULT_MODEL == "Qwen/Qwen2.5-7B-Instruct"
    seen = {}

    class Harness:
        def validate(self, device, output_dir, model, **options):
            seen.update(model=model, options=options)
            return {}

    monkeypatch.setattr(validator, "runtime", Harness)
    validator.validate("cuda", tmp_path / "out")
    assert seen["model"] == validator.DEFAULT_MODEL
    assert seen["options"]["options"]["quantize"] == "nf4"
