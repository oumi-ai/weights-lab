"""The multi-GPU validator's wiring, with a stand-in harness; GPU nodes run it for real."""

import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checks_are_valid_python():
    compile(load("validate_multigpu").CHECKS, "checks.py", "exec")


def test_checks_refuse_to_skip_resume_unless_allowed():
    checks = load("validate_multigpu").CHECKS
    assert "assert room or ALLOW_PARTIAL" in checks


def test_validator_starts_a_balanced_cuda_worker_without_a_pass_cap(tmp_path, monkeypatch):
    validator = load("validate_multigpu")
    calls = []

    class Harness:
        def validate(self, device, output_dir, model, **options):
            calls.append((device, output_dir, model, options))
            return {"state": "succeeded"}

    monkeypatch.setattr(validator, "runtime", Harness)
    assert validator.validate(tmp_path / "out", "org/big", adapter=False, wait=600) == {
        "state": "succeeded"
    }
    ((device, output_dir, model, options),) = calls
    assert (device, output_dir, model) == ("cuda", tmp_path / "out", "org/big")
    assert options["checks"] == "ALLOW_PARTIAL = False\n" + validator.CHECKS
    assert options["options"] == {
        "device-map": "balanced",
        "dtype": "float32",
        "pass-limit": None,
        "snippet-timeout": 600,
    }
    assert options["wait"] == 600
    assert options["assets"](tmp_path, "org/big") == ("org/big", None)


def test_default_assets_are_a_four_block_model_with_an_adapter(tmp_path):
    pytest.importorskip("peft")
    model, adapter = load("validate_multigpu").make_assets(tmp_path)
    assert json.loads((Path(model) / "config.json").read_text())["n_layer"] == 4
    assert (Path(adapter) / "adapter_config.json").is_file()


def test_runtime_harness_takes_start_options_and_assets(tmp_path):
    pytest.importorskip("peft")
    harness = load("validate_runtime")
    checks = (
        "import json\n"
        "config = json.loads((artifacts.parent / 'config.json').read_text())\n"
        "(artifacts / 'runtime.json').write_text(json.dumps({'config': config}))\n"
    )
    report = harness.validate(
        "cpu",
        tmp_path / "runtime",
        checks=checks,
        options={"pass-limit": None, "snippet-timeout": 60},
        assets=lambda root, model: harness.make_assets(root, model),
        wait=60,
    )
    assert report["state"] == "succeeded"
    assert report["config"]["pass_limit"] is None
    assert report["config"]["snippet_timeout"] == 60
    assert report["config"]["adapter"]
