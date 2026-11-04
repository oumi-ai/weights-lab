import importlib.util
import json
from pathlib import Path

import pytest


def harness():
    path = Path(__file__).with_name("validate_runtime.py")
    spec = importlib.util.spec_from_file_location("runtime_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_runtime_harness_checks_real_worker_and_backup(tmp_path):
    pytest.importorskip("peft")
    report = harness().validate("cpu", tmp_path / "runtime")
    assert report["state"] == "succeeded"
    assert report["observed_input_devices"] == ["cpu"]
    assert report["budget_used"] == report["forward_passes"] > 0
    assert report["backup_files_verified"] > 0
    assert report["export_reload_matches"]
    assert not report["worker_alive_after_stop"]
    assert report["stop"]["state"] == "stopped"


def test_runtime_harness_always_stops_worker_when_checks_fail(tmp_path):
    pytest.importorskip("peft")
    root = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="CLI failed"):
        harness().validate("cpu", root, checks="raise RuntimeError('intentional failure')")
    report = json.loads((root / "report.json").read_text())
    assert report["state"] == "failed"
    assert report["stop"]["state"] == "stopped"
    assert report["backup_files_verified"] > 0
    assert not report["worker_alive_after_stop"]
