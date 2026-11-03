"""Offline tests for benchmark input freezing, grading records, and arm isolation."""

import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    # run_arm intentionally changes PATH in its isolated child process. Restore
    # it after in-process unit tests exercise that startup path.
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("scripts.validate_qwen_benchmark")


@pytest.fixture
def grader_source(tmp_path):
    source = tmp_path / "fixture_grader.py"
    source.write_text(
        "from mcqa import loop_signals\n"
        "def build_prompt(problem): return problem + '\\nShow working.'\n"
        "def extract(text): return text, 1\n"
        "def is_correct(pred, answer): return pred == answer\n"
    )
    return source


@pytest.fixture
def prepared(tmp_path, monkeypatch, benchmark, grader_source):
    rows = [
        {"unique_id": f"problem-{i}", "problem": f"Question {i}", "answer": str(i)}
        for i in range(500)
    ]
    requests = []

    class HfApi:
        def dataset_info(self, name, *, revision):
            requests.append(("resolve", name, revision))
            return SimpleNamespace(sha="a" * 40)

    def load_dataset(name, *, revision, split):
        requests.append(("load", name, revision, split))
        return rows

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            assert model == benchmark.MODELS["base"]["model"]
            assert kwargs == {
                "revision": benchmark.MODELS["base"]["revision"],
                "trust_remote_code": False,
            }
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": True,
            }
            return "USER: " + messages[0]["content"] + "\nASSISTANT: <think>"

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=HfApi))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Tokenizer))
    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda _: "fixture-version")
    original_loader = benchmark.load_grader
    digest = hashlib.sha256(grader_source.read_bytes()).hexdigest()
    monkeypatch.setattr(benchmark, "GRADER_SHA256", digest)
    monkeypatch.setattr(benchmark, "load_grader", lambda path: original_loader(path, digest))
    root = tmp_path / "prepared"
    manifest = benchmark.prepare(root, "requested-tag", 2, grader_source)
    return root, manifest, requests


def test_grader_hash_is_checked_and_unused_diagnostic_shim_restored(
    benchmark, grader_source, monkeypatch
):
    sentinel = SimpleNamespace(loop_signals=lambda _: "existing")
    monkeypatch.setitem(sys.modules, "mcqa", sentinel)
    digest = hashlib.sha256(grader_source.read_bytes()).hexdigest()
    grader = benchmark.load_grader(grader_source, digest)
    assert grader.build_prompt("question") == "question\nShow working."
    assert grader.extract("answer") == ("answer", 1)
    assert grader.is_correct("yes", "yes")
    with pytest.raises(NotImplementedError, match="unused|diagnostic"):
        grader.loop_signals("unused")
    assert sys.modules["mcqa"] is sentinel
    with pytest.raises(ValueError, match="SHA256"):
        benchmark.load_grader(grader_source, "0" * 64)


def test_prepare_freezes_resolved_revision_rows_and_thinking_prompts(prepared, benchmark):
    root, manifest, requests = prepared
    assert requests == [
        ("resolve", benchmark.DATASET, "requested-tag"),
        ("load", benchmark.DATASET, "a" * 40, "test"),
    ]
    assert manifest["dataset_revision"] == "a" * 40
    assert manifest["problem_count"] == 2
    assert manifest["full_protocol"] is False
    assert manifest["full_validated"] is False
    loaded, rows, prompts = benchmark.read_prepared(root)
    assert loaded == manifest
    assert [row["id"] for row in rows] == ["problem-0", "problem-1"]
    assert prompts[0]["prompt"] == "USER: Question 0\nShow working.\nASSISTANT: <think>"
    assert manifest["sampling"] == benchmark.SAMPLING
    assert manifest["models"] == benchmark.MODELS
    assert all(benchmark.sha256(root / name) == value for name, value in manifest["files"].items())


def test_changed_frozen_inputs_are_rejected_before_generation(prepared, benchmark):
    root, _, _ = prepared
    with (root / "prompts.jsonl").open("a") as stream:
        stream.write('{"id":"extra","prompt":"changed"}\n')
    with pytest.raises(ValueError, match="Frozen input changed"):
        benchmark.read_prepared(root)


def completions(answer):
    return SimpleNamespace(
        outputs=[
            SimpleNamespace(
                index=i,
                text=answer if i % 2 == 0 else "wrong",
                token_ids=list(range(i + 1)),
                finish_reason="length" if i == 3 else "stop",
            )
            for i in range(4)
        ]
    )


def test_grading_saves_each_raw_sample_and_counts_truncation(benchmark):
    rows = [{"id": "a", "answer": "A"}, {"id": "b", "answer": "B"}]
    grader = SimpleNamespace(
        extract=lambda text: (text, 1), is_correct=lambda pred, gold: pred == gold
    )
    stream = io.StringIO()
    result = benchmark.grade_outputs(rows, [completions("A"), completions("B")], grader, stream)
    saved = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert result == {
        "problems": 2,
        "samples_per_problem": 4,
        "total": 8,
        "correct": 4,
        "accuracy": 0.5,
        "cut": 2,
        "cut_share": 0.25,
        "tokens_sum": 20,
        "mean_tokens": 2.5,
    }
    assert len(saved) == 8
    assert saved[0] == {
        "id": "a",
        "sample_index": 0,
        "text": "A",
        "extracted": "A",
        "answer": "A",
        "correct": True,
        "token_count": 1,
        "finish_reason": "stop",
    }


@pytest.mark.parametrize(
    "invalid", ["missing_problem", "missing_sample", "duplicate_index", "abort"]
)
def test_incomplete_or_aborted_generation_cannot_get_a_score(benchmark, invalid):
    rows = [{"id": "a", "answer": "A"}]
    outputs = [completions("A")]
    if invalid == "missing_problem":
        outputs = []
    elif invalid == "missing_sample":
        outputs[0].outputs.pop()
    elif invalid == "duplicate_index":
        outputs[0].outputs[1].index = 0
    else:
        outputs[0].outputs[0].finish_reason = "abort"
    with pytest.raises(ValueError, match="every problem|samples|finish reason"):
        benchmark.grade_outputs(rows, outputs, None, io.StringIO())


def test_arm_version_mismatch_is_persisted_as_failure(prepared, benchmark, monkeypatch):
    root, _, _ = prepared
    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda _: "wrong-version")
    with pytest.raises(RuntimeError, match="dependency versions"):
        benchmark.run_arm(root, "base")
    summary = json.loads((root / "base" / "summary.json").read_text())
    assert summary["state"] == "failed"
    assert summary["full_validated"] is False
    assert "accuracy" not in summary


def test_diagnostic_arm_uses_pinned_models_and_sampling_without_claiming_full_validation(
    prepared,
    benchmark,
    monkeypatch,
):
    root, _, _ = prepared
    monkeypatch.setattr(benchmark.importlib.metadata, "version", benchmark.REQUIRED.__getitem__)
    monkeypatch.setattr(benchmark, "build_metadata", lambda: {"ninja_executable": "fixture/ninja"})
    fake_torch = SimpleNamespace(
        __version__="test-torch",
        version=SimpleNamespace(cuda="test-cuda"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _: SimpleNamespace(name="fixture GPU", total_memory=123),
            get_device_capability=lambda _: (9, 0),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    calls = []

    class LLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def generate(self, prompts, sampling):
            assert len(prompts) == 2
            assert sampling == benchmark.SAMPLING
            return [completions("0"), completions("1")]

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw)
    )
    result = benchmark.run_arm(root, "edited")
    assert calls == [
        {
            **benchmark.MODELS["edited"],
            "tokenizer": benchmark.MODELS["base"]["model"],
            "tokenizer_revision": benchmark.MODELS["base"]["revision"],
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.85,
            "max_model_len": 34816,
            "seed": 0,
        }
    ]
    assert result["state"] == "passed"
    assert result["total"] == 8
    assert result["full_protocol"] is result["full_validated"] is False
    assert len((root / "edited" / "samples.jsonl").read_text().splitlines()) == 8


def test_parent_uses_separate_child_processes_and_keeps_diagnostic_label(
    prepared, benchmark, monkeypatch
):
    root, _, _ = prepared
    commands = []

    def child(command, *, check):
        assert check is True
        commands.append(command)
        arm = command[-1]
        folder = root / arm
        folder.mkdir()
        benchmark.write_json(
            folder / "summary.json",
            {
                "state": "passed",
                "full_protocol": False,
                "full_validated": False,
                "problems": 2,
                "samples_per_problem": 4,
                "total": 8,
                "accuracy": 0.25 if arm == "base" else 0.5,
                "cut_share": 0.75 if arm == "base" else 0.25,
                "gpu": {"name": "same GPU", "total_memory": 123, "capability": [9, 0]},
            },
        )

    monkeypatch.setattr(benchmark.subprocess, "run", child)
    result = benchmark.run_parent(root)
    assert [command[-1] for command in commands] == ["base", "edited"]
    assert all(command[0] == sys.executable and "--arm" in command for command in commands)
    assert result["state"] == "passed"
    assert result["full_validated"] is False
    assert result["full_protocol"] is False


def test_failed_child_stops_the_parent_before_allocating_second_arm(
    prepared, benchmark, monkeypatch
):
    root, _, _ = prepared
    commands = []

    def child(command, *, check):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(benchmark.subprocess, "run", child)
    with pytest.raises(subprocess.CalledProcessError):
        benchmark.run_parent(root)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["state"] == "failed"
    assert summary["full_validated"] is False
    assert len(commands) == 1


def complete_arm(accuracy, cut_share):
    return {
        "state": "passed",
        "full_protocol": True,
        "problems": 500,
        "samples_per_problem": 4,
        "total": 2000,
        "accuracy": accuracy,
        "cut_share": cut_share,
        "gpu": {"name": "same GPU", "total_memory": 123, "capability": [9, 0]},
    }


@pytest.mark.parametrize(
    "edited_accuracy,edited_cut_share,expected",
    [
        (0.5, 0.3, True),
        (0.05, 0.3, False),
        (0.5, 0.95, False),
        (0.1, 0.3, False),
        (0.5, 0.9, False),
    ],
)
def test_full_validation_requires_both_predeclared_directions(
    benchmark,
    edited_accuracy,
    edited_cut_share,
    expected,
):
    arms = {
        "base": complete_arm(0.1, 0.9),
        "edited": complete_arm(edited_accuracy, edited_cut_share),
    }
    result = benchmark.compare_arms(arms, full_protocol=True)
    assert result["protocol_complete"] is True
    assert result["accuracy_delta"] == pytest.approx(edited_accuracy - 0.1)
    assert result["cut_share_delta"] == pytest.approx(edited_cut_share - 0.9)
    assert result["edited_higher_accuracy"] is (edited_accuracy > 0.1)
    assert result["edited_lower_cut_share"] is (edited_cut_share < 0.9)
    assert result["qualitative_effect_reproduced"] is expected
    assert result["full_validated"] is expected


@pytest.mark.parametrize("diagnostic", [True, False])
def test_positive_directions_do_not_validate_diagnostic_or_incomplete_protocol(
    benchmark, diagnostic
):
    arms = {"base": complete_arm(0.1, 0.9), "edited": complete_arm(0.5, 0.3)}
    if not diagnostic:
        arms["edited"]["total"] = 1999
    result = benchmark.compare_arms(arms, full_protocol=not diagnostic)
    assert result["edited_higher_accuracy"] is result["edited_lower_cut_share"] is True
    assert result["protocol_complete"] is False
    assert result["qualitative_effect_reproduced"] is result["full_validated"] is False


@pytest.mark.parametrize("positive", [True, False])
def test_parent_persists_criteria_before_scores_and_records_effect_validation(
    prepared,
    benchmark,
    monkeypatch,
    positive,
):
    root, _, _ = prepared
    monkeypatch.setattr(benchmark, "read_prepared", lambda _: ({"full_protocol": True}, [], []))
    calls = []

    def child(command, *, check):
        initial = json.loads((root / "summary.json").read_text())
        assert initial["validation_criteria"] == benchmark.VALIDATION_CRITERIA
        assert initial["state"] == "running"
        arm = command[-1]
        calls.append(arm)
        result = (
            complete_arm(0.1, 0.9)
            if arm == "base"
            else complete_arm(0.5 if positive else 0.05, 0.3)
        )
        (root / arm).mkdir()
        benchmark.write_json(root / arm / "summary.json", result)

    monkeypatch.setattr(benchmark.subprocess, "run", child)
    result = benchmark.run_parent(root)
    assert calls == ["base", "edited"]
    assert result["state"] == ("passed" if positive else "validation_failed")
    assert result["protocol_complete"] is True
    assert result["qualitative_effect_reproduced"] is result["full_validated"] is positive
    assert json.loads((root / "summary.json").read_text()) == result
    assert (root / "base" / "summary.json").exists()
    assert (root / "edited" / "summary.json").exists()


def test_absolute_virtualenv_python_exposes_its_bin_once_and_preserves_other_paths(
    benchmark,
    monkeypatch,
):
    monkeypatch.setattr(benchmark.sys, "executable", "/isolated/env/bin/python")
    monkeypatch.setenv("PATH", "/usr/bin:/isolated/env/bin:/bin:/isolated/env/bin")
    assert benchmark.prepare_child_path() == "/isolated/env/bin"
    expected = "/isolated/env/bin:/usr/bin:/bin"
    assert os.environ["PATH"] == expected
    benchmark.prepare_child_path()
    assert os.environ["PATH"] == expected


def test_child_path_keeps_virtualenv_symlink_directory_and_can_be_restored(
    benchmark,
    monkeypatch,
    tmp_path,
):
    original = os.environ.get("PATH")
    virtual_bin = tmp_path / "env" / "bin"
    virtual_bin.mkdir(parents=True)
    virtual_python = virtual_bin / "python"
    virtual_python.symlink_to(sys.executable)
    with monkeypatch.context() as patch:
        patch.setattr(benchmark.sys, "executable", str(virtual_python))
        patch.delenv("PATH", raising=False)
        assert benchmark.prepare_child_path() == str(virtual_bin)
        assert os.environ["PATH"] == str(virtual_bin)
    assert os.environ.get("PATH") == original


def test_optional_ninja_and_driver_metadata_do_not_add_protocol_version_pins(
    benchmark, monkeypatch
):
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/env/bin/{name}")
    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda name: "1.13.2")
    calls = []

    def query(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="570.124.06\n", stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", query)
    assert benchmark.build_metadata() == {
        "ninja_executable": "/env/bin/ninja",
        "ninja_package_version": "1.13.2",
        "nvidia_driver_version": ["570.124.06"],
    }
    assert calls == [["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]]
    assert "ninja" not in benchmark.REQUIRED
