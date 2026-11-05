import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def truncation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent))
    return importlib.import_module("validate_truncation")


def write_engine(root, arm, engine, cuts, prompt_ids=None):
    """Save one engine's samples: cuts maps a problem id to its samples' truncation flags."""
    folder = root / f"{arm}-{engine}"
    folder.mkdir(parents=True)
    samples = [
        {"id": key, "sample_index": i, "truncated": flag}
        for key, flags in cuts.items()
        for i, flag in enumerate(flags)
    ]
    (folder / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    ids = prompt_ids or {key: [1, 2, 3] for key in cuts}
    (folder / "prompt_ids.json").write_text(json.dumps(ids))
    summary = {"accuracy": 0.5, "mean_tokens": 10.0, "duration_seconds": 1.0}
    (folder / "summary.json").write_text(json.dumps(summary))


def test_protocol_is_the_benchmark_without_presence_penalty(truncation):
    from validate_qwen_benchmark import SAMPLING

    assert "presence_penalty" not in truncation.PROTOCOL
    assert truncation.PROTOCOL == {k: v for k, v in SAMPLING.items() if k != "presence_penalty"}
    assert truncation.PROTOCOL["max_tokens"] == 32768 and truncation.SAMPLES == 4


def test_stop_tokens_end_the_turn_and_the_text(truncation):
    vocabulary = {"<|im_end|>": 7, "<|endoftext|>": 8}
    tokenizer = SimpleNamespace(convert_tokens_to_ids=vocabulary.get, unk_token_id=0)
    assert truncation.stop_ids(tokenizer) == [7, 8]
    tokenizer.convert_tokens_to_ids = {"<|im_end|>": 7, "<|endoftext|>": 0}.get
    with pytest.raises(ValueError, match="lacks a stop token"):
        truncation.stop_ids(tokenizer)


def test_comparison_pairs_problems_and_brackets_the_difference(truncation, tmp_path):
    vllm = {f"p{i}": [i % 2 == 0] * 4 for i in range(20)}
    lab = {key: flags[:3] + [not flags[3]] if key == "p0" else flags for key, flags in vllm.items()}
    write_engine(tmp_path, "edited", "vllm", vllm)
    write_engine(tmp_path, "edited", "lab", lab)
    result = truncation.compare(tmp_path, "edited")
    assert result["problems"] == 20
    assert result["vllm_cut_share"] == 0.5
    assert result["lab_cut_share"] == pytest.approx(0.5 - 0.25 / 20)
    assert result["difference"] == pytest.approx(-0.0125)
    low, high = result["interval"]
    assert low <= result["difference"] <= high and result["interval_contains_zero"]
    assert result["prompt_ids_identical"]


def test_a_systematic_gap_falls_outside_the_interval(truncation, tmp_path):
    write_engine(tmp_path, "base", "vllm", {f"p{i}": [False] * 4 for i in range(20)})
    write_engine(tmp_path, "base", "lab", {f"p{i}": [True, True, False, False] for i in range(20)})
    result = truncation.compare(tmp_path, "base")
    assert result["difference"] == 0.5 and not result["interval_contains_zero"]


def test_different_prompts_or_problems_never_match(truncation, tmp_path):
    cuts = {"p0": [True] * 4, "p1": [False] * 4}
    write_engine(tmp_path, "base", "vllm", cuts)
    write_engine(tmp_path, "base", "lab", cuts, prompt_ids={"p0": [1, 2, 3], "p1": [1, 2, 4]})
    assert not truncation.compare(tmp_path, "base")["prompt_ids_identical"]
    write_engine(tmp_path, "edited", "vllm", cuts)
    write_engine(tmp_path, "edited", "lab", {"p0": [True] * 4, "p2": [False] * 4})
    with pytest.raises(ValueError, match="different problems"):
        truncation.compare(tmp_path, "edited")


def test_bootstrap_interval_is_seeded_and_symmetric_for_symmetric_differences(truncation):
    differences = [0.25, -0.25, 0.0, 0.5, -0.5, 0.0] * 5
    first = truncation.bootstrap_interval(differences, 2000, 0, 0.95)
    assert first == truncation.bootstrap_interval(differences, 2000, 0, 0.95)
    assert first[0] < 0 < first[1]
    assert truncation.bootstrap_interval([0.0] * 10, 2000, 0, 0.95) == (0.0, 0.0)


def test_lab_retries_at_half_the_batch_after_running_out_of_memory(
    truncation, tmp_path, monkeypatch
):
    calls = []

    def run(command, env, check):
        batch = int(command[command.index("--lab-batch-size") + 1])
        calls.append((batch, env.get("PYTORCH_CUDA_ALLOC_CONF")))
        folder = tmp_path / "edited-lab"
        folder.mkdir()
        error = "OutOfMemoryError: CUDA out of memory" if batch == 160 else None
        (folder / "summary.json").write_text(json.dumps({"error": error}))
        return SimpleNamespace(returncode=1 if error else 0)

    monkeypatch.setattr(truncation.subprocess, "run", run)
    truncation.run_engine(tmp_path, "edited", "lab", 32768, 160, "cuda")
    assert calls == [(160, "expandable_segments:True"), (80, "expandable_segments:True")]
    assert (tmp_path / "edited-lab-out-of-memory-at-160").is_dir()


def test_other_failures_and_vllm_failures_are_not_retried(truncation, tmp_path, monkeypatch):
    def run(command, env, check):
        engine = command[command.index("--engine") + 1]
        folder = tmp_path / f"base-{engine}"
        folder.mkdir()
        error = "OutOfMemoryError: CUDA out of memory" if engine == "vllm" else "ValueError: x"
        (folder / "summary.json").write_text(json.dumps({"error": error}))
        assert "PYTORCH_CUDA_ALLOC_CONF" not in env or engine == "lab"
        return SimpleNamespace(returncode=1)

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(truncation.subprocess, "run", run)
    for engine in ("vllm", "lab"):
        with pytest.raises(RuntimeError, match=f"base on {engine} failed"):
            truncation.run_engine(tmp_path, "base", engine, 32768, 160, "cuda")


def test_only_the_full_protocol_on_both_models_completes_verification(
    truncation, tmp_path, monkeypatch
):
    monkeypatch.setattr(truncation, "run_engine", lambda *args: None)
    match = {"interval_contains_zero": True, "prompt_ids_identical": True}
    monkeypatch.setattr(truncation, "compare", lambda root, arm: dict(match))
    arms, engines = list(truncation.MODELS), list(truncation.ENGINES)
    for name in ("full", "smoke", "gap"):
        (tmp_path / name).mkdir()  # prepare() creates the output directory first.
    summary = truncation.run_parent(tmp_path / "full", arms, engines, 32768, 160, "cuda")
    assert summary["state"] == "passed" and summary["verification_complete"]
    smoke = truncation.run_parent(tmp_path / "smoke", arms, engines, 64, 160, "cuda")
    assert smoke["state"] == "passed" and not smoke["verification_complete"]
    match["interval_contains_zero"] = False
    gap = truncation.run_parent(tmp_path / "gap", arms, engines, 32768, 160, "cuda")
    assert gap["state"] == "mismatch" and not gap["verification_complete"]
    saved = json.loads((tmp_path / "gap" / "truncation-summary.json").read_text())
    assert saved["criteria"] == truncation.CRITERIA and saved["state"] == "mismatch"
