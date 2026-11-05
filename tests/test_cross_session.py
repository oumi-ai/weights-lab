"""Runs saved by different sessions pair only when the caller opts in."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from oumi_weights_lab import Lab  # noqa: E402

ROWS = [
    {"id": str(i), "prompt": "hello", "answer": "A", "options": [" yes", " no"]} for i in range(4)
]
SETUP_FIELDS = {
    "devices",
    "dtypes",
    "torch",
    "transformers",
    "peft",
    "adapter",
    "attention",
    "device_map",
    "quantization",
    "allow_hub_kernels",
    "model_id",
    "model_commit",
}


def other_lab(lab, tokenizer, tmp_path, name="second", model=None):
    return Lab(model or lab.model, tokenizer, tmp_path / name, model_id="test-model")


def edit_summary(run, change):
    path = Path(run["run_dir"]) / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    change(summary)
    path.write_text(json.dumps(summary), encoding="utf-8")


def test_compare_runs_refuses_other_session_without_opt_in(tiny_lab, tiny_tokenizer, tmp_path):
    second = other_lab(tiny_lab, tiny_tokenizer, tmp_path)
    before = tiny_lab.evaluate(ROWS)
    after = second.evaluate(ROWS)
    with pytest.raises(ValueError, match=r"session_id.*cross_session=True"):
        tiny_lab.compare_runs(before, after)
    with pytest.raises(ValueError, match=r"cross_session=True"):
        tiny_lab.diff(before, after)
    assert "cross_session" not in tiny_lab.compare_runs(before, before)


def test_hint_is_left_out_when_another_key_differs(tiny_lab, tiny_tokenizer, tmp_path):
    second = other_lab(tiny_lab, tiny_tokenizer, tmp_path)
    before = tiny_lab.evaluate(ROWS)
    after = second.evaluate(ROWS, batch_size=1)
    with pytest.raises(ValueError, match="scoring_config") as error:
        tiny_lab.compare_runs(before, after)
    assert "cross_session=True" not in str(error.value)


def test_cross_session_export_reload_has_no_flips(tiny_lab, tiny_tokenizer, tmp_path):
    from transformers import AutoModelForCausalLM

    before = tiny_lab.evaluate(ROWS)
    path = tiny_lab.export(tmp_path / "export")
    reloaded = AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager").eval()
    second = other_lab(tiny_lab, tiny_tokenizer, tmp_path, model=reloaded)
    after = second.evaluate(ROWS)
    report = tiny_lab.compare_runs(before, after["run_dir"], cross_session=True)
    assert report["aggregate"]["fixed"] == report["aggregate"]["broken"] == 0
    cross = report["cross_session"]
    assert cross["before_session"] != cross["after_session"]
    assert isinstance(cross["setup_differences"], list)
    assert "different sessions" in cross["caveat"]
    assert tiny_lab.diff(before, after["run_dir"], cross_session=True) == []


def test_same_session_opt_in_is_harmless(tiny_lab):
    run = tiny_lab.evaluate(ROWS)
    report = tiny_lab.compare_runs(run, run, cross_session=True)
    assert report["cross_session"]["before_session"] == report["cross_session"]["after_session"]
    assert report["cross_session"]["setup_differences"] == []


def test_cross_session_still_requires_protocol(tiny_lab, tiny_tokenizer, tmp_path):
    second = other_lab(tiny_lab, tiny_tokenizer, tmp_path)
    before = tiny_lab.evaluate(ROWS)
    batched = second.evaluate(ROWS, batch_size=1)
    with pytest.raises(ValueError, match="scoring_config"):
        tiny_lab.compare_runs(before, batched, cross_session=True)
    with pytest.raises(ValueError, match="scoring_config"):
        tiny_lab.diff(before, batched, cross_session=True)
    other_mode = second.evaluate(
        [{"id": "0", "prompt": "hello", "answer": "yes"}], mode="sampled", name="sampled"
    )
    with pytest.raises(ValueError, match="mode"):
        tiny_lab.compare_runs(before, other_mode, cross_session=True)


def test_cross_session_requires_same_held_out_status(tiny_lab, tiny_tokenizer, tmp_path):
    second = other_lab(tiny_lab, tiny_tokenizer, tmp_path)
    held = tiny_lab.evaluate(ROWS, held_out=True, max_passes=100)
    dev = second.evaluate(ROWS)
    with pytest.raises(ValueError, match="held_out differs") as error:
        tiny_lab.compare_runs(held, dev, cross_session=True)
    assert "same session" not in str(error.value)
    assert "the same mode and scoring protocol/settings" in str(error.value)
    with pytest.raises(ValueError, match="held_out differs"):
        tiny_lab.diff(held, dev, cross_session=True)


def test_cross_session_validates_the_flag(tiny_lab):
    run = tiny_lab.evaluate(ROWS)
    for value in (1, "yes", None):
        with pytest.raises(ValueError, match="cross_session must be a bool"):
            tiny_lab.compare_runs(run, run, cross_session=value)
        with pytest.raises(ValueError, match="cross_session must be a bool"):
            tiny_lab.diff(run, run, cross_session=value)


def test_setup_block_recorded(tiny_lab):
    summary = tiny_lab.evaluate(ROWS)
    setup = json.loads((Path(summary["run_dir"]) / "summary.json").read_text())["setup"]
    assert set(setup) == SETUP_FIELDS
    assert setup["devices"] == ["cpu"]
    assert setup["dtypes"] == ["torch.float32"]
    assert setup["adapter"] is False
    assert setup["peft"] is None
    assert setup["attention"] == tiny_lab.model.config._attn_implementation
    assert setup["device_map"] is False
    assert setup["quantization"] is None
    assert setup["allow_hub_kernels"] is False
    assert setup["model_id"] == "test-model"
    assert setup["torch"] == torch.__version__
    json.dumps(setup)


def test_cross_session_reports_setup_differences(tiny_lab, tiny_tokenizer, tmp_path):
    second = other_lab(tiny_lab, tiny_tokenizer, tmp_path)
    before = tiny_lab.evaluate(ROWS)
    after = second.evaluate(ROWS)
    edit_summary(after, lambda s: s["setup"].update(dtypes=["torch.bfloat16"]))
    report = tiny_lab.compare_runs(before, after, cross_session=True)
    assert report["cross_session"]["setup_differences"] == ["dtypes"]
    assert report["cross_session"]["after_setup"]["dtypes"] == ["torch.bfloat16"]


def test_legacy_runs_without_setup(tiny_lab, tiny_tokenizer, tmp_path):
    second = other_lab(tiny_lab, tiny_tokenizer, tmp_path)
    before = tiny_lab.evaluate(ROWS)
    after = second.evaluate(ROWS)
    edit_summary(after, lambda s: s.pop("setup"))
    report = tiny_lab.compare_runs(before, after, cross_session=True)
    assert report["cross_session"]["setup_differences"] == ["setup"]
    assert report["cross_session"]["after_setup"] is None


def test_diff_names_the_other_key_when_the_session_also_differs(tiny_lab, tiny_tokenizer, tmp_path):
    second = other_lab(tiny_lab, tiny_tokenizer, tmp_path)
    before = tiny_lab.evaluate(ROWS)
    after = second.evaluate(ROWS, batch_size=1)
    with pytest.raises(ValueError, match="scoring_config") as error:
        tiny_lab.diff(before, after)
    assert "cross_session=True" not in str(error.value)
