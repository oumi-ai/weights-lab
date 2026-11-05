"""Quantized edits save into bundles, resume exactly, and export through save_pretrained."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

import test_quantized_edits as edits  # noqa: E402

from oumi_weights_lab.lab import Lab  # noqa: E402
from oumi_weights_lab.persistence import (  # noqa: E402
    _fingerprint,
    _manifest_digest,
    _state_key,
)
from oumi_weights_lab.quantized import lookup  # noqa: E402

Q_PROJ, PROMPT = edits.Q_PROJ, edits.PROMPT
checkpoint, load, logits = edits.checkpoint, edits.load, edits.logits
METHODS = ("nf4", "fp4", "int8")
# The packed parts the codec snapshots: the bytes, then the scales.
SCALES = {"nf4": "absmax", "fp4": "absmax", "int8": "scales"}
NORM = "model.norm.weight"


def packed(lab, name=Q_PROJ):
    """Copies of the packed bytes and the scales of one quantized weight."""
    parts = lookup(lab.model, name).snapshot()
    return tuple(parts[key] for key in sorted(parts))


def same_packed(first, second):
    return len(first) == len(second) and all(map(torch.equal, first, second))


def rewrite_manifest(bundle, **changes):
    """Change manifest fields of a saved bundle and fix its checksum."""
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("sha256")
    for key, value in changes.items():
        manifest[key] = value(manifest[key]) if callable(value) else value
    manifest["sha256"] = _manifest_digest(manifest)
    path.write_text(json.dumps(manifest))


def make_edits(lab):
    """Two quantized edits that overlap a block, then a float edit."""
    lab.edit(Q_PROJ, lambda value: value * 2 + 0.05, index=(0, slice(0, 8)), note="first")
    lab.edit(Q_PROJ, lambda value: value - 0.3, index=(0, slice(4, 12)), note="second")
    lab.zero(NORM, note="float")


@pytest.mark.parametrize("method", METHODS)
def test_fingerprint_stable_across_loads_and_first_forward(checkpoint, load, method):
    path = checkpoint("llama")
    lab = load(path, quantize=method)
    first = _fingerprint(lab)
    logits(lab)
    assert _fingerprint(lab) == first
    assert _fingerprint(load(path, quantize=method)) == first
    lab.zero(Q_PROJ, index=(0, slice(0, 8)))
    assert _fingerprint(lab) != first
    lab.undo()
    assert _fingerprint(lab) == first


def test_fingerprint_differs_by_quantization(checkpoint, load):
    path = checkpoint("llama")
    found = {_fingerprint(load(path, quantize=m)) for m in ("nf4", "fp4", "int8", None)}
    assert len(found) == 4


@pytest.mark.parametrize("method", METHODS)
def test_bundle_round_trip(checkpoint, load, tmp_path, method):
    path = checkpoint("llama")
    fresh = load(path, quantize=method)
    baseline = {"packed": packed(fresh), "logits": logits(fresh)}
    baseline_norm = fresh.model.get_parameter(NORM).detach().clone()
    lab = load(path, quantize=method)
    make_edits(lab)
    live = {"packed": packed(lab), "logits": logits(lab), "edits": lab.edits()}
    assert not same_packed(live["packed"], baseline["packed"])
    bundle = lab.save_edits(tmp_path / "bundle")
    # Saving reads state without changing it.
    assert same_packed(live["packed"], packed(lab)) and torch.equal(logits(lab), live["logits"])
    resumed = load(path, quantize=method, resume=str(bundle))
    assert same_packed(live["packed"], packed(resumed))
    assert torch.equal(logits(resumed), live["logits"])
    assert resumed.edits() == live["edits"]
    assert [e["quantization"]["type"] for e in resumed.edits()[:2]] == [method, method]
    assert resumed.active_edit_ids == lab.active_edit_ids
    for _ in range(3):
        resumed.undo()
    assert same_packed(baseline["packed"], packed(resumed))
    assert torch.equal(logits(resumed), baseline["logits"])
    assert torch.equal(resumed.model.get_parameter(NORM), baseline_norm)
    assert _fingerprint(resumed) == _fingerprint(fresh)


@pytest.mark.parametrize("method", METHODS)
def test_resume_after_partial_undo(checkpoint, load, tmp_path, method):
    path = checkpoint("llama")
    lab = load(path, quantize=method)
    make_edits(lab)
    lab.undo()
    lab.undo()
    live = packed(lab)
    resumed = load(path, quantize=method, resume=str(lab.save_edits(tmp_path / "bundle")))
    assert same_packed(live, packed(resumed))
    assert resumed.edits() == lab.edits()
    resumed.undo()
    assert same_packed(packed(load(path, quantize=method)), packed(resumed))


@pytest.mark.parametrize("method", METHODS)
def test_bundle_format_version(checkpoint, load, tmp_path, method):
    from safetensors import safe_open

    path = checkpoint("llama")
    lab = load(path, quantize=method)
    lab.zero(NORM)
    plain = lab.save_edits(tmp_path / "plain")
    assert json.loads((plain / "manifest.json").read_text())["format_version"] == 1
    lab.zero(Q_PROJ, index=(0, slice(0, 8)))
    mixed = lab.save_edits(tmp_path / "mixed")
    manifest = json.loads((mixed / "manifest.json").read_text())
    assert manifest["format_version"] == 2
    entry = next(e for e in manifest["active"] if "quantization" in e)
    block = entry["quantization"]
    assert (block["type"], block["blocksize"], block["shape"]) == (method, 64, [64, 64])
    assert block["dtype"] == "torch.float32"
    assert block["report"] == lab.edits()[1]["quantization"]
    with safe_open(str(mixed / "edits.safetensors"), "pt") as stream:
        keys = set(stream.keys())
    assert block["parts"] == sorted(["data", SCALES[method]])
    for part in block["parts"]:
        assert f"{entry['id']}-before.{part}" in keys and f"{entry['id']}-after.{part}" in keys
    assert load(path, quantize=method, resume=str(plain)).active_edit_ids == ["edit-000001"]
    assert len(load(path, quantize=method, resume=str(mixed)).active_edit_ids) == 2
    rewrite_manifest(mixed, format_version=3)
    with pytest.raises(ValueError, match="unsupported edit bundle format"):
        load(path, quantize=method, resume=str(mixed))


def test_resume_refuses_different_quantization(checkpoint, load, tmp_path):
    path = checkpoint("llama")
    lab = load(path, quantize="nf4")
    lab.zero(Q_PROJ, index=(0, slice(0, 8)))
    bundle = lab.save_edits(tmp_path / "bundle")
    with pytest.raises(ValueError, match="nf4") as error:
        load(path, quantize="fp4", resume=str(bundle))
    assert "fp4" in str(error.value) and "quantiz" in str(error.value)
    with pytest.raises(ValueError, match="nf4") as error:
        load(path, resume=str(bundle))
    assert "float" in str(error.value)
    rewrite_manifest(bundle, quantization=lambda saved: {**saved, "blocksize": 128})
    with pytest.raises(ValueError, match="128") as error:
        load(path, quantize="nf4", resume=str(bundle))
    assert "64" in str(error.value)


def test_int8_bundle_does_not_resume_into_a_4bit_lab(checkpoint, load, tmp_path):
    path = checkpoint("llama")
    lab = load(path, quantize="int8")
    lab.zero(Q_PROJ, index=(0, slice(0, 8)))
    bundle = lab.save_edits(tmp_path / "bundle")
    with pytest.raises(ValueError, match="int8") as error:
        load(path, quantize="nf4", resume=str(bundle))
    assert "nf4" in str(error.value) and "threshold" in str(error.value)


def test_float_bundle_into_quantized_lab_mentions_quantization(checkpoint, load, tmp_path):
    path = checkpoint("llama")
    lab = load(path)
    lab.zero(NORM)
    bundle = lab.save_edits(tmp_path / "bundle")
    with pytest.raises(ValueError, match="quantiz"):
        load(path, quantize="nf4", resume=str(bundle))
    quantized = load(path, quantize="nf4")
    quantized.zero(NORM)
    other = quantized.save_edits(tmp_path / "other")
    with pytest.raises(ValueError, match="quantiz"):
        load(path, resume=str(other))


@pytest.mark.parametrize("method", METHODS)
def test_resume_refuses_mismatched_before_state(checkpoint, load, tmp_path, method):
    from safetensors.torch import load_file, save_file

    from oumi_weights_lab.persistence import _file_digest

    path = checkpoint("llama")
    lab = load(path, quantize=method)
    lab.zero(Q_PROJ, index=(0, slice(0, 8)))
    bundle = lab.save_edits(tmp_path / "bundle")
    payload = load_file(str(bundle / "edits.safetensors"))
    payload[f"edit-000001-before.{SCALES[method]}"][0] += 1
    save_file(payload, str(bundle / "edits.safetensors"))
    rewrite_manifest(bundle, payload_sha256=_file_digest(bundle / "edits.safetensors"))
    state = packed(load(path, quantize=method))
    target = load(path, quantize=method)
    with pytest.raises(ValueError, match="does not match replay state"):
        target.load_edits(bundle)
    assert same_packed(state, packed(target))
    assert target.edits() == [] and not (target.artifacts / "evaluations").exists()


@pytest.mark.parametrize("method", METHODS)
def test_export_reload_identical(checkpoint, load, tmp_path, method):
    from transformers import AutoModelForCausalLM

    lab = load(checkpoint("llama"), quantize=method)
    make_edits(lab)
    live = {"packed": packed(lab), "logits": logits(lab)}
    exported = lab.export(tmp_path / "exported")
    manifest = json.loads((exported / "oumi_weights_lab_manifest.json").read_text())
    assert manifest["quantization"]["method"] == method
    model = AutoModelForCausalLM.from_pretrained(exported).eval()
    assert lookup(model, Q_PROJ).method == method
    assert same_packed(packed(Lab(model, lab.tokenizer, tmp_path / "view")), live["packed"])
    reloaded = Lab(model, lab.tokenizer, tmp_path / "reloaded")
    assert torch.equal(logits(reloaded), live["logits"])
    # Exporting changed nothing in the live model.
    assert same_packed(live["packed"], packed(lab)) and torch.equal(logits(lab), live["logits"])


@pytest.mark.parametrize("method", METHODS)
def test_export_bytes_equal_live_bytes(checkpoint, load, tmp_path, method):
    from safetensors import safe_open

    lab = load(checkpoint("llama"), quantize=method)
    logits(lab)  # The first forward is where a CPU repack would happen.
    make_edits(lab)
    exported = lab.export(tmp_path / "exported")
    checked = 0
    with safe_open(str(exported / "model.safetensors"), "pt") as stream:
        for name, _ in lab.model.named_parameters():
            weight = lookup(lab.model, name)
            if weight is not None:
                for part, tensor in weight.live().items():
                    assert torch.equal(stream.get_tensor(_state_key(name, part)), tensor.cpu())
                checked += 1
    assert checked >= 7


def test_export_refuses_peft_merge_into_quantized_base(checkpoint, load, tmp_path):
    peft = pytest.importorskip("peft")
    lab = load(checkpoint("llama"), quantize="nf4")
    config = peft.LoraConfig(task_type="CAUSAL_LM", target_modules=["q_proj"], r=2)
    adapted = Lab(peft.get_peft_model(lab.model, config), lab.tokenizer, tmp_path / "peft")
    with pytest.raises(ValueError, match="merging an adapter into a quantized base"):
        adapted.export(tmp_path / "exported")
    assert not (tmp_path / "exported").exists()
    assert not list(tmp_path.glob(".oumi-weights-lab-export-*"))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("forward_before_save", [False, True])
def test_save_before_or_after_the_first_forward(
    checkpoint, load, tmp_path, method, forward_before_save
):
    """An int8 layer keeps its scales on the weight until its first forward, then on the layer."""
    path = checkpoint("llama")
    baseline = packed(load(path, quantize=method))
    lab = load(path, quantize=method)
    if forward_before_save:
        logits(lab)
    make_edits(lab)
    if forward_before_save:
        logits(lab)
    live = packed(lab)
    bundle = lab.save_edits(tmp_path / "bundle")
    assert same_packed(live, packed(lab))
    resumed = load(path, quantize=method, resume=str(bundle))  # No forward has run in it.
    assert same_packed(live, packed(resumed))
    assert torch.equal(logits(resumed), logits(lab))
    for _ in range(3):
        resumed.undo()
    assert same_packed(baseline, packed(resumed))
    assert torch.equal(logits(resumed), logits(load(path, quantize=method)))


@pytest.mark.parametrize("method", ["nf4", "int8"])
def test_peft_wrapped_quantized_base_saves_resumes_and_undoes(checkpoint, load, tmp_path, method):
    """An adapter over a quantized base keeps the packed weight under ``base_layer``.

    It compares packed parts, not logits: a LoRA forward over bitsandbytes layers fails on
    some torch 2.4 setups, which is unrelated to saving.
    """
    peft = pytest.importorskip("peft")
    from packaging.version import Version
    from transformers import AutoModelForCausalLM

    path = checkpoint("llama")
    config = peft.LoraConfig(task_type="CAUSAL_LM", target_modules=["q_proj"], r=2, lora_alpha=4)
    adapted = peft.get_peft_model(AutoModelForCausalLM.from_pretrained(path), config)
    with torch.no_grad():
        for name, value in adapted.named_parameters():
            if "lora_B" in name:
                value.fill_(0.1)
    adapted.save_pretrained(tmp_path / "adapter")
    if method == "int8" and Version(peft.__version__) < Version("0.14.0"):
        with pytest.raises(ValueError, match="needs peft 0.14.0"):
            load(path, quantize=method, adapter=str(tmp_path / "adapter"))
        return
    lab = load(path, quantize=method, adapter=str(tmp_path / "adapter"))
    name = next(
        n for n, _ in lab.model.named_parameters() if n.endswith("q_proj.base_layer.weight")
    )
    baseline = packed(lab, name)
    lab.edit(name, lambda value: value * 2 + 0.05, index=(0, slice(0, 8)))
    lab.edit(name, lambda value: value - 0.3, index=(0, slice(4, 12)))
    live = packed(lab, name)
    assert not same_packed(live, baseline)
    bundle = lab.save_edits(tmp_path / "bundle")
    resumed = load(path, quantize=method, adapter=str(tmp_path / "adapter"), resume=str(bundle))
    assert same_packed(live, packed(resumed, name))
    resumed.undo()
    resumed.undo()
    assert same_packed(baseline, packed(resumed, name))
    assert resumed.edits()[0]["active"] is False
