"""Quantized loading is read-only: scoring and reads work, edits to packed weights refuse."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from oumi_weights_lab import Lab  # noqa: E402
from oumi_weights_lab.lab import load_lab  # noqa: E402
from oumi_weights_lab.session import create_session, validate_placement  # noqa: E402

METHODS = ("nf4", "fp4", "int8")
ROWS = [
    {"id": str(i), "prompt": "hello", "answer": "A", "options": [" yes", " no"]} for i in range(4)
]
PROMPT, TARGET = "hello world yes", " no hello"
Q_PROJ = "model.layers.0.self_attn.q_proj.weight"


def build_checkpoint(kind, path, tokenizer):
    from transformers import AutoModelForCausalLM, GPT2Config, LlamaConfig

    common = {"vocab_size": 128, "bos_token_id": 1, "eos_token_id": 1, "pad_token_id": 1}
    if kind == "llama":
        config = LlamaConfig(
            **common,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    else:
        config = GPT2Config(**common, n_positions=64, n_embd=64, n_layer=2, n_head=4)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture
def checkpoint(tmp_path, tiny_tokenizer):
    """A factory: ``checkpoint("llama")`` saves a tiny float model and returns its path."""

    def build(kind):
        return build_checkpoint(kind, tmp_path / f"checkpoint-{kind}", tiny_tokenizer)

    return build


@pytest.fixture
def load(tmp_path):
    count = 0

    def build(path, **config):
        nonlocal count
        count += 1
        settings = {"model": path, "device": "cpu", "dtype": "float32", **config}
        return load_lab(settings, tmp_path / f"session-{count}")

    return build


@pytest.fixture
def bnb():
    pytest.importorskip("bitsandbytes")
    import bitsandbytes

    return bitsandbytes


def packed_state(lab, name=Q_PROJ):
    """Copies of the packed bytes and the scales of one quantized weight."""
    weight = lab.model.get_parameter(name)
    state = getattr(weight, "quant_state", None)
    if state is not None:
        return weight.data.clone(), state.absmax.clone()
    scales = weight.SCB
    if scales is None:  # After the first forward the scales live on the layer's state.
        scales = lab.model.get_submodule(name.removesuffix(".weight")).state.SCB
    return weight.data.clone(), scales.clone()


def dequantized(lab, name):
    import bitsandbytes.functional as functional

    weight = lab.model.get_parameter(name)
    if hasattr(weight, "quant_state"):
        return functional.dequantize_4bit(weight.data, weight.quant_state).float()
    return weight.data.float() * weight.SCB[:, None] / 127


@pytest.mark.parametrize("method", METHODS)
def test_quantize_loads_quantized_modules(bnb, checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    expected = bnb.nn.Linear8bitLt if method == "int8" else bnb.nn.Linear4bit
    layer = lab.model.model.layers[0]
    for module in (layer.self_attn.q_proj, layer.self_attn.o_proj, layer.mlp.down_proj):
        assert isinstance(module, expected)
    assert type(lab.model.lm_head) is torch.nn.Linear
    assert type(lab.model.model.embed_tokens) is torch.nn.Embedding
    ids = lab.tokenizer("hello world yes", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        assert torch.isfinite(lab.model(input_ids=ids).logits).all()
    info = lab.quantization
    assert info["method"] == method and info["source"] == "load"
    assert info["compute_dtype"] == "float32" and info["double_quant"] is False
    assert info["bitsandbytes"] == bnb.__version__
    if method == "int8":
        assert info["threshold"] == 6.0 and "blocksize" not in info
    else:
        assert info["blocksize"] == 64 and "threshold" not in info


def test_float_model_has_no_quantization(checkpoint, load):
    lab = load(checkpoint("llama"))
    assert lab.quantization is None
    assert all("quantization" not in part for part in lab.parts())


@pytest.mark.parametrize("method", METHODS)
def test_quantized_parts_report_logical_shape(bnb, checkpoint, load, method):
    path = checkpoint("llama")
    floating, lab = load(path), load(path, quantize=method)
    reference = {part["name"]: part for part in floating.parts()}
    parts = lab.parts()
    assert [part["name"] for part in parts] == list(reference)
    quantized = 0
    for part in parts:
        expected = reference[part["name"]]
        if "quantization" not in part:
            assert part == expected
            continue
        quantized += 1
        assert part["shape"] == expected["shape"] and part["numel"] == expected["numel"]
        assert part["dtype"] == "torch.float32"
        assert part["quantization"]["method"] == method
        assert part["quantization"]["double_quant"] is False
    assert quantized == 2 * 7
    block = next(part for part in parts if part["name"] == Q_PROJ)["quantization"]
    if method == "int8":
        assert block["threshold"] == 6.0 and block["packed_dtype"] == "torch.int8"
        assert block["packed_shape"] == [64, 64]
    else:
        assert block["blocksize"] == 64 and block["packed_dtype"] == "torch.uint8"
        assert block["packed_shape"] == [64 * 64 // 2, 1]
    json.dumps(parts)


@pytest.mark.parametrize("method", METHODS)
def test_measure_uses_dequantized_values(bnb, checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    expected = dequantized(lab, Q_PROJ)
    result = lab.measure(Q_PROJ)
    assert result["shape"] == [64, 64] and result["dtype"] == "torch.float32"
    assert result["numel"] == 64 * 64
    assert result["l2"] == pytest.approx(expected.double().norm().item(), rel=1e-9)
    assert result["mean"] == pytest.approx(expected.double().mean().item(), abs=1e-9)
    assert result["quantization"]["method"] == method
    norm = lab.measure("model.norm.weight")
    assert "quantization" not in norm


def test_measure_matches_after_the_first_forward(bnb, checkpoint, load):
    # int8 keeps its scales on the module after a forward; the numbers must not move.
    lab = load(checkpoint("llama"), quantize="int8")
    before = lab.measure(Q_PROJ)
    ids = lab.tokenizer("hello world", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        lab.model(input_ids=ids)
    assert lab.measure(Q_PROJ) == before


def test_compare_quantized_against_float_model(bnb, checkpoint, load):
    path = checkpoint("llama")
    floating, lab = load(path), load(path, quantize="nf4")
    report = {entry["name"]: entry for entry in lab.compare(floating.model)}
    assert list(report) == [part["name"] for part in floating.parts()]
    assert report[Q_PROJ]["shape"] == [64, 64]
    assert report[Q_PROJ]["quantization"]["method"] == "nf4"
    assert report[Q_PROJ]["l2"] > 0
    assert report["model.norm.weight"]["l2"] == 0
    assert "quantization" not in report["model.norm.weight"]
    expected = (dequantized(lab, Q_PROJ) - floating.model.get_parameter(Q_PROJ)).double()
    assert report[Q_PROJ]["l2"] == pytest.approx(expected.norm().item(), rel=1e-6)
    # The float model can also be the one that compares, against a quantized reference.
    reverse = {entry["name"]: entry for entry in floating.compare(lab.model, "model.layers.0.*")}
    assert reverse[Q_PROJ]["l2"] == pytest.approx(report[Q_PROJ]["l2"], rel=1e-6)
    assert "quantization" not in reverse[Q_PROJ]
    json.dumps(report)


@pytest.mark.parametrize("method", METHODS)
def test_float_parameters_edit_and_undo_exactly(bnb, checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    ids = lab.tokenizer("hello world yes", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        baseline = lab.model(input_ids=ids).logits.clone()
    originals = {name: lab.model.get_parameter(name).detach().clone() for name in names}
    packed = packed_state(lab)
    for name in names:
        lab.edit(name, lambda value: value * 0.5 + 0.25)
    with torch.no_grad():
        assert not torch.equal(lab.model(input_ids=ids).logits, baseline)
    for _ in names:
        lab.undo()
    for name in names:
        assert torch.equal(lab.model.get_parameter(name), originals[name])
    with torch.no_grad():
        assert torch.equal(lab.model(input_ids=ids).logits, baseline)
    assert torch.equal(packed[0], packed_state(lab)[0])


def test_gradient_skips_quantized_parameters(bnb, checkpoint, load):
    lab = load(checkpoint("llama"), quantize="nf4")
    gradients = lab.gradient(PROMPT, TARGET)
    names = {name for name, _ in lab.model.named_parameters()}
    quantized = {part["name"] for part in lab.parts() if "quantization" in part}
    assert quantized and set(gradients) == names - quantized
    assert {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} <= set(gradients)
    assert all(torch.isfinite(value).all() for value in gradients.values())
    with pytest.raises(ValueError, match=Q_PROJ):
        lab.gradient(PROMPT, TARGET, parameters=[Q_PROJ])
    with pytest.raises(ValueError, match=Q_PROJ):
        lab.gradient(PROMPT, TARGET, parameters=["model.norm.weight", Q_PROJ])
    flags = {name: p.requires_grad for name, p in lab.model.named_parameters()}
    module = "model.layers.0.self_attn.q_proj"
    result = lab.gradient(PROMPT, TARGET, parameters=["model.norm.weight"], modules=module)
    assert set(result) == {module, "model.norm.weight"}
    assert torch.isfinite(result[module]).all() and result[module].abs().sum() > 0
    assert flags == {name: p.requires_grad for name, p in lab.model.named_parameters()}
    only = lab.gradient(PROMPT, TARGET, modules=["model.layers.1.mlp.down_proj"], at="input")
    assert list(only) == ["model.layers.1.mlp.down_proj"]
    assert only["model.layers.1.mlp.down_proj"].abs().sum() > 0


def test_gradient_through_int8_layers(bnb, checkpoint, load):
    lab = load(checkpoint("llama"), quantize="int8")
    module = "model.layers.0.mlp.down_proj"
    result = lab.gradient(PROMPT, TARGET, parameters=["lm_head.weight"], modules=module)
    assert torch.isfinite(result[module]).all() and result[module].abs().sum() > 0


@pytest.mark.parametrize("kind", ["llama", "gpt2"])
@pytest.mark.parametrize("method", ["nf4", "int8"])
def test_heads_match_float_model(bnb, checkpoint, load, kind, method):
    path = checkpoint(kind)
    floating, lab = load(path), load(path, quantize=method)
    heads, reference = lab.heads(), floating.heads()
    assert len(heads) == len(reference) == 8
    for entry, expected in zip(heads, reference, strict=True):
        for key in ("layer", "head", "kv_head", "group", "module", "o_input"):
            assert entry[key] == expected[key]
        for key in ("q", "k", "v", "o"):
            assert [part["name"] for part in entry[key]] == [part["name"] for part in expected[key]]
    for layer, head in [(0, 0), (1, 3)]:
        weights, expected = lab.head_weights(layer, head), floating.head_weights(layer, head)
        assert weights.keys() == expected.keys()
        entry = next(e for e in heads if (e["layer"], e["head"]) == (layer, head))
        for key in ("q", "k", "v", "o"):
            assert weights[key].shape == expected[key].shape
            assert weights[key].dtype == torch.float32 and weights[key].device.type == "cpu"
            # Dequantized, the slice is the float one up to quantization error.
            error = (weights[key] - expected[key]).norm() / expected[key].norm()
            assert error < 0.35
            # And it is exactly the dequantized weight at the part's own index.
            part = next(p for p in entry[key] if p["name"].endswith(".weight"))
            assert torch.equal(weights[key], dequantized(lab, part["name"])[part["index"]])


def test_gpt2_head_parts_follow_the_quantized_orientation(bnb, checkpoint, load):
    lab = load(checkpoint("gpt2"), quantize="nf4")
    entry = lab.heads(layers=[0])[1]
    q = next(p for p in entry["q"] if p["name"].endswith("c_attn.weight"))
    o = entry["o"][0]
    # c_attn is stored [out, in] when quantized, so heads are rows; c_proj heads are columns.
    assert q["index"] == (slice(16, 32), slice(None))
    assert o["index"] == (slice(None), slice(16, 32))
    assert dequantized(lab, q["name"])[q["index"]].shape == (16, 64)
    assert dequantized(lab, o["name"])[o["index"]].shape == (64, 16)


@pytest.mark.parametrize("method", ["nf4", "fp4"])
def test_cpu_repack_is_disabled(bnb, checkpoint, load, monkeypatch, method):
    import bitsandbytes.nn.modules as modules

    # Layers read this flag when they are built, so patch it before loading.
    monkeypatch.setattr(modules, "has_avx512bf16", lambda: True)
    lab = load(checkpoint("llama"), quantize=method)
    layers = [m for m in lab.model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    assert layers and not any(m.support_avx512bf16_for_cpu for m in layers)
    before = packed_state(lab)
    ids = lab.tokenizer("hello world yes", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        lab.model(input_ids=ids)
    after = packed_state(lab)
    assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])
    # Without the fix the same forward repacks the weight in place.
    owner = lab.model.get_submodule(Q_PROJ.removesuffix(".weight"))
    owner.support_avx512bf16_for_cpu = True
    with torch.no_grad():
        lab.model(input_ids=ids)
    assert not torch.equal(before[0], packed_state(lab)[0])


def test_prequantized_checkpoint_loads_read_only(bnb, checkpoint, load, tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    path = checkpoint("llama")
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=config, device_map={"": "cpu"}, dtype=torch.float32
    )
    saved = tmp_path / "prequantized"
    model.save_pretrained(saved)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(path).save_pretrained(saved)
    lab = load(str(saved))
    assert lab.quantization["source"] == "checkpoint"
    assert lab.quantization["method"] == "nf4" and lab.quantization["double_quant"] is True
    assert lab.quantization["compute_dtype"] == "float32"
    assert next(p for p in lab.parts() if p["name"] == Q_PROJ)["quantization"]["double_quant"]
    with pytest.raises(ValueError, match="bnb_4bit_use_double_quant"):
        lab.zero(Q_PROJ)
    ids = lab.tokenizer("hello world yes", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        assert torch.isfinite(lab.model(input_ids=ids).logits).all()
    assert load(str(saved), quantize="nf4").quantization["source"] == "checkpoint"

    def explode(*args, **kwargs):
        raise AssertionError("weights were loaded")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", explode)
    for method in ("int8", "fp4"):
        with pytest.raises(ValueError, match=f"nf4.*{method}"):
            load(str(saved), quantize=method)


def test_other_quantization_methods_keep_todays_behavior(bnb, checkpoint, load, monkeypatch):
    from transformers import AutoConfig

    path = checkpoint("llama")
    config = AutoConfig.from_pretrained(path)
    config.quantization_config = {"quant_method": "gptq", "bits": 4}
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    from transformers import AutoModelForCausalLM

    seen = {}

    def fake(model_id, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake)
    with pytest.raises(RuntimeError, match="stop"):
        load(path)
    assert "quantization_config" not in seen and "device_map" not in seen
    with pytest.raises(ValueError, match="already quantized with gptq"):
        load(path, quantize="nf4")


def test_other_quantization_methods_name_themselves_when_an_edit_refuses(tiny_lab):
    model = tiny_lab.model
    model.config.quantization_config = {"quant_method": "gptq", "bits": 4}
    model.transformer.wpe.weight = torch.nn.Parameter(
        torch.zeros(model.transformer.wpe.weight.shape, dtype=torch.int32), requires_grad=False
    )
    with pytest.raises(ValueError, match="Only floating-point.*gptq"):
        tiny_lab.zero("transformer.wpe.weight")
    assert tiny_lab.edits() == []


def test_bias_of_a_quantized_layer_edits_and_undoes(bnb, checkpoint, load):
    lab = load(checkpoint("gpt2"), quantize="nf4")
    name = "transformer.h.0.attn.c_attn.bias"
    before = lab.model.get_parameter(name).detach().clone()
    lab.edit(name, lambda value: value + 1)
    assert torch.equal(lab.model.get_parameter(name), before + 1)
    lab.undo()
    assert torch.equal(lab.model.get_parameter(name), before)
    assert "quantization" not in next(p for p in lab.parts() if p["name"] == name)


def test_quantize_refusals(bnb, checkpoint, load, tmp_path, monkeypatch):
    path = checkpoint("llama")
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_placement({"model": path, "quantize": "nf4", "device_map": "balanced"})
    with pytest.raises(ValueError, match="cannot be combined"):
        create_session(
            tmp_path / "s1", {"model": path, "quantize": "nf4", "device_map": "balanced"}
        )
    for bad in ("nf8", "NF4", 4, True):
        with pytest.raises(ValueError, match="nf4, fp4, int8"):
            validate_placement({"model": path, "quantize": bad})
    validate_placement({"model": path, "quantize": "int8"})
    validate_placement({"model": path, "quantize": None})
    with pytest.raises(ValueError, match="allow_hub_kernels"):
        validate_placement({"model": path, "allow_hub_kernels": "yes"})
    with pytest.raises(ValueError, match="device"):
        load(path, quantize="nf4", device="xla")
    from transformers import AutoModelForCausalLM

    def explode(*args, **kwargs):
        raise AssertionError("weights were loaded")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", explode)
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    with pytest.raises(ImportError) as error:
        load(path, quantize="nf4")
    assert "quantized" in str(error.value) and "oumi-weights-lab[quantized]" in str(error.value)


def test_setup_records_quantization(bnb, checkpoint, load, tmp_path):
    path = checkpoint("llama")
    summary = load(path, quantize="nf4").evaluate(ROWS)
    setup = json.loads((Path(summary["run_dir"]) / "summary.json").read_text())["setup"]
    assert setup["quantization"]["method"] == "nf4"
    assert setup["allow_hub_kernels"] is False
    plain = load(path).evaluate(ROWS)
    setup = json.loads((Path(plain["run_dir"]) / "summary.json").read_text())["setup"]
    assert setup["quantization"] is None and setup["allow_hub_kernels"] is False
    report = load(path).compare_runs(plain, summary, cross_session=True)
    assert "quantization" in report["cross_session"]["setup_differences"]
    assert report["cross_session"]["after_setup"]["quantization"]["method"] == "nf4"


def test_float_edits_save_and_resume_on_a_quantized_model(bnb, checkpoint, load, tmp_path):
    path = checkpoint("llama")
    lab = load(path, quantize="nf4")
    lab.zero("model.norm.weight")
    ids = lab.tokenizer("hello world yes", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        edited = lab.model(input_ids=ids).logits.clone()
    bundle = lab.save_edits(tmp_path / "bundle")
    fresh = load(path, quantize="nf4", resume=str(bundle))
    assert fresh.active_edit_ids == lab.active_edit_ids
    with torch.no_grad():
        assert torch.equal(fresh.model(input_ids=ids).logits, edited)
    fresh.undo()
    assert fresh.model.model.norm.weight.abs().sum() > 0


@pytest.mark.parametrize("method", METHODS)
def test_fingerprint_is_stable_across_forwards_and_loads(bnb, checkpoint, load, method):
    from oumi_weights_lab.persistence import _fingerprint

    path = checkpoint("llama")
    lab = load(path, quantize=method)
    first = _fingerprint(lab)
    ids = lab.tokenizer("hello world yes", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        lab.model(input_ids=ids)
    assert _fingerprint(lab) == first
    assert _fingerprint(load(path, quantize=method)) == first
    assert _fingerprint(load(path)) != first


def test_a_plain_lab_has_no_quantization(tiny_lab):
    assert tiny_lab.quantization is None and tiny_lab.allow_hub_kernels is False


def test_worker_status_reports_quantization_and_stops_cleanly(bnb, checkpoint, tmp_path):
    import threading
    import time

    from oumi_weights_lab.session import read_json
    from oumi_weights_lab.worker import Worker

    session = create_session(
        tmp_path / "worker-session",
        {
            "model": checkpoint("llama"),
            "device": "cpu",
            "dtype": "float32",
            "quantize": "int8",
            "poll_interval": 0.01,
        },
    )
    ready = {}

    def stop_when_ready():
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = read_json(session / "status.json")
            if status["state"] == "ready":
                ready.update(status)
                (session / "shutdown.json").write_text(json.dumps({"reason": "idle"}))
                (session / "STOP").touch()
                return
            time.sleep(0.01)

    thread = threading.Thread(target=stop_when_ready)
    thread.start()
    assert Worker(session).run() == 0
    thread.join()
    assert ready["quantization"] == {"method": "int8", "compute_dtype": "float32"}
    final = read_json(session / "status.json")
    assert final["state"] == "stopped" and final["quantization"] == ready["quantization"]
    assert Path(final["edit_bundle"]).is_dir() and "checkpoint_error" not in final


def run_guard(tmp_path, body):
    """Run ``body`` in a clean interpreter that has a fake ``kernels`` package on its path."""
    package = tmp_path / "fake" / "kernels"
    package.mkdir(parents=True, exist_ok=True)
    marker = tmp_path / "marker"
    marker.unlink(missing_ok=True)
    (package / "__init__.py").write_text(
        textwrap.dedent(
            f"""
            def get_kernel(*args, **kwargs):
                open({str(marker)!r}, "w").write("fetched")
                raise RuntimeError("fake kernel")
            """
        )
    )
    script = "from oumi_weights_lab.lab import block_hub_kernels\n" + textwrap.dedent(body)
    source = Path(__file__).parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": f"{tmp_path / 'fake'}{os.pathsep}{source}"},
        check=False,
    )
    return result, marker


MPS_PROBE = """
        import bitsandbytes.backends.mps.ops as ops
        ops._kernel_load_failed = False
        ops._kernel = None
        print("kernel", ops._get_kernel())
        """


def test_hub_kernels_blocked_without_flag(bnb, tmp_path):
    result, marker = run_guard(
        tmp_path,
        """
        block_hub_kernels(False)
        try:
            import kernels
        except ImportError:
            print("import blocked")
        import importlib.util
        print("spec", importlib.util.find_spec("kernels"))
        """
        + MPS_PROBE,
    )
    assert result.returncode == 0, result.stderr
    assert "import blocked" in result.stdout and "spec None" in result.stdout
    assert "kernel None" in result.stdout
    assert not marker.exists()


def test_hub_kernels_allowed_with_flag(bnb, tmp_path):
    result, marker = run_guard(
        tmp_path,
        """
        block_hub_kernels(True)
        """
        + MPS_PROBE,
    )
    assert result.returncode == 0, result.stderr
    assert "kernel None" in result.stdout  # the fake get_kernel raises, bitsandbytes falls back
    assert marker.read_text() == "fetched"


def test_hub_kernel_guard_raises_when_kernels_was_imported_first(tmp_path):
    result, marker = run_guard(
        tmp_path,
        """
        import kernels
        try:
            block_hub_kernels(False)
        except RuntimeError as error:
            print("raised", "--allow-hub-kernels" in str(error))
        block_hub_kernels(True)
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "raised True" in result.stdout


def test_transformers_still_loads_models_with_the_guard_on(checkpoint, load):
    lab = load(checkpoint("llama"))
    ids = lab.tokenizer("hello world", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        assert torch.isfinite(lab.model(input_ids=ids).logits).all()
    assert sys.modules.get("kernels", 0) is None
    assert Lab is not None


@pytest.mark.parametrize("method", ["nf4", "int8"])
def test_swap_reads_a_quantized_source_as_float(bnb, checkpoint, load, method):
    path = checkpoint("llama")
    floating, source = load(path), load(path, quantize=method)
    before = floating.model.get_parameter(Q_PROJ).detach().clone()
    expected = dequantized(source, Q_PROJ)
    floating.swap(Q_PROJ, source.model)
    swapped = floating.model.get_parameter(Q_PROJ)
    assert torch.equal(swapped, expected) and swapped.abs().max() < 1
    floating.undo()
    assert torch.equal(floating.model.get_parameter(Q_PROJ), before)
    floating.swap(Q_PROJ, source.model, index=(slice(0, 8),))
    assert torch.equal(floating.model.get_parameter(Q_PROJ)[:8], expected[:8])
    floating.undo()
    assert torch.equal(floating.model.get_parameter(Q_PROJ), before)


def test_a_bare_lab_around_a_quantized_model_exports_with_its_settings(
    bnb, checkpoint, load, tmp_path
):
    lab = load(checkpoint("llama"), quantize="nf4")
    bare = Lab(lab.model, lab.tokenizer, tmp_path / "bare")
    assert bare.quantization is None
    out = bare.export(tmp_path / "out")
    manifest = json.loads((out / "oumi_weights_lab_manifest.json").read_text())
    assert manifest["quantization"] == {"method": "nf4", "blocksize": 64}


def test_int8_logical_dtype_is_the_compute_dtype(bnb, checkpoint, load):
    lab = load(checkpoint("llama"), quantize="int8")
    lab.quantization["compute_dtype"] = "bfloat16"
    assert next(p for p in lab.parts() if p["name"] == Q_PROJ)["dtype"] == "torch.bfloat16"
    assert lab.measure(Q_PROJ)["dtype"] == "torch.bfloat16"


def test_allow_hub_kernels_clears_an_earlier_block(monkeypatch):
    from oumi_weights_lab.lab import block_hub_kernels

    monkeypatch.setitem(sys.modules, "kernels", None)
    block_hub_kernels(True)
    assert "kernels" not in sys.modules
    block_hub_kernels(True)  # Nothing to clear is fine.


@pytest.mark.parametrize(
    ("version", "device", "refused"),
    [
        ("2.4.0", "mps", True),
        ("2.4.1+cpu", "mps:0", True),
        ("2.5.1", "mps", False),
        ("2.10.0", "mps", False),
        ("2.4.0", "cpu", False),
        ("2.4.0", "cuda:0", False),
    ],
)
def test_quantized_mps_needs_torch_2_5(version, device, refused):
    from types import SimpleNamespace

    from oumi_weights_lab import lab as lab_module

    torch_like = SimpleNamespace(__version__=version)
    if refused:
        with pytest.raises(RuntimeError, match="torch 2.5 or later"):
            lab_module._check_quantized_device(torch_like, device)
    else:
        lab_module._check_quantized_device(torch_like, device)


@pytest.mark.parametrize("installed, refused", [("0.13.2", True), ("0.14.0", False)])
def test_int8_adapter_needs_a_recent_peft(
    bnb, checkpoint, load, tmp_path, monkeypatch, installed, refused
):
    """peft before 0.14 reads a bitsandbytes attribute that no longer exists on int8 layers."""
    import importlib.metadata

    from transformers import AutoModelForCausalLM

    real = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: installed if name == "peft" else real(name)
    )
    path = checkpoint("llama")

    def explode(*args, **kwargs):
        raise AssertionError("weights loaded")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", explode)
    if refused:
        with pytest.raises(ValueError) as error:
            load(path, quantize="int8", adapter="adapter")
        assert all(text in str(error.value) for text in ("int8", "0.14.0", "0.13.2"))
    else:
        with pytest.raises(AssertionError, match="weights loaded"):
            load(path, quantize="int8", adapter="adapter")
    with pytest.raises(AssertionError, match="weights loaded"):
        load(path, quantize="nf4", adapter="adapter")  # 4-bit has no floor.
