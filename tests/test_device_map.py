"""A model split across devices by accelerate behaves like the same model on one device.

CI splits the model across two CPU placements, which exercises accelerate's hooks.
On a Mac the second half also runs on MPS, and on a node with two GPUs on the second
GPU, which catches tensors left on the wrong device.
"""

import copy
import json
from pathlib import Path

import pytest

accelerate = pytest.importorskip("accelerate")
torch = pytest.importorskip("torch")

from oumi_weights_lab import Lab  # noqa: E402

# transformers warns about saving any map with "cpu" in it; a GPU map has none.
pytestmark = pytest.mark.filterwarnings("ignore:Attempting to save a model with offloaded")

SPLITS = [pytest.param(("cpu", "cpu"), id="cpu")]
if torch.backends.mps.is_available():
    SPLITS += [
        pytest.param(("cpu", "mps"), id="cpu+mps"),
        pytest.param(("mps", "cpu"), id="mps+cpu"),
    ]
if torch.cuda.device_count() >= 2:
    SPLITS.append(pytest.param(("cuda:0", "cuda:1"), id="cuda+cuda"))
# MPS kernels round differently from CPU ones.
TOLERANCE = 1e-4

ROWS = [
    {"prompt": "hello", "options": {"A": " yes", "B": " no"}, "answer": "A"},
    {"prompt": "hello world", "options": {"A": " yes no", "B": " no"}, "answer": "B"},
    {"prompt": "world hello", "options": {"A": " no", "B": " yes", "C": " world"}, "answer": "C"},
]
SAMPLED = [{"prompt": prompt, "answer": "yes"} for prompt in ["hello", "world", "no yes"]]


def _gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=6,
            n_positions=64,
            n_embd=8,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=1,
            # Untied, so the output head can sit on a different device from the inputs.
            tie_word_embeddings=False,
        )
    ).eval()


def _device_map(first, second):
    return {
        "transformer.wte": first,
        "transformer.wpe": first,
        "transformer.drop": first,
        "transformer.h.0": first,
        "transformer.h.1": second,
        "transformer.ln_f": second,
        "lm_head": second,
    }


def _dispatch(model, devices):
    device_map = _device_map(*devices)
    if not ("cpu" in devices and len(set(devices)) > 1):
        return accelerate.dispatch_model(model, device_map=device_map, force_hooks=True)
    # dispatch_model reads "cpu" as offload to the accelerator, so attach the hooks it
    # attaches for a map of GPU indices: one per block, plus one returning outputs to
    # the input device.
    from accelerate.hooks import attach_align_device_hook_on_blocks

    attach_align_device_hook_on_blocks(
        model,
        execution_device={**device_map, "": devices[0]},
        offload=dict.fromkeys(device_map, False),
    )
    model.hf_device_map = device_map
    return model


@pytest.fixture(params=SPLITS)
def labs(request, tmp_path, tiny_tokenizer):
    """A dispatched lab, the same weights on one CPU device, and the split used."""
    model = _gpt2()
    reference = copy.deepcopy(model)
    split = Lab(_dispatch(model, request.param), tiny_tokenizer, tmp_path / "split")
    single = Lab(reference, tiny_tokenizer, tmp_path / "single")
    return split, single, request.param


def _items(summary):
    lines = (Path(summary["run_dir"]) / "items.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _logits(lab, text="hello world yes"):
    ids = lab.tokenizer(text, return_tensors="pt").to(lab.device)
    with torch.no_grad():
        return lab.model(**ids).logits.cpu()


def test_a_model_on_one_device_has_no_device_map(tiny_lab):
    assert tiny_lab.device_map is None


def test_the_device_map_and_input_device_come_from_the_dispatch(labs):
    split, _, (first, second) = labs
    assert split.device_map == _device_map(first, second)
    split.device_map["lm_head"] = "elsewhere"
    assert split.device_map["lm_head"] == second  # callers get a copy
    assert str(split.device).startswith(first)
    devices = {part["device"] for part in split.parts("lm_head.*")}
    assert len(devices) == 1 and devices.pop().startswith(second)
    # accelerate's hooks run every block on its device and return outputs to the inputs'.
    assert split.model._hf_hook.io_same_device
    hook = split.model.transformer.h[1]._hf_hook
    assert str(torch.device(hook.execution_device)).startswith(second)


@pytest.mark.filterwarnings("ignore:You are calling .generate")
@pytest.mark.parametrize("mode", ["likelihood", "sampled"])
def test_inputs_go_to_the_embedding_device_not_model_device(labs, mode, monkeypatch):
    split, _, _ = labs
    # model.device is the first parameter's device, which need not hold the embeddings,
    # for example in a multimodal model whose vision tower comes first.
    monkeypatch.setattr(type(split.model), "device", property(lambda self: torch.device("meta")))
    seen = []
    split.model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["input_ids"].device), with_kwargs=True
    )
    rows = ROWS if mode == "likelihood" else SAMPLED
    config = {"generation_config": {"max_new_tokens": 2}} if mode == "sampled" else {}
    split.evaluate(rows, name="run", mode=mode, batch_size=2, **config)
    assert seen and {device.type for device in seen} == {split.device.type}


@pytest.mark.parametrize("batch_size", [1, 3])
def test_scores_and_pass_counts_match_one_device(labs, batch_size):
    split, single, _ = labs
    options = {"A": " yes", "B": " no", "C": " hello world"}
    scores = split.score("hello world", options, batch_size=batch_size)
    expected = single.score("hello world", options, batch_size=batch_size)
    assert scores == pytest.approx(expected, abs=TOLERANCE)
    assert split.forward_passes == single.forward_passes == 3


@pytest.mark.parametrize("mode", ["likelihood", "sampled"])
def test_evaluations_match_one_device(labs, mode):
    split, single, _ = labs
    rows = ROWS if mode == "likelihood" else SAMPLED
    config = {"generation_config": {"max_new_tokens": 3}} if mode == "sampled" else {}
    results = [
        lab.evaluate(rows, name="run", mode=mode, batch_size=2, **config) for lab in (split, single)
    ]
    assert results[0]["accuracy"] == results[1]["accuracy"]
    assert results[0]["forward_passes"] == results[1]["forward_passes"]
    split_items, single_items = (_items(result) for result in results)
    key = "generated" if mode == "sampled" else "prediction"
    assert [item[key] for item in split_items] == [item[key] for item in single_items]


def test_generate_matches_one_device(labs):
    split, single, _ = labs
    assert split.generate("hello", max_new_tokens=4, do_sample=False) == single.generate(
        "hello", max_new_tokens=4, do_sample=False
    )
    assert split.forward_passes == single.forward_passes > 0


def test_record_and_patch_reach_modules_on_both_devices(labs):
    split, single, _ = labs
    modules = ["transformer.h.0", "transformer.h.1"]
    with split.record(modules, positions=[0, -1]) as got, single.record(modules) as want:
        _logits(split)
        _logits(single)
    for name in modules:
        assert got[name][0].device.type == "cpu"
        assert got[name][0][:, -1] == pytest.approx(want[name][0][:, -1], abs=TOLERANCE)
    for name in modules:
        with split.patch(name, torch.zeros(8)), single.patch(name, torch.zeros(8)):
            assert _logits(split) == pytest.approx(_logits(single), abs=TOLERANCE)


def test_gradients_reach_parameters_on_both_devices(labs):
    split, single, _ = labs
    names = ["transformer.h.0.mlp.c_fc.weight", "transformer.h.1.mlp.c_fc.weight", "lm_head.weight"]
    got = split.gradient("hello world", target=" yes no", parameters=names)
    want = single.gradient("hello world", target=" yes no", parameters=names)
    for name in names:
        assert got[name].device.type == "cpu"
        assert torch.allclose(got[name], want[name], atol=TOLERANCE)


def test_edits_and_undo_are_exact_on_both_devices(labs):
    split, _, _ = labs
    names = ["transformer.h.0.attn.c_proj.weight", "lm_head.weight"]
    target, source = "transformer.h.1.mlp.c_fc.weight", "transformer.h.0.mlp.c_fc.weight"
    before = {
        name: split.model.get_parameter(name).detach().cpu().clone() for name in [*names, target]
    }
    baseline = _logits(split)
    for name in names:
        split.edit(name, lambda value: value * 3, index=(slice(0, 2),))
    assert not torch.equal(_logits(split), baseline)
    # A swap copies a parameter across devices.
    split.swap(target, split.model, source)
    copied = split.model.get_parameter(target).detach().cpu()
    assert torch.equal(copied, split.model.get_parameter(source).detach().cpu())
    for _ in range(len(names) + 1):
        split.undo()
    for name in [*names, target]:
        assert torch.equal(split.model.get_parameter(name).detach().cpu(), before[name])
    assert torch.equal(_logits(split), baseline)


def test_edit_bundles_round_trip_between_split_and_single_device(labs, tmp_path, tiny_tokenizer):
    split, single, devices = labs
    split.edit("lm_head.weight", lambda value: value + 0.5, index=(0,))
    split.edit("transformer.h.0.mlp.c_proj.bias", torch.ones(8))
    bundle = split.save_edits(tmp_path / "edits")
    edited = _logits(split)
    # The bundle's base fingerprint does not depend on where the weights live.
    single.load_edits(bundle)
    assert _logits(single) == pytest.approx(edited, abs=TOLERANCE)
    resumed = Lab(_dispatch(_gpt2(), devices), tiny_tokenizer, tmp_path / "resumed")
    resumed.load_edits(bundle)
    assert torch.equal(_logits(resumed), edited)


def test_exports_reload_with_the_same_logits(labs, tmp_path):
    from transformers import AutoModelForCausalLM

    split, _, _ = labs
    split.edit("transformer.h.1.mlp.c_fc.bias", torch.full((32,), 0.25))
    path = split.export(tmp_path / "export")
    reloaded = AutoModelForCausalLM.from_pretrained(path).eval()
    ids = split.tokenizer("hello world yes", return_tensors="pt")
    with torch.no_grad():
        logits = reloaded(**ids).logits
    assert logits == pytest.approx(_logits(split), abs=TOLERANCE)


def _adapter_pair(devices, tokenizer, root):
    """A LoRA adapter over a dispatched base, and the same adapter on one device."""
    peft = pytest.importorskip("peft")

    def adapted(model):
        config = peft.LoraConfig(target_modules=["c_attn"], r=2, lora_alpha=4, fan_in_fan_out=True)
        return peft.get_peft_model(model, config).eval()

    model = _gpt2()
    single = adapted(copy.deepcopy(model))
    split = adapted(_dispatch(model, devices))
    torch.manual_seed(1)
    # Draw the adapter weights on CPU once; MPS has its own random generator.
    weights = dict(single.named_parameters())
    with torch.no_grad():
        for name, parameter in split.named_parameters():
            if "lora_" in name:
                weights[name].normal_()
                parameter.copy_(weights[name])
    return Lab(split, tokenizer, root / "split"), Lab(single, tokenizer, root / "single")


@pytest.fixture(params=SPLITS)
def adapter_labs(request, tmp_path, tiny_tokenizer):
    return (*_adapter_pair(request.param, tiny_tokenizer, tmp_path / "labs"), request.param)


def test_adapters_over_a_split_base_score_and_export(adapter_labs, tmp_path):
    from transformers import AutoModelForCausalLM

    split, single, _ = adapter_labs
    assert split.device_map is not None and single.device_map is None
    options = {"A": " yes", "B": " no"}
    assert split.score("hello", options) == pytest.approx(
        single.score("hello", options), abs=TOLERANCE
    )
    assert split.forward_passes == 2
    lora = "base_model.model.transformer.h.1.attn.c_attn.lora_B.default.weight"
    got = split.gradient("hello world", target=" yes", parameters=[lora])
    want = single.gradient("hello world", target=" yes", parameters=[lora])
    assert torch.allclose(got[lora], want[lora], atol=TOLERANCE)
    path = split.export(tmp_path / "merged")
    reloaded = AutoModelForCausalLM.from_pretrained(path).eval()
    ids = split.tokenizer("hello world yes", return_tensors="pt")
    with torch.no_grad():
        logits = reloaded(**ids).logits
    assert logits == pytest.approx(_logits(split), abs=TOLERANCE)


def test_adapters_over_a_split_base_record_patch_edit_and_resume(
    adapter_labs, tmp_path, tiny_tokenizer
):
    split, single, devices = adapter_labs
    blocks = ["base_model.model.transformer.h.0", "base_model.model.transformer.h.1"]
    with split.record(blocks) as got, single.record(blocks) as want:
        _logits(split)
        _logits(single)
    for name in blocks:
        assert got[name][0] == pytest.approx(want[name][0], abs=TOLERANCE)
        with split.patch(name, torch.zeros(8)), single.patch(name, torch.zeros(8)):
            assert _logits(split) == pytest.approx(_logits(single), abs=TOLERANCE)
    baseline = _logits(split)
    names = [
        "base_model.model.transformer.h.1.attn.c_attn.base_layer.weight",
        "base_model.model.transformer.h.0.attn.c_attn.lora_B.default.weight",
    ]
    for name in names:
        split.edit(name, lambda value: value * 2)
    edited = _logits(split)
    assert not torch.equal(edited, baseline)
    bundle = split.save_edits(tmp_path / "edits")
    resumed, _ = _adapter_pair(devices, tiny_tokenizer, tmp_path / "resumed")
    resumed.load_edits(bundle)
    assert torch.equal(_logits(resumed), edited)
    for _ in names:
        split.undo()
    assert torch.equal(_logits(split), baseline)


def test_activation_gradients_match_on_split_devices(labs):
    split, single, _ = labs
    modules = ["transformer.h.0.mlp", "transformer.h.1.mlp"]
    got = split.gradient("hello world", target=" yes no", modules=modules)
    want = single.gradient("hello world", target=" yes no", modules=modules)
    for name in modules:
        assert got[name].device.type == "cpu"
        assert got[name].abs().sum() > 0
        assert torch.allclose(got[name], want[name], atol=TOLERANCE)
