"""PEFT exports merge one LoRA layer at a time and leave the live model exactly as it was.

The old export deep-copied the whole model and ran PEFT's merge_and_unload on the
copy. It stays here as the oracle: both paths must write the same tensors.
"""

import copy
import json

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from safetensors.torch import load_file  # noqa: E402

from oumi_weights_lab import Lab  # noqa: E402


def _gpt2():
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=6, n_positions=32, n_embd=8, n_layer=2, n_head=2, eos_token_id=1
        )
    )


def _llama(tie=False):
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=6,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=tie,
            eos_token_id=1,
            bos_token_id=1,
        )
    )


ADAPTERS = {
    "gpt2-conv1d": (_gpt2, {"target_modules": ["c_attn", "c_fc"], "fan_in_fan_out": True}),
    "llama-attention": (_llama, {"target_modules": ["q_proj", "v_proj", "down_proj"]}),
    "llama-embedding-and-head": (
        _llama,
        {"target_modules": ["embed_tokens", "o_proj"], "modules_to_save": ["lm_head"]},
    ),
}


def _adapted(build, options):
    model = peft.get_peft_model(build(), peft.LoraConfig(r=2, lora_alpha=4, **options))
    torch.manual_seed(1)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            # Trained-looking adapters: PEFT starts lora_B (and modules_to_save) at no change.
            if "lora_" in name or "modules_to_save" in name:
                parameter.add_(torch.randn_like(parameter) * 0.1)
    return model.eval()


@pytest.fixture(params=list(ADAPTERS))
def adapter_lab(request, tmp_path, tiny_tokenizer):
    lab = Lab(_adapted(*ADAPTERS[request.param]), tiny_tokenizer, tmp_path / "lab")
    # Exports keep API edits to both base and adapter weights.
    base = next(name for name, _ in lab.model.named_parameters() if "base_layer.weight" in name)
    lora = next(name for name, _ in lab.model.named_parameters() if "lora_A" in name)
    lab.edit(base, lambda value: value + 0.5, index=(0,))
    lab.edit(lora, lambda value: value * 2)
    return lab


def _oracle(lab, path):
    """The previous export: merge a deep copy, then save it."""
    memo = {}  # Share the lab's hook callbacks rather than copying the lab.
    for module in lab.model.modules():
        for attribute in ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks"):
            memo.update({id(hook): hook for hook in getattr(module, attribute).values()})
    merged = copy.deepcopy(lab.model, memo).merge_and_unload(safe_merge=True)
    merged.save_pretrained(path)
    return path


def _tensors(path):
    tensors = {}
    for file in sorted(path.glob("*.safetensors")):
        tensors.update(load_file(file))
    return tensors


def _snapshot(model):
    """Every parameter's bytes and storage, plus PEFT's merge bookkeeping."""
    parameters = {
        name: (value.data_ptr(), value.detach().cpu().clone())
        for name, value in model.named_parameters(remove_duplicate=False)
    }
    merged = {
        name: list(module.merged_adapters)
        for name, module in model.named_modules()
        if hasattr(module, "merged_adapters")
    }
    return parameters, merged


def _assert_unchanged(model, snapshot):
    parameters, merged = _snapshot(model)
    assert merged == snapshot[1]
    assert parameters.keys() == snapshot[0].keys()
    for name, (pointer, value) in parameters.items():
        assert pointer == snapshot[0][name][0], name
        assert torch.equal(value, snapshot[0][name][1]), name


def test_layer_by_layer_export_matches_the_deep_copy_merge(adapter_lab, tmp_path, monkeypatch):
    lab = adapter_lab
    expected = _tensors(_oracle(lab, tmp_path / "oracle"))
    before = _snapshot(lab.model)
    # The live model is never copied and unloaded whole.
    monkeypatch.setattr(type(lab.model.base_model), "merge_and_unload", None)
    path = lab.export(tmp_path / "export")
    monkeypatch.undo()
    exported = _tensors(path)
    assert exported.keys() == expected.keys()
    for name, value in expected.items():
        assert exported[name].dtype == value.dtype, name
        assert torch.equal(exported[name], value), name
    oracle_config = json.loads((tmp_path / "oracle" / "config.json").read_text())
    assert json.loads((path / "config.json").read_text()) == oracle_config
    manifest = json.loads((path / "oumi_weights_lab_manifest.json").read_text())
    assert manifest["merged_adapter"] is True
    assert len(manifest["edits"]) == 2
    _assert_unchanged(lab.model, before)


def test_a_failed_merge_leaves_the_live_model_exactly_as_it_was(adapter_lab, tmp_path):
    lab = adapter_lab
    layers = [m for m in lab.model.modules() if isinstance(m, peft.tuners.lora.LoraLayer)]
    # A non-finite adapter makes PEFT's safe merge raise partway through the model.
    with torch.no_grad():
        next(iter(layers[-1].lora_B.values())).weight.fill_(float("inf"))
    before = _snapshot(lab.model)
    with pytest.raises(ValueError, match="NaN"):
        lab.export(tmp_path / "export")
    assert not (tmp_path / "export").exists()
    _assert_unchanged(lab.model, before)


@pytest.mark.filterwarnings("ignore:Model (has|with) `tie_word_embeddings=True`")
def test_tied_embeddings_under_an_adapter_take_the_copy_path(tmp_path, tiny_tokenizer):
    # Merging into a tied embedding unties it; PEFT's own unload handles that.
    lab = Lab(
        _adapted(lambda: _llama(tie=True), {"target_modules": ["embed_tokens"]}),
        tiny_tokenizer,
        tmp_path / "lab",
    )
    expected = _tensors(_oracle(lab, tmp_path / "oracle"))
    exported = _tensors(lab.export(tmp_path / "export"))
    assert exported.keys() == expected.keys()
    assert all(torch.equal(exported[name], value) for name, value in expected.items())


def test_dora_keeps_the_copy_path_on_one_device_and_is_refused_when_split(tmp_path, tiny_tokenizer):
    lab = Lab(
        _adapted(_llama, {"target_modules": ["q_proj"], "use_dora": True}),
        tiny_tokenizer,
        tmp_path / "lab",
    )
    expected = _tensors(_oracle(lab, tmp_path / "oracle"))
    exported = _tensors(lab.export(tmp_path / "export"))
    assert all(torch.equal(exported[name], value) for name, value in expected.items())
    # Copying a model split across GPUs would need a second copy's memory on each.
    lab.model.get_base_model().hf_device_map = {"model": 0, "lm_head": 1}
    with pytest.raises(ValueError, match="split across devices"):
        lab.export(tmp_path / "split")
    assert not (tmp_path / "split").exists()
