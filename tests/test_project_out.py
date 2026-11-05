from types import SimpleNamespace

import pytest
import torch

from oumi_weights_lab import Lab


def stub_tokenizer():
    return SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [ord(char) for char in text] or [0]
    )


def gpt2(tmp_path, tokenizer=None, name="gpt2"):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=6,
            n_positions=64,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=1,
        )
    ).eval()
    return Lab(model, tokenizer or stub_tokenizer(), tmp_path / name, model_id="tiny")


def llama(tmp_path, dtype=torch.float32, name="llama"):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            mlp_bias=False,
        )
    ).to(dtype)
    return Lab(model.eval(), stub_tokenizer(), tmp_path / name, model_id="tiny")


DOWN = "model.layers.0.mlp.down_proj.weight"


def test_linear_output_side_removes_direction(tmp_path):
    lab = llama(tmp_path)
    before = lab.model.get_parameter(DOWN).detach().clone()
    d = torch.randn(8)
    edit_id = lab.project_out(DOWN, d, index=[0, 3, 5])
    after = lab.model.get_parameter(DOWN).detach()
    unit = d / d.norm()
    assert (unit @ after[:, [0, 3, 5]]).norm() < 1e-5 * before.norm()
    others = [i for i in range(16) if i not in (0, 3, 5)]
    assert torch.equal(after[:, others], before[:, others])
    assert lab.edits()[0]["id"] == edit_id
    assert lab.edits()[0]["note"] == "project_out side=output scale=1.0"
    lab.undo(edit_id)
    assert torch.equal(lab.model.get_parameter(DOWN), before)


def test_linear_input_side_removes_direction(tmp_path):
    lab = llama(tmp_path)
    before = lab.model.get_parameter(DOWN).detach().clone()
    d = torch.randn(16)
    lab.project_out(DOWN, d, side="input")
    after = lab.model.get_parameter(DOWN).detach()
    assert (after @ d).norm() < 1e-5 * before.norm()


def test_linear_input_side_with_row_index_and_reflection(tmp_path):
    lab = llama(tmp_path)
    before = lab.model.get_parameter(DOWN).detach().clone()
    d = torch.randn(16)
    unit = d / d.norm()
    lab.project_out(DOWN, d, side="input", index=slice(2, 4), scale=2.0)
    after = lab.model.get_parameter(DOWN).detach()
    assert torch.allclose(after[2:4] @ unit, -(before[2:4] @ unit), atol=1e-5)
    assert torch.equal(after[:2], before[:2])
    assert torch.equal(after[4:], before[4:])
    assert lab.edits()[0]["index"] == repr((slice(2, 4), slice(None)))


def test_conv1d_output_projection_removes_component_from_output(tmp_path, tiny_tokenizer):
    lab = gpt2(tmp_path, tiny_tokenizer)
    module = "transformer.h.0.mlp.c_proj"
    d = torch.randn(8)
    lab.project_out(f"{module}.weight", d)
    lab.project_out(f"{module}.bias", d)
    inputs = lab.tokenizer("hello world", return_tensors="pt").to(lab.device)
    with lab.record(module) as captured, torch.no_grad():
        lab.model(**inputs)
    y = captured[module][0][0]
    unit = d / d.norm()
    assert ((y @ unit).abs() < 1e-4 * y.norm(dim=-1)).all()


def test_conv1d_input_side(tmp_path):
    lab = gpt2(tmp_path)
    name = "transformer.h.0.mlp.c_fc.weight"
    before = lab.model.get_parameter(name).detach().clone()
    assert before.shape == (8, 32)
    d = torch.randn(8)
    lab.project_out(name, d, side="input")
    after = lab.model.get_parameter(name).detach()
    assert (d @ after).norm() < 1e-5 * before.norm()


def test_embedding_rows(tmp_path):
    lab = gpt2(tmp_path)
    name = "transformer.wte.weight"
    before = lab.model.get_parameter(name).detach().clone()
    d = torch.randn(8)
    lab.project_out(name, d, index=[2, 3])
    after = lab.model.get_parameter(name).detach()
    unit = d / d.norm()
    assert (after[[2, 3]] @ unit).abs().max() < 1e-5
    for row in (0, 1, 4, 5):
        assert torch.equal(after[row], before[row])
    assert not torch.equal(after[2], before[2])
    count = len(lab.edits())
    with pytest.raises(ValueError):
        lab.project_out(name, d, side="input")
    assert len(lab.edits()) == count


def test_bf16_undo_is_exact(tmp_path):
    lab = llama(tmp_path, dtype=torch.bfloat16)
    before = lab.model.get_parameter(DOWN).detach().clone()
    edit_id = lab.project_out(DOWN, torch.randn(8), scale=1.5)
    after = lab.model.get_parameter(DOWN)
    assert after.dtype == torch.bfloat16
    assert not torch.equal(after, before)
    lab.undo(edit_id)
    assert torch.equal(lab.model.get_parameter(DOWN), before)


def test_bundle_round_trip(tmp_path):
    first = llama(tmp_path, name="first")
    base = {k: v.clone() for k, v in first.model.state_dict().items()}
    first.project_out(DOWN, torch.randn(8), index=[1, 4, 9])
    first.project_out(DOWN, torch.randn(16), side="input", index=[0, 2])
    bundle = first.save_edits("projected")
    second = llama(tmp_path, name="second")
    second.model.load_state_dict(base)
    second.load_edits(bundle)
    assert torch.equal(second.model.get_parameter(DOWN), first.model.get_parameter(DOWN))
    assert second.edits() == first.edits()
    assert second.edits()[0]["index"] == repr((slice(None), [1, 4, 9]))


def test_refusals(tmp_path):
    lab = gpt2(tmp_path)
    ln = "transformer.h.0.ln_1.weight"
    proj = "transformer.h.0.mlp.c_proj.weight"
    bias = "transformer.h.0.mlp.c_proj.bias"
    d = torch.ones(8)
    with pytest.raises(ValueError, match="lab.edit"):
        lab.project_out(ln, d)
    bad = [
        (proj, torch.ones(7), {}),
        (proj, torch.zeros(8), {}),
        (proj, torch.tensor([float("nan")] * 8), {}),
        (proj, torch.ones(2, 8), {}),
        (proj, d, {"scale": True}),
        (proj, d, {"scale": float("inf")}),
        (bias, d, {"index": [0]}),
        (bias, d, {"side": "input"}),
        (proj, d, {"side": "both"}),
        (proj, d, {"index": slice(0, 8, 2)}),
        (proj, d, {"index": [1, 1]}),
        (proj, d, {"index": []}),
        (proj, d, {"index": True}),
    ]
    for name, direction, kwargs in bad:
        with pytest.raises(ValueError):
            lab.project_out(name, direction, **kwargs)
    with pytest.raises(ValueError, match="length 8"):
        lab.project_out(proj, torch.ones(7))
    assert lab.edits() == []


def test_peft_base_layer_and_lora_b(tmp_path):
    peft = pytest.importorskip("peft")
    lab = llama(tmp_path)
    model = peft.get_peft_model(
        lab.model,
        peft.LoraConfig(task_type="CAUSAL_LM", target_modules=["down_proj"], r=2),
    ).eval()
    lab = Lab(model, stub_tokenizer(), tmp_path / "lora", model_id="tiny")
    owner = "base_model.model.model.layers.0.mlp.down_proj"
    base = f"{owner}.base_layer.weight"
    lora_b = f"{owner}.lora_B.default.weight"
    with torch.no_grad():
        model.get_parameter(lora_b).normal_()
    d = torch.randn(8)
    lab.project_out(base, d)
    lab.project_out(lora_b, d)
    unit = d / d.norm()
    assert (unit @ model.get_parameter(base)).norm() < 1e-5
    assert (unit @ model.get_parameter(lora_b)).norm() < 1e-5
    # The wrapper forwards `.weight` to its base layer, but it is not a Linear.
    with pytest.raises(ValueError, match="lab.edit"):
        lab.project_out(f"{owner}.weight", d)
    with pytest.raises(ValueError, match="Unknown parameter"):
        lab.project_out(f"{owner}.nothing", d)
