import math

import pytest
import torch
from torch import nn

from oumi_weights_lab import persistence

KINDS = ["gpt2", "llama_gqa"]


def tokens(lab, text="hello world yes no hello"):
    return lab.tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(lab.device)


def run(lab):
    with torch.no_grad():
        return lab.model(**tokens(lab)).logits


@pytest.fixture
def gpt2_lab(tiny_model, make_lab):
    return make_lab(*tiny_model("gpt2"))


def test_gpt2_heads_listing(gpt2_lab):
    heads = gpt2_lab.heads()
    assert [(h["layer"], h["head"]) for h in heads] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    hd = 8
    for entry in heads:
        h = entry["head"]
        assert entry["module"] == f"transformer.h.{entry['layer']}.attn"
        assert entry["kv_head"] == h and entry["group"] == [h]
        assert "gate" not in entry
        assert len(entry["o"]) == 1
        o = entry["o"][0]
        assert o["name"] == f"{entry['module']}.c_proj.weight"
        assert o["index"] == (slice(h * hd, (h + 1) * hd), slice(None))
        for key in "qkv":
            weight, bias = entry[key]
            assert weight["name"] == f"{entry['module']}.c_attn.weight"
            assert bias["name"] == f"{entry['module']}.c_attn.bias"
        assert entry["q"][0]["index"] == (slice(None), slice(h * hd, (h + 1) * hd))
        assert entry["k"][0]["index"] == (slice(None), slice(16 + h * hd, 16 + (h + 1) * hd))
        assert entry["v"][1]["index"] == (slice(32 + h * hd, 32 + (h + 1) * hd),)
        assert entry["o_input"] == {
            "module": f"{entry['module']}.c_proj",
            "at": "input",
            "features": [h * hd, (h + 1) * hd],
        }


def test_parts_survive_edit_bundles(gpt2_lab):
    for part in gpt2_lab.heads()[1]["q"]:
        encoded = persistence._encode_index(part["index"], {})
        assert persistence._decode_index(encoded, {}) == part["index"]
    gpt2_lab.zero(**gpt2_lab.heads()[1]["o"][0])
    assert gpt2_lab.heads(layers=[0])[1]["o"][0]["index"][0] == slice(8, 16)


@pytest.mark.parametrize("kind", KINDS)
def test_zeroing_o_parts_matches_patching_o_input(kind, tiny_model, make_lab):
    lab = make_lab(*tiny_model(kind))
    entry = lab.heads()[1]
    start, stop = entry["o_input"]["features"]

    def drop_head(value):
        value = value.clone()
        value[..., start:stop] = 0
        return value

    baseline = run(lab)
    with lab.patch(entry["o_input"]["module"], drop_head, positions="all", at="input"):
        patched = run(lab)
    assert not torch.allclose(patched, baseline)
    edit_ids = [lab.zero(**part) for part in entry["o"]]
    zeroed = run(lab)
    assert torch.allclose(zeroed, patched, atol=1e-6)
    for _ in edit_ids:
        lab.undo()
    assert torch.equal(run(lab), baseline)


def causal_attention(x, weights, hd):
    scores = (x @ weights["q"].T + weights.get("q_bias", 0)) @ (
        x @ weights["k"].T + weights.get("k_bias", 0)
    ).transpose(-1, -2)
    scores = scores / math.sqrt(hd)
    length = x.shape[-2]
    mask = torch.ones(length, length, dtype=torch.bool).tril()
    return scores.masked_fill(~mask, float("-inf")).softmax(-1)


def test_qk_scores_match_captured_attention_gpt2(gpt2_lab):
    inputs = tokens(gpt2_lab)
    with (
        gpt2_lab.record("transformer.h.0.ln_1", positions="all") as normed,
        gpt2_lab.record("transformer.h.0.attn", positions="all", at="attention") as attention,
        torch.no_grad(),
    ):
        gpt2_lab.model(**inputs)
    x = normed["transformer.h.0.ln_1"][0].float()[0]
    captured = attention["transformer.h.0.attn"][0][0]
    for head in range(2):
        pattern = causal_attention(x, gpt2_lab.head_weights(0, head), 8)
        assert torch.allclose(pattern, captured[head], atol=1e-5)


def test_ov_sum_matches_attention_output_gpt2(gpt2_lab):
    inputs = tokens(gpt2_lab)
    with (
        gpt2_lab.record("transformer.h.0.ln_1", positions="all") as normed,
        gpt2_lab.record("transformer.h.0.attn", positions="all", at="attention") as attention,
        gpt2_lab.record("transformer.h.0.attn", positions="all") as output,
        torch.no_grad(),
    ):
        gpt2_lab.model(**inputs)
    x = normed["transformer.h.0.ln_1"][0].float()[0]
    captured = attention["transformer.h.0.attn"][0][0]
    total = gpt2_lab.model.transformer.h[0].attn.c_proj.bias.detach().clone()
    for head in range(2):
        w = gpt2_lab.head_weights(0, head)
        value = x @ w["v"].T + w["v_bias"]
        total = total + captured[head] @ value @ w["o"].T
    assert torch.allclose(total, output["transformer.h.0.attn"][0][0].float(), atol=1e-5)


def test_llama_gqa_values_and_output(tiny_model, make_lab):
    lab = make_lab(*tiny_model("llama_gqa"))
    heads = lab.heads(layers=[0])
    assert [h["kv_head"] for h in heads] == [0, 0, 1, 1]
    assert [h["group"] for h in heads] == [[0, 1], [0, 1], [2, 3], [2, 3]]
    assert heads[0]["k"] == heads[1]["k"] and heads[0]["v"] == heads[1]["v"]
    assert heads[0]["k"] != heads[2]["k"]
    assert all(len(h["q"]) == len(h["o"]) == 1 for h in heads)
    attn = "model.layers.0.self_attn"
    o_proj = f"{attn}.o_proj"
    inputs = tokens(lab)
    with (
        lab.record(attn, positions="all", at="input") as source,
        lab.record(attn, positions="all", at="attention") as attention,
        lab.record(o_proj, positions="all", at="input") as joined,
        lab.record(o_proj, positions="all") as projected,
        torch.no_grad(),
    ):
        lab.model(**inputs)
    x = source[attn][0].float()[0]
    pattern = attention[attn][0][0]
    heads_in = joined[o_proj][0].float()[0]
    total = 0
    for entry in heads:
        w = lab.head_weights(0, entry["head"])
        assert w["v"].shape == (4, 16) and w["o"].shape == (16, 4)
        start, stop = entry["o_input"]["features"]
        expected = pattern[entry["head"]] @ (x @ w["v"].T)
        assert torch.allclose(heads_in[:, start:stop], expected, atol=1e-5)
        total = total + heads_in[:, start:stop] @ w["o"].T
    assert torch.allclose(total, projected[o_proj][0].float()[0], atol=1e-5)


def test_qwen3_5_gated_split(tiny_model, make_lab):
    lab = make_lab(*tiny_model("qwen3_5"))
    heads = lab.heads()
    assert {h["layer"] for h in heads} == {1}
    assert len(heads) == 4
    attn = "model.layers.1.self_attn"
    inputs = tokens(lab)
    with lab.record(attn, positions="all", at="input") as source, torch.no_grad():
        lab.model(**inputs)
    x = source[attn][0].float()[0]
    full = lab.model.model.layers[1].self_attn.q_proj(x).view(x.shape[0], 4, 8)
    for entry in heads:
        h = entry["head"]
        w = lab.head_weights(1, h)
        assert torch.allclose(x @ w["q"].T, full[:, h, :4], atol=1e-5)
        assert torch.allclose(x @ w["gate"].T, full[:, h, 4:], atol=1e-5)
        assert len(entry["gate"]) == len(entry["q"])
        assert entry["gate"][0]["index"] == (slice(8 * h + 4, 8 * h + 8), slice(None))


def test_opt_heads(tiny_tokenizer, make_lab):
    from transformers import OPTConfig, OPTForCausalLM

    torch.manual_seed(0)
    config = OPTConfig(
        vocab_size=6,
        hidden_size=16,
        ffn_dim=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        word_embed_proj_dim=16,
        max_position_embeddings=64,
        pad_token_id=1,
        bos_token_id=1,
        eos_token_id=1,
    )
    lab = make_lab(OPTForCausalLM(config).eval(), tiny_tokenizer)
    heads = lab.heads()
    assert len(heads) == 4
    entry = heads[1]
    assert entry["o"][0]["name"] == "model.decoder.layers.0.self_attn.out_proj.weight"
    assert entry["o"][0]["index"] == (slice(None), slice(8, 16))
    assert len(entry["q"]) == 2 and entry["q"][1]["name"].endswith("q_proj.bias")
    assert entry["o_input"]["module"] == "model.decoder.layers.0.self_attn.out_proj"
    assert lab.head_weights(0, 1)["o"].shape == (16, 8)


class Fused(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 8, "num_attention_heads": 2})()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].attn = nn.Module()
        setattr(self.layers[0].attn, name, nn.Linear(8, 24))
        self.layers[0].attn.o_proj = nn.Linear(8, 8)


@pytest.mark.parametrize("name", ["qkv_proj", "query_key_value", "q_a_proj", "kv_a_proj_with_mqa"])
def test_unsupported_layouts_raise(name, tiny_tokenizer, make_lab):
    lab = make_lab(Fused(name), tiny_tokenizer)
    with pytest.raises(NotImplementedError, match=name):
        lab.heads()
    with pytest.raises(NotImplementedError, match=name):
        lab.head_weights(0, 0)


def test_model_without_attention_raises(tiny_tokenizer, make_lab):
    lab = make_lab(nn.Sequential(nn.Linear(4, 4)), tiny_tokenizer)
    with pytest.raises(NotImplementedError, match="no attention block"):
        lab.heads()


def test_lora_model_raises(tiny_model, make_lab):
    peft = pytest.importorskip("peft")
    model, tokenizer = tiny_model("gpt2")
    model = peft.get_peft_model(
        model, peft.LoraConfig(r=2, target_modules=["c_attn"], fan_in_fan_out=True)
    )
    lab = make_lab(model, tokenizer)
    with pytest.raises(NotImplementedError, match="merged"):
        lab.heads()
    with pytest.raises(NotImplementedError, match="merged"):
        lab.head_weights(0, 0)


def test_layer_and_head_validation(gpt2_lab):
    assert [h["layer"] for h in gpt2_lab.heads(layers=[1])] == [1, 1]
    for bad in ([2], [0, 0], [True], ["0"], [-1], 0, "0"):
        with pytest.raises(ValueError, match="layers"):
            gpt2_lab.heads(layers=bad)
    for layer, head in [(2, 0), (-1, 0), (0, 2), (0, -1), (True, 0), (0, 1.0)]:
        with pytest.raises(ValueError, match="layer|head"):
            gpt2_lab.head_weights(layer, head)
    with pytest.raises(ValueError, match="0..1"):
        gpt2_lab.head_weights(0, 2)
    weights = gpt2_lab.head_weights(1, 1)
    assert weights["q"].dtype == torch.float32 and weights["q"].device.type == "cpu"
    assert weights["q"].shape == (8, 16) and weights["o"].shape == (16, 8)
    assert set(weights) == {"q", "k", "v", "o", "q_bias", "k_bias", "v_bias"}
    assert weights["q_bias"].shape == (8,)


def test_head_weights_are_copies(gpt2_lab):
    before = gpt2_lab.model.transformer.h[0].attn.c_attn.weight.detach().clone()
    gpt2_lab.head_weights(0, 0)["q"].zero_()
    assert torch.equal(gpt2_lab.model.transformer.h[0].attn.c_attn.weight, before)
