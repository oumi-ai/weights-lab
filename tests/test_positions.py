import pytest
import torch
from torch import nn

from oumi_weights_lab import inspection

KINDS = ["gpt2", "llama_gqa"]
# Layer, attention block, and the output projection whose input holds the per-head outputs.
NAMES = {
    "gpt2": ("transformer.h.0", "transformer.h.0.attn", "transformer.h.0.attn.c_proj"),
    "llama_gqa": ("model.layers.0", "model.layers.0.self_attn", "model.layers.0.self_attn.o_proj"),
}


class Interrupted(BaseException):
    pass


@pytest.fixture(params=KINDS)
def kind(request):
    return request.param


@pytest.fixture
def lab(kind, tiny_model, make_lab):
    return make_lab(*tiny_model(kind))


def tokens(lab, text="hello world yes no hello"):
    encoded = lab.tokenizer(text, return_tensors="pt", return_token_type_ids=False)
    return encoded.to(lab.device)


def run(lab, text="hello world yes no hello"):
    with torch.no_grad():
        return lab.model(**tokens(lab, text)).logits


def test_all_and_slice_positions_record(lab, kind):
    layer = NAMES[kind][0]
    inputs = tokens(lab)
    length = inputs["input_ids"].shape[1]
    with lab.record(layer, positions="all") as everything, torch.no_grad():
        lab.model(**inputs)
    full = everything[layer][0]
    assert full.shape[:2] == (1, length)
    with lab.record(layer, positions=slice(-2, None)) as last, torch.no_grad():
        lab.model(**inputs)
    assert torch.equal(last[layer][0], full[:, -2:])
    with lab.record(layer, positions=slice(0, None, 2)) as alternate, torch.no_grad():
        lab.model(**inputs)
    assert torch.equal(alternate[layer][0], full[:, ::2])


@pytest.mark.parametrize(
    "positions",
    [
        slice(0, None, 0),
        slice(None, None, -1),
        slice(1.5, None),
        slice(2, 2),
        slice(50, None),
        "first",
        "ALL",
        1.5,
        True,
        [],
        [1, 1],
        [0, 1.5],
        [2, -1, 0, 4],  # 4 and -1 are the same token on five tokens
        99,
    ],
)
def test_position_refusals(lab, kind, positions):
    layer = NAMES[kind][0]
    with pytest.raises(ValueError):
        with lab.record(layer, positions=positions):
            run(lab)
    with pytest.raises(ValueError):
        with lab.patch(layer, lambda value: value, positions=positions):
            run(lab)
    assert not lab.model.get_submodule(layer)._forward_hooks
    assert lab.patches() == []


def test_invalid_positions_raise_before_any_hook_is_registered(lab, kind):
    layer = NAMES[kind][0]
    with pytest.raises(ValueError):
        lab.record(layer, positions=slice(0, None, 0)).__enter__()
    with pytest.raises(ValueError):
        lab.record(layer, positions="first").__enter__()
    assert not lab.model.get_submodule(layer)._forward_hooks


def test_padding_positions_helper():
    check = inspection._padding_positions
    assert check(None) == [-1]
    assert check(2) == [2]
    assert check([3, -1]) == [3, -1]
    assert check("all") == []
    assert check(slice(-2, None)) == [-2]
    assert check(slice(-3, -1)) == [-3]
    assert check(slice(-3, None, 1)) == [-3]
    for unsafe in (
        slice(0, 2),
        slice(None, -1),
        slice(-4, None, 2),
        slice(None, None),
        slice(1, None),
    ):
        assert check(unsafe) == [0]


@pytest.mark.parametrize("intervention", ["record", "patch"])
def test_batched_scoring_padding_rules(lab, kind, intervention):
    layer = NAMES[kind][0]
    rows = [{"prompt": "hello world yes", "options": [" yes", " no"], "answer": "A"}]

    def hooked(positions):
        if intervention == "record":
            return lab.record(layer, positions=positions)
        return lab.patch(layer, lambda value: value, positions=positions)

    for refused in (slice(0, 2), slice(None, -1), [0]):
        with hooked(refused):
            with pytest.raises(ValueError, match="batch_size=1"):
                lab.score("hello world", [" yes", " no"], batch_size=2)
            with pytest.raises(ValueError, match="batch_size=1"):
                lab.evaluate(rows, batch_size=2)
    for allowed in ("all", slice(-2, None), -1, None):
        with hooked(allowed):
            assert len(lab.score("hello world", [" yes", " no"], batch_size=2)) == 2
            lab.evaluate(rows, batch_size=2)


def test_input_record_matches_previous_output(tiny_model, make_lab):
    lab = make_lab(*tiny_model("gpt2"))
    act, projection = "transformer.h.0.mlp.act", "transformer.h.0.mlp.c_proj"
    assert isinstance(lab.model.get_submodule(act), nn.Module)
    with lab.record(act, positions="all") as produced, torch.no_grad():
        lab.model(**tokens(lab))
    with lab.record(projection, positions="all", at="input") as consumed, torch.no_grad():
        lab.model(**tokens(lab))
    assert torch.equal(consumed[projection][0], produced[act][0])


def test_input_record_matches_manual_o_proj_input(tiny_model, make_lab):
    lab = make_lab(*tiny_model("llama_gqa"))
    projection = "model.layers.0.self_attn.o_proj"
    with lab.record(projection, positions="all", at="input") as consumed, torch.no_grad():
        lab.model(**tokens(lab))
    seen = consumed[projection][0]
    assert seen.shape[-1] == 16
    with torch.no_grad():
        recomputed = lab.model.get_submodule(projection)(seen)
    with lab.record(projection, positions="all") as output, torch.no_grad():
        lab.model(**tokens(lab))
    assert torch.allclose(recomputed, output[projection][0], atol=1e-6)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Linear(3, 3, bias=False)

    def forward(self, hidden_states=None, other=None):
        return self.scale(hidden_states)


class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.toy = Toy()

    def forward(self, x, keyword):
        return self.toy(hidden_states=x) if keyword else self.toy(x)


def toy_lab(make_lab, tiny_tokenizer):
    model = Outer()
    return make_lab(model, tiny_tokenizer), model


def test_input_hooks_read_and_replace_keyword_and_positional(make_lab, tiny_tokenizer):
    lab, model = toy_lab(make_lab, tiny_tokenizer)
    x = torch.arange(12.0).reshape(1, 4, 3)
    for keyword in (False, True):
        with lab.record("toy", positions=slice(1, 3), at="input") as seen:
            model(x, keyword)
        assert torch.equal(seen["toy"][0], x[:, 1:3])
        with lab.patch("toy", lambda value: value * 0, positions=[0], at="input"):
            patched = model(x, keyword)
        expected = model.toy.scale(x * torch.tensor([0.0, 1, 1, 1]).reshape(1, 4, 1))
        assert torch.allclose(patched, expected)
    with torch.no_grad():
        assert torch.equal(model(x, False), model.toy.scale(x))


def test_input_hook_refuses_modules_without_a_tensor_input(make_lab, tiny_tokenizer):
    lab, model = toy_lab(make_lab, tiny_tokenizer)
    with pytest.raises(ValueError, match="no tensor input"):
        with lab.record("toy", at="input"):
            model.toy(other=torch.zeros(1, 2, 3))
    with pytest.raises(ValueError, match=r"\[batch, tokens"):
        with lab.record("toy", at="input"):
            model.toy(torch.zeros(2, 3))
    assert not model.toy._forward_pre_hooks


def test_input_patch_follows_incoming_device_and_dtype(make_lab, tiny_tokenizer):
    lab, model = toy_lab(make_lab, tiny_tokenizer)
    x = torch.ones(1, 2, 3, dtype=torch.float64)
    seen = []
    model.toy.register_forward_hook(lambda module, args, output: seen.append(args[0].dtype))
    model.toy.double()
    with lab.patch("toy", torch.zeros(3, dtype=torch.float32), positions="all", at="input"):
        model(x, False)
    assert seen == [torch.float64]


def zero_head_logits(lab, projection, head, width):
    def zero(value):
        value = value.clone()
        value[..., head * width : (head + 1) * width] = 0
        return value

    with lab.patch(projection, zero, positions="all", at="input"):
        return run(lab)


def test_zeroing_head_via_o_proj_input_matches_zeroing_weights(lab, kind):
    projection = NAMES[kind][2]
    heads = lab.model.config.num_attention_heads if kind != "gpt2" else lab.model.config.n_head
    width = 16 // heads
    baseline = run(lab)
    for head in range(heads):
        patched = zero_head_logits(lab, projection, head, width)
        assert not torch.allclose(patched, baseline)
        span = slice(head * width, (head + 1) * width)
        index = (span, slice(None)) if kind == "gpt2" else (slice(None), span)
        lab.zero(f"{projection}.weight", index=index)
        edited = run(lab)
        lab.undo()
        assert torch.allclose(patched, edited, atol=1e-6)
    assert torch.equal(run(lab), baseline)
    assert lab.patches() == []


def test_attention_capture_rows_sum_to_one_and_are_causal(lab, kind):
    attention = NAMES[kind][1]
    inputs = tokens(lab)
    length = inputs["input_ids"].shape[1]
    heads = lab.model.config.num_attention_heads if kind != "gpt2" else lab.model.config.n_head
    with lab.record(attention, positions="all", at="attention") as captured, torch.no_grad():
        lab.model(**inputs)
    weights = captured[attention][0]
    assert weights.shape == (1, heads, length, length)
    assert torch.allclose(weights.sum(-1), torch.ones(1, heads, length), atol=1e-5)
    assert torch.equal(weights.triu(1), torch.zeros_like(weights))
    with lab.record(attention, positions=[-1], at="attention") as last, torch.no_grad():
        lab.model(**inputs)
    assert torch.equal(last[attention][0], weights[:, :, -1:])


def test_attention_capture_refuses_non_attention_module(lab, kind):
    layer = NAMES[kind][0]
    mlp = f"{layer}.mlp"
    for module in (layer, mlp, NAMES[kind][2]):
        with pytest.raises(ValueError, match="attention weights"):
            with lab.record(module, at="attention"):
                run(lab)
        assert not lab.model.get_submodule(module)._forward_hooks
    with pytest.raises(ValueError, match="use record"):
        with lab.patch(NAMES[kind][1], lambda value: value, at="attention"):
            pass
    assert lab.patches() == []


def test_attention_capture_skips_linear_attention_blocks(tiny_model, make_lab):
    model, tokenizer = tiny_model("qwen3_5")
    lab = make_lab(model, tokenizer)
    with pytest.raises(ValueError, match="attention weights"):
        with lab.record("model.layers.0.linear_attn", at="attention"):
            run(lab)
    with lab.record("model.layers.1.self_attn", positions="all", at="attention") as captured:
        run(lab)
    assert captured["model.layers.1.self_attn"][0].shape[1] == 4


@pytest.mark.parametrize("at", ["middle", "Output", None, ["input"]])
def test_at_must_be_a_listed_string(lab, kind, at):
    layer = NAMES[kind][0]
    with pytest.raises(ValueError, match="at must be"):
        lab.record(layer, at=at).__enter__()
    with pytest.raises(ValueError, match="at must be"):
        lab.patch(layer, lambda value: value, at=at).__enter__()
    assert not lab.model.get_submodule(layer)._forward_hooks
    assert lab.patches() == []


def test_generation_all_positions_capture_lengths(lab, kind):
    layer = NAMES[kind][0]
    inputs = tokens(lab, "hello world yes")
    length = inputs["input_ids"].shape[1]
    with lab.record(layer, positions="all") as captured, torch.no_grad():
        lab.model.generate(
            **inputs,
            max_new_tokens=3,
            min_new_tokens=3,
            do_sample=False,
            use_cache=True,
            pad_token_id=1,
        )
    assert [value.shape[1] for value in captured[layer]] == [length, 1, 1]


def test_steering_patch_same_tokens_with_and_without_cache(lab, kind):
    layer = NAMES[kind][0]
    inputs = tokens(lab, "hello world yes")
    direction = torch.linspace(-1, 1, 16) * 3

    def generate(use_cache):
        with torch.no_grad():
            out = lab.model.generate(
                **inputs,
                max_new_tokens=5,
                min_new_tokens=5,
                do_sample=False,
                use_cache=use_cache,
                pad_token_id=1,
            )
        return out[0].tolist()

    plain = generate(True)
    with lab.patch(layer, lambda value: value + direction, positions="all"):
        cached, uncached = generate(True), generate(False)
    assert cached == uncached
    assert generate(True) == plain


@pytest.mark.parametrize("registration", ["pre", "forward"])
def test_pre_hook_removed_after_failure(lab, kind, monkeypatch, registration):
    projection = NAMES[kind][2]
    module = lab.model.get_submodule(projection)
    with pytest.raises(RuntimeError, match="boom"):
        with lab.record(projection, at="input"), lab.patch(projection, 0, at="input"):
            assert len(module._forward_pre_hooks) == 2
            raise RuntimeError("boom")
    assert not module._forward_pre_hooks
    assert not module._forward_pre_hooks_with_kwargs
    assert lab.patches() == []

    register = module.register_forward_pre_hook

    def interrupting(*args, **kwargs):
        register(*args, **kwargs)
        raise Interrupted("timeout after hook registration")

    monkeypatch.setattr(module, "register_forward_pre_hook", interrupting)
    with pytest.raises(Interrupted):
        with lab.record(projection, at="input"):
            pytest.fail("the record context must not be entered")
    with pytest.raises(Interrupted):
        with lab.patch(projection, lambda value: value, at="input"):
            pytest.fail("the patch context must not be entered")
    assert not module._forward_pre_hooks
    assert not module._forward_pre_hooks_with_kwargs
    assert lab.patches() == []
    assert not any(module._forward_hooks for module in lab.model.modules())


def test_orphaned_pre_hook_stays_inert(lab, kind, monkeypatch):
    projection = NAMES[kind][2]
    module = lab.model.get_submodule(projection)
    fallback = inspection._remove_by_identity
    monkeypatch.setattr(
        inspection, "_remove_by_identity", lambda module, hook: fallback(object(), hook)
    )
    register = module.register_forward_pre_hook

    def interrupting(*args, **kwargs):
        register(*args, **kwargs)
        raise Interrupted("timeout after hook registration")

    monkeypatch.setattr(module, "register_forward_pre_hook", interrupting)
    baseline = run(lab)
    with pytest.raises(Interrupted):
        with lab.patch(projection, lambda value: value * 0, positions="all", at="input"):
            pass
    assert len(module._forward_pre_hooks) == 1
    assert torch.equal(run(lab), baseline)


def test_patches_metadata_includes_at(lab, kind):
    layer, _, projection = NAMES[kind]
    with lab.patch(layer, lambda value: value):
        with lab.patch(projection, 0, positions=slice(-2, None), at="input"):
            first, second = lab.patches()
    assert (first["at"], second["at"]) == ("output", "input")
    assert second["positions"] == repr(slice(-2, None))
    assert lab.patches() == []
