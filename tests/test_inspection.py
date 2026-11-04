from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from oumi_weights_lab.inspection import InspectionMixin, train_probe


class Inspect(InspectionMixin):
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cpu")


class TupleLayer(torch.nn.Module):
    def forward(self, value):
        return value * 2, "preserved"


class TupleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = TupleLayer()

    def forward(self, value):
        return self.layer(value)


def tiny_model():
    torch.manual_seed(3)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=11,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
    )


def test_record_positions_multiple_calls_and_hook_cleanup():
    model = TupleModel()
    lab = Inspect(model)
    value = torch.arange(24.0).reshape(2, 3, 4)
    with lab.record("layer", positions=[0, -1]) as captured:
        model(value)
        model(value + 1)
    assert len(captured["layer"]) == 2
    assert torch.equal(captured["layer"][0], (value * 2)[:, [0, -1]])
    assert captured["layer"][0].shape == (2, 2, 4)
    assert not captured["layer"][0].requires_grad
    assert not model.layer._forward_hooks
    with pytest.raises(RuntimeError), lab.record("layer"):
        raise RuntimeError("caller failed")
    assert not model.layer._forward_hooks


def test_patch_preserves_structure_positions_and_cleanup():
    model = TupleModel()
    lab = Inspect(model)
    value = torch.ones(1, 3, 4)
    baseline = model(value)[0]
    with lab.patch("layer", torch.tensor(7, dtype=torch.int64), positions=1):
        changed, other = model(value)
    assert other == "preserved"
    assert changed.dtype == value.dtype
    assert torch.equal(changed[:, 0], baseline[:, 0])
    assert torch.equal(changed[:, 1], torch.full((1, 4), 7.0))
    assert torch.equal(changed[:, 2], baseline[:, 2])
    assert torch.equal(model(value)[0], baseline)
    with pytest.raises(RuntimeError), lab.patch("layer", lambda selected: selected * 0):
        assert model(value)[0][:, -1].count_nonzero() == 0
        raise RuntimeError("caller failed")
    assert not model.layer._forward_hooks


def test_hook_model_forward_error_removes_handles():
    lab = Inspect(TupleModel())
    with pytest.raises(ValueError), lab.record("layer", positions=10):
        lab.model(torch.ones(1, 2, 3))
    assert not lab.model.layer._forward_hooks
    with pytest.raises(ValueError), lab.record(["layer", "layer"]):
        pass
    assert not lab.model.layer._forward_hooks


def output_type(model, module, ids):
    types = []
    handle = model.get_submodule(module).register_forward_hook(
        lambda hooked, inputs, output: types.append(type(output))
    )
    try:
        model(ids)
    finally:
        handle.remove()
    return types[0]


@pytest.mark.parametrize(
    "module,kind,position",
    # GPT2Block returns a plain tensor; GPT2Attention returns (output, weights).
    [("transformer.h.0", torch.Tensor, -1), ("transformer.h.0.attn", tuple, 1)],
)
def test_record_and_patch_real_transformer_outputs(module, kind, position):
    model = tiny_model().eval()
    lab = Inspect(model)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        assert issubclass(output_type(model, module, ids), kind)
        with lab.record(module) as captured:
            baseline = model(ids).logits
        assert captured[module][0].shape == (1, 1, 8)
        with lab.patch(module, lambda values: values * 0, positions=position):
            patched = model(ids).logits
    assert patched.shape == baseline.shape == (1, 3, 11)
    # One layer: only the patched position's residual stream, and logits, can move.
    changed = (patched != baseline).any(-1)[0].tolist()
    assert changed == [index == position % 3 for index in range(3)]
    assert not model.get_submodule(module)._forward_hooks


def test_patch_and_record_reject_aliased_positions():
    model = TupleModel()
    lab = Inspect(model)
    value = torch.ones(1, 3, 4)
    for positions in ([2, -1], [0, -3], [1, 1]):
        with (
            pytest.raises(ValueError, match="distinct"),
            lab.patch("layer", torch.zeros(1, 2, 4), positions=positions),
        ):
            model(value)
        with pytest.raises(ValueError, match="distinct"), lab.record("layer", positions=positions):
            model(value)
    assert not model.layer._forward_hooks
    with lab.record("layer", positions=[0, -1]) as captured:
        model(value)
    assert torch.equal(captured["layer"][0], (value * 2)[:, [0, 2]])


def test_gradient_matches_manual_per_row_and_restores_flags():
    model = tiny_model().eval()
    lab = Inspect(model)
    name = "transformer.wte.weight"
    parameter = dict(model.named_parameters())[name]
    ids = torch.tensor([[1, 2, 3, 4], [2, 3, 5, 6]])
    expected = []
    for row in ids:
        loss = model(row[None], labels=row[None]).loss
        expected.append(torch.autograd.grad(loss, parameter)[0])
    parameter.grad = torch.ones_like(parameter)
    parameter.requires_grad_(False)
    model.train()
    model.transformer.h[0].eval()
    modes = [module.training for module in model.modules()]
    original = parameter.detach().clone()
    with torch.no_grad():
        actual = lab.gradient({"input_ids": ids}, parameters=[name])[name]
    torch.testing.assert_close(actual, torch.stack(expected))
    assert not parameter.requires_grad
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(parameter.grad, torch.ones_like(parameter))
    assert torch.equal(parameter, original)
    assert actual.device.type == "cpu" and not actual.requires_grad


def test_gradient_masks_labels_and_rejects_long_rows():
    model = tiny_model().eval()
    lab = Inspect(model)
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 3, 4]])
    name = "transformer.wte.weight"
    expected = torch.autograd.grad(
        model(ids, labels=labels).loss, dict(model.named_parameters())[name]
    )[0]
    actual = lab.gradient({"input_ids": ids, "labels": labels}, parameters=[name])
    torch.testing.assert_close(actual[name][0], expected)
    with pytest.raises(ValueError, match="2 to 3"):
        lab.gradient({"input_ids": ids}, max_tokens=3)
    with pytest.raises(ValueError, match="scored token"):
        lab.gradient({"input_ids": ids, "labels": torch.full_like(ids, -100)})
    with pytest.raises(ValueError, match="parameters"):
        lab.gradient({"input_ids": ids}, parameters=["missing"])


def eager_llama():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=11,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval(), "model.layers.0.mlp.down_proj.weight"


def eager_gpt2():
    model = tiny_model()
    model.config._attn_implementation = "eager"
    return model.eval(), "transformer.h.0.mlp.c_proj.weight"


@pytest.mark.parametrize("build", [eager_llama, eager_gpt2], ids=["rope", "absolute"])
def test_padded_gradients_match_unpadded_rows(build):
    model, name = build()
    lab = Inspect(model)
    expected = lab.gradient({"input_ids": torch.tensor([[3, 4, 5]])}, parameters=[name])[name]
    batch = {
        "input_ids": torch.tensor([[0, 0, 3, 4, 5], [3, 4, 5, 0, 0]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 0, 0]]),
    }
    actual = lab.gradient(batch, parameters=[name])[name]
    torch.testing.assert_close(actual[0], expected[0])  # left padding
    torch.testing.assert_close(actual[1], expected[0])  # right padding
    explicit = dict(batch, position_ids=torch.tensor([[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]))
    torch.testing.assert_close(lab.gradient(explicit, parameters=[name])[name], actual)


def test_gradient_accepts_tied_parameter_aliases():
    model = tiny_model().eval()
    lab = Inspect(model)
    assert model.lm_head.weight is model.transformer.wte.weight
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = torch.autograd.grad(model(ids, labels=ids).loss, model.lm_head.weight)[0]
    tied = lab.gradient({"input_ids": ids}, parameters=["lm_head.weight"])
    assert list(tied) == ["lm_head.weight"]
    torch.testing.assert_close(tied["lm_head.weight"][0], expected)
    both = lab.gradient({"input_ids": ids}, parameters=["lm_head.weight", "transformer.wte.weight"])
    assert list(both) == ["lm_head.weight", "transformer.wte.weight"]
    assert both["lm_head.weight"] is both["transformer.wte.weight"]
    assert "lm_head.weight" not in lab.gradient({"input_ids": ids})
    with pytest.raises(ValueError, match="distinct"):
        lab.gradient({"input_ids": ids}, parameters=["lm_head.weight", "lm_head.weight"])


def test_gradient_failure_restores_model():
    model = tiny_model().train()
    lab = Inspect(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(IndexError):
        lab.gradient({"input_ids": torch.tensor([[1, 99]])}, parameters=["transformer.wte.weight"])
    assert model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_compiled_models_rejected_without_compiling():
    model = TupleModel()
    model._orig_mod = SimpleNamespace()
    lab = Inspect(model)
    with pytest.raises(ValueError, match="uncompiled"), lab.record("layer"):
        pass
    with pytest.raises(ValueError, match="uncompiled"), lab.patch("layer", 0):
        pass
    with pytest.raises(ValueError, match="uncompiled"):
        lab.gradient({"input_ids": torch.tensor([[1, 2]])})


def probe_data():
    features = torch.tensor([[-2.0, 0], [2.0, 0], [-3.0, 1], [3.0, 1], [-4.0, 2], [4.0, 2]])
    return features, torch.tensor([0, 1, 0, 1, 0, 1]), ["doc-a"] * 2 + ["doc-b"] * 2 + ["doc-c"] * 2


def test_probe_splits_whole_documents_and_predicts():
    features, labels, groups = probe_data()
    result = train_probe(features, labels, groups, train_groups=["doc-a", "doc-b"])
    assert result["train_groups"] == ["doc-a", "doc-b"]
    assert result["test_groups"] == ["doc-c"]
    assert result["train_size"] == 4 and result["test_size"] == 2
    assert result["accuracy"] == result["train_accuracy"] == 1
    scores = features.double() @ result["coefficients"] + result["intercept"]
    assert torch.equal(result["classes"][scores.argmax(1)], labels)
    first = train_probe(features, labels, groups, seed=2)
    second = train_probe(features, labels, groups, seed=2)
    assert first["train_groups"] == second["train_groups"]
    assert not set(first["train_groups"]) & set(first["test_groups"])


def test_probe_rejects_leaky_or_invalid_splits():
    features, labels, groups = probe_data()
    with pytest.raises(ValueError, match="two document"):
        train_probe(features, labels, ["same"] * 6)
    with pytest.raises(ValueError, match="proper subset"):
        train_probe(features, labels, groups, train_groups=["unknown"])
    with pytest.raises(ValueError, match="every class"):
        train_probe(features[:4], torch.tensor([0, 0, 1, 1]), groups[:4], train_groups=["doc-a"])
    with pytest.raises(ValueError, match="finite matrix"):
        train_probe(features * float("nan"), labels, groups)


@pytest.mark.parametrize("short", ["features", "labels", "groups"])
def test_probe_rejects_mismatched_row_counts(short):
    data = dict(zip(("features", "labels", "groups"), probe_data(), strict=True))
    data[short] = data[short][:-1]
    with pytest.raises(ValueError, match="same number of rows"):
        train_probe(**data)


def test_probe_tensor_document_ids_keep_related_rows_together():
    features, labels, _ = probe_data()
    result = train_probe(
        features, labels, torch.tensor([1, 1, 2, 2, 3, 3]), train_groups=torch.tensor([1, 2])
    )
    assert result["train_groups"] == [1, 2]
    assert result["test_groups"] == [3]


@pytest.mark.parametrize("positions", [True, [False, 1]])
def test_positions_reject_booleans(positions):
    model = TupleModel()
    lab = Inspect(model)
    with pytest.raises(ValueError, match="integer"), lab.record("layer", positions=positions):
        model(torch.ones(1, 3, 4))
    assert not model.layer._forward_hooks
