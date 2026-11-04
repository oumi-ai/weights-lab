import json

import pytest
import torch

from oumi_weights_lab.inspection import InspectionMixin


class Layer(torch.nn.Module):
    def __init__(self, a, b, scaling=1.0):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict()
        self.lora_B = torch.nn.ModuleDict()
        self.scaling = {}
        self.active_adapters = ["default"]
        self.base_layer = torch.nn.Linear(a.shape[1], b.shape[0], bias=False).double()
        self.add_adapter("default", a, b, scaling)

    def add_adapter(self, name, a, b, scaling):
        self.lora_A[name] = torch.nn.Linear(a.shape[1], a.shape[0], bias=False).double()
        self.lora_B[name] = torch.nn.Linear(b.shape[1], b.shape[0], bias=False).double()
        self.lora_A[name].weight.data.copy_(a)
        self.lora_B[name].weight.data.copy_(b)
        self.scaling[name] = scaling

    def forward(self, *args):
        raise AssertionError("diagnostics must not call forward")


class Inspect(InspectionMixin):
    def __init__(self, layer):
        self.model = torch.nn.ModuleDict({"projection": layer})


def factors():
    generator = torch.Generator().manual_seed(8)
    return torch.randn(3, 7, generator=generator).double(), torch.randn(
        5, 3, generator=generator
    ).double()


@pytest.mark.parametrize("scaling", [0.0, 0.5, -2.0])
def test_scaled_qr_svd_matches_dense(scaling):
    a, b = factors()
    layer = Layer(a, b, scaling)
    row = Inspect(layer).lora_diagnostics(top_k=2)[0]
    dense = scaling * b @ a
    spectrum = torch.linalg.svdvals(dense)
    assert row["top_singular_values"] == pytest.approx(spectrum[:2].tolist())
    assert row["frobenius_norm"] == pytest.approx(dense.norm().item())
    assert row["spectral_norm"] == pytest.approx(spectrum[0].item())
    assert row["base_frobenius_norm"] == pytest.approx(layer.base_layer.weight.norm().item())
    assert row["relative_frobenius_norm"] == pytest.approx(
        dense.norm().item() / layer.base_layer.weight.norm().item()
    )
    assert row["stable_rank"] == pytest.approx(
        (dense.norm() / spectrum[0]).square().item() if scaling else 0
    )
    assert row["numerical_rank"] == (3 if scaling else 0)
    assert row["scaling"] == scaling
    json.dumps(row, allow_nan=False)


def test_rank_deficient_and_zero_base():
    a = torch.tensor([[1.0, 0, 0], [2.0, 0, 0]])
    b = torch.tensor([[1.0, 2], [2.0, 4], [0.0, 0]])
    layer = Layer(a, b)
    layer.base_layer.weight.data.zero_()
    row = Inspect(layer).lora_diagnostics()[0]
    assert row["numerical_rank"] == 1
    assert row["stable_rank"] == pytest.approx(1.0)
    assert row["relative_frobenius_norm"] is None
    layer.lora_B["default"].weight.data.zero_()
    row = Inspect(layer).lora_diagnostics()[0]
    assert row["numerical_rank"] == row["stable_rank"] == row["frobenius_norm"] == 0


def test_pattern_and_multiple_adapter_selection_are_separate():
    a, b = factors()
    layer = Layer(a, b)
    layer.add_adapter("second", a, -b, 2.0)
    lab = Inspect(layer)
    assert [row["adapter"] for row in lab.lora_diagnostics()] == ["default"]
    assert lab.lora_diagnostics("*absent*") == []
    rows = lab.lora_diagnostics("*projection", adapter=["second", "default"])
    assert [row["adapter"] for row in rows] == ["second", "default"]
    assert [row["active"] for row in rows] == [False, True]
    assert rows[0]["frobenius_norm"] == pytest.approx(2 * rows[1]["frobenius_norm"])
    layer.active_adapters = ["default", "second"]
    assert len(lab.lora_diagnostics()) == 2
    layer.disable_adapters = True
    assert lab.lora_diagnostics() == []
    assert lab.lora_diagnostics(adapter="second")[0]["disabled"]
    with pytest.raises(ValueError, match="not found"):
        lab.lora_diagnostics(adapter="absent")


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "8"])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        Inspect(Layer(*factors())).lora_diagnostics(top_k=top_k)


@pytest.mark.parametrize("attribute", ["use_dora", "lora_variant", "use_qalora"])
def test_variants_rejected(attribute):
    layer = Layer(*factors())
    setattr(layer, attribute, {"default": True})
    with pytest.raises(ValueError, match="ordinary linear"):
        Inspect(layer).lora_diagnostics()


def test_embedding_rejected():
    model = torch.nn.Module()
    model.lora_embedding_A = {"default": torch.ones(2, 4)}
    model.lora_embedding_B = {"default": torch.ones(4, 2)}
    model.active_adapters = ["default"]
    with pytest.raises(ValueError, match="embedding"):
        Inspect(model).lora_diagnostics()


def test_merged_base_ratio_unavailable_and_state_preserved():
    layer = Layer(*factors()).train()
    layer.merged_adapters = ["default"]
    before = {name: value.clone() for name, value in layer.state_dict().items()}
    row = Inspect(layer).lora_diagnostics()[0]
    assert row["merged"] and row["base_frobenius_norm"] is None
    assert row["relative_frobenius_norm"] is None
    assert layer.training
    for name, value in layer.state_dict().items():
        assert torch.equal(value, before[name])
    assert all(parameter.grad is None for parameter in layer.parameters())


def test_nonfinite_factors_rejected():
    layer = Layer(*factors())
    layer.lora_A["default"].weight.data[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        Inspect(layer).lora_diagnostics()


def test_raw_tensor_factors_and_unavailable_base():
    a, b = factors()
    layer = torch.nn.Module()
    layer.lora_A, layer.lora_B = {"raw": a}, {"raw": b}
    layer.scaling = {"raw": 0.25}
    layer.active_adapter = "raw"
    row = Inspect(layer).lora_diagnostics()[0]
    assert row["frobenius_norm"] == pytest.approx((0.25 * b @ a).norm().item())
    assert row["base_frobenius_norm"] is None
    assert row["relative_frobenius_norm"] is None


def test_convolutional_factors_and_bias_rejected():
    layer = Layer(*factors())
    layer.lora_A["default"] = torch.nn.Conv1d(7, 3, 1, bias=False)
    with pytest.raises(ValueError, match="matrices"):
        Inspect(layer).lora_diagnostics()
    layer = Layer(*factors())
    layer.lora_B["default"].bias = torch.nn.Parameter(torch.zeros(5))
    with pytest.raises(ValueError, match="biases"):
        Inspect(layer).lora_diagnostics()


def test_only_small_core_matrix_is_multiplied(monkeypatch):
    layer = Layer(*factors())
    original = torch.Tensor.__matmul__

    def bounded_matmul(left, right):
        assert left.shape == right.shape == (3, 3)
        return original(left, right)

    monkeypatch.setattr(torch.Tensor, "__matmul__", bounded_matmul)
    assert Inspect(layer).lora_diagnostics()[0]["shape"] == [5, 7]


@pytest.mark.parametrize("adapter", [0, [], ["default", "default"], [None]])
def test_invalid_adapter_selection(adapter):
    with pytest.raises(ValueError, match="adapter"):
        Inspect(Layer(*factors())).lora_diagnostics(adapter=adapter)


def test_real_peft_linear_scaling_and_selection():
    peft = pytest.importorskip("peft")
    model = torch.nn.ModuleDict({"projection": torch.nn.Linear(7, 5, bias=False)})
    config = peft.LoraConfig(r=3, lora_alpha=6, target_modules=["projection"], use_rslora=True)
    model = peft.get_peft_model(model, config)
    layer = model.base_model.model.projection
    a, b = factors()
    layer.lora_A["default"].weight.data.copy_(a)
    layer.lora_B["default"].weight.data.copy_(b)
    lab = Inspect(model)
    row = lab.lora_diagnostics()[0]
    expected = torch.linalg.svdvals(b @ a) * layer.scaling["default"]
    assert row["scaling"] == pytest.approx(6 / 3**0.5)
    assert row["top_singular_values"] == pytest.approx(expected[:3].tolist(), rel=1e-6)
    model.add_adapter("second", config)
    model.set_adapter("second")
    assert [row["adapter"] for row in lab.lora_diagnostics()] == ["second"]
    assert lab.lora_diagnostics(adapter="default")[0]["active"] is False
