"""Edits to LLM.int8 weights requantize only the rows they touch and undo byte for byte."""

import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

import test_quantized_loading as loading  # noqa: E402

from oumi_weights_lab import quantized  # noqa: E402
from oumi_weights_lab.editing import EditBook  # noqa: E402
from oumi_weights_lab.quantized import lookup  # noqa: E402

PROMPT, Q_PROJ = loading.PROMPT, loading.Q_PROJ
checkpoint, load = loading.checkpoint, loading.load
DOWN = "model.layers.0.mlp.down_proj.weight"
O_PROJ = "model.layers.0.self_attn.o_proj.weight"


def packed(lab, name=Q_PROJ):
    """Copies of the int8 bytes and the row scales, wherever the scales live."""
    return loading.packed_state(lab, name)


def same_packed(first, second):
    return torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def logical(lab, name=Q_PROJ):
    return lookup(lab.model, name).dequantize()


def logits(lab, text=PROMPT):
    ids = lab.tokenizer(text, return_tensors="pt")["input_ids"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            return lab.model(input_ids=ids).logits.clone()


def changed_rows(before, after):
    return {int(r) for r in (before[0] != after[0]).any(dim=1).nonzero()} | {
        int(r) for r in (before[1] != after[1]).nonzero()
    }


def test_int8_untouched_rows_are_bit_identical(checkpoint, load):
    lab = load(checkpoint("llama"), quantize="int8")
    before = packed(lab)
    spots = [(0, 5), (7, 40), (30, 2), (62, 63)]
    for spot in spots:
        lab.edit(Q_PROJ, lambda value: value * 3 + 0.2, index=spot)
    after = packed(lab)
    touched = {row for row, _ in spots}
    assert changed_rows(before, after) <= touched
    keep = [row for row in range(64) if row not in touched]
    assert torch.equal(before[0][keep], after[0][keep])
    assert torch.equal(before[1][keep], after[1][keep])
    assert not same_packed(before, after)
    assert lab.edits()[0]["quantization"]["blocks_requantized"] == 1


def test_int8_row_requantize_equals_whole_tensor(checkpoint, load):
    import bitsandbytes.functional as functional

    lab = load(checkpoint("llama"), quantize="int8")
    before = logical(lab)
    before_packed = packed(lab)
    lab.edit(Q_PROJ, lambda value: value * 4 - 0.1, index=(slice(2, 4), slice(10, 50)))
    expected = before.clone()
    expected[2:4, 10:50] = expected[2:4, 10:50] * 4 - 0.1
    data, scales, _ = functional.int8_vectorwise_quant(expected.to(torch.float32), threshold=0.0)
    got = packed(lab)
    assert torch.equal(got[0][2:4], data[2:4]) and torch.equal(got[1][2:4], scales[2:4])
    assert changed_rows(before_packed, got) == {2, 3}
    assert lab.edits()[0]["quantization"]["blocks_requantized"] == 2


@pytest.mark.parametrize("forward_first", [False, True])
def test_int8_undo_exact_before_and_after_first_forward(checkpoint, load, forward_first):
    lab = load(checkpoint("llama"), quantize="int8")
    layer = lab.model.get_submodule(Q_PROJ.removesuffix(".weight"))
    if forward_first:
        logits(lab)
        assert layer.weight.SCB is None and layer.state.SCB is not None
    else:
        assert layer.weight.SCB is not None
    baseline_packed = {name: packed(lab, name) for name in (Q_PROJ, O_PROJ)}
    baseline = logits(lab)
    layer = lab.model.get_submodule(Q_PROJ.removesuffix(".weight"))  # scales have moved by now
    first = lab.edit(Q_PROJ, lambda value: value * 2 + 0.05, index=(0, slice(0, 4)))
    second = lab.zero(O_PROJ, index=(slice(0, 5), slice(None)))
    edited = logits(lab)
    assert not torch.equal(edited, baseline)  # The forward sees the new bytes and scales.
    live = layer.state.SCB if layer.state.SCB is not None else layer.weight.SCB
    assert torch.equal(live, packed(lab)[1])
    assert (logical(lab)[0, :4] != 0).any()
    lab.undo(second)
    lab.undo(first)
    for name, state in baseline_packed.items():
        assert same_packed(state, packed(lab, name))
    assert torch.equal(logits(lab), baseline)
    assert lab.active_edit_ids == []


def test_int8_vanishing_edit_warns(checkpoint, load):
    lab = load(checkpoint("llama"), quantize="int8")
    before = packed(lab)
    smallest = int(logical(lab)[1].abs().argmin())  # Not the row maximum, so the scale holds.
    with pytest.warns(UserWarning, match="edit-000001.*quantization step"):
        edit_id = lab.edit(Q_PROJ, lambda value: value + 1e-7, index=(1, smallest))
    report = lab.edits()[0]["quantization"]
    assert lab.edits()[0]["id"] == edit_id
    assert report["realized_delta_l2"] == 0 and report["requested_delta_l2"] > 0
    assert report["type"] == "int8" and report["blocksize"] == 64
    assert same_packed(before, packed(lab))
    lab.undo()
    assert lab.active_edit_ids == []


def test_int8_collateral_when_the_row_maximum_changes(checkpoint, load):
    lab = load(checkpoint("llama"), quantize="int8")
    row = logical(lab)[5]
    largest = int(row.abs().argmax())
    smallest = int(row.abs().argmin())
    lab.edit(Q_PROJ, lambda value: value * 0.5 + 0.0, index=(5, smallest))
    assert lab.edits()[-1]["quantization"]["collateral_l2"] == 0
    scale_before = packed(lab)[1][5].item()
    lab.edit(Q_PROJ, lambda value: value * 2.0, index=(5, largest))
    report = lab.edits()[-1]["quantization"]
    assert packed(lab)[1][5].item() > scale_before
    assert report["collateral_l2"] > 0 and report["changed_elements"] > 1
    lab.undo()
    lab.undo()


def test_int8_zero_whole_row(checkpoint, load):
    lab = load(checkpoint("llama"), quantize="int8")
    baseline_packed, baseline = packed(lab), logits(lab)
    lab.zero(Q_PROJ, index=(2, slice(None)))
    values = logical(lab)
    assert (values[2] == 0).all() and torch.isfinite(values).all()
    state = packed(lab)
    assert state[1][2] == 0 and (state[0][2] == 0).all()
    assert torch.isfinite(logits(lab)).all()
    lab.undo()
    assert same_packed(baseline_packed, packed(lab)) and torch.equal(logits(lab), baseline)
    lab.zero(Q_PROJ)  # Every row at once.
    assert (logical(lab) == 0).all() and torch.isfinite(logits(lab)).all()
    lab.undo()
    assert same_packed(baseline_packed, packed(lab))


def test_int8_rollback_after_failure_mid_write(checkpoint, load, monkeypatch):
    lab = load(checkpoint("llama"), quantize="int8")
    before, revision = packed(lab), lab.revision

    def explode(self, rows, scales):
        raise RuntimeError("disk full")

    monkeypatch.setattr(quantized.Int8Codec, "_store_scales", explode)
    with pytest.raises(RuntimeError, match="disk full"):
        lab.edit(Q_PROJ, lambda value: value * 7 + 1, index=(0, slice(0, 8)))
    assert same_packed(before, packed(lab))
    assert lab.edits() == [] and lab.active_edit_ids == [] and lab.revision == revision


def test_int8_project_out_and_head_ablation(checkpoint, load):
    lab = load(checkpoint("llama"), quantize="int8")
    baseline = logits(lab)
    states = {name: packed(lab, name) for name in (DOWN, O_PROJ)}
    direction = torch.randn(64, generator=torch.Generator().manual_seed(3))
    before = logical(lab, DOWN)
    lab.project_out(DOWN, direction)
    unit = direction / direction.norm()
    after = logical(lab, DOWN)
    assert (unit @ after).abs().max() < 0.1 * (unit @ before).abs().max()
    head = lab.heads(layers=[0])[1]
    ids = [lab.zero(**part) for part in head["o"]]
    ablated = logits(lab)
    assert not torch.equal(ablated, baseline) and torch.isfinite(ablated).all()
    columns = head["o"][0]["index"]
    assert (logical(lab, O_PROJ)[columns] == 0).all()
    for _ in ids:
        lab.undo()
    lab.undo()
    for name, state in states.items():
        assert same_packed(state, packed(lab, name))
    assert torch.equal(logits(lab), baseline)


def test_int8_swap_from_float_model(checkpoint, load):
    path = checkpoint("llama")
    floating = load(path)
    lab = load(path, quantize="int8")
    source = floating.model.get_parameter(Q_PROJ).detach().clone() * 1.5 + 0.01
    before = packed(lab)
    lab.swap(Q_PROJ, source)
    values = logical(lab)
    step = packed(lab)[1][:, None] / 127
    assert ((values - source).abs() <= 0.5 * step + 1e-6).all()
    lab.undo()
    assert same_packed(before, packed(lab))
    lab.swap(Q_PROJ, floating.model, index=(slice(0, 4), slice(None)))
    assert (logical(lab)[:4] - floating.model.get_parameter(Q_PROJ)[:4]).abs().max() < 0.02
    lab.undo()
    with pytest.raises(ValueError, match="does not match"):
        lab.swap(Q_PROJ, torch.zeros(64, 32))
    assert same_packed(before, packed(lab))


def test_int8_reads_see_the_edited_values(checkpoint, load):
    lab = load(checkpoint("llama"), quantize="int8")
    lab.zero(Q_PROJ, index=(3, slice(None)))
    measured = lab.measure(Q_PROJ)
    assert measured["quantization"]["method"] == "int8"
    assert measured["l2"] == pytest.approx(logical(lab).double().norm().item(), rel=1e-9)
    logits(lab)  # The scales move to the layer; the numbers do not.
    assert lab.measure(Q_PROJ) == measured
    lab.undo()
    assert lab.measure(Q_PROJ)["l2"] != measured["l2"]


def test_int8_refuses_a_layer_that_keeps_float_weights():
    import bitsandbytes as bnb

    layer = bnb.nn.Linear8bitLt(64, 8, bias=False, has_fp16_weights=True, threshold=0.0)
    model = torch.nn.Sequential(layer)
    book = EditBook(model)
    before = layer.weight.data.clone()
    with pytest.raises(ValueError, match="has_fp16_weights"):
        book.zero("0.weight")
    assert torch.equal(layer.weight.data, before) and book.edits() == []


def test_int8_edits_survive_a_threshold_forward(checkpoint, load):
    """The outlier path reads the same bytes and scales, so an edit changes its output too."""
    lab = load(checkpoint("llama"), quantize="int8")
    layer = lab.model.get_submodule(Q_PROJ.removesuffix(".weight"))
    assert layer.state.threshold == 6.0
    x = torch.randn(3, 64).half()
    x[:, 4] = 30.0  # An outlier input column goes through the float path.
    layer.eval()
    with torch.no_grad():
        first = layer(x).clone()
    lab.zero(Q_PROJ, index=(slice(None), 4))
    with torch.no_grad():
        second = layer(x)
    assert not torch.equal(first, second)
    lab.undo()
    with torch.no_grad():
        assert torch.equal(layer(x), first)


@pytest.mark.parametrize("forward_first", [False, True])
def test_int8_undo_refuses_changed_scale_layout(checkpoint, load, forward_first):
    lab = load(checkpoint("llama"), quantize="int8")
    if forward_first:
        logits(lab)
    layer = lab.model.get_submodule(Q_PROJ.removesuffix(".weight"))
    lab.zero(Q_PROJ, index=(0, slice(0, 8)))
    holder = layer.state if forward_first else layer.weight
    original = holder.SCB
    current = packed(lab)
    for replaced in (original.to(torch.float16), original[:-1].clone()):
        holder.SCB = replaced
        with pytest.raises(RuntimeError, match="changed since the edit"):
            lab.undo()
        assert lab.active_edit_ids == ["edit-000001"]
        assert torch.equal(layer.weight.data, current[0])
    holder.SCB = original
    lab.undo()
    assert lab.active_edit_ids == []


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_int8_edits_use_the_compute_dtype(checkpoint, load, dtype):
    import bitsandbytes.functional as functional

    lab = load(checkpoint("llama"), quantize="int8", dtype=dtype)
    compute = getattr(torch, dtype)
    assert lab.quantization["compute_dtype"] == dtype
    # The first float parameter is no longer in the compute dtype.
    lab.model.get_parameter("model.embed_tokens.weight").data = lab.model.get_parameter(
        "model.embed_tokens.weight"
    ).data.float()
    before, state = logical(lab), packed(lab)
    smallest = int(before[4].abs().argmin())
    lab.edit(Q_PROJ, lambda value: value + 0.001, index=(4, smallest))
    record = lab.edits()[0]
    assert record["dtype"] == str(compute)
    assert [p["dtype"] for p in lab.parts(Q_PROJ)] == [str(compute)]
    assert record["quantization"]["blocks_requantized"] == 1
    expected = before.clone()
    expected[4, smallest] += 0.001
    data, scales, _ = functional.int8_vectorwise_quant(expected[4:5].to(compute), threshold=0.0)
    got = packed(lab)
    assert torch.equal(got[0][4:5], data) and torch.equal(got[1][4:5], scales)
    assert changed_rows(state, got) <= {4}
    lab.zero(Q_PROJ, index=(5, slice(None)))
    assert lab.edits()[1]["dtype"] == str(compute)
    lab.undo()
    lab.undo()
    assert same_packed(state, packed(lab))


def test_int8_edit_and_undo_on_mps(checkpoint, load):
    if not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS hardware")
    if tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) < (2, 5):
        pytest.skip("quantized models on MPS need torch 2.5 or later")
    lab = load(checkpoint("llama"), quantize="int8", device="mps")
    before = packed(lab)
    lab.edit(Q_PROJ, lambda block: block * 0.5, index=(slice(0, 2), slice(None)))
    assert not same_packed(before, packed(lab))
    lab.undo()
    assert same_packed(before, packed(lab))
