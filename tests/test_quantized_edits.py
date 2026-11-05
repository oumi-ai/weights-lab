"""Edits to nf4 and fp4 weights requantize only the blocks they touch and undo byte for byte."""

import json
import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

import test_quantized_loading as loading  # noqa: E402

from oumi_weights_lab import quantized  # noqa: E402
from oumi_weights_lab.editing import EditBook  # noqa: E402
from oumi_weights_lab.quantized import lookup  # noqa: E402

PROMPT, Q_PROJ = loading.PROMPT, loading.Q_PROJ
# Fixtures that build tiny checkpoints and load them quantized come from the loading tests.
checkpoint, load = loading.checkpoint, loading.load
FOUR_BIT = ("nf4", "fp4")
EMBED = "model.embed_tokens.weight"
DOWN = "model.layers.0.mlp.down_proj.weight"
O_PROJ = "model.layers.0.self_attn.o_proj.weight"
BLOCK = 64


def packed(lab, name=Q_PROJ):
    """Copies of the packed bytes and the scales of one 4-bit weight."""
    weight = lab.model.get_parameter(name)
    return weight.data.clone(), weight.quant_state.absmax.clone()


def same_packed(first, second):
    return torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def logical(lab, name=Q_PROJ):
    return lookup(lab.model, name).dequantize()


def logits(lab, text=PROMPT):
    ids = lab.tokenizer(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        return lab.model(input_ids=ids).logits.clone()


def changed_blocks(before, after):
    """Block numbers whose bytes or scale differ, from two ``packed`` results."""
    flat = (before[0] != after[0]).flatten().nonzero().flatten()
    blocks = {int(b) for b in (flat * 2) // BLOCK}
    blocks |= {int(b) for b in (before[1] != after[1]).nonzero().flatten()}
    return blocks


@pytest.mark.parametrize("method", FOUR_BIT)
def test_element_to_block_mapping(checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    width = 64
    for row, column in [(0, 0), (0, 1), (3, 17), (10, 63), (63, 62)]:
        flat = row * width + column
        before = packed(lab)
        scale = before[1][flat // BLOCK].item()
        current = logical(lab)[row, column].item()
        assert abs(current) < scale  # Not the block's largest, so the block keeps its scale.
        target = -0.6 * scale if current > 0.3 * scale else 0.6 * scale
        lab.edit(Q_PROJ, torch.tensor(target), index=(row, column))
        after = packed(lab)
        changed = (before[0] != after[0]).flatten().nonzero().flatten().tolist()
        assert changed == [flat // 2]
        assert changed_blocks(before, after) == {flat // BLOCK}
        lab.undo()
        assert same_packed(before, packed(lab))


@pytest.mark.parametrize("method", FOUR_BIT)
def test_untouched_blocks_are_bit_identical(checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    before = packed(lab)
    spots = [(0, 5), (7, 40), (30, 2), (62, 63)]
    for spot in spots:
        lab.edit(Q_PROJ, lambda value: value * 3 + 0.2, index=spot)
    after = packed(lab)
    touched = {(row * 64 + column) // BLOCK for row, column in spots}
    assert changed_blocks(before, after) <= touched
    keep = torch.ones(before[0].numel(), dtype=torch.bool)
    keep_scale = torch.ones(before[1].numel(), dtype=torch.bool)
    for block in touched:
        keep[block * BLOCK // 2 : (block + 1) * BLOCK // 2] = False
        keep_scale[block] = False
    assert torch.equal(before[0].flatten()[keep], after[0].flatten()[keep])
    assert torch.equal(before[1][keep_scale], after[1][keep_scale])
    assert not torch.equal(before[0], after[0])


def test_nf4_splice_equals_whole_tensor_requantize(checkpoint, load):
    import bitsandbytes.functional as functional

    lab = load(checkpoint("llama"), quantize="nf4")
    before = logical(lab)
    lab.edit(Q_PROJ, lambda value: value * 4 - 0.1, index=(slice(2, 4), slice(10, 50)))
    expected = before.clone()
    expected[2:4, 10:50] = expected[2:4, 10:50] * 4 - 0.1
    state = lab.model.get_parameter(Q_PROJ).quant_state
    data, new_state = functional.quantize_4bit(
        expected.to(state.dtype), blocksize=state.blocksize, quant_type="nf4"
    )
    got = packed(lab)
    assert torch.equal(got[0].flatten(), data.flatten())
    assert torch.equal(got[1], new_state.absmax)


@pytest.mark.parametrize("method", FOUR_BIT)
def test_undo_restores_bytes_absmax_and_logits_exactly(checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    baseline_packed = {name: packed(lab, name) for name in (Q_PROJ, O_PROJ)}
    baseline_embed = lab.model.get_parameter(EMBED).detach().clone()
    baseline = logits(lab)
    first = lab.edit(Q_PROJ, lambda value: value * 2 + 0.05, index=(0, slice(0, 4)))
    second = lab.edit(EMBED, lambda value: value * 0.5 + 0.25)
    third = lab.edit(Q_PROJ, lambda value: value - 0.3, index=(0, slice(4, 8)))
    fourth = lab.zero(O_PROJ, index=(slice(0, 5), slice(None)))
    assert not torch.equal(logits(lab), baseline)
    for edit_id in (fourth, third, second, first):
        assert lab.undo(edit_id) == edit_id
    for name, state in baseline_packed.items():
        assert same_packed(state, packed(lab, name))
    assert torch.equal(lab.model.get_parameter(EMBED), baseline_embed)
    assert torch.equal(logits(lab), baseline)
    assert lab.active_edit_ids == []


@pytest.mark.parametrize("method", FOUR_BIT)
def test_vanishing_edit_warns_and_records(checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    before, scales = packed(lab), logical(lab)
    with pytest.warns(UserWarning, match="edit-000001.*quantization step"):
        edit_id = lab.edit(Q_PROJ, lambda value: value + 1e-5, index=(1, 1))
    record = lab.edits()[0]
    assert record["id"] == edit_id and record["active"]
    report = record["quantization"]
    assert report["requested_delta_l2"] > 0 and report["realized_delta_l2"] == 0
    assert report["changed_elements"] == 0
    assert torch.equal(logical(lab), scales)
    lab.undo()
    assert same_packed(before, packed(lab))
    assert lab.edits()[0]["active"] is False


@pytest.mark.parametrize("method", FOUR_BIT)
def test_quantization_report_values(checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    before = logical(lab)
    region = (0, slice(0, 10))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lab.edit(Q_PROJ, lambda value: value * 5 + 0.1, index=region)
    after = logical(lab)
    candidate = before.clone()
    candidate[region] = before[region] * 5 + 0.1
    inside = torch.zeros(before.shape, dtype=torch.bool)
    inside[region] = True
    record = lab.edits()[0]
    report = record["quantization"]

    def norm(values):
        return values.double().norm().item()

    assert report["type"] == method and report["blocksize"] == BLOCK
    assert report["blocks_requantized"] == 1
    assert report["requested_delta_l2"] == pytest.approx(
        norm((candidate - before)[inside]), rel=1e-6
    )
    assert report["realized_delta_l2"] == pytest.approx(norm((after - before)[inside]), rel=1e-6)
    assert report["residual_l2"] == pytest.approx(norm((after - candidate)[inside]), rel=1e-6)
    assert report["collateral_l2"] == pytest.approx(norm((after - before)[~inside]), rel=1e-6)
    assert report["collateral_l2"] > 0
    assert report["changed_elements"] == int((after != before).sum())
    assert record["shape"] == [64, 64] and record["dtype"] == "torch.float32"
    assert record["index"] == repr(region)
    json.dumps(lab.edits())


@pytest.fixture
def partial_tail():
    """A bare 3x37 4-bit layer: 111 elements, one full block and a 47-element tail."""
    import bitsandbytes as bnb_

    def build(method):
        torch.manual_seed(1)
        layer = bnb_.nn.Linear4bit(
            37,
            3,
            bias=False,
            compress_statistics=False,
            quant_type=method,
            compute_dtype=torch.float32,
            device="cpu",
        )
        layer.weight = bnb_.nn.Params4bit(
            torch.randn(3, 37), requires_grad=False, compress_statistics=False, quant_type=method
        ).to("cpu")
        layer.support_avx512bf16_for_cpu = False
        return torch.nn.Sequential(layer)

    return build


@pytest.mark.parametrize("method", FOUR_BIT)
def test_zero_whole_block_and_tail(checkpoint, load, partial_tail, method):
    lab = load(checkpoint("llama"), quantize=method)
    baseline_packed, baseline = packed(lab), logits(lab)
    lab.zero(Q_PROJ, index=(2, slice(None)))  # row 2 is block 2 exactly
    values = logical(lab)
    assert (values[2] == 0).all() and torch.isfinite(values).all()
    assert packed(lab)[1][2] == 0
    assert torch.isfinite(logits(lab)).all()
    lab.undo()
    assert same_packed(baseline_packed, packed(lab)) and torch.equal(logits(lab), baseline)

    model = partial_tail(method)
    book = EditBook(model)
    start = (model[0].weight.data.clone(), model[0].weight.quant_state.absmax.clone())
    assert start[1].numel() == 2
    book.zero("0.weight", index=(slice(1, 3), slice(None)))  # elements 37..110: tail and more
    values = lookup(model, "0.weight").dequantize()
    assert (values[1:] == 0).all() and torch.isfinite(values).all() and (values[0] != 0).any()
    book.zero("0.weight", index=(slice(0, 1), slice(None)))
    values = lookup(model, "0.weight").dequantize()
    assert (values == 0).all()
    with torch.no_grad():
        assert torch.isfinite(model(torch.randn(2, 37))).all()
    book.undo()
    book.undo()
    assert torch.equal(model[0].weight.data, start[0])
    assert torch.equal(model[0].weight.quant_state.absmax, start[1])
    # A zeroed tail on its own, with the first block untouched.
    book.zero("0.weight", index=(2, slice(None)))
    assert (lookup(model, "0.weight").dequantize()[2] == 0).all()
    book.undo()
    assert torch.equal(model[0].weight.data, start[0])


@pytest.mark.parametrize("method", FOUR_BIT)
def test_rollback_after_failure_mid_write(checkpoint, load, monkeypatch, method):
    lab = load(checkpoint("llama"), quantize=method)
    before, revision = packed(lab), lab.revision

    def explode(self, blocks, absmax):
        raise RuntimeError("disk full")

    monkeypatch.setattr(quantized.FourBitCodec, "_store_absmax", explode)
    with pytest.raises(RuntimeError, match="disk full"):
        lab.edit(Q_PROJ, lambda value: value * 7 + 1, index=(0, slice(0, 8)))
    assert same_packed(before, packed(lab))
    assert lab.edits() == [] and lab.active_edit_ids == [] and lab.revision == revision


@pytest.mark.parametrize("method", FOUR_BIT)
def test_failed_callable_changes_nothing(checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    before = packed(lab)

    def broken(value):
        raise KeyError("bad")

    with pytest.raises(KeyError):
        lab.edit(Q_PROJ, broken, index=(0, slice(0, 8)))
    with pytest.raises(ValueError, match="does not match"):
        lab.edit(Q_PROJ, torch.zeros(3, 3))
    assert same_packed(before, packed(lab)) and lab.edits() == []


def repack(lab, name=Q_PROJ):
    from bitsandbytes.functional import _convert_weight_packed_for_cpu

    weight = lab.model.get_parameter(name)
    weight.data, weight.quant_state = _convert_weight_packed_for_cpu(
        weight.data, weight.quant_state
    )
    assert weight.quant_state.packing_format_for_cpu


@pytest.mark.parametrize("method", FOUR_BIT)
def test_repacked_weight_refuses_edit_and_undo(checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    lab.edit(Q_PROJ, lambda value: value * 2, index=(0, slice(0, 8)))
    repack(lab)
    after_repack = packed(lab)
    with pytest.raises(RuntimeError, match="changed since the edit"):
        lab.undo()
    assert same_packed(after_repack, packed(lab)) and lab.active_edit_ids == ["edit-000001"]
    with pytest.raises(ValueError, match="repacked"):
        lab.zero(Q_PROJ)
    assert same_packed(after_repack, packed(lab)) and len(lab.edits()) == 1


@pytest.mark.parametrize("method", FOUR_BIT)
def test_undo_refuses_a_replaced_parameter_or_state(checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    lab.zero(Q_PROJ, index=(0, slice(0, 8)))
    weight = lab.model.get_parameter(Q_PROJ)
    current = packed(lab)
    original_state = weight.quant_state
    weight.quant_state = type(original_state).__new__(type(original_state))
    weight.quant_state.__dict__.update(original_state.__dict__)
    with pytest.raises(RuntimeError, match="changed since the edit"):
        lab.undo()
    weight.quant_state = original_state
    assert same_packed(current, packed(lab))
    lab.undo()


def test_double_quant_checkpoint_refuses_edits(checkpoint, load, tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    saved = tmp_path / "double"
    model.save_pretrained(saved)
    AutoTokenizer.from_pretrained(path).save_pretrained(saved)
    lab = load(str(saved))
    before = packed(lab)
    for call in (
        lambda: lab.zero(Q_PROJ),
        lambda: lab.edit(Q_PROJ, lambda value: value + 1, index=(0, 0)),
        lambda: lab.project_out(Q_PROJ, torch.ones(64)),
    ):
        with pytest.raises(ValueError, match="bnb_4bit_use_double_quant") as error:
            call()
        assert "--quantize" in str(error.value)
    assert same_packed(before, packed(lab)) and lab.edits() == []


def test_int8_edits_still_refuse(checkpoint, load):
    lab = load(checkpoint("llama"), quantize="int8")
    before = lab.model.get_parameter(Q_PROJ).data.clone()
    for call in (
        lambda: lab.zero(Q_PROJ),
        lambda: lab.edit(Q_PROJ, lambda value: value + 1),
        lambda: lab.swap(Q_PROJ, torch.zeros(64, 64)),
        lambda: lab.project_out(Q_PROJ, torch.ones(64)),
    ):
        with pytest.raises(ValueError, match="not supported yet") as error:
            call()
        assert Q_PROJ in str(error.value) and "int8" in str(error.value)
    assert torch.equal(before, lab.model.get_parameter(Q_PROJ).data) and lab.edits() == []


@pytest.mark.parametrize("method", FOUR_BIT)
def test_project_out_on_quantized_linear(checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    before, state = logical(lab, DOWN), packed(lab, DOWN)
    generator = torch.Generator().manual_seed(3)
    direction = torch.randn(64, generator=generator)
    unit = direction / direction.norm()
    lab.project_out(DOWN, direction)
    after = logical(lab, DOWN)
    # DOWN is [out, in] = [64, 128]; the output side is axis 0.
    residual = (unit @ after).norm() / after.norm()
    untouched = (unit @ before).norm() / before.norm()
    assert untouched > 0.01
    # Rounding to the grid leaves a little of the direction behind. Measured here, the residual
    # is 0.015 ||W|| for nf4 and 0.024 ||W|| for fp4, against 0.12 ||W|| before the edit.
    assert residual < 0.03 and residual < untouched / 4
    report = lab.edits()[0]["quantization"]
    assert report["type"] == method and report["requested_delta_l2"] > 0
    assert report["blocks_requantized"] == 64 * 128 // BLOCK
    lab.undo()
    assert same_packed(state, packed(lab, DOWN))


def test_project_out_input_side_on_gpt2_linear(checkpoint, load):
    lab = load(checkpoint("gpt2"), quantize="nf4")
    name = "transformer.h.0.attn.c_attn.weight"
    state, before = packed(lab, name), logical(lab, name)
    assert before.shape == (192, 64)  # stored [out, in] after quantization
    direction = torch.randn(64, generator=torch.Generator().manual_seed(4))
    lab.project_out(name, direction, side="input")
    after = logical(lab, name)
    unit = direction / direction.norm()
    assert (after @ unit).norm() < 0.05 * after.norm() < (before @ unit).norm()
    lab.undo()
    assert same_packed(state, packed(lab, name))


@pytest.mark.parametrize("method", FOUR_BIT)
def test_swap_from_float_model(checkpoint, load, method):
    path = checkpoint("llama")
    floating = load(path)
    lab = load(path, quantize=method)
    source = floating.model.get_parameter(Q_PROJ).detach().clone() * 1.5 + 0.01
    before = packed(lab)
    lab.swap(Q_PROJ, source)
    values = logical(lab)
    step = packed(lab)[1].repeat_interleave(BLOCK).reshape(64, 64)
    # The widest gap between neighbouring codes is under 0.4 of the block scale in both codes.
    assert ((values - source).abs() <= 0.4 * step + 1e-7).all()
    lab.undo()
    assert same_packed(before, packed(lab))
    lab.swap(Q_PROJ, floating.model, index=(slice(0, 4), slice(None)))
    assert (logical(lab)[:4] - floating.model.get_parameter(Q_PROJ)[:4]).abs().max() < 0.2
    lab.undo()
    with pytest.raises(ValueError, match="does not match"):
        lab.swap(Q_PROJ, torch.zeros(64, 32))
    assert same_packed(before, packed(lab))


def test_swap_between_quantized_models(checkpoint, load):
    path = checkpoint("llama")
    source, lab = load(path, quantize="nf4"), load(path, quantize="nf4")
    source.edit(Q_PROJ, lambda value: value * 2)
    before = packed(lab)
    lab.swap(Q_PROJ, source.model, source_name=Q_PROJ)
    assert torch.equal(packed(lab)[0], packed(source)[0])
    lab.undo()
    assert same_packed(before, packed(lab))


def test_edit_converts_to_the_logical_dtype(checkpoint, load):
    import bitsandbytes.functional as functional

    lab = load(checkpoint("llama"), quantize="nf4", dtype="bfloat16")
    before = logical(lab)
    value = torch.linspace(-0.3, 0.3, 64, dtype=torch.float32) + 1e-4
    lab.edit(Q_PROJ, value, index=(0, slice(None)))
    expected = before.to(torch.bfloat16)
    expected[0] = value.to(torch.bfloat16)
    state = lab.model.get_parameter(Q_PROJ).quant_state
    assert state.dtype == torch.bfloat16
    data, new_state = functional.quantize_4bit(expected, blocksize=BLOCK, quant_type="nf4")
    assert torch.equal(packed(lab)[0].flatten(), data.flatten())
    assert torch.equal(packed(lab)[1], new_state.absmax)
    assert lab.edits()[0]["dtype"] == "torch.bfloat16"


@pytest.mark.parametrize("method", FOUR_BIT)
def test_reads_see_the_edited_values(checkpoint, load, method):
    path = checkpoint("llama")
    floating = load(path)
    lab = load(path, quantize=method)
    before = lab.measure(Q_PROJ)
    gap_before = {e["name"]: e["l2"] for e in lab.compare(floating.model, Q_PROJ)}
    lab.zero(Q_PROJ, index=(slice(0, 8), slice(None)))
    after = lab.measure(Q_PROJ)
    assert after["l2"] < before["l2"]
    assert after["l2"] == pytest.approx(logical(lab).double().norm().item(), rel=1e-9)
    gap = {e["name"]: e["l2"] for e in lab.compare(floating.model, Q_PROJ)}
    assert gap[Q_PROJ] != gap_before[Q_PROJ]
    weights = lab.head_weights(0, 0)
    assert torch.equal(weights["q"], logical(lab)[:16])
    assert (weights["q"][:8] == 0).all() and (weights["q"][8:] != 0).any()
    lab.undo()
    assert lab.measure(Q_PROJ) == before


@pytest.mark.parametrize("method", FOUR_BIT)
def test_head_ablation_on_quantized_model(checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    baseline, state = logits(lab), packed(lab, O_PROJ)
    head = lab.heads(layers=[0])[1]
    ids = []
    for part in head["o"]:
        assert part["name"] == O_PROJ
        ids.append(lab.zero(**part))
    ablated = logits(lab)
    assert not torch.equal(ablated, baseline) and torch.isfinite(ablated).all()
    columns = head["o"][0]["index"]
    assert (logical(lab, O_PROJ)[columns] == 0).all()
    for _ in ids:
        lab.undo()
    assert same_packed(state, packed(lab, O_PROJ))
    assert torch.equal(logits(lab), baseline)


@pytest.mark.parametrize("method", FOUR_BIT)
def test_save_edits_and_export_refuse_with_quantized_edits(checkpoint, load, tmp_path, method):
    lab = load(checkpoint("llama"), quantize=method)
    lab.zero(Q_PROJ, index=(0, slice(0, 8)))
    with pytest.raises(ValueError, match="quantized"):
        lab.save_edits(tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()
    with pytest.raises(ValueError, match="quantized"):
        lab.export(tmp_path / "exported")
    assert not (tmp_path / "exported").exists()
    lab.undo()
    lab.zero("model.norm.weight")
    lab.save_edits(tmp_path / "float-bundle")


def test_codec_refuses_unsupported_weights(checkpoint, load):
    lab = load(checkpoint("llama"), quantize="int8")
    weight = lookup(lab.model, Q_PROJ)
    with pytest.raises(ValueError, match="int8"):
        weight.snapshot()


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_tensor_index_on_any_device_works(checkpoint, load, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no mps device")
    lab = load(checkpoint("llama"), quantize="nf4")
    state = packed(lab)
    rows = torch.tensor([1, 5], device=device)
    mask = torch.zeros(64, dtype=torch.bool, device=device)
    mask[:8] = True
    lab.zero(Q_PROJ, index=(rows, slice(None)))
    lab.zero(Q_PROJ, index=(slice(10, 12), mask))
    values = logical(lab)
    assert (values[[1, 5]] == 0).all() and (values[10:12, :8] == 0).all()
    assert lab.edits()[0]["quantization"]["blocks_requantized"] == 2
    lab.undo()
    lab.undo()
    assert same_packed(state, packed(lab))


@pytest.mark.parametrize("method", FOUR_BIT)
def test_vanishing_edit_raised_as_an_error_changes_nothing(checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    before, revision = packed(lab), lab.revision
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="quantization step"):
            lab.edit(Q_PROJ, lambda value: value + 1e-5, index=(1, 1))
    assert lab.edits() == [] and lab.active_edit_ids == [] and lab.revision == revision
    assert same_packed(before, packed(lab))


@pytest.mark.parametrize("method", FOUR_BIT)
def test_edit_of_tail_elements_keeps_the_rest_of_the_tail(partial_tail, method):
    import bitsandbytes.functional as functional

    model = partial_tail(method)
    weight = model[0].weight
    book = EditBook(model)
    before = (weight.data.clone(), weight.quant_state.absmax.clone())
    values = lookup(model, "0.weight").dequantize()
    scale = before[1][1].item()  # The tail is block 1, elements 64..110 (row 1 col 27 on).
    cells = [(2, 28), (2, 29), (2, 30)]  # flat 102..104, in the tail
    intended = values.clone()
    for row, column in cells:
        assert values[row, column].abs() < scale
        intended[row, column] = -0.6 * scale if values[row, column] > 0.3 * scale else 0.6 * scale
    book.edit("0.weight", intended[2, 28:31].clone(), index=(2, slice(28, 31)))
    after = (weight.data.clone(), weight.quant_state.absmax.clone())
    assert torch.equal(after[1], before[1])
    moved_bytes = (before[0] != after[0]).flatten().nonzero().flatten().tolist()
    assert moved_bytes and set(moved_bytes) <= {51, 52}  # elements 102..104 live in bytes 51, 52
    keep = [i for i in range(before[0].numel()) if i not in (51, 52)]
    assert torch.equal(before[0].flatten()[keep], after[0].flatten()[keep])
    landed = lookup(model, "0.weight").dequantize()
    assert (landed.reshape(-1)[:102] == values.reshape(-1)[:102]).all()
    state = weight.quant_state
    data, whole = functional.quantize_4bit(intended, blocksize=BLOCK, quant_type=method)
    if method == "nf4":
        assert torch.equal(after[0].flatten(), data.flatten())
    expected = functional.dequantize_4bit(data, whole).reshape(3, 37).float()
    assert torch.equal(landed, expected) and state.blocksize == BLOCK
    book.undo()
    assert torch.equal(weight.data, before[0]) and torch.equal(weight.quant_state.absmax, before[1])


@pytest.mark.parametrize("method", FOUR_BIT)
def test_undo_refuses_changed_absmax_layout(checkpoint, load, method):
    lab = load(checkpoint("llama"), quantize=method)
    lab.zero(Q_PROJ, index=(0, slice(0, 8)))
    state = lab.model.get_parameter(Q_PROJ).quant_state
    current = packed(lab)
    original = state.absmax
    for replaced in (original.to(torch.float16), original[:-1].clone()):
        state.absmax = replaced
        with pytest.raises(RuntimeError, match="changed since the edit"):
            lab.undo()
        assert lab.active_edit_ids == ["edit-000001"]
        assert torch.equal(lab.model.get_parameter(Q_PROJ).data, current[0])
    state.absmax = original
    lab.undo()


def test_warning_points_at_the_caller(checkpoint, load):
    lab = load(checkpoint("llama"), quantize="nf4")
    source = logical(lab) + 1e-6
    calls = (
        lambda: lab.edit(Q_PROJ, lambda value: value + 1e-5, index=(1, 1)),
        lambda: lab.swap(Q_PROJ, source, index=(1, slice(0, 1))),
        lambda: lab.project_out(Q_PROJ, torch.eye(64)[0] + 1e-9, index=[1]),
    )
    for call in calls:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        lab.undo()
        ours = [w for w in caught if "quantization step" in str(w.message)]
        assert [w.filename for w in ours] == [__file__]


@pytest.mark.parametrize("method", FOUR_BIT)
def test_edit_and_undo_on_mps(checkpoint, load, method):
    if not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS hardware")
    if tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) < (2, 5):
        pytest.skip("quantized models on MPS need torch 2.5 or later")
    lab = load(checkpoint("llama"), quantize=method, device="mps")
    before = packed(lab)
    lab.edit(Q_PROJ, lambda block: block * 0.5, index=(slice(0, 2), slice(None)))
    assert not same_packed(before, packed(lab))
    lab.undo()
    assert same_packed(before, packed(lab))
