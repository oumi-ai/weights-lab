import json
from types import SimpleNamespace

import pytest

from weights_lab.editing import EditBook, EditingMixin

torch = pytest.importorskip("torch")


@pytest.fixture
def model():
    module = torch.nn.Linear(4, 3)
    with torch.no_grad():
        module.weight.copy_(torch.arange(12).reshape(3, 4))
    return module


def workspace(model, tmp_path, tokenizer=None):
    class Workspace(EditingMixin):
        pass

    lab = Workspace()
    lab.model = model
    lab.tokenizer = tokenizer
    lab.session_path = tmp_path
    lab.artifacts = tmp_path / "artifacts"
    return lab


def test_slice_snapshots_exact_undo_and_monotonic_revisions(model):
    model.to(dtype=torch.float16)
    with torch.no_grad():
        model.weight[0, 0] = -0.0
        model.weight[0, 1] = float("nan")
    before = model.weight.detach().clone().view(torch.int16)
    book = EditBook(model)
    first = book.zero("weight", index=(0, slice(None)))
    second = book.edit("weight", lambda value: value.add_(5), index=(slice(None), 2))
    assert book.revision == 2
    with pytest.raises(ValueError, match="LIFO"):
        book.undo(first)
    assert book.active_edit_ids == [first, second]
    book.undo()
    book.undo(first)
    assert torch.equal(model.weight.view(torch.int16), before)
    assert book.revision == 4
    assert book.active_edit_ids == []
    assert not any(edit["active"] for edit in book.edits())
    with pytest.raises(ValueError, match="no active"):
        book.undo()


def test_callable_failure_and_shape_error_leave_model_and_history_unchanged(model):
    before = model.weight.detach().clone()
    book = EditBook(model)

    def bad(value):
        value.zero_()
        raise RuntimeError("bad edit")

    with pytest.raises(RuntimeError, match="bad edit"):
        book.edit("weight", bad, index=slice(1, 3))
    with pytest.raises(ValueError, match="shape"):
        book.edit("weight", lambda value: value[:, :1])
    with pytest.raises(ValueError, match="shape"):
        book.edit("weight", 0)
    assert torch.equal(model.weight, before)
    assert book.edits() == []
    assert book.revision == 0


def test_advanced_index_is_frozen_and_restored(model):
    book = EditBook(model)
    original = model.weight.detach().clone()
    mask = torch.tensor([True, False, True])
    book.zero("weight", index=mask)
    assert torch.equal(model.weight[[0, 2]], torch.zeros(2, 4))
    mask.fill_(False)
    book.undo()
    assert torch.equal(model.weight, original)


def test_tied_parameter_aliases_share_undo_order():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.arange(4.0))
    model.b = model.a
    book = EditBook(model)
    assert [part["name"] for part in book.parts()] == ["a", "b"]
    first = book.zero("a")
    book.edit("b", lambda value: value.add_(2))
    book.undo()
    assert torch.equal(model.a, torch.zeros(4))
    book.undo(first)
    assert torch.equal(model.a, torch.arange(4.0))


def test_parameter_replacement_prevents_wrong_undo(model):
    book = EditBook(model)
    book.zero("weight")
    model.weight = torch.nn.Parameter(torch.ones_like(model.weight))
    with pytest.raises(RuntimeError, match="changed"):
        book.undo()
    assert book.active_edit_ids


def test_swap_accepts_tensor_mapping_and_model(model):
    book = EditBook(model)
    original = model.weight.detach().clone()
    book.swap("weight", torch.full_like(model.weight, 8), index=1)
    assert torch.equal(model.weight[1], torch.full((4,), 8.0))
    assert torch.equal(model.weight[0], original[0])
    book.swap("bias", {"other": torch.zeros_like(model.bias)}, source_name="other")
    book.swap("weight", torch.nn.Linear(4, 3))
    for _ in range(3):
        book.undo()
    assert torch.equal(model.weight, original)
    with pytest.raises(ValueError, match="shape"):
        book.swap("weight", torch.ones(1))


def test_chunked_stats_compare_noncontiguous_and_nonfinite(monkeypatch):
    import weights_lab.editing as editing

    monkeypatch.setattr(editing, "_CHUNK", 3)
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.arange(12.0).reshape(3, 4).t())
    book = EditBook(module)
    measured = book.measure("weight")
    assert measured["mean"] == pytest.approx(5.5)
    assert measured["std"] == pytest.approx(torch.arange(12.0).std(unbiased=False).item())
    assert measured["l2"] == pytest.approx(torch.arange(12.0).norm().item())
    comparison = book.compare({"weight": module.weight.detach() + 2})[0]
    assert comparison["mean"] == -2
    assert comparison["std"] == 0
    assert comparison["l2"] == pytest.approx(48**0.5)
    with torch.no_grad():
        module.weight[0, 0] = float("nan")
        module.weight[1, 0] = float("inf")
    assert book.measure("weight")["nonfinite_count"] == 2
    with pytest.raises(ValueError, match="Shape"):
        book.compare({"weight": torch.ones(3)})
    assert book.parts("not-present") == []
    with pytest.raises(ValueError, match="Unknown"):
        book.measure("missing")


def tiny_hf():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(
            tokenizers.models.WordLevel(
                {"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3}, unk_token="<unk>"
            )
        ),
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<eos>",
    )
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=4,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            eos_token_id=1,
            bos_token_id=1,
            pad_token_id=1,
        )
    ).eval()
    return model, tokenizer


def test_export_round_trip_preserves_edits_and_tokenizer(tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model, tokenizer = tiny_hf()
    lab = workspace(model, tmp_path, tokenizer)
    first = lab.zero("transformer.h.0.attn.c_attn.weight", index=(slice(None), slice(0, 4)))
    inputs = torch.tensor([[2, 3]])
    with torch.no_grad():
        expected = model(inputs).logits
    destination = lab.export("edited")
    assert destination == tmp_path / "artifacts" / "edited"
    reloaded = AutoModelForCausalLM.from_pretrained(destination).eval()
    with torch.no_grad():
        assert torch.equal(reloaded(inputs).logits, expected)
    assert AutoTokenizer.from_pretrained(destination).get_vocab() == tokenizer.get_vocab()
    manifest = json.loads((destination / "weights_lab_manifest.json").read_text())
    assert manifest["active_edit_ids"] == [first]
    assert manifest["revision"] == 1
    assert manifest["edits"][0]["active"]
    with pytest.raises(ValueError, match="empty"):
        lab.export(destination)
    lab.undo()
    assert lab.revision == 2


def test_export_failure_leaves_destination_absent(tmp_path, model):
    lab = workspace(model, tmp_path, SimpleNamespace(save_pretrained=lambda path: None))
    with pytest.raises(AttributeError):
        lab.export("bad")
    assert not (lab.artifacts / "bad").exists()
    assert not list(lab.artifacts.glob(".weights-lab-export-*"))


def test_lora_full_export_keeps_base_edits_without_merging_live_model(tmp_path):
    from transformers import AutoModelForCausalLM

    peft = pytest.importorskip("peft")
    model, tokenizer = tiny_hf()
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=["c_attn"],
            r=2,
            lora_alpha=4,
            fan_in_fan_out=True,
        ),
    ).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.1)
    lab = workspace(model, tmp_path, tokenizer)
    name = "base_model.model.transformer.h.0.ln_1.weight"
    lab.edit(name, lambda value: value.mul_(0.5))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    inputs = torch.tensor([[2, 3]])
    with torch.no_grad():
        expected = model(inputs).logits
    destination = lab.export("merged")
    restored = AutoModelForCausalLM.from_pretrained(destination).eval()
    with torch.no_grad():
        assert torch.allclose(restored(inputs).logits, expected, atol=1e-6)
    assert torch.equal(restored.transformer.h[0].ln_1.weight, torch.full((8,), 0.5))
    assert all(torch.equal(parameter, before[name]) for name, parameter in model.named_parameters())
    assert isinstance(model, peft.PeftModel)
    assert lab.active_edit_ids == ["edit-000001"]


def test_export_rejects_active_attachments(tmp_path, model):
    lab = workspace(model, tmp_path)
    lab.attachments = lambda: [{"name": "prefix"}]
    with pytest.raises(ValueError, match="Detach"):
        lab.export("unsafe")
    assert not (lab.artifacts / "unsafe").exists()


def test_export_accepts_existing_empty_directory(tmp_path):
    model, tokenizer = tiny_hf()
    lab = workspace(model, tmp_path, tokenizer)
    destination = tmp_path / "empty"
    destination.mkdir()
    assert lab.export(destination) == destination
    assert (destination / "config.json").exists()


def test_stats_stable_for_large_close_values(monkeypatch):
    import weights_lab.editing as editing

    monkeypatch.setattr(editing, "_CHUNK", 2)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor([1e10, 1e10 + 2], dtype=torch.float64))
    assert EditBook(model).measure("weight")["std"] == pytest.approx(1.0)
