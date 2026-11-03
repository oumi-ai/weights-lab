"""Regressions across model interventions, persistence, and evaluation roles."""

import copy

import pytest
import torch

from weights_lab import Lab
from weights_lab.editing import EditBook


class Interrupted(BaseException):
    """Model an asynchronous snippet timeout rather than an ordinary error."""


class InterruptingList(list):
    def __init__(self, values=(), *, operation):
        super().__init__(values)
        self.operation = operation
        self.armed = True

    def append(self, value):
        super().append(value)
        if self.operation == "append" and self.armed:
            self.armed = False
            raise Interrupted("interrupted after append")

    def pop(self, *args):
        value = super().pop(*args)
        if self.operation == "pop" and self.armed:
            self.armed = False
            raise Interrupted("interrupted after pop")
        return value


class InterruptingRecord(dict):
    def __init__(self, record, *, key):
        super().__init__(record)
        self.key = key
        self.armed = True

    def pop(self, key, *default):
        result = super().pop(key, *default)
        if key == self.key and self.armed:
            self.armed = False
            raise Interrupted("interrupted after removing undo backup")
        return result


class InterruptingRevisionBook(EditBook):
    interrupt_revision = False

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "revision" and self.interrupt_revision:
            self.interrupt_revision = False
            raise Interrupted("interrupted after revision commit")


@pytest.mark.parametrize("boundary", ["history_append", "active_append", "revision"])
def test_interrupted_edit_restores_weights_history_and_revision(boundary):
    model = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(12).reshape(3, 4))
    original = model.weight.detach().clone()
    book = InterruptingRevisionBook(model)
    if boundary == "history_append":
        book._history = InterruptingList(operation="append")
    elif boundary == "active_append":
        book._active = InterruptingList(operation="append")
    else:
        book.interrupt_revision = True

    with pytest.raises(Interrupted):
        book.zero("weight", index=(slice(None), 1))

    assert torch.equal(model.weight, original)
    assert book.edits() == []
    assert book.active_edit_ids == []
    assert book.revision == 0
    # Rollback leaves the book usable, including the original undo snapshot.
    identifier = book.zero("weight", index=(slice(None), 1))
    book.undo(identifier)
    assert torch.equal(model.weight, original)


@pytest.mark.parametrize("boundary", ["active_pop", "revision", "original_pop", "parameter_pop"])
def test_interrupted_undo_preserves_edit_and_allows_later_exact_undo(boundary):
    model = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(12).reshape(3, 4))
    original = model.weight.detach().clone()
    book = InterruptingRevisionBook(model)
    first = book.zero("weight", index=0)
    second = book.edit("weight", lambda value: value.add_(5), index=1)
    edited = model.weight.detach().clone()
    history = copy.deepcopy(book.edits())
    revision = book.revision
    if boundary == "active_pop":
        book._active = InterruptingList(book._active, operation="pop")
    elif boundary == "revision":
        book.interrupt_revision = True
    else:
        key = "_original" if boundary == "original_pop" else "_parameter"
        record = InterruptingRecord(book._history[-1], key=key)
        book._history[-1] = record
        book._active[-1] = record

    with pytest.raises(Interrupted):
        book.undo(second)

    assert torch.equal(model.weight, edited)
    assert book.edits() == history
    assert book.active_edit_ids == [first, second]
    assert book.revision == revision
    book.undo(second)
    book.undo(first)
    assert torch.equal(model.weight, original)
    assert book.active_edit_ids == []
    assert book.revision == revision + 2


def test_lora_logit_attachment_affects_direct_forward_and_generation(tiny_lab):
    peft = pytest.importorskip("peft")
    tiny_lab._counter_hook.remove()
    model = peft.get_peft_model(
        tiny_lab.model,
        peft.LoraConfig(task_type="CAUSAL_LM", target_modules=["c_attn"], r=2, fan_in_fan_out=True),
    ).eval()
    lab = Lab(model, tiny_lab.tokenizer, tiny_lab.session_path / "lora")
    inputs = lab.tokenizer("hello world", return_tensors="pt").to(lab.device)
    calls = []

    def force_yes(model, args, kwargs, output):
        calls.append(output.logits.shape)
        delta = output.logits.new_zeros(output.logits.shape[-1])
        delta[4] = 100
        return delta

    with torch.no_grad():
        baseline = lab.model(**inputs).logits.clone()
        with lab.attach(force_yes, mode="logits", name="force-yes"):
            direct = lab.model(**inputs).logits
            assert torch.all(direct.argmax(-1) == 4)
            generated = lab.model.generate(
                **inputs, do_sample=False, min_new_tokens=2, max_new_tokens=2
            )
            assert generated[0, -2:].tolist() == [4, 4]
        restored = lab.model(**inputs).logits
    assert len(calls) == 3  # One direct call and two autoregressive calls.
    assert torch.equal(restored, baseline)
    assert lab.attachments() == []
    assert lab.budget.used == lab.forward_passes == 5


@pytest.mark.parametrize(
    "representation", ["indices", "renamed_labels", "shuffled_rows", "completion", "option_list"]
)
def test_heldout_cannot_reenter_dev_through_equivalent_representation(tiny_lab, representation):
    rows = [
        {
            "prompt": "hello",
            "options": {"positive": " yes", "negative": " no"},
            "answer": "positive",
        },
        {
            "prompt": "world",
            "options": {"positive": " yes", "negative": " no"},
            "answer": "negative",
        },
    ]
    tiny_lab.evaluate(rows, name="heldout", held_out=True, max_passes=4)
    equivalent = copy.deepcopy(rows)
    if representation == "indices":
        equivalent[0]["answer"], equivalent[1]["answer"] = 0, 1
    elif representation == "renamed_labels":
        for index, row in enumerate(equivalent):
            row["options"] = {"Y": " yes", "N": " no"}
            row["answer"] = "Y" if index == 0 else "N"
    elif representation == "shuffled_rows":
        equivalent.reverse()
    elif representation == "completion":
        equivalent[0]["answer"], equivalent[1]["answer"] = " yes", " no"
    else:
        for index, row in enumerate(equivalent):
            row["options"] = [" yes", " no"]
            row["answer"] = index
    passes = tiny_lab.forward_passes
    with pytest.raises(ValueError, match="reserved|held-out"):
        tiny_lab.evaluate(equivalent, name="dev-alias", max_passes=4)
    assert tiny_lab.forward_passes == passes
    assert not (tiny_lab.artifacts / "evaluations" / "dev-alias").exists()


def test_dev_cannot_be_reclassified_as_heldout_after_shuffle(tiny_lab):
    rows = [
        {"prompt": "hello", "options": [" yes", " no"], "answer": 0},
        {"prompt": "world", "options": [" yes", " no"], "answer": 1},
    ]
    tiny_lab.evaluate(rows, name="dev", max_passes=4)
    passes = tiny_lab.forward_passes
    with pytest.raises(ValueError, match="reserved|held-out"):
        tiny_lab.evaluate(list(reversed(rows)), name="heldout-alias", held_out=True, max_passes=4)
    assert tiny_lab.forward_passes == passes


def test_ordered_diff_rejects_shuffled_dev_runs(tiny_lab):
    rows = [
        {"prompt": "hello", "options": [" yes", " no"], "answer": 0},
        {"prompt": "world", "options": [" yes", " no"], "answer": 1},
    ]
    before = tiny_lab.evaluate(rows, name="ordered", max_passes=4)
    after = tiny_lab.evaluate(list(reversed(rows)), name="shuffled", max_passes=4)
    assert before["dataset_identity"] == after["dataset_identity"]
    assert before["dataset_hash"] != after["dataset_hash"]
    with pytest.raises(ValueError, match="same session|dataset"):
        tiny_lab.diff(before, after)


def test_active_patch_is_recorded_and_blocks_export_then_restores(tiny_lab):
    rows = [{"prompt": "hello", "options": [" yes", " no"], "answer": 0}]
    baseline = tiny_lab.evaluate(rows, name="baseline", max_passes=2)
    module = tiny_lab.model.get_submodule("transformer.h.0")
    hook_count = len(module._forward_hooks)
    with tiny_lab.patch("transformer.h.0", lambda value: value.zero_(), positions=[-1]):
        assert len(module._forward_hooks) == hook_count + 1
        active = tiny_lab.patches()
        assert len(active) == 1
        assert active[0]["module"] == "transformer.h.0"
        patched = tiny_lab.evaluate(rows, name="patched", max_passes=2)
        assert patched["patches"] == active
        assert patched["model_revision"] > baseline["model_revision"]
        with pytest.raises(ValueError, match="patch"):
            tiny_lab.export("unsafe-patch")
    assert not (tiny_lab.artifacts / "unsafe-patch").exists()
    assert len(module._forward_hooks) == hook_count
    assert tiny_lab.patches() == []
    restored = tiny_lab.evaluate(rows, name="restored", max_passes=2)
    assert restored["patches"] == []
    assert restored["model_revision"] > patched["model_revision"]
    assert tiny_lab.diff(baseline, restored) == []


def test_patch_metadata_and_hook_cleanup_survive_baseexception(tiny_lab):
    module = tiny_lab.model.get_submodule("transformer.h.0")
    hook_count = len(module._forward_hooks)
    revision = tiny_lab.revision

    def interrupted_replacement(value):
        raise Interrupted("activation experiment cancelled")

    with pytest.raises(Interrupted), tiny_lab.patch("transformer.h.0", interrupted_replacement):
        tiny_lab.model(**tiny_lab.tokenizer("hello", return_tensors="pt"))
    assert len(module._forward_hooks) == hook_count
    assert tiny_lab.patches() == []
    assert tiny_lab.revision > revision
