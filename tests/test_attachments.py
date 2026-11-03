import pytest
import torch

from weights_lab import BudgetExceeded
from weights_lab.circuits import DateFacts


def test_logit_attachment_changes_scores_then_removes_exactly(tiny_lab):
    lab = tiny_lab
    inputs = lab.tokenizer("hello world", return_tensors="pt")
    with torch.no_grad():
        baseline = lab.model(**inputs).logits.clone()
        with lab.attach(lambda *_: torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0, 0.0]), name="yes-bias"):
            changed = lab.model(**inputs).logits
            torch.testing.assert_close(changed[..., 4], baseline[..., 4] + 10)
            assert lab.score("hello", ["yes", "no"])["A"] > lab.score("hello", ["yes", "no"])["B"]
            assert lab.attachments()[0]["name"] == "yes-bias"
            with pytest.raises(ValueError, match="attachment"):
                lab.export("attached")
        restored = lab.model(**inputs).logits
    assert torch.equal(restored, baseline)
    assert lab.attachments() == []
    assert lab.revision == 2


def test_prefix_attachment_scoring_budget_and_cleanup(tiny_lab):
    lab = tiny_lab
    baseline = lab.score("hello", ["yes", "no"])
    with lab.attach(lambda prompt: "world " + prompt, mode="prefix"):
        actual = lab.score("hello", ["yes", "no"])
    assert actual == lab.score("world hello", ["yes", "no"])
    assert baseline != actual
    lab.budget.limit = lab.forward_passes
    with pytest.raises(BudgetExceeded):
        lab.model(**lab.tokenizer("hello", return_tensors="pt"))
    assert lab.budget.used == lab.forward_passes
    with lab.attach(DateFacts(), mode="prefix"):
        assert lab._prepare_prompt("2026-11-03").startswith("Calendar facts:")
    assert lab.attachments() == []


def test_attachment_context_removes_hooks_on_errors(tiny_lab):
    count = len(tiny_lab.model._forward_hooks)
    with pytest.raises(RuntimeError), tiny_lab.attach(lambda *_: 0):
        raise RuntimeError("test")
    assert len(tiny_lab.model._forward_hooks) == count
