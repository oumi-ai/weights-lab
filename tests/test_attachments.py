import pytest
import torch

from oumi_weights_lab import BudgetExceeded


def test_logit_attachment_changes_scores_then_removes_exactly(tiny_lab):
    lab = tiny_lab
    inputs = lab.tokenizer("hello world", return_tensors="pt")
    with torch.no_grad():
        baseline = lab.model(**inputs).logits.clone()
        with lab.attach(lambda *_: torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0, 0.0]), name="yes-bias"):
            changed = lab.model(**inputs).logits
            torch.testing.assert_close(changed[..., 4], baseline[..., 4] + 10)
            scores = lab.score("hello", ["yes", "no"], batch_size=1)
            assert scores["A"] > scores["B"]
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

    class Facts:
        def __call__(self, prompt):
            return "Facts: " + prompt

    with lab.attach(Facts(), mode="prefix"):
        assert lab._prepare_prompt("hello").startswith("Facts:")
    assert lab.attachments() == []


def test_attachment_context_removes_hooks_on_errors(tiny_lab):
    count = len(tiny_lab.model._forward_hooks)
    with pytest.raises(RuntimeError), tiny_lab.attach(lambda *_: 0):
        raise RuntimeError("test")
    assert len(tiny_lab.model._forward_hooks) == count


def test_logit_attachment_updates_tuple_outputs(tiny_lab):
    lab = tiny_lab
    inputs = lab.tokenizer("hello world", return_tensors="pt")
    with torch.no_grad():
        baseline = lab.model(**inputs, return_dict=False)[0].clone()
        with lab.attach(lambda *_: 5.0):
            changed = lab.model(**inputs, return_dict=False)
    assert changed[0].shape == baseline.shape
    torch.testing.assert_close(changed[0], baseline + 5)


@pytest.mark.parametrize("return_dict", [True, False])
def test_logit_attachment_rejects_model_computed_loss(tiny_lab, return_dict):
    lab = tiny_lab
    inputs = lab.tokenizer("hello world hello", return_tensors="pt")
    with torch.no_grad(), lab.attach(lambda *_: torch.tensor([0, 0, 0, 0, 50.0, 0])):
        with pytest.raises(ValueError, match="labels"):
            lab.model(**inputs, labels=inputs["input_ids"], return_dict=return_dict)
        # Without labels the attached logits are available for an explicit loss.
        lab.model(**inputs, labels=None, return_dict=return_dict)


def test_prefix_attachment_runs_once_per_prompt_in_an_evaluation(tiny_lab):
    lab = tiny_lab
    calls = []

    def counted(prompt):
        calls.append(prompt)
        return "world " + prompt

    rows = [
        {"prompt": "hello", "options": [" yes", " no"], "answer": 0},
        {"prompt": "world", "options": [" yes", " no"], "answer": 1},
    ]
    with lab.attach(counted, mode="prefix"):
        lab.evaluate(rows, name="prefixed")
        # The input check prepared each prompt; scoring reused it.
        assert calls == ["hello", "world"]
        lab.score("hello", [" yes", " no"])
    assert calls == ["hello", "world", "hello"]
