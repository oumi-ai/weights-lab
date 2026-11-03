import json
import math
from types import SimpleNamespace

import pytest

from weights_lab.budget import BudgetExceeded
from weights_lab.lab import Lab
from weights_lab.scoring import Scorer, ScoringMixin

torch = pytest.importorskip("torch")


class BoundaryTokenizer:
    bos_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"p": [0, 1], " x": [2, 3], " y": [4], " z": [5], "p x": [5], "": []}[text]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(value)) for value in ids)


class ControlledModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.zeros(6, 6))
        self.table[0, 1] = 3
        self.table[1, 2] = 4
        self.table[2, 3] = 4
        self.calls = []
        self.fail = False
        self.dropout = torch.nn.Dropout(0.1)

    @property
    def device(self):
        return self.table.device

    def forward(self, input_ids, attention_mask=None):
        self.calls.append((input_ids.tolist(), self.training, torch.is_grad_enabled()))
        if self.fail:
            raise RuntimeError("intentional forward failure")
        return SimpleNamespace(logits=self.table[input_ids])

    def generate(self, input_ids, attention_mask=None, max_new_tokens=2, do_sample=False, **kwargs):
        ids = input_ids
        for _ in range(max_new_tokens):
            output = self(input_ids=ids)
            if do_sample:
                token = torch.multinomial(output.logits[:, -1].softmax(-1), num_samples=1)
            else:
                token = output.logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, token], dim=-1)
        return ids


class ScoringLab(ScoringMixin):
    def __init__(self, path, model=None):
        self.session_path = path
        self.artifacts = path / "artifacts"
        self.model = model if model is not None else ControlledModel()
        self.tokenizer = BoundaryTokenizer()
        self.model_id = "controlled"
        self.revision = 0
        self.active_edit_ids = []
        self._forward_passes = 0
        self.model.register_forward_pre_hook(self._count)

    def _count(self, model, inputs):
        self._before_forward()
        self._forward_passes += 1

    @property
    def forward_passes(self):
        return self._forward_passes


@pytest.fixture
def lab(tmp_path):
    return ScoringLab(tmp_path)


def tasks(answer="A"):
    return [{"prompt": "p", "options": [" x", " y"], "answer": answer}]


def test_likelihood_scores_only_continuation_tokens_and_explicit_boundary(lab):
    scores = lab.score("p", [" x", " y"])
    expected_x = 2 * (4 - math.log(math.exp(4) + 5))
    expected_y = -math.log(math.exp(4) + 5)
    assert scores == pytest.approx({"A": expected_x, "B": expected_y}, abs=1e-6)
    assert lab.model.calls[0][0] == [[0, 1, 2, 3]]
    assert lab.model.calls[1][0] == [[0, 1, 4]]
    assert all(not training and not grad for _, training, grad in lab.model.calls)
    assert lab.model.training
    assert lab.forward_passes == lab.budget.used == 2


def test_likelihood_matches_real_huggingface_teacher_forced_loss(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(vocab_size=6, n_positions=16, n_embd=8, n_layer=1, n_head=1)
    )
    lab = Lab(model, BoundaryTokenizer(), tmp_path)
    score = lab.score("p", {"answer": " x"})["answer"]
    model.eval()
    with torch.no_grad():
        output = model(
            input_ids=torch.tensor([[0, 1, 2, 3]]),
            labels=torch.tensor([[-100, -100, 2, 3]]),
        )
    assert score == pytest.approx(-2 * float(output.loss), abs=1e-6)


def test_scoring_restores_each_modules_training_flag(lab):
    lab.model.train()
    lab.model.dropout.eval()
    lab.score("p", [" x"])
    assert lab.model.training
    assert not lab.model.dropout.training


def test_empty_prompt_uses_bos_and_missing_special_token_is_rejected(lab):
    lab.score("", {"answer": " x"})
    assert lab.model.calls[0][0] == [[0, 2, 3]]
    lab.tokenizer.bos_token_id = lab.tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="BOS or EOS"):
        lab.score("", {"answer": " x"})


def test_budget_enforces_direct_forward_and_generation_before_execution(lab):
    lab.budget.limit = 2
    ids = torch.tensor([[0, 1]])
    lab.model(input_ids=ids)
    with pytest.raises(BudgetExceeded):
        lab.model.generate(input_ids=ids, max_new_tokens=2)
    assert lab.forward_passes == lab.budget.used == len(lab.model.calls) == 2
    with pytest.raises(BudgetExceeded):
        lab.model(input_ids=ids)
    assert len(lab.model.calls) == 2


def test_local_limit_does_not_spend_session_budget_on_rejected_call(lab):
    ids = torch.tensor([[0]])
    with lab.limit_passes(1) as local:
        lab.model(input_ids=ids)
        with pytest.raises(BudgetExceeded):
            lab.model(input_ids=ids)
    assert local.used == lab.budget.used == 1
    lab.model(input_ids=ids)
    assert lab.budget.used == 2


def test_failed_forward_still_spends_one_accepted_pass(lab):
    lab.model.fail = True
    with pytest.raises(RuntimeError, match="intentional"):
        lab.score("p", [" x"])
    assert lab.forward_passes == lab.budget.used == 1
    assert lab.model.training


def test_evaluation_writes_per_item_and_provenance_and_never_overwrites(lab):
    lab.revision = 7
    lab.active_edit_ids = ["edit-1"]
    summary = lab.evaluate(tasks(), name="baseline")
    assert summary["accuracy"] == 1
    assert summary["forward_passes"] == 2
    assert summary["model_revision"] == 7
    assert summary["active_edit_ids"] == ["edit-1"]
    saved = json.loads((lab.artifacts / "evaluations/baseline/items.jsonl").read_text())
    assert saved["prompt"] == "p"
    assert saved["prediction"] == "A"
    assert saved["scores"]["A"] > saved["scores"]["B"]
    assert saved == json.loads(
        (lab.artifacts / "evaluations/baseline/items/000000.json").read_text()
    )
    with pytest.raises(FileExistsError):
        lab.evaluate(tasks(), name="baseline")
    with pytest.raises(ValueError, match="safe identifier"):
        lab.evaluate(tasks(), name="../escape")


def test_jsonl_ids_and_names_do_not_bypass_held_out_guard(lab, tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps({**tasks()[0], "id": "one", "source": "elsewhere"}) + "\n")
    with pytest.raises(ValueError, match="explicit max_passes"):
        lab.evaluate(path, held_out=True)
    lab.evaluate(path, name="final", held_out=True, max_passes=2)
    lab.revision += 1
    path.write_text(json.dumps({**tasks()[0], "id": "two"}) + "\n")
    reloaded = ScoringLab(lab.session_path)
    for kwargs in ({"held_out": True, "max_passes": 2}, {"held_out": False}):
        with pytest.raises(ValueError, match="reserved"):
            reloaded.evaluate(path, name="renamed", **kwargs)
    assert reloaded.forward_passes == 0


def test_dev_data_cannot_be_reclassified_as_held_out(lab):
    lab.evaluate(tasks(), name="dev")
    with pytest.raises(ValueError, match="reserved"):
        lab.evaluate(tasks(), held_out=True, max_passes=2)


def test_budget_failure_is_auditable_and_consumes_held_out_reservation(lab):
    with pytest.raises(BudgetExceeded):
        lab.evaluate(tasks(), name="too-small", held_out=True, max_passes=1)
    summary = json.loads((lab.artifacts / "evaluations/too-small/summary.json").read_text())
    assert summary["state"] == "failed"
    assert summary["accuracy"] is None
    assert summary["completed"] == 0
    assert summary["forward_passes"] == lab.budget.used == 1
    with pytest.raises(ValueError, match="reserved"):
        lab.evaluate(tasks(), held_out=True, max_passes=10)


def test_partial_run_failure_never_reports_partial_accuracy(lab):
    with pytest.raises(BudgetExceeded):
        lab.evaluate(tasks() + tasks(), name="partial", max_passes=2)
    summary = json.loads((lab.artifacts / "evaluations/partial/summary.json").read_text())
    assert summary["state"] == "failed"
    assert summary["correct"] == summary["completed"] == 1
    assert summary["accuracy"] is None
    assert summary["total"] == 2
    assert summary["forward_passes"] == 2
    assert len((lab.artifacts / "evaluations/partial/items.jsonl").read_text().splitlines()) == 1


def test_yes_no_and_sampled_checker_modes(lab):
    yes_no = [{"prompt": "p", "options": {"yes": " x", "no": " y"}, "answer": True}]
    assert lab.evaluate(yes_no, mode="yes_no")["accuracy"] == 1
    seen = []

    def checker(generated, expected):
        seen.append((generated, expected))
        return generated == expected

    summary = lab.evaluate(
        [{"prompt": "p", "answer": "2 3"}],
        mode="sampled",
        checker=checker,
        generation_config={"max_new_tokens": 2},
    )
    assert seen == [("2 3", "2 3")]
    assert summary["accuracy"] == 1
    assert summary["forward_passes"] == 2


def test_sampled_seed_repeats_without_changing_callers_rng(lab):
    torch.manual_seed(89)
    rng_before = torch.random.get_rng_state().clone()
    kwargs = {
        "mode": "sampled",
        "generation_config": {"max_new_tokens": 5, "do_sample": True},
        "seed": 73,
    }
    data = [{"prompt": "p", "answer": "irrelevant"}]
    first = lab.evaluate(data, name="sample-one", **kwargs)
    second = lab.evaluate(data, name="sample-two", **kwargs)
    assert first["accuracy"] == second["accuracy"]
    first_item = json.loads((lab.artifacts / "evaluations/sample-one/items.jsonl").read_text())
    second_item = json.loads((lab.artifacts / "evaluations/sample-two/items.jsonl").read_text())
    assert first_item["generated"] == second_item["generated"]
    assert torch.equal(rng_before, torch.random.get_rng_state())


def test_diff_reports_improved_regressed_and_prediction_only_changes(lab):
    data = tasks("A") + tasks("B") + [{"prompt": "p", "options": [" x", " y", " z"], "answer": "C"}]
    before = lab.evaluate(data, name="before")
    lab.model.table[1, 2] = 0
    lab.model.table[1, 4] = 8
    lab.revision += 1
    after = lab.evaluate(data, name="after")
    changes = lab.diff(before, after)
    assert [item["change"] for item in changes] == ["regressed", "improved", "changed"]
    assert lab.diff(after, after) == []
    other = lab.evaluate(tasks(), name="different-data")
    with pytest.raises(ValueError, match="same session, dataset"):
        lab.diff(before, other)


def test_diff_rejects_matching_dataset_from_different_sessions(lab, tmp_path):
    first = lab.evaluate(tasks(), name="first")
    second_lab = ScoringLab(tmp_path / "other-session")
    second = second_lab.evaluate(tasks(), name="second")
    with pytest.raises(ValueError, match="same session"):
        lab.diff(first, second)


def test_scorer_wrapper_shares_the_labs_state(lab):
    scorer = Scorer(lab)
    assert scorer.score("p", [" x"])["A"] < 0
    assert lab.forward_passes == 1
    run = scorer.evaluate(tasks(), name="wrapped")
    assert scorer.diff(run, run) == []
    assert lab.forward_passes == 3


def test_prompt_attachments_affect_scores_without_changing_dataset_identity(lab):
    baseline = lab.evaluate(tasks(), name="baseline")
    lab._prepare_prompt = lambda prompt: ""
    lab.attachments = lambda: [{"name": "empty-prompt"}]
    modified = lab.evaluate(tasks(), name="modified")
    assert baseline["dataset_hash"] == modified["dataset_hash"]
    assert modified["attachments"] == [{"name": "empty-prompt"}]
    assert lab.model.calls[-2][0] == [[0, 2, 3]]
