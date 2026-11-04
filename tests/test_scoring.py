import json
import math
from types import SimpleNamespace

import pytest

from oumi_weights_lab.budget import BudgetExceeded
from oumi_weights_lab.lab import Lab
from oumi_weights_lab.scoring import ScoringMixin

torch = pytest.importorskip("torch")


class BoundaryTokenizer:
    bos_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"p": [0, 1], " x": [2, 3], " y": [4], " z": [5], "p x": [5], "": [], " ": []}[text]

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
        self.call_kwargs = []
        self.fail = False
        self.dropout = torch.nn.Dropout(0.1)

    @property
    def device(self):
        return self.table.device

    def forward(self, input_ids, attention_mask=None, position_ids=None, logits_to_keep=0):
        self.calls.append((input_ids.tolist(), self.training, torch.is_grad_enabled()))
        self.call_kwargs.append(
            {
                "attention_mask": None if attention_mask is None else attention_mask.tolist(),
                "position_ids": None if position_ids is None else position_ids.tolist(),
                "logits_to_keep": logits_to_keep,
            }
        )
        if self.fail:
            raise RuntimeError("intentional forward failure")
        logits = self.table[input_ids]
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)

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
        self.model.register_forward_pre_hook(self._count_forward, with_kwargs=True)

    @property
    def forward_passes(self):
        return self._forward_passes


@pytest.fixture
def lab(tmp_path):
    return ScoringLab(tmp_path)


def tasks(answer="A"):
    return [{"prompt": "p", "options": [" x", " y"], "answer": answer}]


def test_batched_call_costs_one_pass_per_sequence(lab):
    ids = torch.zeros(3, 2, dtype=torch.long)
    lab.model(input_ids=ids)
    lab.model(ids)  # positional
    assert lab.forward_passes == lab.budget.used == 6


def test_batch_over_session_cap_is_refused_before_running(lab):
    lab.budget.limit = 2
    with pytest.raises(BudgetExceeded):
        lab.model(input_ids=torch.zeros(3, 2, dtype=torch.long))
    assert lab.model.calls == [] and lab.budget.used == lab.forward_passes == 0


def test_local_cap_smaller_than_batch_refuses_and_charges_nothing(lab):
    with lab.limit_passes(2) as local:
        with pytest.raises(BudgetExceeded):
            lab.model(input_ids=torch.zeros(3, 2, dtype=torch.long))
        assert local.used == lab.budget.used == 0
    assert lab.model.calls == []


def test_failed_batched_call_charges_the_whole_batch(lab):
    lab.model.fail = True
    with pytest.raises(RuntimeError, match="intentional"):
        lab.model(input_ids=torch.zeros(4, 2, dtype=torch.long))
    assert lab.forward_passes == lab.budget.used == 4


def test_generation_charges_each_step_per_sequence_and_stops_whole_steps(lab):
    prompts = torch.zeros(2, 2, dtype=torch.long)
    lab.model.generate(prompts, max_new_tokens=2)
    assert lab.forward_passes == lab.budget.used == 4  # 2 steps x 2 sequences
    lab.budget.limit = 7  # room for one more step, not two
    with pytest.raises(BudgetExceeded):
        lab.model.generate(prompts, max_new_tokens=2)
    assert lab.forward_passes == lab.budget.used == 6


def _generate_three(lab, **kwargs):
    inputs = lab.tokenizer("hello world", return_tensors="pt")
    with torch.no_grad():
        return lab.model.generate(**inputs, min_new_tokens=3, max_new_tokens=3, **kwargs)


def test_beam_search_counts_a_prompts_beams_once_per_step(tiny_lab):
    _generate_three(tiny_lab, num_beams=2, do_sample=False)
    assert tiny_lab.forward_passes == tiny_lab.budget.used == 3


def test_beam_width_from_generation_config_counts_once(tiny_lab):
    from transformers import GenerationConfig

    _generate_three(tiny_lab, generation_config=GenerationConfig(num_beams=2, do_sample=False))
    assert tiny_lab.forward_passes == tiny_lab.budget.used == 3


def test_sampled_return_sequences_each_count(tiny_lab):
    _generate_three(tiny_lab, do_sample=True, num_return_sequences=2)
    assert tiny_lab.forward_passes == tiny_lab.budget.used == 6  # 3 steps x 2 samples


def test_beam_width_resets_after_a_failed_generate(tiny_lab):
    tiny_lab.budget.limit = 2
    with pytest.raises(BudgetExceeded):
        _generate_three(tiny_lab, num_beams=2, do_sample=False)
    tiny_lab.budget.limit = None
    batch = tiny_lab.tokenizer(["hello world", "world hello"], return_tensors="pt")
    with torch.no_grad():
        tiny_lab.model(**batch)
    assert tiny_lab.forward_passes == tiny_lab.budget.used == 2 + 2


def test_sampled_evaluate_with_beams_fits_a_greedy_sized_cap(tiny_lab):
    rows = [{"prompt": "hello", "answer": "world"}]
    config = {"num_beams": 2, "min_new_tokens": 3, "max_new_tokens": 3}
    result = tiny_lab.evaluate(
        rows, name="beams", mode="sampled", max_passes=3, generation_config=config
    )
    assert result["forward_passes"] == 3


BATCH_ROWS = [
    {"prompt": "hello", "options": {"A": " yes", "B": " no"}},
    {"prompt": "hello world", "options": {"A": " yes no", "B": " no"}},
    {"prompt": "world hello world", "options": {"A": " no yes no", "B": " yes", "C": " world"}},
]


def _scores(lab, rows, batch_size):
    with lab._evaluation_mode():
        return [scores for _, scores in lab._score_rows(rows, batch_size)]


@pytest.mark.parametrize("batch_size", [2, 3, 8])
def test_batched_scores_match_unbatched(tiny_lab, batch_size):
    reference = _scores(tiny_lab, BATCH_ROWS, 1)
    batched = _scores(tiny_lab, BATCH_ROWS, batch_size)
    for expected, actual in zip(reference, batched, strict=True):
        assert actual.keys() == expected.keys()
        assert actual == pytest.approx(expected, abs=1e-5)
        assert max(actual, key=actual.get) == max(expected, key=expected.get)


def test_repeat_batched_scores_are_bitwise_identical(tiny_lab):
    assert _scores(tiny_lab, BATCH_ROWS, 8) == _scores(tiny_lab, BATCH_ROWS, 8)


def test_passes_equal_options_at_every_batch_size(tiny_lab):
    for batch_size in (1, 2, 8):
        before = tiny_lab.forward_passes
        _scores(tiny_lab, BATCH_ROWS, batch_size)
        assert tiny_lab.forward_passes - before == 7


def test_batch_size_one_is_the_v01_call(lab):
    lab.score("p", [" x", " y"], batch_size=1)
    assert lab.model.calls[0][0] == [[0, 1, 2, 3]] and lab.model.calls[1][0] == [[0, 1, 4]]
    for kwargs in lab.model.call_kwargs:
        assert kwargs["position_ids"] is None and kwargs["logits_to_keep"] == 0
        assert all(value == 1 for value in kwargs["attention_mask"][0])


def test_padded_batch_left_pads_and_keeps_only_answer_logits(lab):
    scores = lab.score("p", [" x", " y"], batch_size=8)
    assert lab.model.calls[0][0] == [[0, 1, 2, 3], [0, 0, 1, 4]]
    kwargs = lab.model.call_kwargs[0]
    assert kwargs["attention_mask"] == [[1, 1, 1, 1], [0, 1, 1, 1]]
    assert kwargs["position_ids"] == [[0, 1, 2, 3], [0, 0, 1, 2]]
    assert kwargs["logits_to_keep"] == 3
    assert scores == pytest.approx(lab.score("p", [" x", " y"], batch_size=1), abs=1e-6)


def test_rows_pack_whole_in_order_and_split_only_when_too_big(lab):
    rows = [
        {"prompt": "p", "options": {"A": " x", "B": " y"}},
        {"prompt": "p", "options": {"A": " y", "B": " z"}},
        {"prompt": "", "options": {"A": " x", "B": " y", "C": " z"}},
    ]
    _scores(lab, rows, 4)
    assert [len(ids) for ids, _, _ in lab.model.calls] == [4, 3]
    lab.model.calls.clear()
    split = _scores(lab, rows, 2)
    assert [len(ids) for ids, _, _ in lab.model.calls] == [2, 2, 2, 1]
    for actual, expected in zip(split, _scores(lab, rows, 1), strict=True):
        assert list(actual) == list(expected)
        assert actual == pytest.approx(expected, abs=1e-6)


def test_cap_smaller_than_a_batch_raises_before_running(lab):
    with pytest.raises(BudgetExceeded):
        lab.score("p", [" x", " y"], max_passes=1, batch_size=8)
    assert lab.model.calls == []


def test_likelihood_scores_only_continuation_tokens_and_explicit_boundary(lab):
    scores = lab.score("p", [" x", " y"], batch_size=1)
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
    lab.score("", {"answer": " x"}, batch_size=1)
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
    assert saved["tie"] is False
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


SAMPLED = {"mode": "sampled", "generation_config": {"max_new_tokens": 8, "do_sample": True}}


def generated(lab, name):
    return json.loads((lab.artifacts / f"evaluations/{name}/items.jsonl").read_text())["generated"]


def test_sampled_seed_fixes_output_whatever_the_callers_rng(lab):
    data = [{"prompt": "p", "answer": "irrelevant"}]
    outputs = []
    for caller_seed in (89, 90):
        torch.manual_seed(caller_seed)
        rng_before = torch.random.get_rng_state().clone()
        lab.evaluate(data, name=f"caller-{caller_seed}", seed=73, **SAMPLED)
        assert torch.equal(rng_before, torch.random.get_rng_state())
        outputs.append(generated(lab, f"caller-{caller_seed}"))
    assert outputs[0] == outputs[1]
    lab.evaluate(data, name="other-seed", seed=74, **SAMPLED)
    assert generated(lab, "other-seed") != outputs[0]


def test_failed_generation_restores_callers_rng(lab):
    generate = lab.model.generate

    def fail_after_sampling(**kwargs):
        generate(**kwargs)
        raise RuntimeError("generation failed after sampling")

    lab.model.generate = fail_after_sampling
    torch.manual_seed(89)
    rng_before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="after sampling"):
        lab.evaluate([{"prompt": "p", "answer": "irrelevant"}], seed=73, **SAMPLED)
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


def test_package_does_not_export_a_forwarding_scorer():
    import oumi_weights_lab

    assert "Scorer" not in oumi_weights_lab.__all__
    with pytest.raises(AttributeError):
        oumi_weights_lab.Scorer  # noqa: B018


def test_prompt_attachments_affect_scores_without_changing_dataset_identity(lab):
    baseline = lab.evaluate(tasks(), name="baseline")
    lab._prepare_prompt = lambda prompt: ""
    lab.attachments = lambda: [{"name": "empty-prompt"}]
    modified = lab.evaluate(tasks(), name="modified")
    assert baseline["dataset_hash"] == modified["dataset_hash"]
    assert modified["attachments"] == [{"name": "empty-prompt"}]
    assert lab.model.calls[-2][0] == [[0, 2, 3]]


def test_exact_ties_are_flagged_and_count_as_incorrect(lab):
    lab.model.table.zero_()  # Uniform logits, as after zeroing an output head.
    data = [{"prompt": "p", "options": [" y", " z"], "answer": answer} for answer in "AB"]
    assert lab.evaluate(data, name="degenerate")["correct"] == 0
    lines = (lab.artifacts / "evaluations/degenerate/items.jsonl").read_text().splitlines()
    items = [json.loads(line) for line in lines]
    assert [(item["tie"], item["prediction"], item["correct"]) for item in items] == [
        (True, None, False),
        (True, None, False),
    ]
    with pytest.raises(ValueError, match="distinct"):
        lab.evaluate([{"prompt": "p", "options": [" y", " y"], "answer": "A"}])


@pytest.mark.parametrize(
    "case,error",
    [
        ("generation_key", "generation_config"),
        ("generation_value", "generation_config"),
        ("sampling_range", "temperature must be positive"),
        ("max_length", "leaves no room to generate"),
        ("empty_option", "no continuation tokens"),
        ("token_tie", "distinct token sequences"),
        ("empty_prompt", "BOS or EOS"),
    ],
)
def test_input_errors_fail_before_reserving_held_out_data(lab, case, error):
    rows = valid = [{"prompt": "p", "options": [" x", " y"], "answer": "A"}]
    kwargs = fixed = {}
    generation = {
        "generation_key": {"max_new_tokns": 2},
        "generation_value": {"max_new_tokens": 0},
        # Accepted by GenerationConfig.validate(), refused by generate() itself.
        "sampling_range": {"do_sample": True, "temperature": 0.0},
        "max_length": {"max_new_tokens": None, "max_length": 1},
    }
    if case in generation:
        rows = valid = [{"prompt": "p", "answer": "2 3"}]
        kwargs = {"mode": "sampled", "generation_config": generation[case]}
        fixed = {"mode": "sampled", "generation_config": {"max_new_tokens": 2}}
    elif case == "empty_option":
        rows = [{"prompt": "p", "options": [" x", " "], "answer": "A"}]
    elif case == "token_tie":
        rows = [{"prompt": "p", "options": [" z", "p x"], "answer": "A"}]  # both encode to [5]
    else:
        rows = valid = [{"prompt": "", "options": [" x", " y"], "answer": "A"}]
        lab.tokenizer.bos_token_id = lab.tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match=error):
        lab.evaluate(rows, name="attempt", held_out=True, max_passes=10, **kwargs)
    assert lab.forward_passes == 0
    assert not (lab.artifacts / "evaluations/attempt").exists()
    lab.tokenizer.bos_token_id = lab.tokenizer.eos_token_id = 0
    assert (
        lab.evaluate(valid, name="retry", held_out=True, max_passes=10, **fixed)["completed"] == 1
    )


def test_generate_arguments_are_accepted_but_scorer_inputs_are_not():
    from oumi_weights_lab.scoring import _generation_kwargs

    def generate(inputs=None, generation_config=None, synced_gpus=None, **kwargs):
        pass

    assert _generation_kwargs({"synced_gpus": False}, generate)["synced_gpus"] is False
    for owned in ("input_ids", "attention_mask", "tokenizer"):
        with pytest.raises(ValueError, match="unknown keys"):
            _generation_kwargs({owned: None}, generate)


def test_stop_strings_reach_generate_with_the_tokenizer(lab):
    seen = []
    original = lab.model.generate

    def generate(*args, **kwargs):
        seen.append(kwargs.get("tokenizer"))
        return original(*args, **{k: v for k, v in kwargs.items() if k != "tokenizer"})

    lab.model.generate = generate
    config = {"max_new_tokens": 2, "stop_strings": ["3"]}
    lab.evaluate([{"prompt": "p", "answer": "2 3"}], mode="sampled", generation_config=config)
    assert seen == [lab.tokenizer]


def test_empty_prompt_held_out_rows_cannot_return_with_another_answer(lab):
    row = {"prompt": "", "options": [" x", " y"], "answer": "A"}
    lab.evaluate([row], name="heldout", held_out=True, max_passes=10)
    # Scores do not depend on which option is gold, so this would replay held-out scores.
    with pytest.raises(ValueError, match="overlaps reserved"):
        lab.evaluate([{**row, "answer": "B"}], name="flipped")
