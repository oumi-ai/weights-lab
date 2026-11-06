import itertools
import time

import pytest
import torch

from oumi_weights_lab import BudgetExceeded

KINDS = ["gpt2", "llama_gqa"]
BLOCKS = {"gpt2": "transformer.h", "llama_gqa": "model.layers"}
ROWS = [
    {"prompt": "hello yes", "options": [" yes", " no"], "answer": 0},
    {"prompt": "world hello no", "options": [" yes", " no", " hello"], "answer": 1},
    {"prompt": "yes", "options": [" world", " no"], "answer": 1},
    {"prompt": "hello world yes no", "options": [" yes", " hello"], "answer": 0},
    {"prompt": "no no yes", "options": [" yes", " no"], "answer": 1},
    {"prompt": "world world", "options": [" hello", " world", " yes"], "answer": 2},
]
OPTION_COUNT = sum(len(row["options"]) for row in ROWS)


def make(kind, tiny_model, make_lab, **overrides):
    model, tokenizer = tiny_model(kind, **overrides)
    return make_lab(model, tokenizer), model


def evaluated(lab, rows):
    return lab.evaluate(rows, name=f"manual-{time.monotonic_ns()}", batch_size=8)["accuracy"]


def margin(lab, rows):
    total = 0.0
    for row in rows:
        scores = lab.score(row["prompt"], row["options"], batch_size=1)
        gold = list(scores)[row["answer"]]
        total += scores[gold] - max(v for k, v in scores.items() if k != gold)
    return total / len(rows)


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def hook_count(model):
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules()
    )


def zero_mlp(lab, kind, layer):
    return lab.patch(
        f"{BLOCKS[kind]}.{layer}.mlp", lambda value: torch.zeros_like(value), positions="all"
    )


def head_edit(lab, layer, head):
    entry = next(e for e in lab.heads() if (e["layer"], e["head"]) == (layer, head))
    return lab.zero(**entry["o"][0])


@pytest.mark.parametrize("kind", KINDS)
def test_each_part_matches_zero_evaluate_undo(kind, tiny_model, make_lab):
    lab, _ = make(kind, tiny_model, make_lab)
    result = lab.ablation_scan(ROWS)
    assert set(result) == {"baseline", "parts", "passes"}
    assert result["baseline"] == pytest.approx(evaluated(lab, ROWS))
    assert len(result["parts"]) == len(lab.heads()) + 2
    for entry in result["parts"]:
        part = entry["part"]
        if part[0] == "head":
            edit = head_edit(lab, part[1], part[2])
            expected = evaluated(lab, ROWS)
            lab.undo(edit)
        else:
            with zero_mlp(lab, kind, part[1]):
                expected = evaluated(lab, ROWS)
        assert entry["value"] == pytest.approx(expected)
        assert entry["delta"] == pytest.approx(expected - result["baseline"])
    deltas = [entry["delta"] for entry in result["parts"]]
    assert deltas == sorted(deltas)


@pytest.mark.parametrize("kind", KINDS)
def test_margin_matches_manual_computation(kind, tiny_model, make_lab):
    lab, _ = make(kind, tiny_model, make_lab)
    result = lab.ablation_scan(ROWS, [("head", 0, 1), ("mlp", 1)], metric="margin", batch_size=1)
    assert result["baseline"] == pytest.approx(margin(lab, ROWS), abs=1e-5)
    by_part = {entry["part"]: entry["value"] for entry in result["parts"]}
    edit = head_edit(lab, 0, 1)
    assert by_part[("head", 0, 1)] == pytest.approx(margin(lab, ROWS), abs=1e-5)
    lab.undo(edit)
    with zero_mlp(lab, kind, 1):
        assert by_part[("mlp", 1)] == pytest.approx(margin(lab, ROWS), abs=1e-5)


def test_callable_metric_gets_scores_of_every_row(tiny_model, make_lab):
    lab, _ = make("gpt2", tiny_model, make_lab)
    seen = []

    def metric(rows):
        seen.append(rows)
        return float(len(rows))

    result = lab.ablation_scan(ROWS, [("mlp", 0)], metric=metric)
    assert len(seen) == 2 and all(len(rows) == len(ROWS) for rows in seen)
    assert set(seen[0][1]) == {"A", "B", "C"}
    assert result["baseline"] == len(ROWS) and result["parts"][0]["delta"] == 0


@pytest.mark.parametrize("kind", KINDS)
def test_weights_unchanged_and_nothing_attached(kind, tiny_model, make_lab):
    lab, model = make(kind, tiny_model, make_lab)
    before, hooks = snapshot(model), hook_count(model)
    flags = [p.requires_grad for p in model.parameters()]
    model.train()
    state = torch.get_rng_state()
    lab.ablation_scan(ROWS)
    assert all(m.training for m in model.modules())
    assert torch.equal(torch.get_rng_state(), state)
    assert [p.requires_grad for p in model.parameters()] == flags
    after = snapshot(model)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert hook_count(model) == hooks
    assert lab.edits() == [] and lab.patches() == []


def test_nothing_remains_after_error_in_the_metric(tiny_model, make_lab):
    lab, model = make("gpt2", tiny_model, make_lab)
    before, hooks = snapshot(model), hook_count(model)
    calls = 0

    def metric(rows):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("boom")
        return 0.0

    with pytest.raises(RuntimeError, match="boom"):
        lab.ablation_scan(ROWS, metric=metric)
    assert hook_count(model) == hooks
    assert all(torch.equal(before[k], v) for k, v in snapshot(model).items())
    assert not model.training


def test_nothing_remains_after_error_mid_forward(tiny_model, make_lab):
    lab, model = make("gpt2", tiny_model, make_lab)
    hooks = hook_count(model)
    original = lab._score_batch
    count = 0

    def failing(batch, *, padded):
        nonlocal count
        count += 1
        if count == 5:
            raise RuntimeError("mid-forward")
        return original(batch, padded=padded)

    lab._score_batch = failing
    with pytest.raises(RuntimeError, match="mid-forward"):
        lab.ablation_scan(ROWS, [("head", 0, 0), ("mlp", 1)], batch_size=2)
    assert hook_count(model) == hooks


@pytest.mark.parametrize("kind", KINDS)
def test_passes_equal_rows_options_times_one_plus_parts(kind, tiny_model, make_lab):
    lab, _ = make(kind, tiny_model, make_lab)
    parts = [("head", 0, 0), ("head", 1, 1), ("mlp", 0)]
    before, used = lab.forward_passes, lab.budget.used
    result = lab.ablation_scan(ROWS, parts)
    assert result["passes"] == OPTION_COUNT * 4
    assert lab.forward_passes - before == result["passes"]
    assert lab.budget.used - used == result["passes"]
    assert lab.ablation_scan(ROWS, parts, batch_size=1)["passes"] == OPTION_COUNT * 4


def test_scan_over_the_cap_raises_before_running(tiny_model, make_lab):
    lab, _ = make("gpt2", tiny_model, make_lab)
    parts = [("head", 0, 0), ("mlp", 0)]
    with pytest.raises(BudgetExceeded):
        lab.ablation_scan(ROWS, parts, max_passes=OPTION_COUNT * 3 - 1)
    assert lab.forward_passes == 0
    lab.budget.limit = OPTION_COUNT * 2
    with pytest.raises(BudgetExceeded):
        lab.ablation_scan(ROWS, parts)
    assert lab.forward_passes == 0
    assert lab.ablation_scan(ROWS, [("mlp", 0)])["passes"] == OPTION_COUNT * 2


def test_groups_layers_and_pairs(tiny_model, make_lab):
    lab, _ = make("llama_gqa", tiny_model, make_lab)
    layers = lab.ablation_scan(ROWS, groups="layers")
    assert sorted(tuple(e["part"]) for e in layers["parts"]) == [
        (("head", 0, 0), ("head", 0, 1), ("head", 0, 2), ("head", 0, 3)),
        (("head", 1, 0), ("head", 1, 1), ("head", 1, 2), ("head", 1, 3)),
    ]
    assert layers["passes"] == OPTION_COUNT * 3
    wanted = [("head", 0, 0), ("head", 1, 2), ("mlp", 1)]
    pairs = lab.ablation_scan(ROWS, wanted, groups="pairs")
    assert sorted(tuple(e["part"]) for e in pairs["parts"]) == sorted(
        itertools.combinations(wanted, 2)
    )
    assert pairs["passes"] == OPTION_COUNT * 4
    both = lab.ablation_scan(ROWS, [[wanted[0], wanted[1]]], metric="margin")
    again = lab.ablation_scan(ROWS, wanted, groups="pairs", metric="margin")
    pair = next(e for e in again["parts"] if tuple(e["part"]) == (wanted[0], wanted[1]))
    assert pair["value"] == pytest.approx(both["parts"][0]["value"])


def test_group_zeroes_all_its_heads_together(tiny_model, make_lab):
    lab, _ = make("gpt2", tiny_model, make_lab)
    group = [("head", 0, 0), ("head", 0, 1)]
    result = lab.ablation_scan(ROWS, [group], metric="margin")
    edits = [lab.zero(**e["o"][0]) for e in lab.heads(layers=[0])]
    manual = margin(lab, ROWS)
    for edit in reversed(edits):
        lab.undo(edit)
    assert result["parts"][0]["value"] == pytest.approx(manual, abs=1e-5)
    assert result["parts"][0]["part"] == group


def test_pairs_over_the_cap_say_how_many_passes_they_need(tiny_model, make_lab):
    lab, _ = make("gpt2", tiny_model, make_lab)
    # 4 heads and 2 MLPs: 15 pairs.
    need = OPTION_COUNT * 16
    with pytest.raises(BudgetExceeded, match=f"{need} passes"):
        lab.ablation_scan(ROWS, groups="pairs", max_passes=need - 1)
    assert lab.forward_passes == 0


@pytest.mark.parametrize("kind", KINDS)
def test_batch_size_does_not_change_values(kind, tiny_model, make_lab):
    lab, _ = make(kind, tiny_model, make_lab)
    for metric in ("margin", "accuracy"):
        one = lab.ablation_scan(ROWS, metric=metric, batch_size=1)
        eight = lab.ablation_scan(ROWS, metric=metric, batch_size=8)
        assert one["baseline"] == pytest.approx(eight["baseline"], abs=1e-5)
        by_one = {e["part"]: e["value"] for e in one["parts"]}
        by_eight = {e["part"]: e["value"] for e in eight["parts"]}
        assert by_one.keys() == by_eight.keys()
        for key in by_one:
            assert by_one[key] == pytest.approx(by_eight[key], abs=1e-5)


def test_no_run_directory_is_written(tiny_model, make_lab):
    lab, _ = make("gpt2", tiny_model, make_lab)
    lab.ablation_scan(ROWS, [("mlp", 0)])
    assert not (lab.artifacts / "evaluations").exists()


def test_inputs_are_refused_with_a_reason(tiny_model, make_lab):
    lab, _ = make("gpt2", tiny_model, make_lab)
    with pytest.raises(ValueError, match="metric"):
        lab.ablation_scan(ROWS, metric="f1")
    with pytest.raises(ValueError, match="groups"):
        lab.ablation_scan(ROWS, groups="triples")
    with pytest.raises(ValueError, match="layer"):
        lab.ablation_scan(ROWS, [("head", 9, 0)])
    with pytest.raises(ValueError, match="head"):
        lab.ablation_scan(ROWS, [("head", 0, 7)])
    with pytest.raises(ValueError, match="part"):
        lab.ablation_scan(ROWS, [("neuron", 0)])
    with pytest.raises(ValueError, match="more than once"):
        lab.ablation_scan(ROWS, [("mlp", 0), ("mlp", 0)])
    with pytest.raises(ValueError, match="nonempty"):
        lab.ablation_scan([])
    with pytest.raises(ValueError, match="batch_size"):
        lab.ablation_scan(ROWS, batch_size=0)
    with pytest.raises(ValueError, match="two options"):
        lab.ablation_scan([{"prompt": "hello", "options": [" yes"], "answer": 0}], metric="margin")
    with pytest.raises(ValueError, match="layers"):
        lab.ablation_scan(ROWS, [("mlp", 0)], groups="layers")


def test_an_active_edit_is_part_of_the_baseline(tiny_model, make_lab):
    lab, _ = make("gpt2", tiny_model, make_lab)
    edit = lab.zero("transformer.h.0.mlp.c_proj.weight")
    scan = lab.ablation_scan(ROWS, [("head", 1, 0)])
    assert scan["baseline"] == pytest.approx(evaluated(lab, ROWS))
    lab.undo(edit)
