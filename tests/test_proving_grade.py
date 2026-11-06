import copy
import json
import math

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from oumi_weights_lab.proving import (  # noqa: E402
    CLAIM_FORMAT,
    grade_bundle,
    grade_claim,
    load_claim,
    validate_claim,
)
from oumi_weights_lab.proving.planted import PLANTED_FAMILIES  # noqa: E402

FAMILIES = list(PLANTED_FAMILIES)


def _name(part):
    return f"{part['layer']}.{part['head']}"


def effect(before, after, rows=400):
    return {"accuracy_before": before, "accuracy_after": after, "rows": rows}


def claim_of(key, *, components=None, directions=None, ablation=None, direction_ablation=None):
    """A claim naming the key's parts, with ablation numbers the build measured."""
    check = key["validation"]
    claim = {
        "version": 1,
        "task": key["task"],
        "components": components
        if components is not None
        else [*key["components"], *({"kind": "neurons", **n} for n in key.get("neurons", []))],
        "ablation": ablation or effect(check["baseline_accuracy"], check["key_ablated_accuracy"]),
        "summary": "test",
        "failed_hypotheses": [],
    }
    directions = directions if directions is not None else key.get("directions", [])
    if directions:
        claim["directions"] = directions
        claim["direction_ablation"] = direction_ablation or effect(
            check["baseline_accuracy"], check.get("direction_ablated_accuracy", 0.5)
        )
    return claim


def grade(proving_tasks, family, claim, tmp_path):
    built = proving_tasks[family]
    return grade_claim(built["dir"], built["key"], claim, tmp_path / "work")


@pytest.mark.parametrize("family", FAMILIES)
def test_the_key_written_as_a_claim_scores_one_and_is_solved(family, proving_tasks, tmp_path):
    key = proving_tasks[family]["key"]
    result = grade(proving_tasks, family, claim_of(key), tmp_path)
    assert result["valid"] and result["problems"] == []
    assert result["component_precision"] == result["component_recall"] == 1.0
    assert result["component_f1"] == 1.0
    assert result["ablation_reproduced"] == 1
    assert result["faithful"] is True
    assert result["decoys_named"] == []
    assert result["score"] == 1.0 and result["solved"] is True
    measured = result["measured"]
    assert measured["rows"] == 400 and measured["baseline_accuracy"] >= 0.9
    assert measured["ablated_accuracy"] <= key["chance"] + 0.15
    if key.get("neurons"):
        assert result["neuron_precision"] == result["neuron_recall"] == 1.0
    else:
        assert "neuron_precision" not in result
    if key.get("directions"):
        assert result["direction_recall"] == pytest.approx(1.0)
        assert result["direction_precision"] == pytest.approx(1.0)
        assert result["direction_score"] == pytest.approx(1.0)
        assert result["direction_ablation_reproduced"] == 1
        assert measured["direction_ablated_accuracy"] <= 0.65
        assert result["decoy_direction_alignment"] < 0.2
    else:
        assert "direction_recall" not in result


@pytest.mark.parametrize("family", FAMILIES)
def test_the_decoy_alone_scores_low_is_not_faithful_and_is_reported(
    family, proving_tasks, tmp_path
):
    key = proving_tasks[family]["key"]
    check = key["validation"]
    decoy = key["decoys"]
    claim = claim_of(
        key,
        components=decoy,
        directions=key.get("decoy_directions", []),
        ablation=effect(check["baseline_accuracy"], check["baseline_accuracy"]),
        direction_ablation=effect(check["baseline_accuracy"], 1.0),
    )
    result = grade(proving_tasks, family, claim, tmp_path)
    assert result["component_f1"] == 0.0 and result["component_precision"] == 0.0
    assert result["decoys_named"] == decoy
    assert result["faithful"] is False
    assert result["measured"]["ablated_accuracy"] == pytest.approx(
        result["measured"]["baseline_accuracy"], abs=0.05
    )
    assert result["solved"] is False and result["identification"] == pytest.approx(0.0, abs=1e-6)
    assert result["score"] == pytest.approx(0.0, abs=1e-6)
    if key.get("directions"):
        assert result["direction_recall"] < 0.2
        assert result["decoy_direction_alignment"] == pytest.approx(1.0)


@pytest.mark.parametrize("family", ["copy-head", "feature-direction"])
def test_inflated_ablation_numbers_are_not_reproduced(family, proving_tasks, tmp_path):
    key = proving_tasks[family]["key"]
    check = key["validation"]
    for numbers in (
        effect(check["baseline_accuracy"], 0.9),  # claims the ablation did nothing
        effect(0.5, check["key_ablated_accuracy"]),  # claims a worse baseline
        effect(check["baseline_accuracy"], check["key_ablated_accuracy"] + 0.3),
    ):
        result = grade(proving_tasks, family, claim_of(key, ablation=numbers), tmp_path)
        assert result["ablation_reproduced"] == 0
        assert result["solved"] is False
        assert result["component_f1"] == 1.0
        assert result["identification"] == 1.0
        assert result["score"] == 0.5  # right parts, numbers that do not reproduce


@pytest.mark.parametrize("family", FAMILIES)
def test_a_superset_claim_has_perfect_recall_and_lower_precision(family, proving_tasks, tmp_path):
    key = proving_tasks[family]["key"]
    taken = {(c["layer"], c.get("head")) for c in [*key["components"], *key["decoys"]]}
    extras = [
        {"kind": "head", "layer": layer, "head": head}
        for layer in range(6)
        for head in range(4)
        if (layer, head) not in taken
    ][:2]
    base = claim_of(key)
    claim = {**base, "components": [*base["components"], *extras]}
    result = grade(proving_tasks, family, claim, tmp_path)
    assert result["component_recall"] == 1.0
    wanted = len({(c["kind"], c["layer"], c.get("head")) for c in key["components"]})
    assert result["component_precision"] == pytest.approx(wanted / (wanted + 2))
    assert result["component_precision"] < 1.0
    assert result["component_f1"] < 1.0 and result["solved"] is False
    # Honest numbers: the score is the identification, which here is the F1 (or its mean
    # with a perfect alignment for the family with a direction).
    assert result["ablation_reproduced"] == 1
    expected = result["component_f1"]
    if key.get("directions"):
        expected = (expected + 1.0) / 2
    assert result["score"] == pytest.approx(expected)
    assert result["identification"] == pytest.approx(expected)


def test_a_partial_claim_for_the_backup_family_has_half_recall(proving_tasks, tmp_path):
    key = proving_tasks["copy-backup"]["key"]
    one = key["components"][:1]
    result = grade(proving_tasks, "copy-backup", claim_of(key, components=one), tmp_path)
    assert result["component_precision"] == 1.0 and result["component_recall"] == 0.5
    assert result["component_f1"] == pytest.approx(2 / 3)
    assert result["faithful"] is False  # the other head still copies the flag


def test_a_claim_naming_only_the_primary_of_the_self_repair_family_is_not_faithful(
    proving_tasks, tmp_path
):
    key = proving_tasks["copy-selfrepair"]["key"]
    primary = key["components"][:1]  # the earlier head; the backup repairs its loss
    check = key["validation"]
    one = effect(check["baseline_accuracy"], check["key_head_ablated_alone"][_name(primary[0])])
    result = grade(
        proving_tasks, "copy-selfrepair", claim_of(key, components=primary, ablation=one), tmp_path
    )
    assert result["ablation_reproduced"] == 1  # the numbers are honest
    assert result["faithful"] is False and result["solved"] is False
    assert result["score"] < 0.5


def test_a_claim_naming_one_or_two_of_the_three_redundant_heads_scores_below_half(
    proving_tasks, tmp_path
):
    key = proving_tasks["copy-redundant"]["key"]
    assert len(key["components"]) == 3
    for kept in (1, 2):
        result = grade(
            proving_tasks,
            "copy-redundant",
            claim_of(key, components=key["components"][:kept]),
            tmp_path,
        )
        assert result["faithful"] is False and result["solved"] is False
        assert result["score"] < 0.5, kept


def test_neurons_count_as_their_mlp_layer_and_are_scored_apart(proving_tasks, tmp_path):
    key = proving_tasks["lookup-mlp"]["key"]
    wanted = key["neurons"][0]
    half = {"kind": "neurons", "layer": wanted["layer"], "indices": wanted["indices"][:6]}
    wrong = {"kind": "neurons", "layer": wanted["layer"], "indices": [255]}
    result = grade(proving_tasks, "lookup-mlp", claim_of(key, components=[half]), tmp_path)
    assert result["component_f1"] == 1.0  # the layer is right
    assert result["neuron_precision"] == 1.0 and result["neuron_recall"] == 0.5
    result = grade(proving_tasks, "lookup-mlp", claim_of(key, components=[wrong]), tmp_path)
    assert result["neuron_precision"] == 0.0 and result["neuron_recall"] == 0.0
    assert result["component_f1"] == 1.0


def test_direction_alignment_uses_the_best_claimed_direction_at_or_after_the_layer(
    proving_tasks, tmp_path
):
    key = proving_tasks["feature-direction"]["key"]
    target = key["directions"][0]
    layer = target["after_layer"]
    decoy = key["decoy_directions"][0]

    def run(directions):
        return grade(
            proving_tasks,
            "feature-direction",
            claim_of(key, directions=directions, direction_ablation=effect(1.0, 0.5)),
            tmp_path,
        )

    flipped = {"after_layer": layer, "vector": [-x for x in target["vector"]]}
    assert run([flipped])["direction_recall"] == pytest.approx(1.0)
    later = {"after_layer": 3, "vector": target["vector"]}
    assert run([later])["direction_recall"] == pytest.approx(1.0)
    if layer > 0:
        earlier = {"after_layer": layer - 1, "vector": target["vector"]}
        assert run([earlier])["direction_recall"] == 0.0
    mixed = run([decoy, target])
    assert mixed["direction_recall"] == pytest.approx(1.0)
    bad = run([{"after_layer": layer, "vector": decoy["vector"]}])
    assert bad["direction_recall"] < 0.2
    assert bad["solved"] is False and bad["score"] < 0.6
    assert bad["identification"] == pytest.approx(0.5 * (1 + bad["direction_score"]))
    # A direction that is half the key plus half noise aligns partially, so it is not solved.
    half = [a + b for a, b in zip(target["vector"], decoy["vector"], strict=True)]
    partial = run([{"after_layer": layer, "vector": half}])
    assert partial["direction_recall"] == pytest.approx(1 / math.sqrt(2), abs=0.01)
    assert partial["direction_precision"] == pytest.approx(1 / math.sqrt(2), abs=0.01)
    assert partial["solved"] is False


def test_the_claimed_direction_is_projected_out_where_it_was_named(proving_tasks, tmp_path):
    key = proving_tasks["feature-direction"]["key"]
    decoy = key["decoy_directions"][0]
    honest = grade(proving_tasks, "feature-direction", claim_of(key), tmp_path)
    assert honest["measured"]["direction_ablated_accuracy"] <= 0.65
    off = claim_of(
        key,
        directions=[decoy],
        direction_ablation=effect(key["validation"]["baseline_accuracy"], 0.5),
    )
    result = grade(proving_tasks, "feature-direction", off, tmp_path)
    # Removing the unread direction changes nothing, so the claimed 0.5 is not reproduced.
    assert result["measured"]["direction_ablated_accuracy"] > 0.9
    assert result["direction_ablation_reproduced"] == 0


def test_grading_leaves_the_task_unchanged_and_never_writes_into_it(proving_tasks, tmp_path):
    built = proving_tasks["copy-head"]
    before = {p: p.read_bytes() for p in built["dir"].rglob("*") if p.is_file()}
    result = grade(proving_tasks, "copy-head", claim_of(built["key"]), tmp_path)
    assert result["score"] == 1.0
    after = {p: p.read_bytes() for p in built["dir"].rglob("*") if p.is_file()}
    assert before == after
    assert (tmp_path / "work").is_dir()


# Invalid claims.


def mutated(claim, path, value):
    claim = copy.deepcopy(claim)
    node = claim
    for step in path[:-1]:
        node = node[step]
    if value is KeyError:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return claim


@pytest.mark.parametrize(
    ("path", "value", "fragment"),
    [
        (["task"], "other-task", "task must be"),
        (["version"], 2, "version"),
        (["version"], True, "version"),
        (["ablation"], KeyError, "ablation is required"),
        (["components"], KeyError, "components must be a list"),
        (["components", 0, "kind"], "layer", "unknown kind"),
        (["components", 0, "layer"], 6, "layer must be"),
        (["components", 0, "layer"], True, "layer must be"),
        (["components", 0, "layer"], -1, "layer must be"),
        (["components", 0, "head"], 4, "head must be"),
        (["components", 0, "head"], KeyError, "head must be"),
        (["ablation", "accuracy_after"], 1.5, "accuracy_after"),
        (["ablation", "accuracy_before"], "high", "accuracy_before"),
        (["ablation", "accuracy_after"], float("nan"), "accuracy_after"),
        (["ablation", "rows"], 0, "rows"),
        (["summary"], 3, "summary"),
        (["failed_hypotheses"], "none", "failed_hypotheses"),
        (["components", 0, "role"], 7, "role"),
    ],
)
def test_invalid_claims_score_zero_with_their_problems_listed(
    path, value, fragment, proving_tasks, tmp_path
):
    key = proving_tasks["copy-head"]["key"]
    claim = mutated(claim_of(key), path, value)
    problems = validate_claim(claim, proving_tasks["copy-head"]["dir"])
    assert any(fragment in problem for problem in problems), problems
    result = grade(proving_tasks, "copy-head", claim, tmp_path)
    assert result["valid"] is False and result["problems"] == problems
    assert result["score"] == 0.0 and result["solved"] is False
    assert not (tmp_path / "work").exists()  # nothing was loaded or run


def test_invalid_neurons_and_directions_are_listed(proving_tasks, tmp_path):
    key = proving_tasks["feature-direction"]["key"]
    task = proving_tasks["feature-direction"]["dir"]
    good = claim_of(key)
    vector = good["directions"][0]["vector"]
    cases = {
        "indices must be a nonempty": {"kind": "neurons", "layer": 0, "indices": []},
        "integers from 0 to 255": {"kind": "neurons", "layer": 0, "indices": [256]},
        "must be distinct": {"kind": "neurons", "layer": 0, "indices": [1, 1]},
    }
    for fragment, part in cases.items():
        problems = validate_claim({**good, "components": [part]}, task)
        assert any(fragment in problem for problem in problems), (fragment, problems)
    bad_directions = {
        "list of 64 numbers": {"after_layer": 0, "vector": vector[:-1]},
        "finite numbers": {"after_layer": 0, "vector": [float("inf")] + vector[1:]},
        "finite numbers ": {"after_layer": 0, "vector": ["x"] + vector[1:]},
        "all zeros": {"after_layer": 0, "vector": [0.0] * 64},
        "after_layer must be": {"after_layer": 9, "vector": vector},
    }
    for fragment, item in bad_directions.items():
        problems = validate_claim({**good, "directions": [item]}, task)
        assert any(fragment.strip() in problem for problem in problems), (fragment, problems)
    problems = validate_claim({**good, "directions": []}, task)
    assert any("needs at least one claimed direction" in p for p in problems)
    assert validate_claim("not a claim", task) == ["claim must be a JSON object"]
    assert validate_claim(good, task) == []


def test_load_claim_reads_only_data(tmp_path):
    path = tmp_path / "claim.json"
    path.write_text('{"version": 1, "components": []}')
    assert load_claim(path) == {"version": 1, "components": []}
    path.write_text('{"vector": [NaN]}')
    with pytest.raises(ValueError, match="NaN"):
        load_claim(path)
    path.write_text("{" + " " * 1_000_001 + "}")
    with pytest.raises(ValueError, match="larger"):
        load_claim(path)
    path.write_text("__import__('os')")
    with pytest.raises(json.JSONDecodeError):
        load_claim(path)


def test_claim_format_describes_every_field_and_its_example_is_valid_json(proving_tasks):
    for word in ("components", "directions", "ablation", "direction_ablation", "neurons"):
        assert word in CLAIM_FORMAT
    start = CLAIM_FORMAT.index("{")
    end = CLAIM_FORMAT.index("\n\nRules")
    example = CLAIM_FORMAT[start:end].replace("[64 numbers]", "[0.1]")
    parsed = json.loads(example)
    assert parsed["version"] == 1 and parsed["ablation"]["rows"] == 200


# Bundles.


def test_grade_bundle_applies_the_bundle_to_a_fresh_lab_and_runs_the_check(tmp_path):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    from oumi_weights_lab.lab import load_lab

    # A test-only family: the task is "zero the first MLP's output weight".
    task = tmp_path / "task"
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=6, n_positions=16, n_embd=8, n_layer=2, n_head=2)
    ).eval()
    model.save_pretrained(task / "model")
    import tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"<eos>": 0, "a": 1}, unk_token="a"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>").save_pretrained(
        task / "model"
    )
    (task / "task.json").write_text(json.dumps({"id": "toy-dev-0", "family": "toy"}))
    key = {"task": "toy-dev-0", "layer": 0}

    agent = load_lab(
        {"model": str(task / "model"), "device": "cpu", "dtype": "float32"}, tmp_path / "agent"
    )
    agent.zero("transformer.h.0.mlp.c_proj.weight")
    agent.save_edits(tmp_path / "bundle")

    def check(lab, key):
        weight = lab.model.transformer.h[key["layer"]].mlp.c_proj.weight
        zeroed = float(weight.detach().abs().sum()) == 0
        return {"score": 1.0 if zeroed else 0.0, "edits": len(lab.edits())}

    result = grade_bundle(task, key, tmp_path / "bundle", tmp_path / "work", check)
    assert result == {"score": 1.0, "edits": 1}

    # A fresh lab without the bundle fails the same check, so the bundle is what passed it.
    empty = tmp_path / "empty"
    load_lab(
        {"model": str(task / "model"), "device": "cpu", "dtype": "float32"}, tmp_path / "e"
    ).save_edits(empty)
    assert grade_bundle(task, key, empty, tmp_path / "work", check)["score"] == 0.0

    # A bundle that was altered does not load; the grade is 0 and says why.
    manifest = tmp_path / "bundle" / "manifest.json"
    text = json.loads(manifest.read_text())
    text["pass_limit"] = 123
    manifest.write_text(json.dumps(text))
    broken = grade_bundle(task, key, tmp_path / "bundle", tmp_path / "work", check)
    assert broken["score"] == 0.0 and broken["valid"] is False
    assert "did not load" in broken["problems"][0]

    with pytest.raises(ValueError, match="score from 0 to 1"):
        grade_bundle(task, key, empty, tmp_path / "work", lambda lab, key: {"score": 2})
    with pytest.raises(ValueError, match="score from 0 to 1"):
        grade_bundle(task, key, empty, tmp_path / "work", lambda lab, key: 1.0)


def test_one_wrong_neuron_in_the_key_layer_with_honest_numbers_is_not_solved(
    proving_tasks, tmp_path
):
    key = proving_tasks["lookup-mlp"]["key"]
    wrong = {"kind": "neurons", "layer": key["neurons"][0]["layer"], "indices": [255]}
    base = key["validation"]["baseline_accuracy"]
    claim = claim_of(key, components=[wrong], ablation=effect(base, base))
    result = grade(proving_tasks, "lookup-mlp", claim, tmp_path)
    assert result["component_f1"] == 1.0 and result["ablation_reproduced"] == 1
    assert result["faithful"] is False
    assert result["solved"] is False
    assert result["score"] == pytest.approx(0.5)


def test_the_key_direction_plus_the_decoy_direction_is_not_solved(proving_tasks, tmp_path):
    key = proving_tasks["feature-direction"]["key"]
    both = [*key["directions"], *key["decoy_directions"]]
    result = grade(
        proving_tasks,
        "feature-direction",
        claim_of(key, directions=both),
        tmp_path,
    )
    assert result["direction_recall"] == pytest.approx(1.0)
    assert result["direction_precision"] == pytest.approx(0.5, abs=0.1)
    assert result["direction_score"] < 0.9
    assert result["solved"] is False and result["score"] < 1.0


def test_a_calibration_style_key_has_no_faithful_factor(proving_tasks, tmp_path):
    import copy

    key = copy.deepcopy(proving_tasks["copy-head"]["key"])
    key["split"] = "calibration"
    built = proving_tasks["copy-head"]
    result = grade_claim(built["dir"], key, claim_of(key), tmp_path / "work")
    assert "faithful" not in result and result["score"] == pytest.approx(1.0)


def test_grading_restores_the_callers_rng_and_kernels_entry(proving_tasks, tmp_path):
    import random
    import sys

    import numpy as np
    import torch

    key = proving_tasks["copy-head"]["key"]
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    expected = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())
    had = "kernels" in sys.modules
    before = sys.modules.get("kernels")
    grade(proving_tasks, "copy-head", claim_of(key), tmp_path)
    assert random.getstate() == expected[0]
    assert (np.random.get_state()[1] == expected[1]).all()
    assert torch.equal(torch.get_rng_state(), expected[2])
    assert ("kernels" in sys.modules) == had and sys.modules.get("kernels") is before


def test_a_huge_integer_or_deeply_nested_claim_is_invalid_not_a_crash(proving_tasks, tmp_path):
    key = proving_tasks["copy-head"]["key"]
    huge = claim_of(key)
    huge["ablation"]["accuracy_after"] = 10**400
    result = grade(proving_tasks, "copy-head", huge, tmp_path)
    assert result["valid"] is False and result["score"] == 0.0 and result["problems"]
