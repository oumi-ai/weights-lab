"""Deterministic offline fixtures; these tests never download a model."""

import ast
import copy
import json
import math
import unicodedata
from pathlib import Path

import pytest

import oumi_weights_lab
from oumi_weights_lab.budget import BudgetExceeded
from oumi_weights_lab.evaluation import FAILED_NLL, mcnemar
from oumi_weights_lab.scoring import ScoringMixin


class WordTokenizer:
    bos_token_id = eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())


class OfflineLab(ScoringMixin):
    def __init__(self, root):
        self.session_path = root
        self.artifacts = root / "artifacts"
        self.tokenizer = WordTokenizer()
        self.scores = {}
        self._forward_passes = 0

    @property
    def forward_passes(self):
        return self._forward_passes

    def score(self, prompt, options, **kwargs):
        for _ in options:
            self._before_forward()
            self._forward_passes += 1
        return self.scores.get(prompt, {"A": 0.0, "B": -2.0}).copy()

    def _sample(self, prompt, generation_config, seed):
        return "A"


@pytest.fixture
def lab(tmp_path):
    return OfflineLab(tmp_path)


def rows(n=4):
    return [
        {
            "id": str(i),
            "prompt": f"p{i}",
            "answer": "A",
            "options": [" a", " b"],
            "metadata": {
                "source": "math" if i < n // 2 else "language",
                "document_id": f"doc-{i // 2}",
            },
        }
        for i in range(n)
    ]


def test_metadata_retained_but_not_a_heldout_identity_escape(lab):
    data = rows(1)
    first = lab.evaluate(data, held_out=True, max_passes=2)
    item = json.loads((Path(first["run_dir"]) / "items.jsonl").read_text())
    assert item["metadata"] == data[0]["metadata"]
    data[0]["metadata"] = {"source": "renamed", "nested": [1, True, None]}
    data[0]["id"] = "renamed"
    with pytest.raises(ValueError, match="reserved"):
        lab.evaluate(data)


def test_same_prediction_score_changes_and_candidate_set_metrics(lab):
    data = rows(1)
    before = lab.evaluate(data)
    lab.scores["p0"] = {"A": 0.0, "B": -1.0}
    after = lab.evaluate(data)
    assert lab.diff(before, after) == []
    assert lab.diff(before, after, include_scores=True)[0]["change"] == "score_changed"
    passes = lab.forward_passes
    result = lab.compare_runs(before, after, bootstrap_samples=20)
    aggregate = result["aggregate"]
    assert aggregate["delta"]["accuracy"] == 0
    assert aggregate["delta"]["nll"] > 0
    assert aggregate["delta"]["brier"] > 0
    assert aggregate["delta"]["ece"] > 0
    assert aggregate["before"]["nll"] == pytest.approx(math.log1p(math.exp(-2)))
    assert result["probability_scope"] == "candidate_set"
    assert result["top_probability_movers"][0]["gold_probability_delta"] < 0
    assert lab.forward_passes == passes


def test_raw_score_movers_without_probability_change(lab):
    data = rows(1)
    before = lab.evaluate(data)
    lab.scores["p0"] = {"A": 5.0, "B": 3.0}
    after = lab.evaluate(data)
    result = lab.compare_runs(before, after, bootstrap_samples=10)
    assert result["aggregate"]["delta"]["nll"] == 0
    assert result["top_score_movers"][0]["gold_score_delta"] == 5


@pytest.mark.parametrize("mutation", ["order", "metadata", "id", "answer"])
def test_comparison_rejects_unpaired_data_and_metadata(lab, mutation):
    data = rows(2)
    first = lab.evaluate(data)
    changed = copy.deepcopy(data)
    if mutation == "order":
        changed.reverse()
    elif mutation == "metadata":
        changed[0]["metadata"]["source"] = "decoy"
    else:
        changed[0][mutation] = "B" if mutation == "answer" else "decoy-id"
    second = lab.evaluate(changed)
    with pytest.raises(ValueError, match="identical ordered"):
        lab.compare_runs(first, second, bootstrap_samples=5)


def test_duplicate_ids_rejected_for_pairing(lab):
    data = rows(2)
    data[1]["id"] = data[0]["id"]
    run = lab.evaluate(data)
    with pytest.raises(ValueError, match="unique"):
        lab.compare_runs(run, run)


def test_grouped_losses_paired_flips_and_document_bootstrap(lab):
    data = rows()
    for row in data[2:]:
        lab.scores[row["prompt"]] = {"A": -2.0, "B": 0.0}
    before = lab.evaluate(data)
    for row in data:
        lab.scores[row["prompt"]] = (
            {"A": -2.0, "B": 0.0} if int(row["id"]) < 2 else {"A": 0.0, "B": -2.0}
        )
    after = lab.evaluate(data)
    kwargs = {"group_by": "metadata.source", "bootstrap_samples": 500, "seed": 23}
    result = lab.compare_runs(before, after, **kwargs)
    assert result == lab.compare_runs(before, after, **kwargs)
    assert result["aggregate"]["fixed"] == result["aggregate"]["broken"] == 2
    assert result["aggregate"]["mcnemar_p"] == 1
    assert result["groups"]["math"]["delta"]["accuracy"] == -1
    assert result["groups"]["math"]["delta"]["nll"] == pytest.approx(2)
    assert result["groups"]["language"]["delta"]["accuracy"] == 1
    assert result["aggregate"]["bootstrap"]["clusters"] == 2
    assert result["aggregate"]["bootstrap"]["delta_ci"]["accuracy"] == [-1, 1]


def test_missing_document_or_requested_group_never_silently_drops_out(lab):
    data = rows(2)
    del data[1]["metadata"]["document_id"]
    run = lab.evaluate(data)
    with pytest.raises(ValueError, match="missing document"):
        lab.compare_runs(run, run)
    data[0]["metadata"].pop("document_id")
    run = lab.evaluate(data)
    with pytest.raises(ValueError, match="missing group"):
        lab.compare_runs(run, run, group_by="metadata.absent")


@pytest.mark.parametrize("fixed,broken", [(0, 0), (1, 0), (10, 0), (2, 8), (15, 20)])
def test_exact_mcnemar_matches_small_binomial_tail(fixed, broken):
    n = fixed + broken
    expected = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(fixed, broken) + 1)) / 2**n)
    assert mcnemar(fixed, broken) == pytest.approx(expected)


def test_exact_mcnemar_stable_at_large_n():
    assert mcnemar(0, 0) == 1
    assert mcnemar(100_000, 100_000) == 1
    assert mcnemar(50_000, 49_999) == pytest.approx(1, abs=1e-8)
    assert 0 < mcnemar(51_000, 49_000) < 1e-6
    assert mcnemar(100_000, 0) == 0


def test_failed_items_stay_in_pair_and_suite_gate(lab):
    data = rows(2)
    baseline = lab.evaluate_suite({"task": data}, name="before")
    candidate = lab.evaluate_suite({"task": {"data": data, "max_passes": 2}}, name="after")
    assert candidate["state"] == "failed"
    report = lab.compare_runs(
        baseline["datasets"]["task"], candidate["datasets"]["task"], bootstrap_samples=20
    )
    assert report["aggregate"]["after"]["n"] == 2
    assert report["aggregate"]["after"]["failures"] == 1
    assert report["aggregate"]["after"]["accuracy"] == 0.5
    assert report["aggregate"]["after"]["nll"] >= FAILED_NLL / 2
    assert not lab.compare_suites(
        baseline, candidate, accuracy_tolerance=1, nll_tolerance=100, bootstrap_samples=10
    )["pass"]


def test_suite_group_regression_cannot_hide_in_aggregate(lab):
    data = rows()
    lab.scores.update({"p2": {"A": -2.0, "B": 0.0}, "p3": {"A": -2.0, "B": 0.0}})
    baseline = lab.evaluate_suite({"task": data}, name="before")
    lab.scores = {"p0": {"A": -2.0, "B": 0.0}, "p1": {"A": -2.0, "B": 0.0}}
    candidate = lab.evaluate_suite({"task": data}, name="after")
    assert lab.compare_suites(baseline, candidate, bootstrap_samples=10)["pass"]
    result = lab.compare_suites(
        baseline, candidate, group_by="metadata.source", bootstrap_samples=10
    )
    assert not result["pass"]
    assert result["checks"]["task"]["surfaces"]["aggregate"]
    assert not result["checks"]["task"]["surfaces"]["group:math"]
    assert lab.compare_suites(
        baseline["suite_path"],
        candidate["suite_path"],
        group_by="metadata.source",
        accuracy_tolerance={"task": 1},
        nll_tolerance={"task": 2.01},
        bootstrap_samples=10,
    )["pass"]
    assert not lab.compare_suites(
        baseline,
        candidate,
        group_by="metadata.source",
        required_groups=["missing"],
        bootstrap_samples=10,
    )["pass"]
    missing = {"datasets": {}}
    assert not lab.compare_suites(baseline, missing)["pass"]


def test_suite_records_invalid_dataset_and_continues(lab):
    suite = lab.evaluate_suite({"invalid": [], "valid": rows(1)})
    assert suite["state"] == "failed"
    assert suite["datasets"]["valid"]["state"] == "succeeded"
    assert "nonempty" in suite["datasets"]["invalid"]["error"]
    assert not lab.compare_suites(suite, suite)["pass"]


def test_protocol_and_custom_checker_requirements(lab):
    data = [{"prompt": "p", "answer": "A"}]
    first = lab.evaluate(data, mode="sampled", seed=1)
    other = lab.evaluate(data, mode="sampled", seed=2)
    with pytest.raises(ValueError, match="protocol/settings"):
        lab.compare_runs(first, other)
    custom = lab.evaluate(data, mode="sampled", checker=lambda a, b: True)
    with pytest.raises(ValueError, match="checker_id"):
        lab.compare_runs(custom, custom)
    with pytest.raises(ValueError, match="checker_id"):
        lab.diff(custom, custom)
    identified = lab.evaluate(
        data, mode="sampled", checker=lambda a, b: True, checker_id="grade-v1"
    )
    report = lab.compare_runs(identified, identified, bootstrap_samples=5)
    assert report["aggregate"]["before"]["nll"] is None
    assert report["aggregate"]["bootstrap"]["delta_ci"]["accuracy"] == [0, 0]


def test_extreme_likelihood_does_not_underflow_nll(lab):
    lab.scores["p0"] = {"A": -10000.0, "B": 0.0}
    run = lab.evaluate(rows(1))
    assert lab.compare_runs(run, run, bootstrap_samples=5)["aggregate"]["before"]["nll"] == 10000


def test_dev_screen_deterministic_stratified_and_reserves_full_source(lab):
    data = rows(20)
    one = lab.create_screen(
        data, name="one", size=8, stratify_by=["metadata.source"], min_per_cell=2, seed=7
    )
    two = lab.create_screen(
        list(reversed(data)),
        name="two",
        size=8,
        stratify_by=["metadata.source"],
        min_per_cell=2,
        seed=7,
    )
    assert lab.verify_screen(one) == lab.verify_screen(two)
    assert one["selected"] == 8
    assert all(cell["selected"] >= 2 for cell in one["cells"].values())
    assert lab.evaluate(lab.verify_screen(one))["total"] == 8
    with pytest.raises(ValueError, match="reserved"):
        lab.evaluate(data, held_out=True, max_passes=100)
    with pytest.raises(ValueError, match="min_per_cell"):
        lab.create_screen(data, name="too-small", size=1, stratify_by=["metadata.source"])


@pytest.mark.parametrize("target", ["source", "manifest", "rows", "missing"])
def test_screen_tamper_and_missing_content_refused(lab, tmp_path, target):
    source = tmp_path / "data.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows(8)))
    manifest = lab.create_screen(source, name="screen", size=4, seed=3)
    path = Path(manifest["screen_path"])
    if target == "missing":
        source.unlink()
    elif target == "source":
        source.write_text(source.read_text().replace("p0", "changed"))
    elif target == "manifest":
        content = json.loads(path.read_text())
        content["seed"] += 1
        path.write_text(json.dumps(content))
    else:
        (path.parent / "rows.json").write_text(json.dumps({"tasks": []}))
    with pytest.raises((ValueError, FileNotFoundError), match="hash|No such file"):
        lab.verify_screen(manifest)


def test_screen_never_selects_declared_or_reserved_heldout(lab):
    data = rows(2)
    with pytest.raises(ValueError, match="dev-only"):
        lab.create_screen(data, name="explicit", held_out=True)
    declared = copy.deepcopy(data)
    declared[0]["metadata"]["split"] = "test"
    with pytest.raises(ValueError, match="dev-only"):
        lab.create_screen(declared, name="declared")
    lab.evaluate(data[:1], held_out=True, max_passes=2)
    with pytest.raises(ValueError, match="overlaps"):
        lab.create_screen(data, name="overlap")


def test_screen_rows_and_explicit_bootstrap_group(lab):
    manifest = lab.create_screen(rows(4), name="screen", size=2)
    run = lab.evaluate(lab.verify_screen(manifest))
    report = lab.compare_runs(run, run, bootstrap_group_by="metadata.source", bootstrap_samples=5)
    assert report["bootstrap_group_by"] == "metadata.source"


def test_row_reservations_survive_registry_only_resume(lab, tmp_path):
    data = rows(2)
    lab.evaluate(data[:1], held_out=True, max_passes=2)
    registry = (lab.artifacts / "evaluations/_registry.json").read_text()
    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    restored = OfflineLab(restored_root)
    eval_root = restored.artifacts / "evaluations"
    eval_root.mkdir(parents=True)
    (eval_root / "_registry.json").write_text(registry)
    assert restored.create_screen(data[1:], name="unrelated")["selected"] == 1
    with pytest.raises(ValueError, match="overlaps reserved"):
        restored.create_screen(data, name="overlap")
    with pytest.raises(ValueError, match="overlaps reserved"):
        restored.evaluate(data)
    changed = copy.deepcopy(data[:1])
    changed[0]["metadata"] = {"source": "renamed"}
    with pytest.raises(ValueError, match="overlaps reserved"):
        restored.evaluate(changed + data[1:], held_out=True, max_passes=8)


def test_screen_source_cannot_later_be_heldout_subset_after_resume(lab, tmp_path):
    data = rows(4)
    lab.create_screen(data, name="dev", size=1)
    registry = (lab.artifacts / "evaluations/_registry.json").read_text()
    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    restored = OfflineLab(restored_root)
    eval_root = restored.artifacts / "evaluations"
    eval_root.mkdir(parents=True)
    (eval_root / "_registry.json").write_text(registry)
    with pytest.raises(ValueError, match="overlaps reserved"):
        restored.evaluate(data[2:], held_out=True, max_passes=10)
    assert restored.forward_passes == 0


def test_sampled_structured_answers_do_not_require_hashable_gold(lab):
    run = lab.evaluate(
        [{"prompt": "p", "answer": {"fields": [1, 2]}}],
        mode="sampled",
        checker=lambda generated, expected: True,
        checker_id="structured-v1",
    )
    report = lab.compare_runs(run, run, bootstrap_samples=5)
    assert report["aggregate"]["delta"]["accuracy"] == 0
    assert report["top_score_movers"][0]["gold_score_delta"] is None


def test_group_values_and_document_ids_preserve_json_types(lab):
    data = rows(3)
    for row, value in zip(data, [1, "1", "@json:1"], strict=True):
        row["metadata"] = {"source": value, "document_id": value}
    run = lab.evaluate(data)
    report = lab.compare_runs(run, run, group_by="metadata.source", bootstrap_samples=5)
    assert set(report["groups"]) == {"@json:1", "1", "@str:@json:1"}
    assert report["aggregate"]["bootstrap"]["clusters"] == 3


def test_explicit_null_document_ids_are_missing_not_independent_rows(lab):
    data = rows(2)
    for row in data:
        row["metadata"]["document_id"] = None
    run = lab.evaluate(data)
    with pytest.raises(ValueError, match="missing document"):
        lab.compare_runs(run, run)


def test_running_or_interrupted_suite_never_passes_with_only_finished_surfaces(lab):
    suite = lab.evaluate_suite({"task": rows(1)})
    interrupted = copy.deepcopy(suite)
    interrupted["state"] = "running"
    assert not lab.compare_suites(suite, interrupted, bootstrap_samples=5)["pass"]
    incomplete = copy.deepcopy(suite)
    incomplete["expected_datasets"].append("not-started")
    gate = lab.compare_suites(incomplete, incomplete, bootstrap_samples=5)
    assert not gate["pass"]
    assert not gate["checks"]["not-started"]["pass"]


def test_diff_pairs_by_position_ignoring_ids_but_comparison_pairs_strictly(lab):
    data = rows(2)
    first = lab.evaluate(data)
    renamed = copy.deepcopy(data)
    for row in renamed:
        row["id"] = "renamed-" + row["id"]
    second = lab.evaluate(renamed)
    assert first["dataset_hash"] == second["dataset_hash"]
    assert first["dataset_identity"] == second["dataset_identity"]
    assert lab.diff(first, second) == []
    with pytest.raises(ValueError, match="identical ordered"):
        lab.compare_runs(first, second)


def test_diff_refuses_equally_truncated_successful_runs(lab):
    data = rows(2)
    first, second = lab.evaluate(data), lab.evaluate(data)
    for run in (first, second):
        path = Path(run["run_dir"]) / "items.jsonl"
        path.write_text(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="missing evaluation items"):
        lab.diff(first, second)


def test_sampled_identity_is_the_text_the_default_checker_compares(lab):
    runs = [
        lab.evaluate([{"prompt": "What is 2+2?", "answer": answer}], mode="sampled")
        for answer in ("4", 4, " 4\n")
    ]
    assert len({run["dataset_identity"] for run in runs}) == 1


def test_identity_normalizes_unicode_and_surrounding_whitespace(lab):
    nfc = [{"prompt": "Ångström café", "options": [" Å", " b"], "answer": "A"}]
    nfd = [
        {
            "prompt": unicodedata.normalize("NFD", nfc[0]["prompt"]) + " ",
            "options": [unicodedata.normalize("NFD", "Å"), "b "],
            "answer": "A",
        }
    ]
    assert lab.evaluate(nfc)["dataset_identity"] == lab.evaluate(nfd)["dataset_identity"]
    lab.evaluate([{**nfc[0], "prompt": "Café capital is"}], held_out=True, max_passes=2)
    renamed = unicodedata.normalize("NFD", "Café capital is")
    with pytest.raises(ValueError, match="reserved"):
        lab.evaluate([{**nfd[0], "prompt": renamed, "options": [" c", " d"]}])


@pytest.mark.parametrize("held_mode", ["likelihood", "sampled"])
@pytest.mark.parametrize("held_first", [True, False])
def test_prompt_cannot_cross_between_dev_and_held_out_in_any_mode(lab, held_mode, held_first):
    prompt = "France's capital is"
    data = {
        "likelihood": [
            {"prompt": prompt, "options": {"paris": " Paris", "rome": " Rome"}, "answer": "paris"}
        ],
        "sampled": [{"prompt": prompt, "answer": " Paris"}],
    }
    dev_mode = "sampled" if held_mode == "likelihood" else "likelihood"
    held = {"mode": held_mode, "held_out": True, "max_passes": 2}
    dev = {"mode": dev_mode}
    first, second = (held, dev) if held_first else (dev, held)
    lab.evaluate(data[first["mode"]], **first)
    with pytest.raises(ValueError, match="reserved"):
        lab.evaluate(data[second["mode"]], **second)


def test_empty_prompt_rows_reserve_only_their_full_identity(lab):
    lab.evaluate(
        [{"prompt": "", "options": [" a", " b"], "answer": "A"}], held_out=True, max_passes=2
    )
    lab.evaluate([{"prompt": "", "options": [" c", " d"], "answer": "A"}])
    with pytest.raises(ValueError, match="reserved"):
        lab.evaluate([{"prompt": "", "options": [" b", " a"], "answer": "B"}])


def test_screen_reserves_prompts_through_the_shared_registry(lab):
    data = rows(2)
    lab.create_screen(data, name="dev")
    with pytest.raises(ValueError, match="reserved"):
        lab.evaluate(
            [{"prompt": "p0", "answer": "anything"}], mode="sampled", held_out=True, max_passes=2
        )


def test_registry_record_without_identities_fails_closed(lab):
    lab.evaluate(rows(1), held_out=True, max_passes=2)
    path = lab.artifacts / "evaluations/_registry.json"
    registry = json.loads(path.read_text())
    for record in registry["datasets"].values():
        del record["item_identities"]
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="identities"):
        lab.evaluate(rows(2)[1:])


@pytest.mark.parametrize("tolerance,lost,passes", [(0.1, 1, True), (0.1, 2, False), (0.3, 3, True)])
def test_accuracy_tolerance_is_compared_exactly(lab, tolerance, lost, passes):
    data = [
        {"id": str(i), "prompt": f"p{i}", "answer": "A", "options": [" a", " b"]} for i in range(10)
    ]
    wrong = {"A": -2.0, "B": 0.0}
    lab.scores.update({"p8": wrong, "p9": wrong})
    baseline = lab.evaluate_suite({"task": data}, name="before")
    lab.scores.update({f"p{i}": wrong for i in range(8 - lost, 8)})
    candidate = lab.evaluate_suite({"task": data}, name="after")
    gate = lab.compare_suites(
        baseline, candidate, accuracy_tolerance=tolerance, nll_tolerance=100, bootstrap_samples=5
    )
    assert gate["pass"] is passes


def test_unfinished_rows_are_listed_apart_from_movers(lab):
    data = rows(4)
    baseline = lab.evaluate(data)
    lab.scores["p0"] = {"A": 0.0, "B": -1.0}
    with pytest.raises(BudgetExceeded):
        lab.evaluate(data, name="truncated", max_passes=4)
    report = lab.compare_runs(baseline, "truncated", bootstrap_samples=5)
    assert [(row["index"], row["failed"]) for row in report["failed_rows"]] == [
        (2, True),
        (3, True),
    ]
    assert report["failed_rows"][0]["gold_probability_after"] is None
    for key in ("top_probability_movers", "top_score_movers"):
        assert [(row["index"], row["failed"]) for row in report[key]] == [(0, False), (1, False)]
    assert report["top_probability_movers"][0]["gold_probability_delta"] < 0


def test_answer_matching_one_label_and_another_completion_is_rejected(lab):
    with pytest.raises(ValueError, match="ambiguous"):
        lab.evaluate([{"prompt": "p", "options": ["B", "A"], "answer": "A"}])
    assert lab.evaluate([{"prompt": "p", "options": ["A", "B"], "answer": "A"}])["accuracy"] == 1


def test_non_ascii_rows_survive_evaluation_comparison_and_screens(lab):
    data = [
        {
            "id": f"ид-{i}",
            "prompt": prompt,
            "options": [" grün", " 東京"],
            "answer": "A",
            "metadata": {"tags": [tag], "document_id": f"文書-{i}"},
        }
        for i, (prompt, tag) in enumerate([("Ünïcödé façade", "café"), ("日本の首都", "日本")])
    ]
    run = lab.evaluate(data, name="unicode")
    item = json.loads((Path(run["run_dir"]) / "items.jsonl").read_text().splitlines()[0])
    assert (item["id"], item["options"]["B"]) == ("ид-0", " 東京")
    report = lab.compare_runs(run, run, group_by="metadata.tags", bootstrap_samples=5)
    assert set(report["groups"]) == {'@json:["café"]', '@json:["日本"]'}
    assert report["top_probability_movers"][0]["id"] in {"ид-0", "ид-1"}
    screen = lab.create_screen(data, name="unicode-screen", stratify_by=["metadata.tags"])
    assert set(screen["cells"]) == {'@json:[["café"]]', '@json:[["日本"]]'}
    assert sorted(row["id"] for row in lab.verify_screen(screen)) == ["ид-0", "ид-1"]


def test_evaluation_modules_never_import_scoring():
    package = Path(oumi_weights_lab.__file__).parent
    for module, forbidden in (
        ("evaluation", {"scoring"}),
        ("evaluation_data", {"scoring", "evaluation"}),
    ):
        tree = ast.parse((package / f"{module}.py").read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not imported & forbidden, module


@pytest.mark.parametrize("method", ["evaluate", "create_screen"])
def test_unknown_mode_is_rejected_before_any_reservation(lab, method):
    kwargs = {"name": "bad-mode"} if method == "create_screen" else {}
    with pytest.raises(ValueError, match="mode must be one of likelihood, yes_no, sampled"):
        getattr(lab, method)(rows(1), mode="ranking", **kwargs)
    assert not (lab.artifacts / "evaluations").exists()
