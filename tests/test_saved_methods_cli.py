"""``proving-ground grade --method`` and ``oumi-weights-lab methods``: record, list, promote."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from oumi_weights_lab.cli import main  # noqa: E402
from oumi_weights_lab.methods_store import SavedMethodsMixin  # noqa: E402
from oumi_weights_lab.proving import record_method_result  # noqa: E402


class FakeLab(SavedMethodsMixin):
    def __init__(self, setup=None):
        self.setup_block = setup or {"model_id": "fake"}

    def _setup(self):
        return dict(self.setup_block)


def strong(lab, x):
    """Find the copy head by ablating each head alone."""
    return {"x": x}


def weak(lab, x):
    return {"x": x}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("OUMI_WEIGHTS_LAB_METHODS", str(tmp_path / "methods"))
    lab = FakeLab()
    return {
        "dir": tmp_path / "methods",
        "strong": lab.save_method(strong),
        "weak": lab.save_method(weak, note="baseline"),
    }


def run(capsys, *argv):
    code = main([*map(str, argv)])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def grade_all(store, name, scores, split="dev", setup=None):
    for index, score in enumerate(scores):
        record_method_result(
            store["dir"],
            name,
            store[name]["version"],
            {
                "task": f"{split}-{index:010d}",
                "split": split,
                "score": {"score": score, "solved": score >= 0.9},
                **({"setup": setup} if setup is not None else {}),
            },
        )


def test_grade_with_a_method_records_the_result(proving_tasks, store, tmp_path, capsys):
    built = proving_tasks["copy-head"]
    key = built["key"]
    key_file = built["dir"].parent.parent / "keys" / f"{key['task']}.json"
    check = key["validation"]
    claim = {
        "version": 1,
        "task": key["task"],
        "components": key["components"],
        "ablation": {
            "accuracy_before": check["baseline_accuracy"],
            "accuracy_after": check["key_ablated_accuracy"],
            "rows": check["rows"],
        },
    }
    claim_file = tmp_path / "claim.json"
    claim_file.write_text(json.dumps(claim))
    reference = f"strong@{store['strong']['version']}"
    code, out, _ = run(
        capsys, "proving-ground", "grade", "--task", built["dir"], "--key", key_file,
        "--claim", claim_file, "--method", reference,
    )  # fmt: skip
    graded = json.loads(out)
    assert code == 0 and graded["score"] == pytest.approx(1.0)
    assert graded["saved_method"] == {"name": "strong", "version": store["strong"]["version"]}
    path = store["dir"] / "strong" / store["strong"]["version"] / "results.jsonl"
    (record,) = [json.loads(line) for line in path.read_text().splitlines()]
    assert record["task"] == key["task"] and record["split"] == "dev"
    assert record["score"]["score"] == pytest.approx(1.0) and record["time"]
    (listed,) = [m for m in json.loads(run(capsys, "methods", "list")[1]) if m["name"] == "strong"]
    assert listed["versions"][0]["results"]["tasks"] == 1


def test_grade_refuses_an_unknown_method_or_version_and_records_nothing(
    proving_tasks, store, tmp_path, capsys
):
    built = proving_tasks["copy-head"]
    key_file = built["dir"].parent.parent / "keys" / f"{built['key']['task']}.json"
    claim_file = tmp_path / "claim.json"
    claim_file.write_text("{}")
    base = ["proving-ground", "grade", "--task", built["dir"], "--key", key_file]
    base += ["--claim", claim_file]
    for method in ("nothing@" + "0" * 12, f"strong@{'0' * 12}", "strong"):
        code, out, err = run(capsys, *base, "--method", method)
        assert code == 1 and out == "" and "oumi-weights-lab:" in err
    assert (
        store["dir"] / "strong" / store["strong"]["version"] / "results.jsonl"
    ).read_text() == ""


def test_methods_list_prints_json_with_notes_and_summaries(store, capsys):
    grade_all(store, "weak", [0.5, 1.0])
    code, out, _ = run(capsys, "methods", "list")
    listing = {m["name"]: m for m in json.loads(out)}
    assert code == 0 and set(listing) == {"strong", "weak"}
    (version,) = listing["weak"]["versions"]
    assert version["note"] == "baseline"
    assert version["results"]["tasks"] == 2
    assert version["results"]["mean_score"] == pytest.approx(0.75)
    assert version["results"]["solve_rate"] == pytest.approx(0.5)
    assert version["results"]["splits"] == ["dev"]


def test_promote_writes_a_draft_with_the_table(store, tmp_path, capsys):
    grade_all(store, "strong", [0.9, 1.0, 0.8])
    output = tmp_path / "draft.md"
    code, out, _ = run(capsys, "methods", "promote", "strong", "--output", output)
    assert code == 0 and json.loads(out) == {"draft": str(output.resolve())}
    text = output.read_text()
    assert text.startswith("<!-- Draft for maintainer review")
    assert "# Strong\n\nFind the copy head by ablating each head alone." in text
    assert "def strong(lab, x):" in text and "```python" in text
    assert '"model_id": "fake"' in text
    assert "| Task | Split | Runs | Mean score | Solve rate | Setup |" in text
    assert "| dev-0000000001 | dev | 1 | 1.000 | 1.00 | - |" in text
    assert "## Compared with" not in text


def test_promote_refuses_without_results_and_never_writes_into_skills(store, tmp_path, capsys):
    code, out, err = run(capsys, "methods", "promote", "strong", "--output", tmp_path / "d.md")
    assert code == 1 and "no scored proving-ground results" in err
    assert not (tmp_path / "d.md").exists()
    grade_all(store, "strong", [1.0])
    skills = tmp_path / "skills" / "x" / "draft.md"
    code, _, err = run(capsys, "methods", "promote", "strong", "--output", skills)
    assert code == 1 and "skills" in err and not skills.exists()
    (tmp_path / "d.md").write_text("mine")
    code, _, err = run(capsys, "methods", "promote", "strong", "--output", tmp_path / "d.md")
    assert code == 1 and "--overwrite" in err and (tmp_path / "d.md").read_text() == "mine"
    code, _, err = run(
        capsys, "methods", "promote", "strong", "--output", tmp_path / "d.md", "--force"
    )
    assert code == 1 and (tmp_path / "d.md").read_text() == "mine"
    code, _, _ = run(
        capsys, "methods", "promote", "strong", "--output", tmp_path / "d.md", "--overwrite"
    )
    assert code == 0 and (tmp_path / "d.md").read_text().startswith("<!-- Draft")


def test_promote_against_needs_the_method_to_win(store, tmp_path, capsys):
    grade_all(store, "strong", [0.9, 0.95, 0.85, 0.9, 1.0, 0.9])
    grade_all(store, "weak", [0.2, 0.3, 0.1, 0.25, 0.3, 0.2])
    weak_ref = f"weak@{store['weak']['version']}"
    output = tmp_path / "won.md"
    code, _, _ = run(
        capsys, "methods", "promote", "strong", "--against", weak_ref, "--output", output
    )
    text = output.read_text()
    assert code == 0 and f"## Compared with {weak_ref}" in text
    assert "On the 6 tasks both were graded on" in text and "beats the other" in text

    lost = tmp_path / "lost.md"
    code, _, err = run(
        capsys, "methods", "promote", "weak", "--against", "strong", "--output", lost
    )
    assert code == 1 and "does not beat" in err and not lost.exists()
    assert "--force" in err
    code, _, _ = run(
        capsys, "methods", "promote", "weak", "--against", "strong", "--output", lost, "--force"
    )
    assert code == 0 and "does not beat" in lost.read_text()
    code, _, err = run(
        capsys, "methods", "promote", "weak", "--against", "strong", "--output", lost, "--force"
    )
    assert code == 1 and "--overwrite" in err


def test_promote_against_refuses_a_method_with_no_results_or_no_shared_task(
    store, tmp_path, capsys
):
    grade_all(store, "strong", [0.9, 0.9])
    code, _, err = run(
        capsys, "methods", "promote", "strong", "--against", "weak", "--output", tmp_path / "a.md"
    )
    assert code == 1 and "no scored proving-ground results" in err
    grade_all(store, "weak", [0.1], split="calibration")
    code, _, err = run(
        capsys, "methods", "promote", "strong", "--against", "weak", "--output", tmp_path / "a.md",
        "--force",
    )  # fmt: skip
    assert code == 1 and "nothing to pair" in err


def test_promote_shows_every_setup_the_method_was_saved_or_graded_in(store, tmp_path, capsys):
    FakeLab({"model_id": "second-model"}).save_method(strong)
    grade_all(store, "strong", [0.9, 1.0], setup={"model_id": "third-model"})
    output = tmp_path / "setups.md"
    code, _, _ = run(capsys, "methods", "promote", "strong", "--output", output)
    text = output.read_text()
    assert code == 0
    for model in ("fake", "second-model", "third-model"):
        assert f'"model_id": "{model}"' in text
    assert "Setup 3" in text


def test_against_compares_only_matching_setups_and_says_what_it_left_out(store, tmp_path, capsys):
    a, b = {"model_id": "a", "dtype": "float32"}, {"model_id": "a", "dtype": "bfloat16"}
    grade_all(store, "strong", [0.9, 0.95, 0.85, 0.9], setup=a)
    grade_all(store, "strong", [0.0, 0.0, 0.0, 0.0], setup=b)
    grade_all(store, "weak", [0.2, 0.3, 0.1, 0.25], setup=a)
    output = tmp_path / "setups.md"
    code, _, _ = run(
        capsys, "methods", "promote", "strong", "--against", "weak", "--output", output
    )
    text = output.read_text()
    assert code == 0 and "On the 4 tasks both were graded on" in text
    assert "Left out" in text and "dtype" in text and "4 results" in text
    assert "| 1 |" in text


def test_against_refuses_when_no_setups_match(store, tmp_path, capsys):
    grade_all(store, "strong", [0.9, 0.9], setup={"model_id": "a"})
    grade_all(store, "weak", [0.1, 0.1], setup={"model_id": "b"})
    code, _, err = run(
        capsys, "methods", "promote", "strong", "--against", "weak",
        "--output", tmp_path / "x.md", "--force",
    )  # fmt: skip
    assert code == 1 and "setup" in err and not (tmp_path / "x.md").exists()


def test_a_nan_score_is_recorded_as_null_and_left_out_of_summaries(store, capsys):
    grade_all(store, "strong", [1.0])
    record_method_result(
        store["dir"],
        "strong",
        store["strong"]["version"],
        {
            "task": "dev-0000000009",
            "split": "dev",
            "score": {"score": float("nan"), "solved": False},
        },
    )
    path = store["dir"] / "strong" / store["strong"]["version"] / "results.jsonl"
    last = json.loads(path.read_text().splitlines()[-1])
    assert last["score"]["score"] is None
    (listed,) = [m for m in json.loads(run(capsys, "methods", "list")[1]) if m["name"] == "strong"]
    summary = listed["versions"][0]["results"]
    assert summary["results"] == 2 and summary["tasks"] == 1 and summary["mean_score"] == 1.0


def test_grade_prints_the_grade_before_it_records_it(
    proving_tasks, store, tmp_path, capsys, monkeypatch
):
    import oumi_weights_lab.proving as proving

    def failing(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(proving, "record_method_result", failing)
    built = proving_tasks["copy-head"]
    key_file = built["dir"].parent.parent / "keys" / f"{built['key']['task']}.json"
    claim_file = tmp_path / "claim.json"
    claim_file.write_text("[1, 2")
    code, out, err = run(
        capsys, "proving-ground", "grade", "--task", built["dir"], "--key", key_file,
        "--claim", claim_file, "--method", f"strong@{store['strong']['version']}",
    )  # fmt: skip
    assert code == 1 and "disk full" in err
    assert json.loads(out)["score"] == 0.0
