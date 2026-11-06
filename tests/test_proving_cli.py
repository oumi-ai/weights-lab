"""``oumi-weights-lab proving-ground``: build, grade and compare, and its argument errors."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from oumi_weights_lab import cli  # noqa: E402
from oumi_weights_lab.cli import main  # noqa: E402


def call(capsys, *argv):
    code = main(["proving-ground", *map(str, argv)])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_build_makes_a_dev_task_and_its_key_and_prints_json(tmp_path, capsys):
    code, out, _ = call(
        capsys,
        "build",
        "--family",
        "copy-head",
        "--split",
        "dev",
        "--count",
        "1",
        "--start",
        "3",
        "--out",
        tmp_path / "tasks",
        "--keys",
        tmp_path / "keys",
    )
    assert code == 0
    (built,) = json.loads(out)["tasks"]
    assert built["family"] == "copy-head" and built["index"] == 3
    assert (tmp_path / "tasks" / built["task"] / "task.json").is_file()
    assert (tmp_path / "keys" / f"{built['task']}.json").is_file()


def test_build_heldout_uses_the_secret_and_does_not_write_it(tmp_path, capsys):
    secret = "ab" * 32
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text(secret)
    code, out, err = call(
        capsys,
        "build",
        "--family",
        "copy-head",
        "--split",
        "heldout",
        "--count",
        "1",
        "--out",
        tmp_path / "tasks",
        "--keys",
        tmp_path / "keys",
        "--secret-file",
        secret_file,
    )
    assert code == 0
    assert secret not in out + err
    for path in (tmp_path / "tasks").rglob("*"):
        if path.is_file() and path.suffix != ".safetensors":
            assert secret.encode() not in path.read_bytes()
    for path in (tmp_path / "keys").iterdir():
        assert secret.encode() not in path.read_bytes()


def test_a_heldout_build_without_a_secret_is_refused_and_builds_nothing(tmp_path, capsys):
    code, out, err = call(
        capsys,
        "build",
        "--family",
        "copy-head",
        "--split",
        "heldout",
        "--count",
        "2",
        "--out",
        tmp_path / "tasks",
        "--keys",
        tmp_path / "keys",
    )
    assert code == 1 and out == "" and "--secret-file" in err
    assert not (tmp_path / "tasks").exists()


def test_a_secret_for_a_dev_build_is_refused(tmp_path, capsys):
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("cd" * 32)
    code, out, err = call(
        capsys,
        "build",
        "--family",
        "copy-head",
        "--split",
        "dev",
        "--count",
        "1",
        "--out",
        tmp_path / "tasks",
        "--keys",
        tmp_path / "keys",
        "--secret-file",
        secret_file,
    )
    assert code == 1 and out == "" and "public" in err
    assert not (tmp_path / "tasks").exists()


def test_a_short_secret_and_a_wrong_family_for_the_split_are_refused(tmp_path, capsys):
    short = tmp_path / "short.txt"
    short.write_text("abc")
    common = ["--count", "1", "--out", tmp_path / "tasks", "--keys", tmp_path / "keys"]
    code, _, err = call(
        capsys, "build", "--family", "copy-head", "--split", "heldout", *common,
        "--secret-file", short,
    )  # fmt: skip
    assert code == 1 and "32 bytes" in err
    code, _, err = call(capsys, "build", "--family", "gpt2-ioi", "--split", "dev", *common)
    assert code == 1 and "calibration" in err
    code, _, err = call(capsys, "build", "--family", "nope", "--split", "dev", *common)
    assert code == 1 and "unknown family" in err
    assert not (tmp_path / "tasks").exists()


def test_build_requires_a_family_and_a_positive_count(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(
            [
                "proving-ground",
                "build",
                "--split",
                "dev",
                "--count",
                "1",
                "--out",
                "a",
                "--keys",
                "b",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            ["proving-ground", "build", "--family", "copy-head", "--split", "dev", "--count", "0",
             "--out", "a", "--keys", "b"]
        )  # fmt: skip
    capsys.readouterr()


def test_grade_scores_an_honest_claim_one_and_a_missing_claim_zero(proving_tasks, tmp_path, capsys):
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
    code, out, _ = call(
        capsys, "grade", "--task", built["dir"], "--key", key_file, "--claim", claim_file
    )
    result = json.loads(out)
    assert code == 0 and result["valid"] and result["score"] == pytest.approx(1.0)
    assert result["solved"] is True

    code, out, _ = call(
        capsys, "grade", "--task", built["dir"], "--key", key_file, "--claim", tmp_path / "none"
    )
    result = json.loads(out)
    assert code == 0 and result["score"] == 0.0 and not result["valid"]
    assert "could not be read" in result["problems"][0]

    claim_file.write_text("[1, 2")
    code, out, _ = call(
        capsys, "grade", "--task", built["dir"], "--key", key_file, "--claim", claim_file
    )
    assert json.loads(out)["score"] == 0.0


def record(task, arm, repeat, score, split="dev"):
    return {
        "task": task,
        "split": split,
        "arm": arm,
        "repeat": repeat,
        "score": {"score": score, "solved": score == 1.0},
    }


def write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_compare_pairs_two_arms_of_one_file(tmp_path, capsys):
    tasks = [f"dev-{i:010d}" for i in range(8)]
    records = [record(t, "control", n, 0.2) for t in tasks for n in range(3)]
    records += [record(t, "skill", n, 0.8) for t in tasks for n in range(3)]
    results = tmp_path / "results.jsonl"
    write(results, records)
    code, out, _ = call(capsys, "compare", f"{results}:control", f"{results}:skill")
    report = json.loads(out)
    assert code == 0 and report["tasks"] == 8
    assert report["left_mean"] == pytest.approx(0.2) and report["right_mean"] == pytest.approx(0.8)
    assert report["delta"] == pytest.approx(0.6) and report["beats"] is True
    assert report["unpaired_left"] == report["unpaired_right"] == []


def test_compare_one_arm_across_two_files_with_a_split_filter(tmp_path, capsys):
    first, latest = tmp_path / "first.jsonl", tmp_path / "latest.jsonl"
    tasks = [f"dev-{i:010d}" for i in range(6)]
    write(first, [record(t, "skill", 0, 0.4) for t in tasks])
    write(
        latest,
        [record(t, "skill", 0, 0.4) for t in tasks]
        + [record("heldout-1", "skill", 0, 1.0, "heldout")],
    )
    code, out, _ = call(capsys, "compare", first, f"{latest}:skill", "--split", "dev")
    report = json.loads(out)
    assert code == 0 and report["split"] == "dev" and report["delta"] == pytest.approx(0.0)
    assert report["beats"] is False


def test_compare_refuses_a_missing_file_an_empty_arm_and_no_pairs(tmp_path, capsys):
    results = tmp_path / "results.jsonl"
    write(results, [record("dev-0000000001", "skill", 0, 1.0)])
    code, _, err = call(capsys, "compare", tmp_path / "none.jsonl", results)
    assert code == 1 and "not a results file" in err
    code, _, err = call(capsys, "compare", f"{results}:control", results)
    assert code == 1 and "no results" in err
    other = tmp_path / "other.jsonl"
    write(other, [record("dev-0000000002", "skill", 0, 1.0)])
    code, _, err = call(capsys, "compare", results, other)
    assert code == 1 and "nothing to pair" in err


def test_the_run_command_needs_its_arguments(capsys):
    with pytest.raises(SystemExit):
        main(["proving-ground", "run", "--tasks", "a"])
    capsys.readouterr()


def test_run_refuses_keys_inside_tasks_before_touching_docker(tmp_path, capsys):
    (tmp_path / "tasks" / "keys").mkdir(parents=True)
    code, out, err = call(
        capsys, "run", "--tasks", tmp_path / "tasks", "--keys", tmp_path / "tasks" / "keys",
        "--arm", "skill", "--image", "img", "--agent-cmd", "agent", "--agent-label", "m",
        "--results", tmp_path / "r.jsonl",
    )  # fmt: skip
    assert code == 1 and out == "" and "contain each other" in err


def test_the_command_is_hidden_when_the_package_is_missing(monkeypatch):
    monkeypatch.setattr(cli, "_proving_available", lambda: False)
    with pytest.raises(SystemExit):
        main(["proving-ground", "grade"])
    assert "proving-ground" not in cli._parser().format_help()
    monkeypatch.undo()
    assert "proving-ground" in cli._parser().format_help()
