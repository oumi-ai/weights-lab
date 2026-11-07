import pytest

pytest.importorskip("torch")

from oumi_weights_lab.proving import beats, compare  # noqa: E402


def record(task, score, *, repeat=0, split="dev", solved=None):
    return {
        "task": task,
        "split": split,
        "repeat": repeat,
        "score": {"score": score, "solved": score >= 1.0 if solved is None else solved},
    }


def arm(scores, **kwargs):
    return [record(f"t{i}", score, **kwargs) for i, score in enumerate(scores)]


def test_a_known_delta_and_its_interval():
    left = arm([0.2, 0.1, 0.3, 0.2, 0.2, 0.1, 0.3, 0.2])
    right = arm([0.5, 0.4, 0.6, 0.5, 0.5, 0.4, 0.6, 0.5])
    report = compare(left, right)
    assert report["tasks"] == 8
    assert report["left_mean"] == pytest.approx(0.2)
    assert report["right_mean"] == pytest.approx(0.5)
    assert report["delta"] == pytest.approx(0.3)
    low, high = report["delta_ci"]
    assert low == pytest.approx(0.3) and high == pytest.approx(0.3)  # every task moved by 0.3
    assert report["permutation_p"] < 0.05
    assert report["samples"] == 10_000 and report["seed"] == 0
    assert beats(report) is True


def test_no_difference_does_not_beat():
    scores = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2]
    report = compare(arm(scores), arm(scores))
    assert report["delta"] == 0 and report["delta_ci"] == [0, 0]
    assert beats(report) is False
    worse = compare(arm([0.5] * 6), arm([0.2] * 6))
    assert worse["delta"] == pytest.approx(-0.3) and beats(worse) is False


def test_an_interval_that_includes_zero_does_not_beat():
    left = arm([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    right = arm([0.9, 0.1, 0.8, 0.2, 0.6, 0.4])
    report = compare(left, right)
    assert report["delta_ci"][0] < 0 < report["delta_ci"][1]
    assert beats(report) is False


def test_repeats_are_averaged_per_task_before_pairing():
    left = [record("a", 0.0, repeat=0), record("a", 1.0, repeat=1), record("b", 0.5)]
    right = [record("a", 1.0, repeat=r) for r in range(3)] + [record("b", 0.5)]
    report = compare(left, right)
    assert report["tasks"] == 2
    assert report["left_mean"] == pytest.approx((0.5 + 0.5) / 2)
    assert report["right_mean"] == pytest.approx((1.0 + 0.5) / 2)
    # Task a was solved in one of two left runs and all right runs; b never.
    assert report["left_solve_rate"] == pytest.approx((0.5 + 0.0) / 2)
    assert report["right_solve_rate"] == pytest.approx((1.0 + 0.0) / 2)


def test_unpaired_tasks_are_reported_and_left_out():
    left = [record("a", 0.1), record("b", 0.2), record("only-left", 0.9)]
    right = [record("a", 0.4), record("b", 0.5), record("only-right", 0.0)]
    report = compare(left, right)
    assert report["tasks"] == 2
    assert report["unpaired_left"] == ["only-left"]
    assert report["unpaired_right"] == ["only-right"]
    assert report["delta"] == pytest.approx(0.3)


def test_comparing_arms_with_no_common_task_is_refused():
    with pytest.raises(ValueError, match="no task appears on both sides"):
        compare([record("a", 0.1)], [record("b", 0.2)])
    with pytest.raises(ValueError, match="no task appears on both sides"):
        compare([], [])


def test_split_keeps_only_that_splits_records():
    left = [record("d1", 0.1), record("d2", 0.1), record("h1", 0.9, split="heldout")]
    right = [record("d1", 0.3), record("d2", 0.3), record("h1", 0.1, split="heldout")]
    assert compare(left, right, split="dev")["delta"] == pytest.approx(0.2)
    held = compare(left, right, split="heldout")
    assert held["tasks"] == 1 and held["split"] == "heldout"
    assert held["delta"] == pytest.approx(-0.8)
    assert compare(left, right)["tasks"] == 3
    with pytest.raises(ValueError, match="no task appears"):
        compare(left, right, split="calibration")


def test_the_report_is_deterministic():
    left = arm([0.1, 0.4, 0.2, 0.8, 0.3, 0.5, 0.6])
    right = arm([0.3, 0.5, 0.1, 0.9, 0.6, 0.4, 0.9])
    first = compare(left, right)
    assert compare(left, right) == first
    assert compare(list(reversed(left)), list(reversed(right))) == first


def test_compare_uses_the_labs_own_paired_statistics(monkeypatch):
    from oumi_weights_lab import evaluation

    seen = {}
    real = evaluation._paired_report

    def spy(pairs, clusters, samples, seed, fractional=None):
        seen.update(pairs=pairs, clusters=clusters, samples=samples, seed=seed, f=fractional)
        return real(pairs, clusters, samples, seed, fractional)

    monkeypatch.setattr(evaluation, "_paired_report", spy)
    compare(arm([0.1, 0.2]), arm([0.3, 0.4]))
    assert seen["clusters"] == ["t0", "t1"] and seen["samples"] == 10_000
    assert seen["seed"] == 0 and seen["f"] is True
    assert seen["pairs"][0] == (
        {"correct": 0.1, "failed": False, "fractional": True},
        {"correct": 0.3, "failed": False, "fractional": True},
    )


def setup_record(task, **setup):
    base = {
        "image_id": "sha256:a",
        "agent_cmd": "agent -p",
        "agent_label": "m",
        "timeout": 600,
        "network": {"mode": "proxy", "allow": ["api.anthropic.com"]},
    }
    return {**record(task, 0.5), **base, **setup}


def test_each_sides_setup_is_reported():
    left = [setup_record(f"t{i}") for i in range(4)]
    right = [setup_record(f"t{i}", image_id="sha256:b") for i in range(4)]
    report = compare(left, right)
    assert report["left_setup"]["image_id"] == "sha256:a"
    assert report["right_setup"] == {
        "image_id": "sha256:b",
        "agent_cmd": "agent -p",
        "agent_label": "m",
        "timeout": 600,
        "network": {"mode": "proxy", "allow": ["api.anthropic.com"]},
    }


@pytest.mark.parametrize(
    "field,other",
    [
        ("image_id", "sha256:z"),
        ("agent_cmd", "x"),
        ("agent_label", "n"),
        ("timeout", 60),
        ("network", {"mode": "open"}),
        ("network", {"mode": "proxy", "allow": ["api.anthropic.com", "pypi.org"]}),
        ("network", None),
    ],
)
def test_a_side_that_mixes_setups_is_refused(field, other):
    mixed = [setup_record("t0"), setup_record("t1", **{field: other})]
    with pytest.raises(ValueError, match=field):
        compare(mixed, [setup_record("t0"), setup_record("t1")])
    with pytest.raises(ValueError, match=field):
        compare([setup_record("t0"), setup_record("t1")], mixed)
