"""The proving-ground runner: what a container is given, and what is recorded.

The isolation checks here are the point: the agent's container may see the task directory
(read-only), an empty output directory, a prompt and named environment variables, and
nothing else. No Docker runs; ``launch`` and ``probe`` are fakes.
"""

import copy
import json
import os
import re
import shlex
import stat
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from oumi_weights_lab.proving import build_task, load_key, runner  # noqa: E402
from oumi_weights_lab.proving.planted import PLANTED_FAMILIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "org.oumi.weights-lab.arm": "skill",
    "org.oumi.weights-lab.version": "9.9.9",
    "org.oumi.weights-lab.revision": "abc123",
}


def probe(image, agent):
    return {"agent_version": "1.2.3 (Fake Agent)", "image_id": "sha256:feed", "labels": LABELS}


def honest_claim(key, **changes):
    check = key["validation"]
    claim = {
        "version": 1,
        "task": key["task"],
        "components": [
            *key["components"],
            *({"kind": "neurons", **n} for n in key.get("neurons", [])),
        ],
        "ablation": {
            "accuracy_before": check["baseline_accuracy"],
            "accuracy_after": check["key_ablated_accuracy"],
            "rows": check["rows"],
        },
        "summary": "test",
        "failed_hypotheses": [],
    }
    if key.get("directions"):
        claim["directions"] = key["directions"]
        claim["direction_ablation"] = {
            "accuracy_before": check["baseline_accuracy"],
            "accuracy_after": check.get("direction_ablated_accuracy", 0.5),
            "rows": check["rows"],
        }
    claim.update(changes)
    return claim


@pytest.fixture
def tasks(proving_tasks):
    """(tasks directory, keys directory, {task id: key}); shared and never modified."""
    dirs = {f: v["dir"] for f, v in proving_tasks.items()}
    some = next(iter(dirs.values()))
    keys = {v["key"]["task"]: v["key"] for v in proving_tasks.values()}
    return some.parent, some.parent.parent / "keys", keys


class Fake:
    """A launcher that records invocations and plays a scripted agent."""

    def __init__(self, keys, behaviour="honest", timed_out=False):
        self.keys, self.behaviour, self.timed_out = keys, behaviour, timed_out
        self.invocations = []

    def __call__(self, invocation):
        self.invocations.append(invocation)
        task = Path(invocation.mounts[0][0]).name
        out = invocation.out_dir
        if self.behaviour == "honest":
            (out / "claim.json").write_text(json.dumps(honest_claim(self.keys[task])))
        elif callable(self.behaviour):
            self.behaviour(out, task)
        return {"exit_code": None if self.timed_out else 0, "timed_out": self.timed_out}


class FakeEgress:
    """Stands in for ``docker_egress``: records each network it was asked for and whether
    it was taken down."""

    def __init__(self):
        self.made, self.removed = [], []

    @contextmanager
    def __call__(self, name, image, allow):
        self.made.append((name, image, allow))
        try:
            yield runner.Egress(f"{name}-net", f"http://{name}-proxy:3128", allow)
        finally:
            self.removed.append(name)


def run_all(tasks, tmp_path, fake, **overrides):
    tasks_dir, keys_dir, _ = tasks
    settings = {
        "arm": "skill",
        "image": "weights-lab-proving:skill",
        "agent_cmd": "fake-agent -p --model small",
        "agent_label": "small-model",
        "repeats": 1,
        "timeout": 600,
        "results": tmp_path / "results.jsonl",
        "launch": fake,
        "probe": probe,
        "egress": FakeEgress(),
    }
    settings.update(overrides)
    return runner.run(tasks_dir, keys_dir, **settings)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# --- the docker command line ---------------------------------------------------------


def argv_for(tmp_path, **changes):
    task_dir, out_dir = tmp_path / "t" / "dev-aaaaaaaaaa", tmp_path / "run" / "out"
    task_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    options = {
        "name": "wl-1",
        "image": "img:skill",
        "task_dir": task_dir,
        "out_dir": out_dir,
        "command": ["agent", "-p"],
        "prompt": "PROMPT TEXT",
    }
    options.update(changes)
    return runner.docker_argv(**options), task_dir, out_dir


def mounts_in(argv):
    found = []
    for i, word in enumerate(argv):
        if word == "--mount":
            found.append(dict(part.partition("=")[::2] for part in argv[i + 1].split(",")))
    return found


def test_only_the_task_and_the_out_directory_are_mounted(tmp_path):
    (argv, mounts), task_dir, out_dir = argv_for(tmp_path)
    found = mounts_in(argv)
    assert len(found) == 2
    task, out = found
    assert task["source"] == str(task_dir.resolve()) and task["target"] == "/task"
    assert "readonly" in task
    assert out["source"] == str(out_dir.resolve()) and out["target"] == "/out"
    assert "readonly" not in out
    assert mounts == [(task["source"], "/task", "ro"), (out["source"], "/out", "rw")]
    assert "-v" not in argv and "--volume" not in argv and "--volumes-from" not in argv
    assert "--privileged" not in argv and "--network" not in argv


def test_the_container_runs_unprivileged_and_is_removed(tmp_path):
    (argv, _), _, _ = argv_for(tmp_path)
    assert argv[:5] == ["docker", "run", "--rm", "--name", "wl-1"]
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"


def test_the_prompt_is_the_last_argument_or_replaces_the_token(tmp_path):
    (argv, _), _, _ = argv_for(tmp_path)
    assert argv[-3:] == ["agent", "-p", "PROMPT TEXT"]
    (argv, _), _, _ = argv_for(tmp_path / "b", command=["agent", "--prompt", "{prompt}", "--x"])
    assert argv[-4:] == ["agent", "--prompt", "PROMPT TEXT", "--x"]
    assert "{prompt}" not in argv


def test_env_is_passed_by_name_and_values_never_appear(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "sk-very-secret-value")
    (argv, _), _, _ = argv_for(tmp_path, env=["FAKE_API_KEY", "OTHER"])
    pairs = [argv[i + 1] for i, w in enumerate(argv) if w == "-e"]
    assert pairs == ["FAKE_API_KEY", "OTHER"]
    assert "sk-very-secret-value" not in " ".join(argv)


@pytest.mark.parametrize("bad", ["KEY=value", "a b", "", "1X", "-e"])
def test_env_given_as_a_value_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="environment variable name"):
        argv_for(tmp_path, env=[bad])


def test_a_path_with_a_comma_is_refused(tmp_path):
    with pytest.raises(ValueError, match="comma"):
        argv_for(tmp_path, out_dir=tmp_path / "a,b")


# --- end to end ----------------------------------------------------------------------


def test_every_task_and_repeat_is_run_and_an_honest_claim_scores_one(tasks, tmp_path):
    fake = Fake(tasks[2])
    records = run_all(tasks, tmp_path, fake, repeats=2)
    assert len(records) == len(fake.invocations) == 2 * len(PLANTED_FAMILIES)
    saved = lines(tmp_path / "results.jsonl")
    assert saved == records
    assert {(r["task"], r["repeat"]) for r in saved} == {(t, n) for t in tasks[2] for n in (0, 1)}
    for record in saved:
        key = tasks[2][record["task"]]
        assert record["family"] == key["family"] and record["split"] == "dev"
        assert record["arm"] == "skill"
        assert record["agent_cmd"] == "fake-agent -p --model small"
        assert record["agent_label"] == "small-model"
        assert record["agent_version"] == "1.2.3 (Fake Agent)"
        assert record["image"] == "weights-lab-proving:skill"
        assert record["image_id"] == "sha256:feed"
        assert record["lab_version"] == "9.9.9" and record["revision"] == "abc123"
        assert record["exit_code"] == 0 and record["timed_out"] is False
        assert record["timeout"] == 600
        assert record["claim_valid"] is True and record["problems"] == []
        assert record["score"]["score"] == pytest.approx(1.0)
        assert record["score"]["solved"] is True
        assert record["seconds"] >= 0
        assert re.match(r"\d{4}-\d\d-\d\dT.*\+00:00$", record["started"])
        assert Path(record["run_dir"]).is_dir()


def test_each_run_gets_a_fresh_writable_out_directory_with_a_transcript(tasks, tmp_path):
    fake = Fake(tasks[2])
    run_all(tasks, tmp_path, fake, repeats=2)
    outs = [i.out_dir for i in fake.invocations]
    assert len(set(outs)) == len(outs)
    for out in outs:
        assert out.parent.parent == (tmp_path / "results.runs")
        assert stat.S_IMODE(out.stat().st_mode) == 0o777
        assert (out / "transcript.txt").exists()
    assert not any(tmp_path.joinpath("results.jsonl") == o for o in outs)


def test_an_agent_that_writes_nothing_scores_zero_with_a_problem(tasks, tmp_path):
    records = run_all(tasks, tmp_path, Fake(tasks[2], behaviour="nothing"))
    for record in records:
        assert record["claim_valid"] is False
        assert record["score"]["score"] == 0.0 and record["score"]["solved"] is False
        assert any("no claim was written" in p for p in record["problems"])


def test_an_invalid_or_unreadable_claim_scores_zero(tasks, tmp_path):
    def junk(out, task):
        (out / "claim.json").write_text("{not json")

    def wrong_task(out, task):
        (out / "claim.json").write_text(json.dumps({"version": 1, "task": "dev-0000000000"}))

    def oversized(out, task):
        (out / "claim.json").write_text(" " * 1_100_000)

    for behaviour in (junk, wrong_task, oversized):
        path = tmp_path / f"{behaviour.__name__}.jsonl"
        records = run_all(tasks, tmp_path, Fake(tasks[2], behaviour=behaviour), results=path)
        assert all(r["score"]["score"] == 0.0 and r["problems"] for r in records)


def test_a_timeout_is_recorded_and_a_claim_written_before_it_is_graded(tasks, tmp_path):
    records = run_all(tasks, tmp_path, Fake(tasks[2], timed_out=True))
    for record in records:
        assert record["timed_out"] is True and record["exit_code"] is None
        assert record["score"]["score"] == pytest.approx(1.0)
    none = run_all(
        tasks,
        tmp_path,
        Fake(tasks[2], behaviour="nothing", timed_out=True),
        results=tmp_path / "late.jsonl",
    )
    assert all(r["timed_out"] and r["score"]["score"] == 0.0 for r in none)


def test_rerunning_skips_finished_runs_and_resumes_the_rest(tasks, tmp_path):
    first = Fake(tasks[2])
    run_all(tasks, tmp_path, first, repeats=1)
    assert len(first.invocations) == len(PLANTED_FAMILIES)
    again = Fake(tasks[2])
    assert run_all(tasks, tmp_path, again, repeats=1) == []
    assert again.invocations == []
    more = Fake(tasks[2])
    records = run_all(tasks, tmp_path, more, repeats=2)
    assert len(more.invocations) == len(PLANTED_FAMILIES)
    assert {r["repeat"] for r in records} == {1}
    assert len(lines(tmp_path / "results.jsonl")) == 2 * len(PLANTED_FAMILIES)


class Slow(Fake):
    """A launcher that waits so that runs overlap, and counts how many run at once."""

    def __init__(self, keys):
        super().__init__(keys)
        self.lock, self.now, self.peak = threading.Lock(), 0, 0

    def __call__(self, invocation):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        try:
            time.sleep(0.05)
            return super().__call__(invocation)
        finally:
            with self.lock:
                self.now -= 1


def test_jobs_run_every_task_and_repeat_once_and_write_each_record_once(tasks, tmp_path):
    fake = Slow(tasks[2])
    records = run_all(tasks, tmp_path, fake, repeats=2, jobs=3)
    wanted = {(t, n) for t in tasks[2] for n in range(2)}
    written = lines(tmp_path / "results.jsonl")
    assert len(fake.invocations) == len(records) == len(written) == len(wanted)
    assert {(r["task"], r["repeat"]) for r in written} == wanted
    assert 1 < fake.peak <= 3
    assert len({i.name for i in fake.invocations}) == len(fake.invocations)
    assert len({r["run_dir"] for r in written}) == len(written)
    assert all(r["score"]["valid"] for r in written)


def test_jobs_resume_after_a_partial_run_skips_the_finished_runs(tasks, tmp_path):
    one = next(iter(tasks[2]))
    run_all(tasks, tmp_path, Fake(tasks[2]), repeats=1, task_ids=[one])
    records = run_all(tasks, tmp_path, Slow(tasks[2]), repeats=1, jobs=3)
    assert one not in {r["task"] for r in records}
    assert len(records) == len(tasks[2]) - 1
    assert len(lines(tmp_path / "results.jsonl")) == len(tasks[2])
    assert run_all(tasks, tmp_path, Fake(tasks[2]), repeats=1, jobs=3) == []


def test_jobs_must_be_at_least_one(tasks, tmp_path):
    with pytest.raises(ValueError, match="jobs"):
        run_all(tasks, tmp_path, Fake(tasks[2]), jobs=0)


@pytest.mark.parametrize("jobs", [1, 2])
def test_an_interrupt_launches_nothing_more_and_writes_no_record(
    tasks, tmp_path, jobs, monkeypatch
):
    """Ctrl-C reaches the main thread; the container it also killed must not be recorded."""
    import _thread

    launched, lock = [], threading.Lock()
    noticed = threading.Event()
    real_wait = runner.wait

    def watching_wait(*args, **kwargs):
        try:
            return real_wait(*args, **kwargs)
        except BaseException:
            noticed.set()
            raise

    monkeypatch.setattr(runner, "wait", watching_wait)

    def launch(invocation):
        with lock:
            launched.append(invocation.name)
            first = len(launched) == 1
        if first:
            _thread.interrupt_main()
        # Every container dies with the same Ctrl-C, but each launch returns only once the
        # main thread has caught the interrupt, so no timing decides the outcome. (At jobs=1
        # the interrupt lands in this very call and nothing after it runs.)
        noticed.wait(timeout=30)
        return {"exit_code": 130, "timed_out": False}

    with pytest.raises(KeyboardInterrupt):
        run_all(tasks, tmp_path, launch, repeats=3, jobs=jobs)
    assert len(launched) <= jobs
    assert runner.read_results(tmp_path / "results.jsonl") == []


def test_the_other_arm_is_not_skipped_by_the_first(tasks, tmp_path):
    run_all(tasks, tmp_path, Fake(tasks[2]))
    control = Fake(tasks[2])
    labels = dict(LABELS, **{"org.oumi.weights-lab.arm": "control"})
    records = run_all(
        tasks,
        tmp_path,
        control,
        arm="control",
        probe=lambda image, agent: {"labels": labels},
    )
    assert len(records) == len(PLANTED_FAMILIES)
    assert {r["arm"] for r in lines(tmp_path / "results.jsonl")} == {"skill", "control"}


def test_only_the_named_tasks_run(tasks, tmp_path):
    one = next(iter(tasks[2]))
    fake = Fake(tasks[2])
    records = run_all(tasks, tmp_path, fake, task_ids=[one])
    assert [r["task"] for r in records] == [one]
    with pytest.raises(ValueError, match="no task"):
        run_all(tasks, tmp_path, fake, task_ids=["../keys"])


def test_agent_version_and_image_come_from_the_probe_once_per_call(tasks, tmp_path):
    seen = []

    def counting(image, agent):
        seen.append((image, agent))
        return probe(image, agent)

    run_all(tasks, tmp_path, Fake(tasks[2]), repeats=2, probe=counting)
    assert seen == [("weights-lab-proving:skill", "fake-agent")]


# --- isolation -----------------------------------------------------------------------


def all_text(invocation):
    return " ".join([*invocation.argv, *map(str, invocation.mounts)])


def test_no_invocation_mentions_keys_results_or_any_family(tasks, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "sk-very-secret-value")
    fake = Fake(tasks[2])
    run_all(tasks, tmp_path, fake, env=["FAKE_API_KEY"])
    keys_dir = str(tasks[1].resolve())
    for invocation in fake.invocations:
        text = all_text(invocation)
        assert keys_dir not in text
        assert str((tmp_path / "results.jsonl").resolve()) not in text
        assert "sk-very-secret-value" not in text
        assert [m[1] for m in invocation.mounts] == ["/task", "/out"]
        assert invocation.mounts[0][2] == "ro"
        assert invocation.env == ("FAKE_API_KEY",)
        for family in PLANTED_FAMILIES:
            assert family not in text
    # the secret value reaches no result record either
    assert "sk-very-secret-value" not in (tmp_path / "results.jsonl").read_text()


FORBIDDEN = (
    "decoy",
    "planted",
    "plant ",
    "skill",
    "key",
    "secret",
    "seed",
    "heldout",
    "held-out",
    "oumi",
    "weights-lab",
    "weights_lab",
    "family",
    "families",
    "mechanism is",
)


def test_the_prompt_holds_no_answers_and_no_family_names(tasks, tmp_path):
    tasks_dir, _, keys = tasks
    for ident, key in keys.items():
        prompt = runner.build_prompt(tasks_dir / ident, timeout=1800)
        low = prompt.lower()
        for word in FORBIDDEN:
            # the claim format may say "true mechanism"; nothing else may name these words
            assert word not in low.replace("true mechanism", ""), (ident, word)
        for family in PLANTED_FAMILIES:
            assert family not in low
        assert ident in prompt and "/task/model" in prompt and "/out/claim.json" in prompt
        assert "30 minutes" in prompt
        assert str(key["eval_seed"]) not in prompt
        for part in key["components"]:
            assert json.dumps(part) not in prompt
        for direction in key.get("directions", []):
            assert str(direction["vector"][0]) not in prompt
        for decoy in key["decoys"]:
            assert json.dumps(decoy) not in prompt


def test_both_arms_get_the_same_prompt(tasks, tmp_path):
    skill, control = Fake(tasks[2]), Fake(tasks[2])
    run_all(tasks, tmp_path, skill)
    labels = dict(LABELS, **{"org.oumi.weights-lab.arm": "control"})
    run_all(
        tasks,
        tmp_path,
        control,
        arm="control",
        probe=lambda image, agent: {"labels": labels},
        image="weights-lab-proving:control",
    )
    prompts = lambda fake: sorted(i.argv[-1] for i in fake.invocations)  # noqa: E731
    assert prompts(skill) == prompts(control)


def test_a_heldout_run_passes_neither_the_secret_nor_the_family(tmp_path, monkeypatch):
    secret = "7f" * 32
    secret_file = tmp_path / "private" / "secret.txt"
    secret_file.parent.mkdir()
    secret_file.write_text(secret + "\n")
    info = build_task(
        "copy-head", "heldout", 0, tmp_path / "tasks", tmp_path / "keys", secret_file=secret_file
    )
    key = load_key(info["key_path"])
    assert key["family"] == "copy-head" and key["split"] == "heldout"
    fake = Fake({key["task"]: key})
    records = runner.run(
        tmp_path / "tasks",
        tmp_path / "keys",
        arm="skill",
        image="img",
        agent_cmd="fake-agent -p",
        agent_label="m",
        repeats=1,
        results=tmp_path / "results.jsonl",
        launch=fake,
        probe=probe,
        egress=FakeEgress(),
    )
    (invocation,) = fake.invocations
    text = all_text(invocation)
    assert secret not in text and str(secret_file) not in text
    assert "copy-head" not in text
    # the family is recorded on the host, from the key
    assert records[0]["family"] == "copy-head" and records[0]["split"] == "heldout"
    # the task directory the container mounts holds no family, seed or secret
    mounted = Path(invocation.mounts[0][0])
    blob = b"".join(
        p.read_bytes() for p in mounted.rglob("*") if p.is_file() and p.suffix != ".safetensors"
    )
    assert b"copy-head" not in blob and secret.encode() not in blob
    assert str(key["eval_seed"]).encode() not in blob


def test_the_mounted_task_directory_is_exactly_what_the_builder_wrote(tasks):
    tasks_dir, _, keys = tasks
    for ident in keys:
        info = runner.check_task_dir(tasks_dir / ident)
        assert info["id"] == ident and "family" not in info
        assert {p.name for p in (tasks_dir / ident).iterdir()} == runner.TASK_ENTRIES


def copy_task(tasks, tmp_path):
    import shutil

    tasks_dir, keys_dir, keys = tasks
    ident = next(iter(keys))
    shutil.copytree(tasks_dir / ident, tmp_path / "tasks" / ident)
    shutil.copytree(keys_dir, tmp_path / "keys")
    return tmp_path / "tasks", tmp_path / "keys", ident


def run_copy(tasks_dir, keys_dir, tmp_path, **overrides):
    settings = {
        "arm": "skill",
        "image": "img",
        "agent_cmd": "fake-agent",
        "agent_label": "m",
        "repeats": 1,
        "results": tmp_path / "results.jsonl",
        "launch": lambda invocation: {"exit_code": 0, "timed_out": False},
        "probe": probe,
        "egress": FakeEgress(),
    }
    settings.update(overrides)
    return runner.run(tasks_dir, keys_dir, **settings)


def test_a_task_directory_with_a_stray_file_is_refused(tasks, tmp_path):
    tasks_dir, keys_dir, ident = copy_task(tasks, tmp_path)
    (tasks_dir / ident / "key.json").write_text("{}")
    with pytest.raises(ValueError, match="must not have"):
        run_copy(tasks_dir, keys_dir, tmp_path)


def test_a_task_json_with_a_family_or_seed_is_refused(tasks, tmp_path):
    tasks_dir, keys_dir, ident = copy_task(tasks, tmp_path)
    path = tasks_dir / ident / "task.json"
    info = json.loads(path.read_text())
    path.write_text(json.dumps({**info, "family": "copy-head"}))
    with pytest.raises(ValueError, match="family"):
        run_copy(tasks_dir, keys_dir, tmp_path)


def test_a_symlink_in_a_task_directory_is_refused(tasks, tmp_path):
    tasks_dir, keys_dir, ident = copy_task(tasks, tmp_path)
    target = keys_dir / f"{ident}.json"
    (tasks_dir / ident / "model" / "link.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        run_copy(tasks_dir, keys_dir, tmp_path)


def test_keys_inside_tasks_or_tasks_inside_keys_is_refused(tasks, tmp_path):
    tasks_dir, keys_dir, ident = copy_task(tasks, tmp_path)
    inside = tasks_dir / "keys"
    inside.mkdir()
    with pytest.raises(ValueError, match="contain each other"):
        run_copy(tasks_dir, inside, tmp_path)
    with pytest.raises(ValueError, match="contain each other"):
        run_copy(tasks_dir, tasks_dir.parent, tmp_path)
    with pytest.raises(ValueError, match="contain each other"):
        run_copy(tasks_dir, tasks_dir, tmp_path)


def test_a_results_file_inside_the_tasks_directory_is_refused(tasks, tmp_path):
    tasks_dir, keys_dir, ident = copy_task(tasks, tmp_path)
    with pytest.raises(ValueError, match="results file"):
        run_copy(tasks_dir, keys_dir, tmp_path, results=tasks_dir / ident / "results.jsonl")
    with pytest.raises(ValueError, match="results file"):
        run_copy(tasks_dir, keys_dir, tmp_path, results=tasks_dir / "r.jsonl")


def test_a_task_without_a_key_is_refused(tasks, tmp_path):
    tasks_dir, keys_dir, ident = copy_task(tasks, tmp_path)
    (keys_dir / f"{ident}.json").unlink()
    with pytest.raises(ValueError, match="no key"):
        run_copy(tasks_dir, keys_dir, tmp_path)


def test_an_image_of_the_other_arm_is_refused(tasks, tmp_path):
    tasks_dir, keys_dir, _ = copy_task(tasks, tmp_path)
    with pytest.raises(ValueError, match="labelled arm"):
        run_copy(tasks_dir, keys_dir, tmp_path, arm="control")
    with pytest.raises(ValueError, match="labelled arm"):
        run_copy(tasks_dir, keys_dir, tmp_path, probe=lambda i, a: {"labels": {}})


def test_an_unset_env_name_and_a_bad_arm_are_refused(tasks, tmp_path, monkeypatch):
    tasks_dir, keys_dir, _ = copy_task(tasks, tmp_path)
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    with pytest.raises(ValueError, match="not set"):
        run_copy(tasks_dir, keys_dir, tmp_path, env=["NOT_SET_ANYWHERE"])
    with pytest.raises(ValueError, match="environment variable name"):
        run_copy(tasks_dir, keys_dir, tmp_path, env=["A=b"])
    with pytest.raises(ValueError, match="arm"):
        run_copy(tasks_dir, keys_dir, tmp_path, arm="both")


def test_a_claim_that_is_a_symlink_is_not_followed(tasks, tmp_path):
    tasks_dir, keys_dir, keys = tasks
    ident = next(iter(keys))
    key_file = keys_dir / f"{ident}.json"

    def link(out, task):
        (out / "claim.json").symlink_to(key_file)

    results = tmp_path / "results.jsonl"
    records = run_all(
        tasks, tmp_path, Fake(keys, behaviour=link), task_ids=[ident], results=results
    )
    assert records[0]["score"]["score"] == 0.0
    assert any("symlink" in p for p in records[0]["problems"])
    assert str(keys[ident]["eval_seed"]) not in results.read_text()


# --- the default launcher ------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="uses a shell script")
def test_a_timed_out_container_is_killed_by_name(tmp_path, monkeypatch):
    log = tmp_path / "docker.log"
    script = tmp_path / "bin" / "docker"
    script.parent.mkdir()
    script.write_text(
        f'#!/bin/sh\necho "$@" >> {shlex.quote(str(log))}\n'
        'if [ "$1" = run ]; then echo started; exec sleep 30; fi\n'
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    out = tmp_path / "out"
    out.mkdir()
    transcript = out / "transcript.txt"
    transcript.touch()
    invocation = runner.Invocation(
        name="wl-test",
        image="img",
        argv=("docker", "run", "--name", "wl-test", "img"),
        mounts=(),
        env=(),
        timeout=1,
        out_dir=out,
        transcript=transcript,
    )
    outcome = runner.docker_launch(invocation)
    assert outcome == {"exit_code": None, "timed_out": True}
    assert "kill wl-test" in log.read_text()
    assert "started" in transcript.read_text()


# --- the agent's network -------------------------------------------------------------


def network_flags(argv):
    """The ``--network`` values and the proxy variables an argv sets."""
    networks = [argv[i + 1] for i, w in enumerate(argv) if w == "--network"]
    proxies = dict(
        argv[i + 1].split("=", 1)
        for i, w in enumerate(argv)
        if w == "-e" and "=" in argv[i + 1] and "PROXY" in argv[i + 1].upper()
    )
    return networks, proxies


def test_a_restricted_container_joins_only_the_internal_network_and_uses_the_proxy(tmp_path):
    route = runner.Egress("wl-net", "http://wl-proxy:3128", ("api.anthropic.com",))
    (argv, _), _, _ = argv_for(tmp_path, egress=route, env=["FAKE_API_KEY"])
    networks, proxies = network_flags(argv)
    assert networks == ["wl-net"]
    assert proxies == {name: "http://wl-proxy:3128" for name in runner.PROXY_ENV}
    assert {"HTTPS_PROXY", "HTTP_PROXY"} <= set(proxies)
    assert argv.index("--network") < argv.index("img:skill")
    assert [argv[i + 1] for i, w in enumerate(argv) if w == "-e" and "=" not in argv[i + 1]] == [
        "FAKE_API_KEY"
    ]


@pytest.mark.parametrize("name", ["HTTPS_PROXY", "http_proxy", "NO_PROXY"])
def test_a_proxy_variable_from_the_caller_is_refused_when_the_runner_sets_them(tmp_path, name):
    route = runner.Egress("wl-net", "http://wl-proxy:3128", ("api.anthropic.com",))
    with pytest.raises(ValueError, match="set by the runner"):
        argv_for(tmp_path, egress=route, env=[name])


def test_runs_are_restricted_to_the_api_host_by_default(tasks, tmp_path):
    fake, egress = Fake(tasks[2]), FakeEgress()
    records = run_all(tasks, tmp_path, fake, egress=egress)
    assert len(egress.made) == 1 and egress.removed == [egress.made[0][0]]
    name, image, allow = egress.made[0]
    assert image == "weights-lab-proving:skill" and allow == ("api.anthropic.com",)
    for invocation in fake.invocations:
        networks, proxies = network_flags(list(invocation.argv))
        assert networks == [f"{name}-net"]
        assert set(proxies.values()) == {f"http://{name}-proxy:3128"}
    assert {json.dumps(r["network"]) for r in records} == {
        json.dumps({"mode": "proxy", "allow": ["api.anthropic.com"]})
    }


def test_allowed_hosts_and_the_proxy_image_are_passed_on(tasks, tmp_path):
    egress = FakeEgress()
    records = run_all(
        tasks,
        tmp_path,
        Fake(tasks[2]),
        egress=egress,
        allow_hosts=["pypi.org", "api.anthropic.com", "pypi.org"],
        proxy_image="weights-lab-proving:control",
    )
    assert egress.made[0][1:] == ("weights-lab-proving:control", ("api.anthropic.com", "pypi.org"))
    assert records[0]["network"] == {"mode": "proxy", "allow": ["api.anthropic.com", "pypi.org"]}


def test_an_open_network_run_records_it_and_starts_no_proxy(tasks, tmp_path):
    fake, egress = Fake(tasks[2]), FakeEgress()
    records = run_all(tasks, tmp_path, fake, egress=egress, open_network=True)
    assert egress.made == []
    assert all(network_flags(list(i.argv)) == ([], {}) for i in fake.invocations)
    assert all(r["network"] == {"mode": "open"} for r in records)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"open_network": True, "allow_hosts": ["pypi.org"]}, "restricted network"),
        ({"allow_hosts": ["pypi.org/simple"]}, "host"),
    ],
)
def test_bad_network_settings_are_refused_before_anything_runs(tasks, tmp_path, change, message):
    fake, egress = Fake(tasks[2]), FakeEgress()
    with pytest.raises(ValueError, match=message):
        run_all(tasks, tmp_path, fake, egress=egress, **change)
    assert fake.invocations == [] and egress.made == []


def test_the_network_is_taken_down_when_a_run_raises(tasks, tmp_path):
    egress = FakeEgress()

    def broken(invocation):
        raise RuntimeError("the container did not start")

    with pytest.raises(RuntimeError, match="did not start"):
        run_all(tasks, tmp_path, broken, egress=egress, jobs=2)
    assert len(egress.made) == 1 and egress.removed == [egress.made[0][0]]


@pytest.mark.skipif(sys.platform == "win32", reason="uses a shell script")
def test_docker_egress_makes_an_internal_network_and_a_bare_proxy_and_removes_both(
    tmp_path, monkeypatch
):
    log = tmp_path / "docker.log"
    script = tmp_path / "bin" / "docker"
    script.parent.mkdir()
    script.write_text(
        # One line per call, cut short: the proxy's source is one of the arguments.
        f'#!/bin/sh\nprintf "%s" "$*" | tr "\\n" " " | head -c 200 >> {shlex.quote(str(log))}\n'
        f"echo >> {shlex.quote(str(log))}\n"
        'if [ "$1" = logs ]; then echo "listening on 0.0.0.0:3128"; fi\n'
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(KeyError):
        with runner.docker_egress("wl-x", "img:control", ("api.anthropic.com", "pypi.org")) as e:
            assert e == runner.Egress(
                "wl-x-net", "http://wl-x-proxy:3128", ("api.anthropic.com", "pypi.org")
            )
            raise KeyError("a run failed")
    calls = [line for line in log.read_text().splitlines() if line]
    assert calls[0] == "network create --internal wl-x-net"
    start = calls[1]
    assert start.startswith(
        "run --detach --rm --name wl-x-proxy --cap-drop ALL --security-opt no-new-privileges "
        "img:control python -c "
    )
    assert "--mount" not in start and "-v " not in start and "--network" not in start
    assert calls[2] == "network connect wl-x-net wl-x-proxy"
    assert calls[3] == "logs wl-x-proxy"
    assert calls[4:] == ["rm --force wl-x-proxy", "network rm wl-x-net"]


def test_the_proxy_command_allows_exactly_the_given_hosts(monkeypatch):
    seen = []

    def docker(*args, timeout=120):
        seen.append(args)
        return "listening on 0.0.0.0:3128" if args[0] == "logs" else ""

    monkeypatch.setattr(runner, "_docker", docker)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: None)
    with runner.docker_egress("wl-y", "img", ("api.anthropic.com", "pypi.org")):
        pass
    command = list(next(args for args in seen if args[0] == "run"))
    source = command[command.index("-c") + 1]
    assert source == (ROOT / "src/oumi_weights_lab/proving/egress.py").read_text()
    rest = command[command.index("-c") + 2 :]
    assert rest == ["--port", "3128", "--allow", "api.anthropic.com", "--allow", "pypi.org"]


# --- the image files -----------------------------------------------------------------


def ignored(patterns, path):
    """Whether a docker-ignore file excludes ``path`` (last matching rule wins)."""

    def regex(pattern):
        out, i = "", 0
        while i < len(pattern):
            if pattern[i : i + 2] == "**":
                out += ".*"
                i += 2
                if pattern[i : i + 1] == "/":
                    i += 1
                    out = out[:-2] + "(?:.*/)?"
            elif pattern[i] == "*":
                out += "[^/]*"
                i += 1
            elif pattern[i] == "?":
                out += "[^/]"
                i += 1
            else:
                out += re.escape(pattern[i])
                i += 1
        return re.compile(out + r"\Z")

    parts = path.split("/")
    prefixes = ["/".join(parts[: n + 1]) for n in range(len(parts))]
    state = False
    for line in patterns:
        negate = line.startswith("!")
        compiled = regex(line.lstrip("!"))
        if any(compiled.match(prefix) for prefix in prefixes):
            state = not negate
    return state


@pytest.fixture(scope="module")
def image_files():
    docker = ROOT / "docker"
    dockerfile = (docker / "proving-ground.Dockerfile").read_text()
    patterns = [
        line.strip()
        for line in (docker / "proving-ground.Dockerfile.dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return dockerfile, patterns


def test_the_build_context_keeps_answers_secrets_and_the_proving_package_out(image_files):
    _, patterns = image_files
    for path in (
        "keys/dev-1.json",
        "tasks/keys/x.json",
        "secret.txt",
        "my-secret-file",
        ".env",
        ".venv/bin/python",
        "sessions/abc/status.json",
        ".git/config",
        "tests/test_proving_tasks.py",
        "docs/proving-ground.md",
        "src/oumi_weights_lab/proving/planted.py",
        "src/oumi_weights_lab/proving/tasks.py",
        "skills/oumi-weights-lab/references/proving-ground.md",
        "src/oumi_weights_lab/__pycache__/cli.cpython-312.pyc",
    ):
        assert ignored(patterns, path), path
    for path in (
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "src/oumi_weights_lab/cli.py",
        "src/oumi_weights_lab/lab.py",
        "skills/oumi-weights-lab/SKILL.md",
        "skills/oumi-weights-lab/references/heads.md",
        "docker/proving-ground-stub.md",
    ):
        assert not ignored(patterns, path), path


def test_the_dockerfile_copies_only_whitelisted_paths(image_files):
    dockerfile, patterns = image_files
    copies = re.findall(r"^COPY (?!--from)(?:--\S+ )*(.+) \S+$", dockerfile, re.M)
    assert copies
    for sources in copies:
        for source in sources.split():
            assert (ROOT / source).exists(), source
            assert not ignored(patterns, source if "." in source else source + "/x.py"), source
    for word in ("tests", "docs", "keys", "secret", "examples", "scripts"):
        assert not re.search(rf"^COPY .*\b{word}\b", dockerfile, re.M), word


def test_the_skill_image_strips_the_proving_package_and_its_recipe(image_files):
    dockerfile, _ = image_files
    assert "rm -rf src/oumi_weights_lab/proving" in dockerfile
    # the recipe becomes the stub; SKILL.md is left alone so links to it still land
    assert "cp /stub.md skills/oumi-weights-lab/references/proving-ground.md" in dockerfile
    assert "cmp /stub.md skills/oumi-weights-lab/references/proving-ground.md" in dockerfile
    assert "test ! -e src/oumi_weights_lab/proving" in dockerfile
    assert "sed -i" not in dockerfile and "grep -r" not in dockerfile
    stub = (ROOT / "docker" / "proving-ground-stub.md").read_text()
    recipe = (ROOT / "skills/oumi-weights-lab/references/proving-ground.md").read_text()
    assert stub.splitlines()[0] == recipe.splitlines()[0] == "# Test a method on known answers"
    assert "The proving ground is not installed in this environment." in stub
    assert len(stub.splitlines()) <= 5 and "oumi-weights-lab proving-ground" not in stub
    # the lab and the skill reach the image only through the stripped source stage
    for line in re.findall(r"^COPY --from=(\S+) .*$", dockerfile, re.M):
        assert line in {"uv", "source", "pins"}


def test_the_skill_stage_points_the_agent_at_the_skill_and_control_does_not(image_files):
    dockerfile, _ = image_files
    skill = dockerfile.split("FROM base AS arm-skill")[1].split("FROM base AS arm-control")[0]
    control = dockerfile.split("FROM base AS arm-control")[1].split("FROM arm-")[0]
    assert "${SKILL_DIR}" in skill and "/work/CLAUDE.md" in skill and "/work/AGENTS.md" in skill
    assert "CLAUDE.md" not in control and "AGENTS.md" not in control
    # only the skill stage names the skill or writes the agent's instruction files
    base = dockerfile.split("AS base")[1].split("FROM base AS arm-skill")[0]
    final = "FROM arm-" + dockerfile.split("FROM base AS arm-control")[1].split("FROM arm-", 1)[1]
    for shared in (base, final):
        for word in ("CLAUDE.md", "AGENTS.md", "SKILL_DIR", "skills/oumi-weights-lab"):
            assert word not in shared, word


def test_the_skill_and_control_arms_share_one_dockerfile_and_label_themselves(image_files):
    dockerfile, _ = image_files
    assert "ARG ARM=skill" in dockerfile and "FROM arm-${ARM} AS final" in dockerfile
    for label in ("arm", "version", "revision"):
        assert f"org.oumi.weights-lab.{label}=" in dockerfile
    assert 'ARG AGENT_INSTALL="npm install -g @anthropic-ai/claude-code"' in dockerfile
    assert "ARG SKILL_DIR=/home/agent/.claude/skills/oumi-weights-lab" in dockerfile
    assert "\nUSER agent\n" in dockerfile
    assert "FROM node:22-bookworm-slim" in dockerfile and "uv python install 3.12" in dockerfile
    control = dockerfile.split("FROM base AS arm-control")[1].split("FROM arm-")[0]
    assert "oumi" not in control and "skill" not in control.lower().replace("arm-control", "")


def test_both_arms_switch_off_the_agents_traffic_to_other_hosts(image_files):
    dockerfile, _ = image_files
    final = dockerfile.split("FROM arm-${ARM} AS final")[1]
    for name in (
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
        "DISABLE_TELEMETRY",
        "DISABLE_ERROR_REPORTING",
        "DISABLE_AUTOUPDATER",
        "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL",
    ):
        assert f"    {name}=1" in final, name
    # The image's Python runs the proxy: the base stage, which both arms share, installs it.
    assert "uv venv --python 3.12 /opt/venv" in dockerfile.split("FROM base AS pins")[0]


def test_the_extras_are_listed_once_and_match_the_plugin_wrapper(image_files):
    dockerfile, _ = image_files
    declared = re.findall(r'^ARG LAB_EXTRAS="([^"]*)"', dockerfile, re.M)
    assert len(declared) == 1
    wrapper = (ROOT / "bin" / "oumi-weights-lab").read_text()
    extras = re.findall(r"--extra ([a-z-]+)", wrapper)
    assert sorted(declared[0].split()) == sorted(extras)
    assert not re.search(
        r"\b(adapters|datasets|multi-gpu|quantized)\b", dockerfile.replace(declared[0], "")
    )


def test_the_main_cli_holds_only_a_hook_for_the_proving_command():
    # The image strips the proving package, and the command's code lives inside it.
    text = (ROOT / "src" / "oumi_weights_lab" / "cli.py").read_text()
    mentions = [line for line in text.splitlines() if "proving" in line.lower()]
    assert len(mentions) <= 5, mentions
    assert not re.search(r"decoy|heldout|held-out|family|secret", " ".join(mentions), re.I)


def test_a_huge_integer_or_deeply_nested_claim_scores_zero_and_the_run_goes_on(tasks, tmp_path):
    def huge(out, task):
        claim = copy.deepcopy(honest_claim(tasks[2][task]))
        claim["ablation"]["accuracy_after"] = 10**400
        claim["components"][0]["layer"] = 10**400
        (out / "claim.json").write_text(json.dumps(claim))

    def nested(out, task):
        (out / "claim.json").write_text("[" * 200_000 + "]" * 200_000)

    for behaviour in (huge, nested):
        path = tmp_path / f"{behaviour.__name__}.jsonl"
        records = run_all(tasks, tmp_path, Fake(tasks[2], behaviour=behaviour), results=path)
        assert len(records) == len(PLANTED_FAMILIES)
        assert all(r["score"]["score"] == 0.0 and r["problems"] for r in records)


@pytest.mark.parametrize(
    "change,named",
    [
        ({"agent_cmd": "other-agent -p"}, "agent_cmd"),
        ({"agent_label": "other-model"}, "agent_label"),
        ({"timeout": 60}, "timeout"),
        ({"open_network": True}, "network"),
        ({"allow_hosts": ["pypi.org"]}, "network"),
    ],
)
def test_resuming_with_a_different_setup_is_refused_and_names_the_difference(
    tasks, tmp_path, change, named
):
    run_all(tasks, tmp_path, Fake(tasks[2]))
    fake = Fake(tasks[2])
    with pytest.raises(ValueError, match=named):
        run_all(tasks, tmp_path, fake, repeats=2, **change)
    assert fake.invocations == []


def test_resuming_with_a_new_image_is_refused_even_when_nothing_is_left_to_run(tasks, tmp_path):
    run_all(tasks, tmp_path, Fake(tasks[2]))

    def new_image(image, agent):
        return {**probe(image, agent), "image_id": "sha256:other"}

    with pytest.raises(ValueError, match="image id"):
        run_all(tasks, tmp_path, Fake(tasks[2]), probe=new_image)


def test_the_other_arms_records_do_not_block_a_different_setup(tasks, tmp_path):
    run_all(tasks, tmp_path, Fake(tasks[2]))
    labels = dict(LABELS, **{"org.oumi.weights-lab.arm": "control"})
    records = run_all(
        tasks,
        tmp_path,
        Fake(tasks[2]),
        arm="control",
        probe=lambda i, a: {"labels": labels, "image_id": "sha256:control"},
    )
    assert len(records) == len(PLANTED_FAMILIES)


def test_every_documented_agent_command_places_the_prompt_before_variadic_flags(tmp_path):
    texts = [
        (ROOT / "docs" / "proving-ground.md").read_text(),
        (ROOT / "tests" / "live" / "validate_proving_ground.py").read_text(),
    ]
    found = [m for text in texts for m in re.findall(r'--agent-cmd "([^"]*)"', text)]
    assert len(found) == 2
    for number, command in enumerate(found):
        words = shlex.split(command)
        assert "{prompt}" in words
        if "--disallowedTools" in words:
            assert words.index("{prompt}") < words.index("--disallowedTools")
        (argv, _), _, _ = argv_for(tmp_path / str(number), command=words)
        assert argv[-len(words) :].count("PROMPT TEXT") == 1
