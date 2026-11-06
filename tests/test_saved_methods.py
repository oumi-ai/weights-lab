"""Saved methods: save, list, load and run by name, through a Lab and through the worker."""

import json
import multiprocessing
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oumi_weights_lab.methods_store import Method, SavedMethodsMixin
from oumi_weights_lab.session import create_session, read_json, submit
from oumi_weights_lab.worker import Worker

LIMIT = 7


@pytest.fixture
def methods_dir(tmp_path, monkeypatch):
    path = tmp_path / "saved"
    monkeypatch.setenv("OUMI_WEIGHTS_LAB_METHODS", str(path))
    return path


class FakeLab(SavedMethodsMixin):
    """The part of a Lab that saved methods use, for tests that need no model."""

    model_id = "fake"

    def __init__(self, setup=None):
        self.setup_block = setup or {"model_id": "fake", "torch": "0"}

    def _setup(self):
        return dict(self.setup_block)


def double(lab, x):
    """Double a number."""
    return {"value": x * 2}


def test_save_list_load_and_call_round_trip(tiny_lab, methods_dir):
    meta = tiny_lab.save_method(double, note="first try")
    assert meta["name"] == "double" and meta["note"] == "first try"
    assert len(meta["version"]) == 12 and meta["signature"] == "(lab, x)"
    assert meta["function"] == "double" and meta["docstring"] == "Double a number."
    assert meta["setup"]["model_id"] == "test-model"
    from oumi_weights_lab import __version__

    assert meta["lab_version"] == __version__
    directory = methods_dir / "double" / meta["version"]
    assert sorted(p.name for p in directory.iterdir()) == [
        "meta.json",
        "method.py",
        "results.jsonl",
    ]
    assert json.loads((directory / "meta.json").read_text()) == meta
    (listed,) = tiny_lab.methods()
    assert listed["name"] == "double"
    (version,) = listed["versions"]
    assert version["version"] == meta["version"] and version["note"] == "first try"
    assert version["results"] == {
        "results": 0,
        "tasks": 0,
        "mean_score": None,
        "solve_rate": None,
        "splits": [],
    }
    method = tiny_lab.load_method("double")
    assert (method.name, method.version) == ("double", meta["version"])
    assert "def double(lab, x)" in method.source and method.meta == meta
    assert method(4) == {
        "value": 8,
        "saved_method": {"name": "double", "version": meta["version"]},
    }
    assert tiny_lab.run_method("double", 5)["value"] == 10


def test_version_is_stable_for_the_same_source_and_changes_with_it(tiny_lab, methods_dir):
    first = tiny_lab.save_method(double)
    again = tiny_lab.save_method(double, note="ignored")
    assert again == first

    def triple(lab, x):
        return x * 3

    renamed = tiny_lab.save_method(triple, name="double")
    assert renamed["version"] != first["version"]
    assert len(tiny_lab.methods()[0]["versions"]) == 2


def test_source_that_cannot_be_read_is_refused(tiny_lab, methods_dir):
    namespace = {}
    exec("def lost(lab):\n    return 1\n", namespace)
    with pytest.raises(ValueError, match="source"):
        tiny_lab.save_method(namespace["lost"])


def test_a_module_global_becomes_an_import_line(tiny_lab, methods_dir):
    def norm(lab, values):
        return float(torch.tensor(np.asarray(values, dtype="float32")).norm())

    meta = tiny_lab.save_method(norm)
    source = (methods_dir / "norm" / meta["version"] / "method.py").read_text()
    assert source.startswith("import numpy as np\nimport torch\n")
    assert tiny_lab.run_method("norm", [3, 4]) == 5.0


def test_refused_sources_name_the_offending_names(tiny_lab, methods_dir):
    helper = 3

    def uses_global(lab):
        return LIMIT

    def uses_closure(lab):
        return helper

    def wrong_first(model):
        return model

    def no_parameters():
        return 1

    with pytest.raises(ValueError, match="LIMIT") as error:
        tiny_lab.save_method(uses_global)
    assert "argument" in str(error.value) and "import inside" in str(error.value)
    with pytest.raises(ValueError, match="helper"):
        tiny_lab.save_method(uses_closure)
    with pytest.raises(ValueError, match="first parameter.*lab"):
        tiny_lab.save_method(wrong_first)
    with pytest.raises(ValueError, match="first parameter.*lab"):
        tiny_lab.save_method(no_parameters)
    with pytest.raises(ValueError, match="plain function"):
        tiny_lab.save_method(lambda lab: 1, name="anon")
    with pytest.raises(ValueError, match="name"):
        tiny_lab.save_method(double, name="Bad Name")
    assert not methods_dir.exists() or not any(methods_dir.iterdir())


def test_names_in_defaults_and_nested_code_are_checked(tiny_lab, methods_dir):
    def with_default(lab, k=LIMIT):
        return k

    def nested(lab):
        return (lambda: LIMIT)()

    for function in (with_default, nested):
        with pytest.raises(ValueError, match="LIMIT"):
            tiny_lab.save_method(function)


def test_a_dict_result_is_copied_and_a_saved_method_key_is_refused(tiny_lab):
    meta = {"name": "x", "version": "0" * 12}
    original = {"value": 1, "method": "the method's own key"}
    result = Method(tiny_lab, meta, "", lambda lab: original)()
    assert result is not original and "saved_method" not in original
    assert result["method"] == "the method's own key"
    assert result["saved_method"] == {"name": "x", "version": "0" * 12}
    with pytest.raises(ValueError, match="saved_method"):
        Method(tiny_lab, meta, "", lambda lab: {"saved_method": 1})()


def test_a_non_dict_result_comes_back_unchanged(tiny_lab, methods_dir):
    def plain(lab):
        return [1, 2]

    tiny_lab.save_method(plain)
    assert tiny_lab.run_method("plain") == [1, 2]


def test_load_picks_the_newest_version_unless_one_is_named(tiny_lab, methods_dir):
    def pick(lab):
        return 1

    first = tiny_lab.save_method(pick)

    def pick(lab):  # noqa: F811
        return 2

    time.sleep(0.01)
    second = tiny_lab.save_method(pick)
    assert tiny_lab.load_method("pick")() == 2
    assert tiny_lab.load_method("pick", first["version"])() == 1
    assert [v["version"] for v in tiny_lab.methods()[0]["versions"]] == [
        second["version"],
        first["version"],
    ]
    with pytest.raises(ValueError, match="no version"):
        tiny_lab.load_method("pick", "0" * 12)
    with pytest.raises(ValueError, match="no saved method"):
        tiny_lab.load_method("missing")


def test_a_changed_saved_source_is_refused_on_load(tiny_lab, methods_dir):
    meta = tiny_lab.save_method(double)
    path = methods_dir / "double" / meta["version"] / "method.py"
    path.write_text(path.read_text().replace("* 2", "* 5"))
    with pytest.raises(ValueError, match="changed"):
        tiny_lab.load_method("double")


def test_default_directory_is_methods_where_start_ran(tiny_lab, tmp_path, monkeypatch):
    monkeypatch.delenv("OUMI_WEIGHTS_LAB_METHODS", raising=False)
    monkeypatch.chdir(tmp_path)
    meta = tiny_lab.save_method(double)
    assert (tmp_path / "methods" / "double" / meta["version"] / "method.py").is_file()


def test_the_environment_variable_names_the_directory(tiny_lab, methods_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tiny_lab.save_method(double)
    assert (methods_dir / "double").is_dir() and not (tmp_path / "methods").exists()


# Through the worker: a function defined in one snippet is saved, a later snippet runs it.


def fake_loader(config, path):
    lab = FakeLab()
    return SimpleNamespace(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        forward_passes=0,
        save_method=lab.save_method,
        load_method=lab.load_method,
        run_method=lab.run_method,
        methods=lab.methods,
    )


def run_worker(path):
    raise SystemExit(Worker(path, loader=fake_loader).run())


def wait_for(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_a_method_defined_in_one_snippet_is_saved_and_run_by_a_later_one(tmp_path, methods_dir):
    path = tmp_path / "session"
    create_session(path, {"model": "fake", "snippet_timeout": 30, "poll_interval": 0.01})
    process = multiprocessing.get_context("spawn").Process(target=run_worker, args=(path,))
    process.start()
    try:
        wait_for(lambda: read_json(path / "status.json").get("state") == "ready")

        def finish(snippet_id):
            target = path / "results" / f"{snippet_id}.json"
            wait_for(target.exists)
            return read_json(target)

        first = submit(
            path,
            "import json\n\n"
            "def count_chars(lab, text):\n"
            '    """Count characters."""\n'
            "    return {'n': len(json.dumps(text))}\n\n"
            "print('defined')\n",
        )
        assert finish(first)["state"] == "succeeded"
        saved = submit(path, "lab.save_method(count_chars, note='from a snippet')\n")
        assert finish(saved)["state"] == "succeeded"
        later = submit(path, "print(lab.run_method('count_chars', 'abc'))\n")
        assert finish(later)["state"] == "succeeded"
        output = (path / "outputs" / f"{later}.log").read_text()
        assert (
            "'n': 5" in output and "'saved_method'" in output and "'name': 'count_chars'" in output
        )
        (version,) = (methods_dir / "count_chars").iterdir()
        assert "def count_chars" in (version / "method.py").read_text()
    finally:
        (path / "STOP").touch()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)


# Snippet sources: kept on disk, not in memory


def test_a_snippet_source_is_read_from_disk_and_not_kept_in_linecache(tmp_path, methods_dir):
    import inspect
    import linecache

    path = tmp_path / "session"
    create_session(path, {"model": "fake", "snippet_timeout": 30, "poll_interval": 0.01})
    worker = Worker(path, loader=fake_loader)
    worker.lab = SimpleNamespace(forward_passes=0)
    worker.namespace = {}
    snippet = submit(path, "def f(lab):\n    return 1\n")
    worker._execute(next((path / "inbox").glob("*.py")))
    completed = str(path / "completed" / f"{snippet}.py")
    assert read_json(path / "results" / f"{snippet}.json")["state"] == "succeeded"
    assert not [key for key in linecache.cache if snippet in key]
    assert inspect.getsource(worker.namespace["f"]) == "def f(lab):\n    return 1\n"
    linecache.cache.pop(completed, None)


def test_a_failing_snippet_traceback_shows_its_source_line(tmp_path, methods_dir):
    path = tmp_path / "session"
    create_session(path, {"model": "fake", "snippet_timeout": 30, "poll_interval": 0.01})
    process = multiprocessing.get_context("spawn").Process(target=run_worker, args=(path,))
    process.start()
    try:
        wait_for(lambda: read_json(path / "status.json").get("state") == "ready")
        snippet = submit(path, "x = 1\nraise ValueError('boom from a snippet')\n")
        target = path / "results" / f"{snippet}.json"
        wait_for(target.exists)
        assert read_json(target)["state"] == "failed"
        log = (path / "outputs" / f"{snippet}.log").read_text()
        assert "raise ValueError('boom from a snippet')" in log
    finally:
        (path / "STOP").touch()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)


# Setups


def test_saving_the_same_source_in_another_setup_adds_it_to_the_setups_list(methods_dir):
    first = FakeLab({"model_id": "a"}).save_method(double)
    assert first["setups"] == [{"model_id": "a"}] and first["setup"] == {"model_id": "a"}
    assert FakeLab({"model_id": "a"}).save_method(double) == first
    second = FakeLab({"model_id": "b"}).save_method(double)
    assert second["setups"] == [{"model_id": "a"}, {"model_id": "b"}]
    assert second["setup"] == {"model_id": "a"} and second["created_at"] == first["created_at"]
    assert FakeLab({"model_id": "b"}).save_method(double) == second
    stored = json.loads((methods_dir / "double" / first["version"] / "meta.json").read_text())
    assert stored == second


def test_a_lab_records_a_graded_result_with_its_setup(methods_dir):
    lab = FakeLab({"model_id": "a"})
    meta = lab.save_method(double)
    graded = {"task": "dev-0000000001", "split": "dev", "score": {"score": 0.5, "solved": False}}
    record = lab.record_method_result("double", meta["version"], graded)
    assert record["setup"] == {"model_id": "a"} and record["time"]
    path = methods_dir / "double" / meta["version"] / "results.jsonl"
    assert json.loads(path.read_text()) == record
    with pytest.raises(ValueError, match="no version"):
        lab.record_method_result("double", "0" * 12, graded)


# Names that are not module globals


def test_class_bodies_lambdas_and_comprehensions_are_not_refused(tiny_lab, methods_dir):
    def with_class(lab):
        class Holder:
            first = 1
            second = first + 1

            def get(self):
                return self.second

        return Holder().get()

    def with_lambda_default(lab, key=lambda v: v + 1):
        return key(1)

    def with_comprehension_default(lab, values=[i * 2 for i in range(3)]):  # noqa: B006
        return values

    def with_nested(lab):
        return sorted([x for x in (3, 1, 2)], key=lambda value: -value)

    for function, expected in (
        (with_class, 2),
        (with_lambda_default, 2),
        (with_comprehension_default, [0, 2, 4]),
        (with_nested, [3, 2, 1]),
    ):
        tiny_lab.save_method(function)
        assert tiny_lab.run_method(function.__name__) == expected


def test_true_global_refusals_survive_the_scope_analysis(tiny_lab, methods_dir):
    def in_class(lab):
        class Holder:
            value = LIMIT

        return Holder

    def in_comprehension_default(lab, values=[LIMIT for _ in range(2)]):  # noqa: B006
        return values

    def in_lambda_default(lab, key=lambda v: LIMIT):
        return key

    def writes_a_global(lab):
        global LIMIT
        LIMIT = 8

    for function in (in_class, in_comprehension_default, in_lambda_default, writes_a_global):
        with pytest.raises(ValueError, match="LIMIT"):
            tiny_lab.save_method(function)
