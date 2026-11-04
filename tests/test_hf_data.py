"""Offline tests: stand-in datasets and huggingface_hub modules replace the network."""

import sys
from types import SimpleNamespace

import pytest

EXAMPLES = [
    {"question": "hello", "label": "yes"},
    {"question": "hello world", "label": "no"},
    {"question": "world", "label": "yes"},
]


@pytest.fixture
def hub(monkeypatch):
    calls = {"info": [], "load": []}

    class HfApi:
        def dataset_info(self, path, revision=None):
            calls["info"].append((path, revision))
            return SimpleNamespace(sha="0123abcd")

    def load_dataset(path, name=None, split=None, revision=None):
        calls["load"].append((path, name, split, revision))
        return [dict(example) for example in EXAMPLES]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=HfApi))
    return calls


def to_row(example):
    return {
        "prompt": example["question"],
        "options": [" yes", " no"],
        "answer": " " + example["label"],
    }


def test_rows_are_pinned_to_a_commit_and_carry_their_provenance(tiny_lab, hub):
    rows = tiny_lab.hf_rows("org/qa", split="test", to_row=to_row, name="easy", revision="v1")
    assert hub["info"] == [("org/qa", "v1")]
    assert hub["load"] == [("org/qa", "easy", "test", "0123abcd")]
    assert [row["prompt"] for row in rows] == ["hello", "hello world", "world"]
    assert rows[1]["id"] == "org/qa/easy/test/1"
    assert rows[1]["metadata"]["hf"] == {
        "path": "org/qa",
        "name": "easy",
        "split": "test",
        "revision": "0123abcd",
        "index": 1,
    }


def test_limit_keeps_the_first_rows_and_ids_from_to_row_survive(tiny_lab, hub):
    def with_id(example):
        return {**to_row(example), "id": example["question"], "metadata": {"source": "qa"}}

    rows = tiny_lab.hf_rows("org/qa", split="train", to_row=with_id, limit=2)
    assert [row["id"] for row in rows] == ["hello", "hello world"]
    assert rows[0]["metadata"]["source"] == "qa"
    assert rows[0]["metadata"]["hf"]["name"] is None


def test_a_test_split_can_still_seed_a_dev_screen_and_evaluate(tiny_lab, hub):
    rows = tiny_lab.hf_rows("org/qa", split="test", to_row=to_row)
    tiny_lab.create_screen(rows, name="dev-screen", size=2)
    assert tiny_lab.evaluate(rows, name="hf")["total"] == 3


def test_local_dataset_directories_are_not_resolved_on_the_hub(tiny_lab, hub, tmp_path):
    rows = tiny_lab.hf_rows(str(tmp_path), split="train", to_row=to_row)
    assert hub["info"] == []
    assert rows[0]["metadata"]["hf"]["revision"] is None
    with pytest.raises(ValueError, match="revision"):
        tiny_lab.hf_rows(str(tmp_path), split="train", to_row=to_row, revision="v1")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"to_row": "not callable"}, "to_row"),
        ({"to_row": lambda example: ["not", "a", "dict"]}, "dict"),
        ({"to_row": lambda example: {**to_row(example), "metadata": "x"}}, "metadata"),
        ({"to_row": to_row, "limit": 0}, "limit"),
    ],
)
def test_invalid_arguments_and_rows_are_refused(tiny_lab, hub, kwargs, message):
    with pytest.raises(ValueError, match=message):
        tiny_lab.hf_rows("org/qa", split="test", **kwargs)


def test_a_missing_datasets_extra_names_the_install(tiny_lab, monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match=r"\[datasets\]"):
        tiny_lab.hf_rows("org/qa", split="test", to_row=to_row)


COMMIT = "a" * 40


def test_a_full_commit_hash_needs_no_hub_lookup(tiny_lab, hub):
    rows = tiny_lab.hf_rows("org/qa", split="test", to_row=to_row, revision=COMMIT)
    assert hub["info"] == []
    assert hub["load"] == [("org/qa", None, "test", COMMIT)]
    assert rows[0]["metadata"]["hf"]["revision"] == COMMIT


def test_offline_lookups_ask_for_a_commit(tiny_lab, hub, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ValueError, match="commit"):
        tiny_lab.hf_rows("org/qa", split="test", to_row=to_row, revision="main")
    assert hub["info"] == []
    rows = tiny_lab.hf_rows("org/qa", split="test", to_row=to_row, revision=COMMIT)
    assert len(rows) == 3


def test_local_directories_are_recorded_by_their_resolved_path(
    tiny_lab, hub, tmp_path, monkeypatch
):
    (tmp_path / "qa").mkdir()
    monkeypatch.chdir(tmp_path)
    rows = tiny_lab.hf_rows("qa", split="train", to_row=to_row)
    resolved = str((tmp_path / "qa").resolve())
    assert hub["load"][0][0] == resolved
    assert rows[0]["metadata"]["hf"]["path"] == resolved
    assert rows[0]["id"].startswith(resolved)


def test_a_file_path_is_not_a_local_dataset(tiny_lab, hub, tmp_path, monkeypatch):
    (tmp_path / "notes").write_text("not a dataset")
    monkeypatch.chdir(tmp_path)
    tiny_lab.hf_rows("notes", split="train", to_row=to_row)
    assert hub["info"] == [("notes", None)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"split": None}, "split"),
        ({"split": ""}, "split"),
        ({"to_row": lambda example: {**to_row(example), "metadata": {"hf": "mine"}}}, "hf"),
    ],
)
def test_split_and_reserved_provenance_are_checked(tiny_lab, hub, kwargs, message):
    options = {"split": "test", "to_row": to_row, **kwargs}
    with pytest.raises(ValueError, match=message):
        tiny_lab.hf_rows("org/qa", **options)
