import json


def test_item_ids_are_saved_without_changing_dataset_identity(tiny_lab):
    rows = [{"id": "source-1", "prompt": "hello", "options": ["yes", "no"], "answer": 0}]
    first = tiny_lab.evaluate(rows, name="named")
    rows[0]["id"] = "renamed"
    second = tiny_lab.evaluate(rows, name="renamed")
    assert first["dataset_hash"] == second["dataset_hash"]
    assert first["dataset_identity"] == second["dataset_identity"]
    item = json.loads((tiny_lab.artifacts / "evaluations/named/items/000000.json").read_text())
    assert item["id"] == "source-1"
    assert tiny_lab.diff(first, second) == []
