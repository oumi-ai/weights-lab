"""Exercise released-edit replay mechanics using synthetic local BF16 weights.

Synthetic hashes replace the published hashes only inside these tests. No model
checkpoint or network download is needed, and these tests make no score claim.
"""

import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from weights_lab import Lab


def raw_hash(tensor):
    # Independent reference: hash the BF16 storage without a float conversion.
    return hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy())).hexdigest()


@pytest.fixture
def replay(monkeypatch):
    # The examples namespace is source-only; pytest's console entry point does
    # not necessarily put the repository root on sys.path.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.replay_qwen_edit")


@pytest.fixture
def local_release(tmp_path, monkeypatch, replay):
    base, edited, targets = {}, {}, {}
    for number, name in enumerate(replay.TARGETS):
        original = torch.full((1024, 3584), float(number), dtype=torch.bfloat16)
        if number == 0:
            original[0, 0] = -0.0
        replacement = original.clone()
        replacement[:, :64] = number + 1
        base[name], edited[name] = original, replacement
        targets[name] = {
            "base_sha256": raw_hash(original),
            "edited_sha256": raw_hash(replacement),
            "changed_entries": 1024 * 64,
        }
    base["unchanged"] = torch.tensor([7.0], dtype=torch.bfloat16)
    edited["unchanged"] = base["unchanged"].clone()
    base_path, edited_path = tmp_path / "base.safetensors", tmp_path / "edited.safetensors"
    save_file(base, base_path)
    save_file(edited, edited_path)
    index = dict.fromkeys(base, "fixture.safetensors")

    def checkpoint_files(model, revision):
        assert (model, revision) in {
            (replay.BASE_MODEL, replay.BASE_REVISION),
            (replay.EDITED_MODEL, replay.EDITED_REVISION),
        }
        path = base_path if model == replay.BASE_MODEL else edited_path
        return index.copy(), {"fixture.safetensors": str(path)}

    monkeypatch.setattr(replay, "TARGETS", targets)
    monkeypatch.setattr(replay, "_checkpoint_files", checkpoint_files)
    return base_path, edited_path


def test_local_bf16_release_audits_swaps_and_undoes_without_forwards(
    local_release, tmp_path, replay
):
    base_path, _ = local_release
    # Build the actual public parameter paths without allocating unused layers or
    # a second language model. The real Lab and EditBook perform the intervention.
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList(torch.nn.Module() for _ in range(20))
    with safe_open(base_path, framework="pt", device="cpu") as checkpoint:
        for name in replay.TARGETS:
            layer_number = int(name.split(".")[3])
            layer = model.model.language_model.layers[layer_number]
            layer.mlp = torch.nn.Module()
            layer.mlp.down_proj = torch.nn.Module()
            layer.mlp.down_proj.weight = torch.nn.Parameter(checkpoint.get_tensor(name).clone())
    lab = Lab(model, tokenizer=None, session_path=tmp_path / "session", pass_limit=0)
    (lab.session_path / "config.json").write_text(
        json.dumps({"model": replay.BASE_MODEL, "revision": replay.BASE_REVISION})
    )
    report = replay.replay_qwen_edit(lab)

    assert report["verification"]["audit_all"] is True
    assert report["verification"]["tensor_count_checked"] == 3
    assert set(report["verification"]["changed_tensors"]) == set(replay.TARGETS)
    assert report["restored_exactly"] is True
    assert report["forward_passes"] == lab.forward_passes == lab.budget.used == 0
    assert lab.active_edit_ids == []
    assert lab.revision == 4
    assert len(lab.edits()) == 2
    for edit in lab.edits():
        assert edit["active"] is False
        assert edit["shape"] == [1024, 64]  # Snapshot selected columns, not the full matrix.
    for name, expected in replay.TARGETS.items():
        parameter = lab.model.get_parameter(name)
        assert raw_hash(parameter.detach()) == expected["base_sha256"]
        actual = report["verification"]["tensors"][name]
        assert actual["edited_sha256"] == expected["edited_sha256"]
        assert actual["changed_entries"] == 1024 * 64
        assert actual["columns"] == list(range(64))
    output = lab.artifacts / "qwen-replay"
    persisted = json.loads((output / "report.json").read_text())
    assert persisted == report
    assert (
        json.loads((output / "verification.json").read_text())["verification"]
        == report["verification"]
    )


def test_full_release_audit_rejects_an_additional_changed_tensor(local_release, replay):
    _, edited_path = local_release
    with safe_open(edited_path, framework="pt", device="cpu") as checkpoint:
        tensors = {name: checkpoint.get_tensor(name).clone() for name in checkpoint.keys()}
    tensors["unchanged"].fill_(8)
    save_file(tensors, edited_path)
    with pytest.raises(ValueError, match="exactly.*changed tensors"):
        replay.verify_release()
    # The opt-in target-only mode remains explicitly distinguishable in evidence.
    _, report = replay.verify_release(audit_all=False)
    assert report["audit_all"] is False
    assert report["tensor_count_checked"] == 2


def test_unsharded_checkpoint_fallback_reads_names_without_network(tmp_path, monkeypatch, replay):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    path = tmp_path / "model.safetensors"
    save_file({"a": torch.ones(2, dtype=torch.bfloat16), "b": torch.zeros(1)}, path)
    calls = []

    def download(model, filename, *, revision):
        calls.append((model, filename, revision))
        if filename == "model.safetensors.index.json":
            raise EntryNotFoundError("Synthetic checkpoint has no shard index")
        assert filename == "model.safetensors"
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    index, files = replay._checkpoint_files("synthetic/unsharded", "pinned-revision")
    assert index == {"a": "model.safetensors", "b": "model.safetensors"}
    assert files == {"model.safetensors": str(path)}
    assert calls == [
        ("synthetic/unsharded", "model.safetensors.index.json", "pinned-revision"),
        ("synthetic/unsharded", "model.safetensors", "pinned-revision"),
    ]


def test_sharded_checkpoint_index_resolves_each_file_at_requested_revision(
    tmp_path, monkeypatch, replay
):
    import huggingface_hub

    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {"a": "part1", "b": "part2", "c": "part1"}}))
    calls = []

    def download(model, filename, *, revision):
        assert model == "synthetic/sharded"
        assert revision == "pinned-sharded"
        calls.append(filename)
        return str(index_path if filename.endswith(".json") else tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    index, files = replay._checkpoint_files("synthetic/sharded", "pinned-sharded")
    assert index == {"a": "part1", "b": "part2", "c": "part1"}
    assert files == {"part1": str(tmp_path / "part1"), "part2": str(tmp_path / "part2")}
    assert calls == ["model.safetensors.index.json", "part1", "part2"]
