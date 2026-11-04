"""Replay the released Qwen edit inside a oumi-weights-lab worker.

Submit this file with oumi-weights-lab send --file. Optionally set a qwen_replay
configuration dictionary in an earlier snippet; see docs/qwen-replay.md.
This replays released tensor values, not the discovery or published benchmark.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
EDITED_MODEL = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
EDITED_REVISION = "d1671a178705bcab7644806d23c535517f35723b"
LOGS_REVISION = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
# Raw contiguous BF16 bytes, from the pinned upstream exp8 manifest.
TARGETS = {
    "model.language_model.layers.15.mlp.down_proj.weight": {
        "base_sha256": "ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca",
        "edited_sha256": "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
        "changed_entries": 65260,
    },
    "model.language_model.layers.19.mlp.down_proj.weight": {
        "base_sha256": "638eedeb701b94aa861f4438e615c07ec6c60ee703db1bfee2de577bb6498c3e",
        "edited_sha256": "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2",
        "changed_entries": 64928,
    },
}


def raw_tensor_sha256(tensor: Any) -> str:
    """Hash actual dtype bits, including BF16, without converting to float32."""
    import torch

    array = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    raw = memoryview(array).cast("B")
    digest = hashlib.sha256()
    for offset in range(0, len(raw), 1024 * 1024):
        digest.update(raw[offset : offset + 1024 * 1024])
    return digest.hexdigest()


def _checkpoint_files(model: str, revision: str) -> tuple[dict, dict]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        index_path = hf_hub_download(model, "model.safetensors.index.json", revision=revision)
    except EntryNotFoundError:
        from safetensors import safe_open

        path = hf_hub_download(model, "model.safetensors", revision=revision)
        with safe_open(path, framework="pt", device="cpu") as tensors:
            index = {name: "model.safetensors" for name in tensors.keys()}
        return index, {"model.safetensors": path}
    index = json.loads(Path(index_path).read_text())["weight_map"]
    files = {
        filename: hf_hub_download(model, filename, revision=revision)
        for filename in sorted(set(index.values()))
    }
    return index, files


def verify_release(audit_all: bool = True) -> tuple[dict, dict]:
    """Download pinned safetensors and check released bytes; never load model code."""
    import torch
    from safetensors import safe_open

    base_index, base_files = _checkpoint_files(BASE_MODEL, BASE_REVISION)
    edited_index, edited_files = _checkpoint_files(EDITED_MODEL, EDITED_REVISION)
    if set(base_index) != set(edited_index):
        raise ValueError("Pinned checkpoints contain different tensor names")
    names = sorted(base_index if audit_all else TARGETS)
    changed, replacements, details = [], {}, {}
    with ExitStack() as stack:
        base = {
            name: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for name, path in base_files.items()
        }
        edited = {
            name: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for name, path in edited_files.items()
        }
        for name in names:
            original = base[base_index[name]].get_tensor(name)
            replacement = edited[edited_index[name]].get_tensor(name)
            if original.shape != replacement.shape or original.dtype != replacement.dtype:
                raise ValueError(f"Checkpoint shape or dtype differs for {name}")
            original_hash = raw_tensor_sha256(original)
            replacement_hash = raw_tensor_sha256(replacement)
            if original_hash != replacement_hash:
                changed.append(name)
            if name in TARGETS:
                expected = TARGETS[name]
                if (
                    original.dtype != torch.bfloat16
                    or list(original.shape) != [1024, 3584]
                    or original_hash != expected["base_sha256"]
                    or replacement_hash != expected["edited_sha256"]
                ):
                    raise ValueError(f"Published tensor verification failed: {name}")
                unequal = original.view(torch.int16) != replacement.view(torch.int16)
                columns = unequal.any(dim=0).nonzero().flatten()
                entries = int(unequal.sum().item())
                if len(columns) != 64 or entries != expected["changed_entries"]:
                    raise ValueError(f"Published changed-entry count differs: {name}")
                replacements[name] = replacement.clone()
                details[name] = {
                    "base_sha256": original_hash,
                    "edited_sha256": replacement_hash,
                    "shape": list(original.shape),
                    "dtype": str(original.dtype),
                    "changed_entries": entries,
                    "columns": columns.tolist(),
                }
    if set(changed) != set(TARGETS):
        raise ValueError(f"Expected exactly the published changed tensors; found {changed}")
    return replacements, {
        "audit_all": audit_all,
        "tensor_count_checked": len(names),
        "changed_tensors": changed,
        "tensors": details,
    }


def replay_qwen_edit(
    lab: Any,
    *,
    data: str | Path | list | None = None,
    checker: Any = None,
    max_passes: int | None = None,
    generation_config: dict | None = None,
    seed: int = 0,
    audit_all: bool = True,
    run_prefix: str = "qwen-replay",
) -> dict:
    """Verify, optionally score, swap the released slices, and restore exactly.

    Optional data is a dev set, never a held-out benchmark. A checker and explicit
    per-condition pass cap are required. Prompts get the pinned tokenizer's
    thinking-enabled chat template. Sampling here is a local diagnostic protocol.
    """
    import torch

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_prefix):
        raise ValueError("run_prefix must be a safe identifier of at most 80 characters")
    config = json.loads((Path(lab.session_path) / "config.json").read_text())
    if (
        config.get("model") != BASE_MODEL
        or config.get("revision") != BASE_REVISION
        or config.get("adapter")
        or lab.active_edit_ids
        or lab.attachments()
        or lab.patches()
    ):
        raise ValueError("Use a fresh, unedited session of the pinned base without adapters")
    tasks = None
    if data is not None:
        if not callable(checker) or max_passes is None:
            raise ValueError("Sampled evaluation requires a checker and explicit max_passes")
        tasks = (
            [json.loads(line) for line in Path(data).read_text().splitlines() if line.strip()]
            if isinstance(data, (str, Path))
            else data
        )
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("data must be a nonempty JSONL path or list")
        tasks = [
            {
                **row,
                "prompt": lab.tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                ),
            }
            for row in tasks
        ]
    output_dir = Path(lab.artifacts) / run_prefix
    output_dir.mkdir(parents=True, exist_ok=False)
    replacements, verification = verify_release(audit_all=audit_all)
    for name, expected in TARGETS.items():
        parameter = lab.model.get_parameter(name)
        if (
            parameter.dtype != torch.bfloat16
            or raw_tensor_sha256(parameter) != expected["base_sha256"]
        ):
            raise ValueError(f"Loaded model must contain the original BF16 target bytes: {name}")
    report = {
        "purpose": "Replay a released edit; no discovery or benchmark reproduction claim.",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "edited_model": EDITED_MODEL,
        "edited_revision": EDITED_REVISION,
        "logs_revision": LOGS_REVISION,
        "verification": verification,
        "seed": seed,
        "generation_config": generation_config
        or {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 128,
        },
        "max_passes_per_condition": max_passes,
    }
    (output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")

    def evaluate(condition):
        return lab.evaluate(
            tasks,
            name=f"{run_prefix}-{condition}",
            mode="sampled",
            checker=checker,
            max_passes=max_passes,
            generation_config=report["generation_config"],
            seed=seed,
        )

    before = lab.forward_passes
    if tasks is not None:
        report["baseline"] = evaluate("baseline")
    edit_ids = []
    try:
        for name, replacement in replacements.items():
            columns = verification["tensors"][name]["columns"]
            edit_ids.append(
                lab.swap(
                    name,
                    replacement,
                    index=(slice(None), columns),
                    note=f"Released Qwen edit {EDITED_REVISION}",
                )
            )
            if raw_tensor_sha256(lab.model.get_parameter(name)) != TARGETS[name]["edited_sha256"]:
                raise ValueError(f"Live edited tensor hash differs: {name}")
        report["edit_ids"] = edit_ids.copy()
        if tasks is not None:
            report["edited"] = evaluate("edited")
            report["changed_items"] = lab.diff(report["baseline"], report["edited"])
    finally:
        for edit_id in reversed(edit_ids):
            lab.undo(edit_id)
        report["restored_exactly"] = all(
            raw_tensor_sha256(lab.model.get_parameter(name)) == expected["base_sha256"]
            for name, expected in TARGETS.items()
        )
        report["forward_passes"] = lab.forward_passes - before
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["restored_exactly"]:
        raise RuntimeError("Undo did not restore the published base tensor hashes")
    if tasks is not None:
        report["restored"] = evaluate("restored")
        report["restored_changes"] = lab.diff(report["baseline"], report["restored"])
    report["forward_passes"] = lab.forward_passes - before
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__oumi_weights_lab__":
    print(json.dumps(replay_qwen_edit(lab, **globals().get("qwen_replay", {})), indent=2))
elif __name__ == "__main__":
    raise SystemExit("Submit this file with oumi-weights-lab send --file; see docs/qwen-replay.md")
