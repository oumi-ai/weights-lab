"""Portable exact edit snapshots, without pickled Python objects or code replay."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .budget import PassBudget
from .editing import EditBook, _regions
from .session import read_json, session_lock, write_json


def _digest(value: Any) -> str:
    import torch

    digest = hashlib.sha256()
    digest.update(str((str(value.dtype), tuple(value.shape))).encode())
    for _, chunk in _regions(value.detach()):
        data = chunk.to(device="cpu").contiguous().reshape(-1).view(torch.uint8)
        # Hash the buffer in place; hashlib releases the GIL for large updates.
        digest.update(data.numpy())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _interleaved(targets: dict[str, Any]) -> list[str]:
    """Names ordered round-robin across devices, so parallel hashing reads every GPU at once."""
    groups: dict[str, list[str]] = {}
    for name, value in targets.items():
        groups.setdefault(str(value.device), []).append(name)
    rounds = itertools.zip_longest(*groups.values())
    return [name for names in rounds for name in names if name is not None]


def _fingerprint(lab: Any, originals: dict | None = None) -> str:
    originals = originals or {}
    state = lab.model.state_dict(keep_vars=True)
    storage = {}
    for value in state.values():
        if value.numel():
            key = (str(value.device), value.untyped_storage().data_ptr())
            if key in storage and storage[key] is not value:
                raise ValueError("distinct tensors sharing storage cannot be saved as edit bundles")
            storage[key] = value
    targets = {name: originals.get(id(value), value) for name, value in state.items()}
    # Hashing every tensor dominates a deadline save on large models, so hash in
    # parallel, taking tensors from each device in turn so copies from every GPU
    # overlap. MPS is not safe to drive from several threads; hash it serially.
    serial = any(value.device.type == "mps" for value in targets.values())
    names = _interleaved(targets)
    with ThreadPoolExecutor(max_workers=1 if serial else min(32, os.cpu_count() or 1)) as pool:
        digests = pool.map(_digest, (targets[name] for name in names))
        tensors = dict(zip(names, digests, strict=True))
    # Tied tensors are part of the model contract, not just equal-valued weights.
    aliases: dict[int, list[str]] = {}
    for name, value in lab.model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(value), []).append(name)
    config = getattr(lab.model, "config", None)
    config = config.to_dict() if hasattr(config, "to_dict") else {}
    for key in ("_name_or_path", "transformers_version", "_commit_hash"):
        config.pop(key, None)
    tokenizer = lab.tokenizer
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_state = json.loads(backend.to_str()) if backend is not None else None
    if backend_state is not None:
        for key in ("padding", "truncation"):
            backend_state.pop(key, None)
    identity = {
        "tensors": tensors,
        "aliases": sorted(sorted(names) for names in aliases.values()),
        "config": config,
        "model_class": f"{type(lab.model).__module__}.{type(lab.model).__qualname__}",
        "adapters": {
            name: value.to_dict() for name, value in getattr(lab.model, "peft_config", {}).items()
        },
        "scaling": {
            name: {
                "scaling": dict(module.scaling),
                "active": list(getattr(module, "active_adapters", [])),
                "disabled": bool(getattr(module, "disable_adapters", False)),
                "merged": list(getattr(module, "merged_adapters", [])),
            }
            for name, module in lab.model.named_modules()
            if hasattr(module, "lora_A") and hasattr(module, "scaling")
        },
        "tokenizer": {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "vocab": vocab,
            "backend": backend_state,
            "special_tokens": str(getattr(tokenizer, "special_tokens_map", {})),
            "chat_template": getattr(tokenizer, "chat_template", None),
        },
    }

    def encode(value):
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"unsupported model identity value: {type(value).__name__}")

    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=encode).encode()).hexdigest()


def _encode_index(value: Any, tensors: dict[str, Any]) -> Any:
    import torch

    if value is None or isinstance(value, (bool, int)):
        return value
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if isinstance(value, slice):
        return {"kind": "slice", "parts": [value.start, value.stop, value.step]}
    if isinstance(value, (tuple, list)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "parts": [_encode_index(part, tensors) for part in value],
        }
    if isinstance(value, torch.Tensor):
        name = f"index-{len(tensors)}"
        tensors[name] = value.detach().cpu().contiguous().clone()
        return {"kind": "tensor", "name": name}
    raise ValueError(f"unsupported portable edit index: {type(value).__name__}")


def _decode_index(value: Any, tensors: dict[str, Any]) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if not isinstance(value, dict):
        raise ValueError("invalid edit index")
    kind = value.get("kind")
    if kind == "ellipsis":
        return Ellipsis
    if kind == "slice":
        parts = value["parts"]
        if len(parts) != 3 or any(p is not None and type(p) is not int for p in parts):
            raise ValueError("invalid slice index")
        return slice(*parts)
    if kind in {"tuple", "list"}:
        parts = [_decode_index(part, tensors) for part in value["parts"]]
        return tuple(parts) if kind == "tuple" else parts
    if kind == "tensor":
        return tensors[value["name"]].clone()
    raise ValueError("invalid edit index kind")


def _inactive_components(lab: Any) -> None:
    for method in ("attachments", "patches"):
        if getattr(lab, method)():
            raise ValueError("Remove active attachments and patches before saving or loading edits")
    if getattr(lab, "_evaluation_budgets", []):
        raise ValueError("Save or load edits outside an evaluation or pass-limit context")


class PersistenceMixin:
    def _bundle_path(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        return path if path.is_absolute() else self.artifacts / path

    def save_edits(self, path: str | Path) -> Path:
        """Save exact active edits, undo history, budgets and dataset reservations.

        Only API edits are portable. Arbitrary Python variables, hooks and direct
        tensor writes are not replayed. No model forward calls are needed.
        """
        from safetensors.torch import save_file

        _inactive_components(self)
        destination = self._bundle_path(path)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        book = self.editbook
        tensors, working, active = {}, {}, []
        for record in reversed(book._active):
            parameter = record["_parameter"]
            encoded_index = _encode_index(record["_index"], tensors)
            index = _decode_index(encoded_index, tensors)
            if self.model.get_parameter(record["name"]) is not parameter:
                raise ValueError("parameter storage changed since the edit")
            if id(parameter) not in working:
                working[id(parameter)] = parameter.detach().cpu().clone()
            current = working[id(parameter)]
            selected = current if index is None else current[index]
            original = record["_original"]
            if selected.shape != original.shape or selected.dtype != original.dtype:
                raise ValueError("parameter dtype or shape changed since the edit")
            key = record["id"]
            tensors[f"{key}-after"] = selected.contiguous().clone()
            tensors[f"{key}-before"] = original.contiguous().clone()
            active.append({"id": key, "index": encoded_index})
            if index is None:
                current.copy_(original)
            else:
                current[index] = original
        registry_path = self.artifacts / "evaluations" / "_registry.json"
        with session_lock(self.session_path):
            registry = read_json(registry_path) if registry_path.exists() else None
            manifest = {
                "format_version": 1,
                "base_sha256": _fingerprint(self, working),
                "model_id": self.model_id,
                "model_commit": getattr(self, "model_commit", None),
                "book_revision": book.revision,
                "component_revision": self.revision - book.revision,
                "history": book.edits(),
                "active": list(reversed(active)),
                "forward_passes": self.forward_passes,
                "pass_limit": self.budget.limit,
                "registry": registry,
            }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".oumi-weights-lab-edits-", dir=destination.parent
        ) as tmp:
            staged = Path(tmp) / "bundle"
            staged.mkdir()
            save_file(tensors, str(staged / "edits.safetensors"))
            manifest["payload_sha256"] = _file_digest(staged / "edits.safetensors")
            manifest["sha256"] = _manifest_digest(manifest)
            write_json(staged / "manifest.json", manifest)
            staged.rename(destination)
        return destination

    def load_edits(self, path: str | Path) -> None:
        """Resume a saved bundle into a fresh, identical model and experiment.

        A new session directory is required. The inherited pass limit may be
        tightened, but cannot be raised. Evaluation reservations are carried on.
        """
        from safetensors.torch import load_file

        _inactive_components(self)
        registry_path = self.artifacts / "evaluations" / "_registry.json"
        if self.forward_passes or self.editbook.edits() or registry_path.exists():
            raise ValueError("load_edits requires a fresh model and experiment")
        bundle = self._bundle_path(path)
        manifest = read_json(bundle / "manifest.json")
        checksum = manifest.pop("sha256", None)
        if _manifest_digest(manifest) != checksum:
            raise ValueError("edit bundle manifest checksum mismatch")
        if manifest.get("format_version") != 1:
            raise ValueError("unsupported edit bundle format")
        for key in ("book_revision", "component_revision"):
            PassBudget._validate(manifest[key], key)
        if _file_digest(bundle / "edits.safetensors") != manifest["payload_sha256"]:
            raise ValueError("edit bundle payload checksum mismatch")
        if _fingerprint(self) != manifest["base_sha256"]:
            raise ValueError(
                "base model, tokenizer or checkpoint content does not match edit bundle"
            )
        used, inherited = manifest["forward_passes"], manifest["pass_limit"]
        PassBudget._validate(used, "saved forward passes")
        PassBudget(inherited)
        requested = self.budget.limit
        if inherited is not None and requested is not None and requested > inherited:
            raise ValueError("resume cannot raise the saved pass limit")
        budget = PassBudget(inherited if requested is None else requested)
        budget.consume(used)
        tensors = load_file(str(bundle / "edits.safetensors"))
        history = copy.deepcopy(manifest["history"])
        by_id = {record["id"]: record for record in history}
        if len(by_id) != len(history):
            raise ValueError("duplicate edit IDs")
        if list(by_id) != [f"edit-{i + 1:06d}" for i in range(len(history))]:
            raise ValueError("invalid edit history order")
        book = EditBook(self.model)
        book._history = history
        book.revision = manifest["book_revision"]
        working, originals = {}, {}
        for entry in manifest["active"]:
            record = by_id[entry["id"]]
            parameter = self.model.get_parameter(record["name"])
            index = _decode_index(entry["index"], tensors)
            if id(parameter) not in working:
                originals[id(parameter)] = (parameter, parameter.detach().cpu().clone())
                working[id(parameter)] = originals[id(parameter)][1].clone()
            current = working[id(parameter)]
            before, after = tensors[f"{record['id']}-before"], tensors[f"{record['id']}-after"]
            selected = current if index is None else current[index]
            if _digest(selected) != _digest(before) or after.dtype != selected.dtype:
                raise ValueError("saved edit original does not match replay state")
            if after.shape != selected.shape:
                raise ValueError("saved edit shape mismatch")
            if index is None:
                current.copy_(after)
            else:
                current[index] = after
            record.update(_index=index, _original=before, _parameter=parameter)
            book._active.append(record)
        if book.active_edit_ids != [r["id"] for r in history if r["active"]]:
            raise ValueError("edit history and active edits disagree")
        import torch

        with session_lock(self.session_path):
            if registry_path.exists():
                raise ValueError("load_edits requires a fresh experiment")
            try:
                with torch.no_grad():
                    for key, value in working.items():
                        originals[key][0].copy_(value)
                if manifest["registry"] is not None:
                    registry_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(registry_path, manifest["registry"], durable=True)
            except BaseException:
                with torch.no_grad():
                    for parameter, original in originals.values():
                        parameter.copy_(original)
                registry_path.unlink(missing_ok=True)
                raise
            self._editbook = book
            self._budget = budget
            self._forward_passes = used
            # Keep Lab.revision monotonic across resumes without touching the
            # attachment and patch counters, which may already be nonzero here.
            self._resumed_component_revision = manifest["component_revision"]
