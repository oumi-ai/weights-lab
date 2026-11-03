"""Reversible parameter edits and portable checkpoint exports.

Only edits made through this API enter the undo history. Direct tensor writes by
snippets are intentionally possible, but cannot be tracked or undone here.
"""

from __future__ import annotations

import copy
import fnmatch
import json
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_CHUNK = 262_144


def _regions(tensor: Any, index: tuple = ()):
    """Yield bounded views, including for non-contiguous parameters."""
    if tensor.numel() <= _CHUNK:
        yield index, tensor
        return
    dim = max(range(tensor.ndim), key=lambda d: tensor.shape[d])
    step = max(1, _CHUNK // (tensor.numel() // tensor.shape[dim]))
    for start in range(0, tensor.shape[dim], step):
        region = [slice(None)] * tensor.ndim
        region[dim] = slice(start, min(start + step, tensor.shape[dim]))
        yield from _regions(tensor[tuple(region)], (*index, tuple(region)))


def _stats(tensor: Any, other: Any = None) -> dict[str, Any]:
    import torch

    count = 0
    mean = moment = square = 0.0
    minimum, maximum = math.inf, -math.inf
    for indices, chunk in _regions(tensor.detach()):
        values = chunk.to(device="cpu", dtype=torch.float64)
        if other is not None:
            reference = other.detach()
            for index in indices:
                reference = reference[index]
            values = values - reference.to(device="cpu", dtype=torch.float64)
        finite = values[torch.isfinite(values)]
        if finite.numel():
            chunk_count = finite.numel()
            chunk_mean = finite.mean().item()
            chunk_moment = (finite - chunk_mean).square().sum().item()
            delta = chunk_mean - mean
            combined = count + chunk_count
            moment += chunk_moment + delta * delta * count * chunk_count / combined
            mean += delta * chunk_count / combined
            count = combined
            square += finite.square().sum().item()
            minimum = min(minimum, finite.min().item())
            maximum = max(maximum, finite.max().item())
    return {
        "numel": tensor.numel(),
        "finite_count": count,
        "nonfinite_count": tensor.numel() - count,
        "mean": mean if count else None,
        "std": math.sqrt(moment / count) if count else None,
        "min": minimum if count else None,
        "max": maximum if count else None,
        "l2": math.sqrt(square),
    }


def _parameter(model: Any, name: str) -> Any:
    try:
        return model.get_parameter(name)
    except (AttributeError, KeyError) as error:
        raise ValueError(f"Unknown parameter {name!r}") from error


class EditBook:
    """Exact CPU snapshots of edited slices, with strictly LIFO undo.

    Callables receive an isolated tensor and may return a replacement or mutate
    their argument and return None. Shapes must match exactly; broadcasting is
    rejected. Revisions increase after both successful edits and undo operations.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.revision = 0
        self._history: list[dict[str, Any]] = []
        self._active: list[dict[str, Any]] = []

    @property
    def active_edit_ids(self) -> list[str]:
        return [record["id"] for record in self._active]

    def parts(self, pattern: str = "*") -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "numel": value.numel(),
                "requires_grad": value.requires_grad,
            }
            for name, value in self.model.named_parameters(remove_duplicate=False)
            if fnmatch.fnmatchcase(name, pattern)
        ]

    def measure(self, name: str) -> dict[str, Any]:
        parameter = _parameter(self.model, name)
        return {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            **_stats(parameter),
        }

    def compare(self, other: Any, pattern: str = "*") -> list[dict[str, Any]]:
        results = []
        for part in self.parts(pattern):
            name = part["name"]
            parameter = _parameter(self.model, name)
            reference = other[name] if isinstance(other, Mapping) else _parameter(other, name)
            if parameter.shape != reference.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: {parameter.shape} != {reference.shape}"
                )
            results.append(
                {"name": name, "shape": list(parameter.shape), **_stats(parameter, reference)}
            )
        return results

    def edit(self, name: str, value: Any, index: Any = None, note: str = "") -> str:
        import torch

        parameter = _parameter(self.model, name)
        if not parameter.is_floating_point():
            raise ValueError("Only floating-point parameters can be edited")
        index = copy.deepcopy(index)
        selected = parameter if index is None else parameter[index]
        original = selected.detach().to(device="cpu", copy=True)
        candidate = selected.detach().clone()
        # The callable never sees model storage, including for ordinary slices.
        with torch.no_grad():
            replacement = value(candidate) if callable(value) else value
            if replacement is None:
                if not callable(value):
                    raise ValueError("An edit value is required")
                replacement = candidate
            replacement = torch.as_tensor(
                replacement, device=parameter.device, dtype=parameter.dtype
            )
            if replacement.shape != original.shape:
                raise ValueError(f"Edit shape {replacement.shape} does not match {original.shape}")
        previous_revision = self.revision
        history_size, active_size = len(self._history), len(self._active)
        record = {
            "id": f"edit-{len(self._history) + 1:06d}",
            "name": name,
            "index": repr(index),
            "shape": list(original.shape),
            "dtype": str(original.dtype),
            "note": note,
            "revision": previous_revision + 1,
            "active": True,
            "_index": index,
            "_original": original,
            "_parameter": parameter,
        }
        try:
            with torch.no_grad():
                if index is None:
                    parameter.copy_(replacement)
                else:
                    parameter[index] = replacement
            self._history.append(record)
            self._active.append(record)
            self.revision = previous_revision + 1
        except BaseException:
            with torch.no_grad():
                if index is None:
                    parameter.copy_(original.to(parameter.device))
                else:
                    parameter[index] = original.to(parameter.device)
            del self._history[history_size:]
            del self._active[active_size:]
            self.revision = previous_revision
            raise
        return record["id"]

    def zero(self, name: str, index: Any = None, note: str = "") -> str:
        return self.edit(name, lambda value: value.zero_(), index=index, note=note)

    def swap(
        self,
        name: str,
        source: Any,
        source_name: str | None = None,
        index: Any = None,
        note: str = "",
    ) -> str:
        import torch

        source_name = source_name or name
        if not isinstance(source, torch.Tensor):
            source = (
                source[source_name]
                if isinstance(source, Mapping)
                else _parameter(source, source_name)
            )
        target = _parameter(self.model, name)
        if source.shape != target.shape:
            raise ValueError(f"Swap shape {source.shape} does not match {target.shape}")
        value = source if index is None else source[index]
        return self.edit(name, value, index=index, note=note)

    def undo(self, edit_id: str | None = None) -> str:
        import torch

        if not self._active:
            raise ValueError("There are no active edits")
        record = self._active[-1]
        if edit_id is not None and edit_id != record["id"]:
            raise ValueError("Undo must follow reverse edit order (LIFO)")
        parameter = _parameter(self.model, record["name"])
        original = record["_original"]
        index = record["_index"]
        selected = parameter if index is None else parameter[index]
        if (
            parameter is not record["_parameter"]
            or parameter.dtype != original.dtype
            or selected.shape != original.shape
        ):
            raise RuntimeError("Parameter storage, dtype, or shape changed since the edit")
        current = selected.detach().to(device="cpu", copy=True)
        record_before, active_before, previous_revision = (
            record.copy(),
            self._active.copy(),
            self.revision,
        )
        try:
            with torch.no_grad():
                if index is None:
                    parameter.copy_(original.to(parameter.device))
                else:
                    parameter[index] = original.to(parameter.device)
            self._active.pop()
            record["active"] = False
            self.revision = previous_revision + 1
            record["undone_revision"] = self.revision
            record.pop("_original")
            record.pop("_parameter")
            record.pop("_index")
        except BaseException:
            with torch.no_grad():
                if index is None:
                    parameter.copy_(current.to(parameter.device))
                else:
                    parameter[index] = current.to(parameter.device)
            self._active[:] = active_before
            record.clear()
            record.update(record_before)
            self.revision = previous_revision
            raise
        return record["id"]

    def edits(self) -> list[dict[str, Any]]:
        return [
            {key: copy.deepcopy(value) for key, value in record.items() if not key.startswith("_")}
            for record in self._history
        ]


class EditingMixin:
    """Lab methods for parameter inspection, reversible edits, and export."""

    @property
    def editbook(self) -> EditBook:
        if not hasattr(self, "_editbook"):
            self._editbook = EditBook(self.model)
        return self._editbook

    @property
    def revision(self) -> int:
        return self.editbook.revision

    @property
    def active_edit_ids(self) -> list[str]:
        return self.editbook.active_edit_ids

    def parts(self, pattern: str = "*") -> list[dict[str, Any]]:
        return self.editbook.parts(pattern)

    def measure(self, name: str) -> dict[str, Any]:
        return self.editbook.measure(name)

    def compare(self, other: Any, pattern: str = "*") -> list[dict[str, Any]]:
        return self.editbook.compare(other, pattern)

    def edit(self, name: str, value: Any, index: Any = None, note: str = "") -> str:
        return self.editbook.edit(name, value, index, note)

    def zero(self, name: str, index: Any = None, note: str = "") -> str:
        return self.editbook.zero(name, index, note)

    def swap(
        self,
        name: str,
        source: Any,
        source_name: str | None = None,
        index: Any = None,
        note: str = "",
    ) -> str:
        return self.editbook.swap(name, source, source_name, index, note)

    def undo(self, edit_id: str | None = None) -> str:
        return self.editbook.undo(edit_id)

    def edits(self) -> list[dict[str, Any]]:
        return self.editbook.edits()

    def export(self, path: str | Path) -> Path:
        """Save a standalone HF checkpoint; relative paths use the artifacts dir.

        PEFT exports clone and merge the complete current model, preserving base
        weight edits as well as adapters. This temporarily needs a second model's
        memory. Unsupported merges fail without changing the live model or output.
        """
        attachments = getattr(self, "attachments", None)
        if callable(attachments) and attachments():
            raise ValueError("Detach active attachments before exporting a standalone checkpoint")
        patches = getattr(self, "patches", None)
        if callable(patches) and patches():
            raise ValueError("Remove active patches before exporting a standalone checkpoint")
        destination = Path(path).expanduser()
        if not destination.is_absolute():
            destination = self.artifacts / destination
        if destination.is_symlink() or (
            destination.exists() and (not destination.is_dir() or any(destination.iterdir()))
        ):
            raise ValueError("Export destination must be absent or an empty directory")
        model = self.model
        merged_adapter = hasattr(model, "peft_config")
        if merged_adapter:
            if not callable(getattr(model, "merge_and_unload", None)):
                raise ValueError("This adapter cannot be exported as a full merged model")
            # Bound hook callbacks can refer back to the Lab and its undo history.
            # Preserve callbacks during cloning rather than cloning that workspace.
            memo = {}
            for module in model.modules():
                for attribute in ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks"):
                    memo.update({id(hook): hook for hook in getattr(module, attribute).values()})
            model = copy.deepcopy(model, memo).merge_and_unload(safe_merge=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".weights-lab-export-", dir=destination.parent
        ) as temp:
            staged = Path(temp) / "checkpoint"
            model.save_pretrained(staged)
            self.tokenizer.save_pretrained(staged)
            manifest = {
                "format_version": 1,
                "model_id": getattr(self, "model_id", None),
                "model_commit": getattr(self, "model_commit", None),
                "revision": self.revision,
                "active_edit_ids": self.active_edit_ids,
                "edits": self.edits(),
                "merged_adapter": merged_adapter,
            }
            (staged / "weights_lab_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            if destination.exists():
                destination.rmdir()  # Refuse a destination populated during export.
            staged.rename(destination)
        return destination
