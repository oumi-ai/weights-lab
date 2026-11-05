"""Reversible parameter edits and portable checkpoint exports.

Only edits made through this API enter the undo history. Direct tensor writes by
snippets are intentionally possible, but cannot be tracked or undone here.
"""

from __future__ import annotations

import copy
import fnmatch
import json
import math
import sys
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .quantized import QuantizedWeight, editable, has_quantized_layers, lookup, other_method

_CHUNK = 262_144
# Exports write shards of at most this size, which bounds the CPU memory each shard
# needs while it is written.
SHARD_SIZE = "5GB"


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
        # MPS has no float64: move to CPU before widening.
        values = chunk.cpu().to(torch.float64)
        if other is not None:
            reference = other.detach()
            for index in indices:
                reference = reference[index]
            values = values - reference.cpu().to(torch.float64)
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


def _outside_package() -> int:
    """The ``stacklevel`` of the first caller outside this package, seen from the caller of this."""
    frame, level = sys._getframe(1), 1
    while frame is not None and frame.f_globals.get("__name__", "").startswith(__package__):
        frame, level = frame.f_back, level + 1
    return level


def _cpu_index(index: Any) -> Any:
    """``index`` with any tensor moved to the CPU, where a quantized weight is edited."""
    import torch

    if isinstance(index, torch.Tensor):
        return index.cpu()
    if isinstance(index, (tuple, list)):
        return type(index)(_cpu_index(part) for part in index)
    return index


def _logical(model: Any, name: str, dtype: Any = None) -> Any:
    """A parameter, or a quantized weight dequantized to float32 on the CPU."""
    packed = lookup(model, name, dtype)
    return _parameter(model, name) if packed is None else packed.dequantize()


def _projection_axis(owner: Any, attribute: str, side: str, shape: Any = None) -> tuple[int, int]:
    """Return the axis a direction lives on and the length it must have.

    ``shape`` is the logical shape of the weight, which differs from ``owner.weight.shape``
    when the weight is packed.
    """
    import torch
    from transformers.pytorch_utils import Conv1D

    kind = f"{type(owner).__module__}.{type(owner).__qualname__}"
    if isinstance(owner, (torch.nn.Linear, Conv1D)):
        out_axis = 0 if isinstance(owner, torch.nn.Linear) else 1
        if attribute == "bias":
            if side != "output":
                raise ValueError("A bias has no input side; use side='output'")
            return 0, owner.bias.shape[0]
        if attribute == "weight":
            axis = out_axis if side == "output" else 1 - out_axis
            return axis, (shape if shape is not None else owner.weight.shape)[axis]
    elif isinstance(owner, torch.nn.Embedding) and attribute == "weight":
        if side != "output":
            raise ValueError("An embedding has no input side; use side='output'")
        return 1, owner.weight.shape[1]
    raise ValueError(
        f"project_out does not support {attribute!r} of {kind}; use lab.edit with your own callable"
    )


def _projection_index(index: Any, size: int) -> Any:
    """Check a selection along the axis that is not projected and return a list or slice."""
    if index is None:
        return slice(None)
    if isinstance(index, int) and not isinstance(index, bool):
        index = [index]
    if isinstance(index, slice):
        parts = (index.start, index.stop)
        if index.step not in (None, 1) or any(
            part is not None and (not isinstance(part, int) or isinstance(part, bool))
            for part in parts
        ):
            raise ValueError("A slice index needs integer bounds and a step of 1")
        if len(range(size)[index.start : index.stop]) == 0:
            raise ValueError(f"The slice selects nothing on an axis of length {size}")
        return slice(index.start, index.stop)
    if (
        isinstance(index, list)
        and index
        and all(isinstance(i, int) and not isinstance(i, bool) for i in index)
    ):
        if any(i < -size or i >= size for i in index):
            raise ValueError(f"An index entry is outside the axis of length {size}")
        if len({i % size for i in index}) != len(index):
            raise ValueError("An index list must not repeat entries")
        return list(index)
    raise ValueError("index must be None, an int, a slice with step 1, or a nonempty list of ints")


class EditBook:
    """Exact CPU snapshots of edited slices, with strictly LIFO undo.

    Callables receive an isolated tensor and may return a replacement or mutate
    their argument and return None. Shapes must match exactly; broadcasting is
    rejected. Revisions increase after both successful edits and undo operations.
    """

    def __init__(self, model: Any, quantization: dict[str, Any] | None = None) -> None:
        self.model = model
        self._quantization = quantization
        self.revision = 0
        self._history: list[dict[str, Any]] = []
        self._active: list[dict[str, Any]] = []

    @property
    def active_edit_ids(self) -> list[str]:
        return [record["id"] for record in self._active]

    def _packed(self, name: str, model: Any = None) -> Any:
        """The quantized weight ``name``, whose int8 dtype is the Lab's compute dtype."""
        dtype = self._compute_dtype() if model in (None, self.model) else None
        return lookup(self.model if model is None else model, name, dtype)

    def _compute_dtype(self) -> Any:
        """The Lab's compute dtype, or None for a bare model."""
        import torch

        if not self._quantization:
            return None
        return getattr(torch, self._quantization["compute_dtype"], None)

    def parts(self, pattern: str = "*") -> list[dict[str, Any]]:
        """List parameters; a quantized weight reports its logical shape, dtype and size.

        A quantized weight's entry also has a ``quantization`` block that says how it is packed.
        """
        found = []
        for name, value in self.model.named_parameters(remove_duplicate=False):
            if not fnmatch.fnmatchcase(name, pattern):
                continue
            entry = {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "numel": value.numel(),
                "requires_grad": value.requires_grad,
            }
            packed = self._packed(name)
            if packed is not None:
                entry.update(
                    shape=list(packed.shape),
                    dtype=str(packed.dtype),
                    numel=packed.shape.numel(),
                    quantization=packed.describe(),
                )
            found.append(entry)
        return found

    def measure(self, name: str) -> dict[str, Any]:
        """Statistics of one parameter; a quantized weight is measured after dequantizing."""
        parameter = _parameter(self.model, name)
        packed = self._packed(name)
        if packed is None:
            return {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                **_stats(parameter),
            }
        return {
            "name": name,
            "shape": list(packed.shape),
            "dtype": str(packed.dtype),
            **_stats(packed.dequantize()),
            "quantization": packed.describe(),
        }

    def compare(self, other: Any, pattern: str = "*") -> list[dict[str, Any]]:
        results = []
        for part in self.parts(pattern):
            name = part["name"]
            packed = self._packed(name)
            parameter = _parameter(self.model, name) if packed is None else packed.dequantize()
            reference = other[name] if isinstance(other, Mapping) else _logical(other, name)
            if parameter.shape != reference.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: {parameter.shape} != {reference.shape}"
                )
            results.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    **_stats(parameter, reference),
                    **({"quantization": part["quantization"]} if "quantization" in part else {}),
                }
            )
        return results

    def edit(self, name: str, value: Any, index: Any = None, note: str = "") -> str:
        import torch

        parameter = _parameter(self.model, name)
        packed = editable(self.model, name, self._compute_dtype())
        if packed is not None:
            return self._edit_quantized(packed, value, index, note)
        if not parameter.is_floating_point():
            method = other_method(self.model)
            raise ValueError(
                "Only floating-point parameters can be edited"
                + (f"; {name} is stored by {method} quantization" if method else "")
            )
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

    def _edit_quantized(self, weight: QuantizedWeight, value: Any, index: Any, note: str) -> str:
        """Edit a packed weight: change it in float32, then requantize only the blocks it moved."""
        import torch

        index = _cpu_index(copy.deepcopy(index))
        parameter = weight.parameter
        state = weight.state
        snapshot = weight.snapshot()
        previous_revision = self.revision
        history_size, active_size = len(self._history), len(self._active)
        try:
            with torch.no_grad():
                before = weight.dequantize()
                selected = before if index is None else before[index]
                # The callable never sees the dequantized copy itself.
                candidate_part = selected.detach().to(
                    device=parameter.device, dtype=weight.dtype, copy=True
                )
                replacement = value(candidate_part) if callable(value) else value
                if replacement is None:
                    if not callable(value):
                        raise ValueError("An edit value is required")
                    replacement = candidate_part
                replacement = torch.as_tensor(
                    replacement, device=parameter.device, dtype=weight.dtype
                )
                if replacement.shape != selected.shape:
                    raise ValueError(
                        f"Edit shape {replacement.shape} does not match {selected.shape}"
                    )
                candidate = before.clone()
                inside = torch.zeros(before.shape, dtype=torch.bool)
                if index is None:
                    candidate.copy_(replacement.cpu().float())
                    inside[...] = True
                else:
                    candidate[index] = replacement.cpu().float()
                    inside[index] = True
                written = weight.write(candidate, before)
                landed = weight.dequantize()
            changed = landed != before

            def norm(difference: Any) -> float:
                return float(torch.linalg.vector_norm(difference))

            quantization = {
                "type": weight.method,
                "blocksize": weight.blocksize,
                "blocks_requantized": written["blocks_requantized"],
                "requested_delta_l2": norm((candidate - before)[inside]),
                "realized_delta_l2": norm((landed - before)[inside]),
                "residual_l2": norm((landed - candidate)[inside]),
                "collateral_l2": norm((landed - before)[~inside]),
                "changed_elements": int(changed.sum()),
            }
            record = {
                "id": f"edit-{len(self._history) + 1:06d}",
                "name": weight.name,
                "index": repr(index),
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
                "note": note,
                "revision": previous_revision + 1,
                "active": True,
                "quantization": quantization,
                "_index": index,
                "_snapshot": snapshot,
                "_parameter": parameter,
                "_state": state,
            }
            if quantization["requested_delta_l2"] > 0 and quantization["realized_delta_l2"] == 0:
                # Inside the try: a warning raised as an error rolls the edit back.
                warnings.warn(
                    f"{record['id']}: the change to {weight.name} was smaller than the "
                    "quantization step, so no stored value changed; the edit is recorded and "
                    "undo still works",
                    UserWarning,
                    stacklevel=_outside_package(),
                )
            self._history.append(record)
            self._active.append(record)
            self.revision = previous_revision + 1
        except BaseException:
            weight.restore(snapshot)
            del self._history[history_size:]
            del self._active[active_size:]
            self.revision = previous_revision
            raise
        return record["id"]

    def zero(self, name: str, index: Any = None, note: str = "") -> str:
        return self.edit(name, lambda value: value.zero_(), index=index, note=note)

    def project_out(
        self,
        name: str,
        direction: Any,
        *,
        side: str = "output",
        index: Any = None,
        scale: float = 1.0,
        note: str = "",
    ) -> str:
        """Remove a direction from one weight or bias with a single reversible edit.

        `side="output"` removes it from what the layer writes; `side="input"` removes
        it from what the layer reads. `index` picks entries along the other axis.
        Scale 1.0 removes the direction and 2.0 reflects across it. Supports the
        weight and bias of `Linear` and `Conv1D` and the weight of `Embedding`; refuses
        anything else, a wrong-length, zero, or non-finite direction, a bias index, and
        an embedding input side.
        """
        import torch

        if side not in ("output", "input"):
            raise ValueError("side must be 'output' or 'input'")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
        ):
            raise ValueError("scale must be a finite int or float")
        parameter = _parameter(self.model, name)
        packed = editable(self.model, name, self._compute_dtype())
        shape = parameter.shape if packed is None else packed.shape
        owner_name, _, attribute = name.rpartition(".")
        owner = self.model.get_submodule(owner_name)
        axis, length = _projection_axis(owner, attribute, side, shape)
        vector = torch.as_tensor(direction)
        if vector.ndim != 1 or vector.shape[0] != length:
            raise ValueError(f"direction must be 1-D with length {length}")
        vector = vector.detach().to(torch.promote_types(vector.dtype, torch.float32))
        if not bool(torch.isfinite(vector).all()):
            raise ValueError("direction must be finite")
        largest = vector.abs().max()
        if largest == 0:
            raise ValueError("direction must not be zero")
        # Scaling by the largest entry first keeps huge or tiny vectors from
        # overflowing or underflowing in the norm.
        vector = vector / largest
        unit = vector / vector.norm()
        if len(shape) == 1:
            if index is not None:
                raise ValueError("A bias takes no index")
            full = None
        else:
            other = shape[1 - axis]
            selected = _projection_index(index, other)
            full = (slice(None), selected) if axis == 0 else (selected, slice(None))

        def project(block: Any) -> Any:
            work = block.to(torch.promote_types(block.dtype, torch.float32))
            direction_ = unit.to(work.device, work.dtype)
            if work.ndim == 1:
                work = work - scale * direction_ * (direction_ @ work)
            elif axis == 0:
                work = work - scale * torch.outer(direction_, direction_ @ work)
            else:
                work = work - scale * torch.outer(work @ direction_, direction_)
            return work.to(block.dtype)

        return self.edit(
            name,
            project,
            index=full,
            note=note or f"project_out side={side} scale={scale}",
        )

    def swap(
        self,
        name: str,
        source: Any,
        source_name: str | None = None,
        index: Any = None,
        note: str = "",
    ) -> str:
        import torch

        packed = editable(self.model, name, self._compute_dtype())
        source_name = source_name or name
        if not isinstance(source, torch.Tensor):
            if isinstance(source, Mapping):
                source = source[source_name]
            else:
                # A quantized source stands for its dequantized float values.
                held = lookup(source, source_name)
                source = _parameter(source, source_name) if held is None else held.dequantize()
        shape = _parameter(self.model, name).shape if packed is None else packed.shape
        if source.shape != shape:
            raise ValueError(f"Swap shape {source.shape} does not match {shape}")
        value = source if index is None else source[index]
        return self.edit(name, value, index=index, note=note)

    def undo(self, edit_id: str | None = None) -> str:
        import torch

        if not self._active:
            raise ValueError("There are no active edits")
        record = self._active[-1]
        if edit_id is not None and edit_id != record["id"]:
            raise ValueError("Undo must follow reverse edit order (LIFO)")
        if "_snapshot" in record:
            return self._undo_quantized(record)
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

    def _undo_quantized(self, record: dict[str, Any]) -> str:
        """Restore a packed weight's bytes and scales, after checking nothing was swapped out."""
        snapshot = record["_snapshot"]
        weight = self._packed(record["name"])
        parameter = _parameter(self.model, record["name"])
        if (
            weight is None
            or parameter is not record["_parameter"]
            or getattr(parameter, "quant_state", None) is not record["_state"]
            or weight.repacked
            or any(
                part.dtype != snapshot[name].dtype or part.shape != snapshot[name].shape
                for name, part in weight.live().items()
            )
        ):
            raise RuntimeError("Parameter storage, dtype, or shape changed since the edit")
        current = weight.snapshot()
        record_before, active_before, previous_revision = (
            record.copy(),
            self._active.copy(),
            self.revision,
        )
        try:
            weight.restore(snapshot)
            self._active.pop()
            record["active"] = False
            self.revision = previous_revision + 1
            record["undone_revision"] = self.revision
            for key in ("_snapshot", "_parameter", "_state", "_index"):
                record.pop(key)
        except BaseException:
            weight.restore(current)
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
            self._editbook = EditBook(self.model, getattr(self, "quantization", None))
        return self._editbook

    @property
    def revision(self) -> int:
        """Increases whenever API edits, undo, attachments, or patches change outputs."""
        return (
            self.editbook.revision
            + getattr(self, "_resumed_component_revision", 0)
            + getattr(self, "_attachment_revision", 0)
            + getattr(self, "_patch_revision", 0)
        )

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

    def project_out(
        self,
        name: str,
        direction: Any,
        *,
        side: str = "output",
        index: Any = None,
        scale: float = 1.0,
        note: str = "",
    ) -> str:
        return self.editbook.project_out(
            name, direction, side=side, index=index, scale=scale, note=note
        )

    def undo(self, edit_id: str | None = None) -> str:
        return self.editbook.undo(edit_id)

    def edits(self) -> list[dict[str, Any]]:
        return self.editbook.edits()

    def export(self, path: str | Path) -> Path:
        """Save a standalone HF checkpoint; relative paths use the artifacts dir.

        PEFT exports merge the adapter into the current weights, preserving base
        weight edits as well as adapters. LoRA layers merge one at a time into a CPU
        copy and are then restored exactly, so the GPUs never hold a second copy; CPU
        memory holds every merged layer until the checkpoint is written. Other
        adapters (DoRA, quantized layers, other PEFT methods, tied embeddings under
        an adapter) merge a clone of the whole model, which needs a second model's
        memory and is refused for a model split across devices. Failed merges
        leave the live model and the destination unchanged.
        """
        if getattr(self, "quantization", None) or has_quantized_layers(self.model):
            raise ValueError(
                "Exporting a quantized model is not supported yet; export from a float session"
            )
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
        layerwise = False
        if merged_adapter:
            from .merging import layerwise_blocker

            blocker = layerwise_blocker(model)
            layerwise = blocker is None
            # Transformers maps GPU indices, accelerate also accepts names.
            devices = {
                f"cuda:{device}" if isinstance(device, int) else str(device)
                for device in (getattr(self, "device_map", None) or {}).values()
            }
            if not layerwise and len(devices) > 1:
                raise ValueError(
                    f"Cannot export this adapter from a model split across devices ({blocker}): "
                    "merging it needs a copy of the whole model"
                )
        if merged_adapter and not layerwise:
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
            prefix=".oumi-weights-lab-export-", dir=destination.parent
        ) as temp:
            staged = Path(temp) / "checkpoint"
            if layerwise:
                from .merging import merged_state_dict

                model.get_base_model().save_pretrained(
                    staged, state_dict=merged_state_dict(model), max_shard_size=SHARD_SIZE
                )
            else:
                model.save_pretrained(staged, max_shard_size=SHARD_SIZE)
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
            (staged / "oumi_weights_lab_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            if destination.exists():
                destination.rmdir()  # Refuse a destination populated during export.
            staged.rename(destination)
        return destination
