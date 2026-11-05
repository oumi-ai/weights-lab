"""Access to bitsandbytes-quantized weights: reads and edits for nf4, fp4 and int8.

Only package code uses this module. bitsandbytes is never imported here: a model can hold
its layers only after the package has been imported, so the module looks for it in
``sys.modules`` and the package works without it.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import Any


def _layer_classes() -> tuple[type, ...]:
    """The bitsandbytes layer classes, or none when bitsandbytes was never imported."""
    package = sys.modules.get("bitsandbytes")
    nn = getattr(package, "nn", None)
    if nn is None:
        return ()
    return nn.Linear4bit, nn.Linear8bitLt


def is_quantized(module: Any) -> bool:
    """Whether ``module`` is a bitsandbytes 4-bit or 8-bit linear layer."""
    classes = _layer_classes()
    return bool(classes) and isinstance(module, classes)


def _on_cpu(state: Any) -> Any:
    """A copy of a 4-bit quant state whose tensors are on the CPU. The model's is untouched."""
    import torch

    clone = copy.copy(state)
    for attribute in ("absmax", "code", "offset"):
        value = getattr(state, attribute, None)
        if isinstance(value, torch.Tensor):
            setattr(clone, attribute, value.detach().cpu())
    if getattr(state, "nested", False):
        clone.state2 = _on_cpu(state.state2)
    return clone


@dataclass(frozen=True)
class QuantizedWeight:
    """One packed weight: its owner, its method, and the logical float weight it stands for."""

    name: str
    module: Any
    parameter: Any
    method: str
    shape: Any
    dtype: Any
    double_quant: bool

    def describe(self) -> dict[str, Any]:
        """How the weight is stored, as the ``quantization`` block of ``parts`` and ``measure``."""
        if self.method == "int8":
            setting = {"threshold": float(self.module.state.threshold)}
        else:
            setting = {"blocksize": int(self.parameter.quant_state.blocksize)}
        return {
            "method": self.method,
            **setting,
            "double_quant": self.double_quant,
            "packed_shape": list(self.parameter.data.shape),
            "packed_dtype": str(self.parameter.data.dtype),
        }

    def dequantize(self) -> Any:
        """The logical weight as a float32 tensor on the CPU."""
        import torch

        if self.method == "int8":
            return Int8Codec(self.module, self.dtype).dequantize()
        import bitsandbytes.functional as functional

        data = self.parameter.data.detach().cpu()
        values = functional.dequantize_4bit(data, _on_cpu(self.parameter.quant_state))
        return values.reshape(self.shape).to(torch.float32)

    def check_editable(self) -> None:
        """Raise ValueError unless an edit can change this weight block by block."""
        if self.method == "int8":
            if self.parameter.has_fp16_weights:
                raise ValueError(
                    f"{self.name} keeps float weights (has_fp16_weights), so it has no int8 "
                    "bytes to edit; edit the float parameter instead"
                )
            return
        if self.double_quant:
            raise ValueError(
                f"{self.name} uses double quantization (bnb_4bit_use_double_quant), which shares "
                "its scales across the whole tensor, so no edit can stay inside one block; "
                "reload the float checkpoint with --quantize nf4 or fp4 to edit it"
            )
        if self.repacked:
            raise ValueError(
                f"{self.name} was repacked by bitsandbytes for CPU kernels "
                "(packing_format_for_cpu), so its bytes no longer follow the stored layout"
            )

    @property
    def repacked(self) -> bool:
        """Whether bitsandbytes has rewritten a 4-bit weight into its CPU kernel layout."""
        state = getattr(self.parameter, "quant_state", None)
        return bool(getattr(state, "packing_format_for_cpu", False))

    @property
    def state(self) -> Any:
        """The 4-bit quant state, which undo checks is the same object; int8 has none."""
        return getattr(self.parameter, "quant_state", None)

    @property
    def blocksize(self) -> int:
        """Weights per scale: the 4-bit block size, or the row length of an int8 weight."""
        if self.method == "int8":
            return int(self.shape[1])
        return int(self.state.blocksize)

    def _codec(self) -> FourBitCodec | Int8Codec:
        self.check_editable()
        if self.method == "int8":
            return Int8Codec(self.module, self.dtype)
        return FourBitCodec(self.parameter)

    def snapshot(self) -> dict[str, Any]:
        """CPU copies of the packed bytes and the scales, for ``restore``."""
        return self._codec().snapshot()

    def live(self) -> dict[str, Any]:
        """The tensors a snapshot copies, under the same part names, for undo to compare."""
        return self._codec().live()

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Write a ``snapshot`` back in place, bit for bit."""
        self._codec().restore(snapshot)

    def write(self, after: Any, before: Any) -> dict[str, Any]:
        """Requantize the blocks where ``after`` differs from ``before``, in place.

        A block is 64 weights of a 4-bit weight, or one row of an int8 weight. Both tensors are
        float32 CPU tensors in the logical shape. Returns ``blocks_requantized``.
        """
        return self._codec().write(after, before)


class FourBitCodec:
    """Block-local writes into a ``Params4bit``: nf4 or fp4 with one float32 scale per block.

    Element ``i`` of the flattened weight is in block ``i // blocksize`` and in byte ``i // 2``,
    the high nibble for even ``i``. A block quantizes on its own to exactly the bytes and scale
    the whole tensor gives it, so a block can be rewritten without touching any other.
    """

    def __init__(self, parameter: Any) -> None:
        self.parameter = parameter
        self.state = parameter.quant_state

    def live(self) -> dict[str, Any]:
        return {"data": self.parameter.data, "absmax": self.state.absmax}

    def snapshot(self) -> dict[str, Any]:
        return {name: part.detach().cpu().clone() for name, part in self.live().items()}

    def restore(self, snapshot: dict[str, Any]) -> None:
        import torch

        with torch.no_grad():
            self.parameter.data.copy_(snapshot["data"])
            self.state.absmax.copy_(snapshot["absmax"])

    def write(self, after: Any, before: Any) -> dict[str, Any]:
        import bitsandbytes.functional as functional
        import torch

        size = int(self.state.blocksize)
        flat = after.detach().reshape(-1).to(self.state.dtype)
        moved = after.detach().reshape(-1) != before.detach().reshape(-1)
        blocks = torch.unique(moved.nonzero().flatten() // size)
        count, full = flat.numel(), flat.numel() // size
        whole = blocks[blocks < full]
        tail = blocks[blocks >= full]  # at most the one partial block
        half = size // 2
        kind = self.state.quant_type
        pieces = []
        if whole.numel():
            values = flat[: full * size].reshape(full, size)[whole].reshape(-1)
            data, state = functional.quantize_4bit(values, blocksize=size, quant_type=kind)
            elements = (whole[:, None] * size + torch.arange(size)).reshape(-1)
            offsets = (whole[:, None] * half + torch.arange(half)).reshape(-1)
            pieces.append((whole, elements, offsets, data.reshape(-1), state.absmax))
        if tail.numel() and full * size < count:
            data, state = functional.quantize_4bit(
                flat[full * size :], blocksize=size, quant_type=kind
            )
            elements = torch.arange(full * size, count)
            offsets = full * half + torch.arange(data.numel())
            pieces.append((tail, elements, offsets, data.reshape(-1), state.absmax))
        with torch.no_grad():
            for numbers, elements, offsets, data, absmax in pieces:
                data = self._keep_unmoved(numbers, elements, offsets, data, absmax, moved)
                device = self.parameter.device
                # Index assignment, because MPS lacks index_copy before torch 2.8.
                self.parameter.data.view(-1)[offsets.to(device)] = data.to(device)
                self._store_absmax(numbers, absmax)
        return {"blocks_requantized": int(blocks.numel())}

    def _keep_unmoved(
        self, numbers: Any, elements: Any, offsets: Any, data: Any, absmax: Any, moved: Any
    ) -> Any:
        """``data`` with the old code restored for elements that kept both value and scale.

        Requantizing a dequantized value usually gives the same code back, but fp4 has two zeros
        and can swap them. Keeping the old code means an edit changes only what it has to.
        """
        size = int(self.state.blocksize)
        old = self.parameter.data.detach().view(-1)[offsets.to(self.parameter.device)].cpu()
        old_scale = self.state.absmax.detach()[numbers.to(self.state.absmax.device)].cpu()
        same_scale = (old_scale == absmax.cpu().to(old_scale.dtype)).repeat_interleave(size)
        keep = ~moved[elements] & same_scale[: elements.numel()]
        new = self._nibbles(data)
        before = self._nibbles(old)
        new[: keep.numel()][keep] = before[: keep.numel()][keep]
        return (new[0::2] << 4) | new[1::2]

    @staticmethod
    def _nibbles(data: Any) -> Any:
        """The 4-bit codes of packed bytes in element order: the high nibble comes first."""
        import torch

        return torch.stack((data >> 4, data & 15), dim=1).reshape(-1)

    def _store_absmax(self, blocks: Any, absmax: Any) -> None:
        scales = self.state.absmax
        scales[blocks.to(scales.device)] = absmax.to(scales.device, scales.dtype)


class Int8Codec:
    """Row-local writes into an ``Int8Params``: LLM.int8 with one float32 scale per row.

    A row quantizes on its own to exactly the bytes and scale the whole tensor gives it. The
    scales sit on the weight until the first forward and on the layer's state after that, and
    the bytes are shared by ``weight.data``, ``weight.CB`` and ``state.CB``, so every writer
    and reader goes through the live copies listed here.
    """

    def __init__(self, module: Any, dtype: Any) -> None:
        self.module = module
        self.parameter = module.weight
        self.dtype = dtype

    def _tensors(self, name: str) -> list[Any]:
        """The distinct live tensors that hold the bytes (``CB``) or the scales (``SCB``)."""
        owners = (self.parameter, self.module.state)
        if name == "CB":
            found = [self.parameter.data] + [getattr(o, "CB", None) for o in owners]
        else:
            found = [getattr(o, "SCB", None) for o in owners]
        unique: dict[Any, Any] = {}
        for tensor in found:
            if tensor is not None:
                unique.setdefault((tensor.device, tensor.untyped_storage().data_ptr()), tensor)
        return list(unique.values())

    def _scales(self) -> Any:
        scales = self._tensors("SCB")
        if not scales:
            raise RuntimeError("an int8 weight has no row scales to read")
        return scales[0]

    def dequantize(self) -> Any:
        values = self.parameter.data.detach().cpu().float()
        return values * self._scales().detach().cpu().float()[:, None] / 127

    def live(self) -> dict[str, Any]:
        return {"data": self.parameter.data, "scales": self._scales()}

    def snapshot(self) -> dict[str, Any]:
        return {name: part.detach().cpu().clone() for name, part in self.live().items()}

    def restore(self, snapshot: dict[str, Any]) -> None:
        import torch

        with torch.no_grad():
            for data in self._tensors("CB"):
                data.copy_(snapshot["data"])
            for scales in self._tensors("SCB"):
                scales.copy_(snapshot["scales"])

    def write(self, after: Any, before: Any) -> dict[str, Any]:
        import bitsandbytes.functional as functional
        import torch

        rows = (after.detach() != before.detach()).any(dim=1).nonzero().flatten()
        if rows.numel():
            values = after.detach()[rows].to(self.dtype)
            data, scales, _ = functional.int8_vectorwise_quant(values, threshold=0.0)
            with torch.no_grad():
                for target in self._tensors("CB"):
                    # Index assignment, because MPS lacks index_copy before torch 2.8.
                    target[rows.to(target.device)] = data.to(target.device)
                self._store_scales(rows, scales)
        return {"blocks_requantized": int(rows.numel())}

    def _store_scales(self, rows: Any, scales: Any) -> None:
        for target in self._tensors("SCB"):
            target[rows.to(target.device)] = scales.to(target.device, target.dtype)


def lookup(model: Any, name: str, dtype: Any = None) -> QuantizedWeight | None:
    """The quantized weight called ``name``, or None for ordinary parameters and unknown names.

    ``dtype`` is the compute dtype that an int8 weight reports as its logical dtype. Without
    it, a bare model's first float parameter stands in.
    """
    owner_name, _, attribute = name.rpartition(".")
    if attribute != "weight" or not _layer_classes():
        return None
    try:
        owner = model.get_submodule(owner_name)
    except AttributeError:
        return None
    if not is_quantized(owner):
        return None
    if dtype is None:
        dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), None)
    return of(owner, name, dtype)


def of(module: Any, name: str = "", dtype: Any = None) -> QuantizedWeight | None:
    """The quantized weight of ``module``, or None when it is not a bitsandbytes layer.

    ``dtype`` is the logical dtype of an int8 weight, which the file does not record;
    it defaults to float32.
    """
    import torch

    if not is_quantized(module):
        return None
    parameter = module.weight
    if isinstance(module, sys.modules["bitsandbytes"].nn.Linear8bitLt):
        # LLM.int8 keeps the logical shape, with one float scale per row.
        return QuantizedWeight(
            name, module, parameter, "int8", parameter.data.shape, dtype or torch.float32, False
        )
    state = getattr(parameter, "quant_state", None)
    if state is None:
        return None
    return QuantizedWeight(
        name,
        module,
        parameter,
        str(state.quant_type),
        torch.Size(state.shape),
        state.dtype,
        bool(getattr(state, "nested", False)),
    )


def has_quantized_layers(model: Any) -> bool:
    """Whether any module of ``model`` is a bitsandbytes layer."""
    return any(is_quantized(module) for module in model.modules())


def editable(model: Any, name: str, dtype: Any = None) -> QuantizedWeight | None:
    """The quantized weight called ``name`` after checking an edit can change it, or None.

    Raises ValueError for int8 layers that keep float weights, double-quantized weights, and
    weights bitsandbytes has repacked for CPU kernels.
    """
    weight = lookup(model, name, dtype)
    if weight is not None:
        weight.check_editable()
    return weight


def other_method(model: Any) -> str | None:
    """The quantization method a checkpoint declares when it is not bitsandbytes, if any."""
    config = getattr(model, "config", None)
    declared = getattr(config, "quantization_config", None)
    method = declared.get("quant_method") if isinstance(declared, dict) else None
    method = method or getattr(declared, "quant_method", None)
    method = getattr(method, "value", method)
    return None if method in (None, "bitsandbytes") else str(method)


def quantization_settings(lab: Any) -> dict[str, Any] | None:
    """How a lab's model is quantized, as a dict of ``method`` and ``blocksize`` or ``threshold``.

    Bundles and exports record it. It is None for a float model. A lab that was not loaded
    with a ``quantization`` summary reads the settings off the model's first packed layer.
    """
    declared = getattr(lab, "quantization", None)
    if declared is None:
        weight = next((found for m in lab.model.modules() if (found := of(m)) is not None), None)
        declared = weight.describe() if weight is not None else None
    if declared is None:
        return None
    return {key: declared[key] for key in ("method", "blocksize", "threshold") if key in declared}
