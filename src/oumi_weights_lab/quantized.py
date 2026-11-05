"""Read-only access to bitsandbytes-quantized weights.

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

        data = self.parameter.data.detach().cpu()
        if self.method == "int8":
            scales = getattr(self.parameter, "SCB", None)
            if scales is None:
                scales = getattr(self.module.state, "SCB", None)
            if scales is None:
                raise RuntimeError(f"{self.name} has no int8 row scales to dequantize with")
            return data.float() * scales.detach().cpu().float()[:, None] / 127
        import bitsandbytes.functional as functional

        values = functional.dequantize_4bit(data, _on_cpu(self.parameter.quant_state))
        return values.reshape(self.shape).to(torch.float32)


def lookup(model: Any, name: str) -> QuantizedWeight | None:
    """The quantized weight called ``name``, or None for ordinary parameters and unknown names."""
    owner_name, _, attribute = name.rpartition(".")
    if attribute != "weight" or not _layer_classes():
        return None
    try:
        owner = model.get_submodule(owner_name)
    except AttributeError:
        return None
    if not is_quantized(owner):
        return None
    float_dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), None)
    return of(owner, name, float_dtype)


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


def refuse_edit(model: Any, name: str) -> None:
    """Raise ValueError when ``name`` is a packed weight; edits to those are not supported yet."""
    weight = lookup(model, name)
    if weight is not None:
        raise ValueError(
            f"{name} is a {weight.method} quantized weight; "
            "editing quantized weights is not supported yet"
        )


def other_method(model: Any) -> str | None:
    """The quantization method a checkpoint declares when it is not bitsandbytes, if any."""
    config = getattr(model, "config", None)
    declared = getattr(config, "quantization_config", None)
    method = declared.get("quant_method") if isinstance(declared, dict) else None
    method = method or getattr(declared, "quant_method", None)
    method = getattr(method, "value", method)
    return None if method in (None, "bitsandbytes") else str(method)
