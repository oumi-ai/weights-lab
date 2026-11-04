"""Forward-free diagnostics for ordinary linear LoRA factors."""

from __future__ import annotations

import math
from fnmatch import fnmatchcase


def _names(adapter):
    if adapter is None:
        return None
    try:
        names = [adapter] if isinstance(adapter, str) else list(adapter)
    except TypeError as exc:
        raise ValueError("adapter must be a name or a nonempty list of names") from exc
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("adapter must be a name or a nonempty list of names")
    if len(set(names)) != len(names):
        raise ValueError("adapter names must be distinct")
    return names


def _option(module, name, adapter):
    value = getattr(module, name, {})
    return value.get(adapter, False) if hasattr(value, "get") else value


def _base_norm(module, shape):
    """Read ordinary dense base weights in bounded chunks; never dequantize."""
    import torch

    if getattr(module, "merged_adapters", []):
        return None
    base = getattr(module, "base_layer", None)
    weight = getattr(base, "weight", None)
    if (
        not isinstance(weight, torch.Tensor)
        or type(weight) not in (torch.Tensor, torch.nn.Parameter)
        or weight.layout != torch.strided
        or not weight.is_floating_point()
        or weight.device.type == "meta"
        or weight.ndim != 2
        or tuple(weight.shape) not in (shape, shape[::-1])
    ):
        return None
    total = 0.0
    # Bound the converted CPU copy, including for noncontiguous base weights.
    detached = weight.detach()
    pieces = [detached.view(-1)] if detached.is_contiguous() else detached
    for row in pieces:
        for chunk in row.split(65536):
            # MPS has no float64: move to CPU before widening.
            value = chunk.cpu().to(torch.float64)
            if not torch.isfinite(value).all():
                raise ValueError("base weights must be finite")
            total = math.hypot(total, torch.linalg.vector_norm(value).item())
    return total


def _measure(module, name, adapter, top_k, active):
    import torch

    label = f"{name or '<root>'}:{adapter}"
    if (
        _option(module, "use_dora", adapter)
        or _option(module, "lora_variant", adapter)
        or _option(module, "use_qalora", adapter)
        or adapter in getattr(module, "lora_embedding_A", {})
        or adapter in getattr(module, "lora_embedding_B", {})
    ):
        raise ValueError(f"{label}: only ordinary linear LoRA is supported (no DoRA/variants)")
    a_layer, b_layer = module.lora_A[adapter], module.lora_B[adapter]
    if any(getattr(layer, "bias", None) is not None for layer in (a_layer, b_layer)):
        raise ValueError(f"{label}: LoRA factor biases are unsupported")
    a, b = getattr(a_layer, "weight", a_layer), getattr(b_layer, "weight", b_layer)
    if any(
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.layout != torch.strided
        or value.ndim != 2
        or value.device.type == "meta"
        for value in (a, b)
    ):
        raise ValueError(f"{label}: LoRA factors must be materialized floating point matrices")
    if a.shape[0] != b.shape[1] or min(*a.shape, *b.shape) == 0:
        raise ValueError(f"{label}: incompatible LoRA factor shapes")
    try:
        scaling = float(module.scaling[adapter])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: missing or invalid per-adapter scaling") from exc
    if not math.isfinite(scaling):
        raise ValueError(f"{label}: scaling must be finite")
    a = a.detach().cpu().to(torch.float64)
    b = b.detach().cpu().to(torch.float64)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError(f"{label}: LoRA factors must be finite")
    # B = Qb Rb, A.T = Qa Ra. Orthogonal Q factors preserve singular values.
    rb = torch.linalg.qr(b, mode="r").R
    ra = torch.linalg.qr(a.T, mode="r").R
    core = rb @ ra.T
    if not torch.isfinite(core).all():
        raise ValueError(f"{label}: diagnostic values overflowed; rescale the factors")
    singular = torch.linalg.svdvals(core) * abs(scaling)
    if not torch.isfinite(singular).all():
        raise ValueError(f"{label}: diagnostic values overflowed; rescale the factors")
    spectral = singular[0].item()
    frobenius = torch.linalg.vector_norm(singular).item()
    tolerance = max(b.shape[0], a.shape[1]) * torch.finfo(torch.float64).eps * spectral
    shape = (b.shape[0], a.shape[1])
    base_norm = _base_norm(module, shape)
    row = {
        "module": name,
        "adapter": adapter,
        "active": adapter in active,
        "disabled": bool(getattr(module, "disable_adapters", False)),
        "merged": adapter in getattr(module, "merged_adapters", []),
        "shape": list(shape),
        "factor_rank": a.shape[0],
        "scaling": scaling,
        "top_singular_values": singular[:top_k].tolist(),
        "numerical_rank": int((singular > tolerance).sum().item()),
        "rank_tolerance": tolerance,
        "stable_rank": (frobenius / spectral) ** 2 if spectral else 0.0,
        "frobenius_norm": frobenius,
        "spectral_norm": spectral,
        "a_frobenius_norm": torch.linalg.vector_norm(a).item(),
        "b_frobenius_norm": torch.linalg.vector_norm(b).item(),
        "base_frobenius_norm": base_norm,
        "relative_frobenius_norm": frobenius / base_norm if base_norm else None,
    }
    numbers = [value for value in row.values() if isinstance(value, (int, float))]
    if not all(math.isfinite(value) for value in numbers + row["top_singular_values"]):
        raise ValueError(f"{label}: diagnostic values overflowed; rescale the factors")
    return row


def lora_diagnostics(model, pattern="*", adapter=None, top_k=8):
    """Return per-adapter scaled B@A spectra for matching module-name globs."""
    import torch

    if not isinstance(pattern, str):
        raise ValueError("pattern must be a module-name glob string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    requested = _names(adapter)
    rows, found = [], set()
    with torch.no_grad():
        for name, module in model.named_modules():
            if not fnmatchcase(name, pattern):
                continue
            available = set(getattr(module, "lora_A", {}))
            available |= set(getattr(module, "lora_embedding_A", {}))
            if not available:
                continue
            active = getattr(module, "active_adapters", None)
            if active is None:
                active = getattr(module, "active_adapter", [])
            active = [active] if isinstance(active, str) else list(active or [])
            selected = active if requested is None else requested
            if requested is None and getattr(module, "disable_adapters", False):
                continue
            for adapter_name in selected:
                if adapter_name not in available:
                    continue
                found.add(adapter_name)
                if adapter_name not in getattr(module, "lora_B", {}):
                    raise ValueError(
                        f"{name}:{adapter_name}: unsupported embedding or missing B factor"
                    )
                rows.append(_measure(module, name, adapter_name, top_k, active))
    if requested is not None and set(requested) - found:
        raise ValueError(
            f"adapters not found in matching modules: {sorted(set(requested) - found)}"
        )
    return rows
