"""Per-head views of attention weights, for circuit analysis.

A head is a set of slices of the model's own parameters, so the parts that
``heads`` returns work with ``lab.zero`` and ``lab.edit``. Only base weights are read.
"""

from __future__ import annotations

from typing import Any

from .quantized import is_quantized, lookup, of

# Children that mark an attention layout this module does not split into heads.
_REFUSED = {
    "qkv_proj": "a fused qkv_proj (Phi-3)",
    "query_key_value": "a fused query_key_value (GPT-NeoX, Falcon, Bloom)",
    "q_a_proj": "multi-head latent attention (q_a_proj)",
    "kv_a_proj_with_mqa": "multi-head latent attention (kv_a_proj_with_mqa)",
}
_PROJECTIONS = ("q_proj", "k_proj", "v_proj")


def _dense(model: Any, name: str, index: Any = None) -> Any:
    """A detached float32 CPU copy of a slice of the parameter ``name``.

    Every weight read goes through here, which dequantizes quantized weights.
    """
    import torch

    packed = lookup(model, name)
    if packed is not None:
        dense = packed.dequantize()
        return (dense if index is None else dense[index]).clone()
    selected = model.get_parameter(name).detach()
    if index is not None:
        selected = selected[index]
    return selected.to(device="cpu", dtype=torch.float32, copy=True)


def _rows(start: int, stop: int) -> tuple:
    return (slice(start, stop), slice(None))


def _columns(start: int, stop: int) -> tuple:
    return (slice(None), slice(start, stop))


def _part(name: str, index: tuple) -> dict[str, Any]:
    return {"name": name, "index": index}


def _head_dim(module: Any, model: Any) -> int:
    value = getattr(module, "head_dim", None)
    if value is None:
        config = model.config
        config = config.get_text_config() if hasattr(config, "get_text_config") else config
        value = getattr(config, "head_dim", None)
        if value is None:
            value = config.hidden_size // config.num_attention_heads
    return int(value)


def _layer_index(name: str) -> int:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    raise ValueError(f"cannot find a layer index in the module name {name!r}")


def _split(module: Any, children: dict[str, Any], name: str, model: Any) -> dict[str, Any] | None:
    """The head layout of one module, or None when it is not an attention block."""
    from torch import nn
    from transformers.pytorch_utils import Conv1D

    if set(_PROJECTIONS) <= children.keys():
        output = next((key for key in ("o_proj", "out_proj") if key in children), None)
        if output is None or not all(
            isinstance(children[key], nn.Linear) for key in (*_PROJECTIONS, output)
        ):
            return None
        hd = _head_dim(module, model)
        o_in, q_out = children[output].in_features, children["q_proj"].out_features
        k_out, v_out = children["k_proj"].out_features, children["v_proj"].out_features
        if o_in % hd or k_out % hd or v_out != k_out:
            raise ValueError(f"{name}: projection sizes are not whole {hd}-wide heads")
        heads = o_in // hd
        if q_out not in (heads * hd, 2 * heads * hd):
            raise ValueError(f"{name}: q_proj has {q_out} outputs for {heads} heads of width {hd}")
        return {
            "kind": "gated" if q_out == 2 * heads * hd else "separate",
            "head_dim": hd,
            "heads": heads,
            "kv_heads": k_out // hd,
            "output": output,
            "biased": {key: children[key].bias is not None for key in _PROJECTIONS},
        }
    if {"c_attn", "c_proj"} <= children.keys() and not getattr(module, "is_cross_attention", False):
        # transformers quantizes GPT-2's Conv1D into a Linear4bit or Linear8bitLt that stores
        # [out, in], so the layers are transposed from their float form.
        fused = [children["c_attn"], children["c_proj"]]
        if not all(isinstance(child, Conv1D) or is_quantized(child) for child in fused):
            return None
        stored = [of(child).shape if is_quantized(child) else child.weight.shape for child in fused]
        transposed = isinstance(fused[0], Conv1D)
        if transposed != isinstance(fused[1], Conv1D):
            return None
        hd = _head_dim(module, model)
        width = stored[1][0]
        if width % hd or stored[0][1 if transposed else 0] != 3 * width:
            raise ValueError(f"{name}: projection sizes are not whole {hd}-wide heads")
        return {
            "kind": "fused",
            "transposed": transposed,
            "head_dim": hd,
            "heads": width // hd,
            "kv_heads": width // hd,
            "width": width,
        }
    return None


def _blocks(model: Any) -> list[tuple[str, dict[str, Any]]]:
    if hasattr(model, "peft_config"):
        raise NotImplementedError(
            "head slices read base weights only; export the merged model and start a session on it"
        )
    found = []
    owners: dict[int, str] = {}
    for name, module in model.named_modules():
        children = dict(module.named_children())
        for child, layout in _REFUSED.items():
            if child in children:
                raise NotImplementedError(
                    f"{name} uses {layout}; heads cannot be split into per-head slices"
                )
        layout = _split(module, children, name, model)
        if layout is not None:
            layer = _layer_index(name)
            if layer in owners:
                raise NotImplementedError(
                    f"{owners[layer]} and {name} have the same layer index {layer}; "
                    "heads supports one stack of numbered layers"
                )
            owners[layer] = name
            found.append((name, layout))
    if not found:
        raise NotImplementedError(
            "no attention block with separate q/k/v projections or GPT-2's fused c_attn was found"
        )
    return found


def _linear_parts(name: str, child: str, biased: bool, start: int, stop: int) -> list[dict]:
    parts = [_part(f"{name}.{child}.weight", _rows(start, stop))]
    if biased:
        parts.append(_part(f"{name}.{child}.bias", (slice(start, stop),)))
    return parts


def _fused_parts(attn: str, offset: int, lo: int, hi: int, transposed: bool) -> list[dict]:
    heads = _columns if transposed else _rows
    return [
        _part(f"{attn}.weight", heads(offset + lo, offset + hi)),
        _part(f"{attn}.bias", (slice(offset + lo, offset + hi),)),
    ]


def _entries(name: str, layout: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per head of an attention block."""
    hd, heads, kv_heads = layout["head_dim"], layout["heads"], layout["kv_heads"]
    if heads % kv_heads:
        raise ValueError(f"{name}: {heads} heads do not divide into {kv_heads} key/value heads")
    size = heads // kv_heads
    entries = []
    for head in range(heads):
        kv = head // size
        lo, hi = head * hd, (head + 1) * hd
        if layout["kind"] == "fused":
            width, attn = layout["width"], f"{name}.c_attn"
            output = f"{name}.c_proj"

            flip = layout["transposed"]
            parts = {
                "q": _fused_parts(attn, 0, lo, hi, flip),
                "k": _fused_parts(attn, width, lo, hi, flip),
                "v": _fused_parts(attn, 2 * width, lo, hi, flip),
                "o": [_part(f"{output}.weight", _rows(lo, hi) if flip else _columns(lo, hi))],
            }
        else:
            biased = layout["biased"]
            output = f"{name}.{layout['output']}"
            gated = layout["kind"] == "gated"
            q_start = 2 * lo if gated else lo
            parts = {
                "q": _linear_parts(name, "q_proj", biased["q_proj"], q_start, q_start + hd),
                "k": _linear_parts(name, "k_proj", biased["k_proj"], kv * hd, (kv + 1) * hd),
                "v": _linear_parts(name, "v_proj", biased["v_proj"], kv * hd, (kv + 1) * hd),
                "o": [_part(f"{output}.weight", _columns(lo, hi))],
            }
            if gated:
                parts["gate"] = _linear_parts(
                    name, "q_proj", biased["q_proj"], q_start + hd, q_start + 2 * hd
                )
        entries.append(
            {
                "layer": _layer_index(name),
                "head": head,
                "kv_head": kv,
                "group": list(range(kv * size, (kv + 1) * size)),
                "module": name,
                **parts,
                "o_input": {"module": output, "at": "input", "features": [lo, hi]},
            }
        )
    return entries


def _all_entries(model: Any) -> list[dict[str, Any]]:
    entries = [entry for name, layout in _blocks(model) for entry in _entries(name, layout)]
    return sorted(entries, key=lambda entry: (entry["layer"], entry["head"]))


class CircuitsMixin:
    """Lab methods that expose attention heads as slices of the model's weights."""

    def heads(self, layers: list[int] | None = None) -> list[dict[str, Any]]:
        """List each attention head with the weight slices that make it up.

        Each entry has ``q``, ``k``, ``v`` and ``o`` lists of parts, plus ``gate`` for
        models whose ``q_proj`` carries a gate. A part is ``{"name", "index"}``, so
        ``lab.zero(**part)`` ablates it. Grouped-query heads in one group share their
        ``k`` and ``v`` parts. ``o_input`` names the module input whose features are
        the head's output. ``layers`` limits the listing to those layer indexes.

        Raises ValueError for layers the model lacks, and NotImplementedError for
        LoRA models and for fused qkv, latent-attention, or attention-free models.
        """
        entries = _all_entries(self.model)
        if layers is None:
            return entries
        present = sorted({entry["layer"] for entry in entries})
        valid = (
            isinstance(layers, (list, tuple))
            and all(type(layer) is int for layer in layers)
            and len(set(layers)) == len(layers)
            and set(layers) <= set(present)
        )
        if not valid:
            raise ValueError(f"layers must be distinct layer indexes from {present}")
        return [entry for entry in entries if entry["layer"] in set(layers)]

    def head_weights(self, layer: int, head: int) -> dict[str, Any]:
        """One head's weights as float32 CPU tensors in ``nn.Linear`` orientation.

        ``q``, ``k`` and ``v`` are ``[head_dim, hidden]`` (``k`` and ``v`` from the head's
        key/value head), ``o`` is ``[hidden, head_dim]``, and ``gate`` is
        ``[head_dim, hidden]`` where queries are gated. Biases come back as
        ``q_bias``, ``k_bias``, ``v_bias`` and ``gate_bias`` when the model has them.

        Raises ValueError for an unknown layer or head, and NotImplementedError where
        ``heads`` does.
        """
        entries = _all_entries(self.model)
        present = sorted({entry["layer"] for entry in entries})
        if type(layer) is not int or layer not in present:
            raise ValueError(f"layer must be one of {present}")
        in_layer = [entry for entry in entries if entry["layer"] == layer]
        if type(head) is not int or not 0 <= head < len(in_layer):
            raise ValueError(f"head must be in 0..{len(in_layer) - 1} for layer {layer}")
        entry = in_layer[head]
        # GPT-2's Conv1D stores in x out; nn.Linear (and its quantized form) stores out x in.
        from transformers.pytorch_utils import Conv1D

        fused = isinstance(self.model.get_submodule(entry["o_input"]["module"]), Conv1D)
        weights = {}
        for key in ("q", "k", "v", "o", "gate"):
            for part in entry.get(key, []):
                tensor = _dense(self.model, part["name"], part["index"])
                if part["name"].endswith(".bias"):
                    weights[f"{key}_bias"] = tensor
                else:
                    weights[key] = tensor.T.contiguous() if fused else tensor
        return weights
