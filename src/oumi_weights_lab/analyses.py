"""Circuit analyses: direct logit attribution, path patching, and edge attribution patching.

All three read a model as an additive residual stream. The embeddings, each attention head,
each MLP, and (for layers whose attention is not split into heads, such as Qwen3.5's
linear-attention layers) each attention block write into it, and each reader sees the sum of
what came before, through its own norm. Models whose blocks do anything else to a sublayer's
output are refused, and so are models where the recomputation below does not reproduce what
the model itself does. A refusal is a ``NotImplementedError`` naming the module, never an
approximate number.

A metric is a callable ``metric(logits) -> scalar tensor`` over the last-position logits of a
batch (shape ``[batch, vocab]``, float32), or an ``(answer, counterfactual)`` pair of strings,
meaning the mean logit difference between the first token of each at the last position.
Prompt pairs with equal token counts run together in one batch, so no padding is involved.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import Any

from .circuits import _all_entries, _dense
from .inspection import (
    _activation,
    _attention_weights,
    _eager,
    _input_activation,
    _TemporaryHook,
    _with_input,
    _with_output,
)
from .quantized import is_quantized

_PRE_ATTENTION = ("input_layernorm", "ln_1")
_PRE_MLP = ("post_attention_layernorm", "ln_2")
_FINAL_NORMS = ("norm", "ln_f", "final_layernorm", "norm_f")
_BY = ("component", "position")


def _refuse(message: str) -> NotImplementedError:
    return NotImplementedError(f"{message}; refusing rather than return approximate numbers")


def _is_norm(name: str, module: Any) -> bool:
    from torch import nn

    return (
        isinstance(module, nn.LayerNorm)
        or "norm" in type(module).__name__.lower()
        or "norm" in name.lower()
        or name.startswith("ln_")
    )


def _name_of(model: Any, target: Any) -> str:
    for name, module in model.named_modules():
        if module is target:
            return name
    raise _refuse("the model's embedding or output module has no name")


def _head_tensors(model: Any, entry: dict[str, Any]) -> dict[str, Any]:
    """One head's weights in ``nn.Linear`` orientation, as ``head_weights`` returns them."""
    from transformers.pytorch_utils import Conv1D

    fused = isinstance(model.get_submodule(entry["o_input"]["module"]), Conv1D)
    out = {}
    for key in ("q", "k", "v", "o", "gate"):
        for part in entry.get(key, []):
            tensor = _dense(model, part["name"], part["index"])
            if part["name"].endswith(".bias"):
                out[f"{key}_bias"] = tensor
            else:
                out[key] = tensor.T.contiguous() if fused else tensor
    return out


class _Net:
    """The additive residual structure of a model, or a refusal that says why there is none."""

    def __init__(self, model: Any) -> None:
        if any(is_quantized(module) for module in model.modules()):
            raise _refuse(
                "circuit analyses recompute head inputs in float and are not exact for "
                "bitsandbytes layers; start a session on an unquantized model"
            )
        entries = _all_entries(model)  # refuses PEFT models and unsupported layouts
        self.model = model
        self._weights: dict[int, dict[str, Any]] = {}
        by_layer: dict[int, list[dict[str, Any]]] = {}
        for entry in entries:
            by_layer.setdefault(entry["layer"], []).append(entry)
        containers = set()
        for entry in entries:
            layer_name, _, _ = entry["module"].rpartition(".")
            container, _, index = layer_name.rpartition(".")
            if not index.isdigit():
                raise _refuse(f"{entry['module']} is not a direct child of a numbered layer")
            containers.add(container)
        if len(containers) != 1:
            raise _refuse("attention blocks sit in more than one stack of layers")
        container = containers.pop()
        stack = model.get_submodule(container)
        self.layers: dict[int, dict[str, Any]] = {}
        for child in stack._modules:
            if child.isdigit():
                self.layers[int(child)] = self._layer(f"{container}.{child}", int(child), by_layer)
        self.layer_ids = sorted(self.layers)
        self.embedding = _name_of(model, model.get_input_embeddings())
        head = model.get_output_embeddings()
        if head is None:
            raise _refuse("the model has no output head")
        self.head = head
        self.final_norm = self._final_norm(model)
        self.nodes: list[tuple] = [("embed",)]
        self.first: dict[int, int] = {}
        self.mlp_node: dict[int, int] = {}
        for layer in self.layer_ids:
            info = self.layers[layer]
            self.first[layer] = len(self.nodes)
            if info["heads"]:
                self.nodes += [("head", layer, h) for h in range(info["nh"])]
            else:
                self.nodes.append(("attn", layer))
            self.mlp_node[layer] = len(self.nodes)
            self.nodes.append(("mlp", layer))
        self.node_id = {key: index for index, key in enumerate(self.nodes)}
        self.inputs: list[tuple] = []
        self.limit: list[int] = []
        for layer in self.layer_ids:
            info = self.layers[layer]
            if info["heads"]:
                for kind, count in (("q", info["nh"]), ("k", info["nkv"]), ("v", info["nkv"])):
                    for unit in range(count):
                        self.inputs.append((kind, layer, unit))
                        self.limit.append(self.first[layer])
            else:
                self.inputs.append(("attn", layer))
                self.limit.append(self.first[layer])
            self.inputs.append(("mlp", layer))
            self.limit.append(self.mlp_node[layer])
        self.inputs.append(("logits",))
        self.limit.append(len(self.nodes))
        self.row = {key: index for index, key in enumerate(self.inputs)}

    def _layer(self, name: str, index: int, by_layer: dict) -> dict[str, Any]:
        module = self.model.get_submodule(name)
        children = dict(module.named_children())
        norms = sorted(child for child, value in children.items() if _is_norm(child, value))
        pre_attention = [n for n in norms if n in _PRE_ATTENTION]
        pre_mlp = [n for n in norms if n in _PRE_MLP]
        if len(norms) != 2 or len(pre_attention) != 1 or len(pre_mlp) != 1:
            raise _refuse(
                f"{name} has the norms {norms}, which are not exactly one norm before attention "
                "and one before the MLP. Norms on a sublayer's output (Gemma 2, OLMo 2) and "
                "parallel attention and MLP blocks are not supported"
            )
        if "mlp" not in children:
            raise _refuse(f"{name} has no mlp child (it has {sorted(children)})")
        others = [child for child in children if child not in norms and child != "mlp"]
        if len(others) != 1:
            raise _refuse(
                f"{name} has {others} besides its norms and mlp; expected one attention block"
            )
        info: dict[str, Any] = {
            "name": name,
            "norm1": f"{name}.{pre_attention[0]}",
            "norm2": f"{name}.{pre_mlp[0]}",
            "mlp": f"{name}.mlp",
            "attn": f"{name}.{others[0]}",
            "heads": None,
        }
        heads = by_layer.get(index)
        if heads is None:
            return info
        if heads[0]["module"] != info["attn"]:
            raise _refuse(f"{name}: the attention block is not {info['attn']}")
        hd = heads[0]["o_input"]["features"][1] - heads[0]["o_input"]["features"][0]
        nh = len(heads)
        nkv = len({entry["kv_head"] for entry in heads})
        gated = "gate" in heads[0]
        fused = heads[0]["q"][0]["name"].endswith(".c_attn.weight")
        attn = info["attn"]
        if fused:
            module_name = f"{attn}.c_attn"
            proj = {
                "q": (module_name, 0, hd, nh),
                "k": (module_name, nh * hd, hd, nh),
                "v": (module_name, 2 * nh * hd, hd, nh),
            }
        else:
            unit = 2 * hd if gated else hd
            proj = {
                "q": (f"{attn}.q_proj", 0, unit, nh),
                "k": (f"{attn}.k_proj", 0, hd, nkv),
                "v": (f"{attn}.v_proj", 0, hd, nkv),
            }
        info.update(
            heads=heads,
            nh=nh,
            nkv=nkv,
            hd=hd,
            gated=gated,
            proj=proj,
            o_module=heads[0]["o_input"]["module"],
        )
        return info

    @staticmethod
    def _final_norm(model: Any) -> str:
        found = [
            name
            for name, module in model.named_modules()
            if name.rpartition(".")[2] in _FINAL_NORMS
            and not any(part.isdigit() for part in name.split("."))
        ]
        if len(found) != 1:
            raise _refuse(f"cannot find the final norm (candidates: {found})")
        return found[0]

    # -- weights -----------------------------------------------------------------------

    def weights(self, layer: int) -> dict[str, Any]:
        """Stacked float32 CPU weights of a layer's heads, cached for the life of the call."""
        import torch

        if layer in self._weights:
            return self._weights[layer]
        info = self.layers[layer]
        heads = info["heads"]
        tensors = [_head_tensors(self.model, entry) for entry in heads]
        size = info["nh"] // info["nkv"]

        def stack(key, rows, extra=None):
            parts = []
            for index in rows:
                value = tensors[index][key]
                if extra is not None:
                    value = torch.cat([value, tensors[index][extra]], dim=0)
                parts.append(value)
            return torch.stack(parts)

        def bias(key, rows, extra=None):
            if f"{key}_bias" not in tensors[0]:
                return None
            parts = []
            for index in rows:
                value = tensors[index][f"{key}_bias"]
                if extra is not None:
                    value = torch.cat([value, tensors[index][f"{extra}_bias"]], dim=0)
                parts.append(value)
            return torch.stack(parts)

        every = range(info["nh"])
        groups = range(0, info["nh"], size)
        try:
            o_bias = _dense(self.model, f"{info['o_module']}.bias")
        except AttributeError:
            o_bias = None
        weights = {
            "q": stack("q", every, "gate" if info["gated"] else None),
            "q_bias": bias("q", every, "gate" if info["gated"] else None),
            "k": stack("k", groups),
            "k_bias": bias("k", groups),
            "v": stack("v", groups),
            "v_bias": bias("v", groups),
            "o": stack("o", every),
            "o_bias": o_bias,
        }
        self._weights[layer] = weights
        return weights

    # -- names -------------------------------------------------------------------------

    def label(self, node: tuple) -> Any:
        """The user-facing name of a node."""
        if node[0] == "embed":
            return "embed"
        if node[0] == "head":
            return (node[1], node[2])
        return (node[0], node[1])

    def input_label(self, key: tuple) -> Any:
        if key[0] == "logits":
            return "logits"
        if key[0] in ("q", "k", "v"):
            return (key[1], key[2], key[0])
        return (key[0], key[1])

    def node_key(self, spec: Any) -> tuple:
        """A sender spec as a node key, or ValueError."""
        if isinstance(spec, list):
            spec = tuple(spec)
        valid = False
        key: tuple = ()
        if spec == "embed":
            key, valid = ("embed",), True
        elif isinstance(spec, tuple) and len(spec) == 2:
            if spec[0] in ("mlp", "attn") and type(spec[1]) is int:
                key = (spec[0], spec[1])
                valid = key in self.node_id
            elif type(spec[0]) is int and type(spec[1]) is int:
                key = ("head", spec[0], spec[1])
                valid = key in self.node_id
        if not valid:
            raise ValueError(
                f"unknown sender {spec!r}: use 'embed', (layer, head), ('mlp', layer), or "
                f"('attn', layer) for a layer whose attention is not split into heads"
            )
        return key

    def input_key(self, spec: Any, kv: bool = False) -> tuple:
        """A receiver spec as an input key, or ValueError.

        ``(layer, head, "k")`` names the key head that ``head`` reads; with ``kv`` true the
        middle number is the key/value head itself, as edge lists give it.
        """
        if isinstance(spec, list):
            spec = tuple(spec)
        key: tuple | None = None
        if spec == "logits":
            key = ("logits",)
        elif isinstance(spec, tuple) and len(spec) == 2 and spec[0] in ("mlp", "attn"):
            key = (spec[0], spec[1]) if type(spec[1]) is int else None
        elif (
            isinstance(spec, tuple)
            and len(spec) == 3
            and type(spec[0]) is int
            and type(spec[1]) is int
            and spec[2] in ("q", "k", "v")
        ):
            info = self.layers.get(spec[0])
            if info is None:
                raise ValueError(f"unknown receiver {spec!r}: the model has no layer {spec[0]}")
            if not info["heads"]:
                raise ValueError(
                    f"unknown receiver {spec!r}: layer {spec[0]} has no attention heads; "
                    f"use ('attn', {spec[0]}) for its attention block"
                )
            limit = info["nkv"] if kv and spec[2] != "q" else info["nh"]
            if 0 <= spec[1] < limit:
                unit = spec[1]
                if spec[2] != "q" and not kv:
                    unit = info["heads"][spec[1]]["kv_head"]
                key = (spec[2], spec[0], unit)
        if key is None or key not in self.row:
            raise ValueError(
                f"unknown receiver {spec!r}: use (layer, head, 'q'|'k'|'v'), ('mlp', layer), "
                "('attn', layer), or 'logits'"
            )
        return key

    def owners(self, row: int) -> list[int]:
        """Node indexes whose computation reads input ``row`` (-1 stands for the logits)."""
        key = self.inputs[row]
        if key[0] == "logits":
            return [-1]
        if key[0] == "q":
            return [self.node_id[("head", key[1], key[2])]]
        if key[0] in ("k", "v"):
            info = self.layers[key[1]]
            size = info["nh"] // info["nkv"]
            return [self.node_id[("head", key[1], key[2] * size + i)] for i in range(size)]
        return [self.node_id[(key[0], key[1])]]


# -- one forward pass with hooks -----------------------------------------------------------


def _tolerance(dtype: Any, scale: float, floor: float, factor: float = 16.0) -> float:
    """How far a recomputation may differ from the model: the larger of ``floor * scale`` and
    ``factor`` rounding steps of ``dtype`` at ``scale``.

    ``floor`` is the allowance in float32 (and float64); half-precision dtypes round far more
    coarsely, so their allowance grows with the dtype's epsilon. A different norm or a
    different graph differs by order ``scale``, far above either.
    """
    import torch

    eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-7
    return max(floor, factor * eps) * scale


def _head_outputs(o_in: Any, wo: Any) -> Any:
    """Each head's write to the residual stream, ``[heads, batch, tokens, hidden]``."""
    import torch

    batch, tokens, width = o_in.shape
    heads, _, hd = wo.shape
    return torch.einsum("bthd,hrd->hbtr", o_in.reshape(batch, tokens, heads, hd), wo.to(o_in))


class _Rec:
    """What one forward pass left behind, detached."""

    def __init__(self, state: _State, logits: Any, net: _Net, nodes: bool) -> None:
        self.pre0 = state.pre[net.layer_ids[0]].detach()
        self.emb = state.emb.detach()
        self.o_in = {layer: value.detach() for layer, value in state.o_in.items()}
        self.attn_out = {layer: value.detach() for layer, value in state.attn_out.items()}
        self.mlp_out = {layer: value.detach() for layer, value in state.mlp_out.items()}
        self.final_in = state.final_in.detach()
        self.logits = logits.detach()
        self.nodes = (
            [state.cur[index].detach() for index in range(len(net.nodes))] if nodes else None
        )
        self.proj = {key: value.detach() for key, value in state.proj.items()}
        self.attn_weights = {layer: value.detach() for layer, value in state.attn_weights.items()}


class _State:
    def __init__(self) -> None:
        self.emb: Any = None
        self.pre: dict[int, Any] = {}
        self.o_in: dict[int, Any] = {}
        self.attn_out: dict[int, Any] = {}
        self.mlp_out: dict[int, Any] = {}
        self.final_in: Any = None
        self.cur: dict[int, Any] = {}
        self.proj: dict[tuple, Any] = {}
        self.attn_weights: dict[int, Any] = {}
        self.inputs: dict[tuple, Any] = {}
        self.leaves: dict[tuple, Any] = {}


class _Pass:
    """The hooks for one forward pass, and the record they fill.

    Hooks only record unless asked to change something: overwrite head outputs (``o_override``,
    a function of layer and the head outputs), MLP or attention-block outputs, the embeddings,
    or the inputs listed in ``replace``. With ``engine`` set, every reader gets its own
    residual, as ``_Engine`` describes.
    """

    def __init__(
        self,
        net: _Net,
        *,
        nodes: bool = False,
        proj: bool = False,
        attention: bool = False,
        o_override: Any = None,
        mlp_override: dict | None = None,
        attn_override: dict | None = None,
        embed_override: Any = None,
        emb_replace: Any = None,
        record: tuple = (),
        replace: dict | None = None,
        engine: _Engine | None = None,
    ) -> None:
        self.net, self.state = net, _State()
        self.nodes = nodes or (engine is not None and engine.drop is not None)
        self.proj, self.attention = proj, attention
        self.o_override, self.mlp_override = o_override, mlp_override or {}
        self.attn_override, self.embed_override = attn_override or {}, embed_override
        self.emb_replace = emb_replace
        self.record, self.replace = set(record), replace or {}
        self.engine = engine

    # Input hooks (path patching and the engine) ------------------------------------------

    def _swap(self, key: tuple, value: Any, state_value: Any) -> Any:
        """Record ``value`` for ``key`` if asked, and return what the module should receive."""
        if key in self.record:
            self.state.inputs[key] = state_value.detach().clone()
        if key in self.replace:
            return self.replace[key].to(device=value.device, dtype=value.dtype)
        return None

    def _emb_hook(self, module, args, output):
        value = _activation(output)
        if self.emb_replace is not None:
            value = self.emb_replace.to(device=value.device, dtype=value.dtype)
            self.state.emb = value
            return _with_output(output, value)
        self.state.emb = value
        return None

    def _layer_pre(self, layer, module, args, kwargs):
        value, slot = _input_activation(args, kwargs)
        changed = None
        if layer == self.net.layer_ids[0] and self.embed_override is not None:
            changed = self.embed_override.to(device=value.device, dtype=value.dtype)
            value = changed
        self.state.pre[layer] = value
        if layer == self.net.layer_ids[0] and self.nodes:
            self.state.cur[0] = value
        if changed is None:
            return None
        return _with_input(args, kwargs, slot, changed)

    def _o_pre(self, layer, module, args, kwargs):
        value, slot = _input_activation(args, kwargs)
        changed = None
        if self.o_override is not None:
            changed = self.o_override(layer, value)
            value = changed
        self.state.o_in[layer] = value
        if self.nodes:
            outputs = _head_outputs(value, self.net.weights(layer)["o"])
            for head in range(outputs.shape[0]):
                self.state.cur[self.net.node_id[("head", layer, head)]] = outputs[head]
        return None if changed is None else _with_input(args, kwargs, slot, changed)

    def _attn_post(self, layer, module, args, output):
        info = self.net.layers[layer]
        value = _activation(output)
        if info["heads"] is None:
            if layer in self.attn_override:
                value = self.attn_override[layer].to(device=value.device, dtype=value.dtype)
                output = _with_output(output, value)
                self.state.attn_out[layer] = value
                if self.nodes:
                    self.state.cur[self.net.node_id[("attn", layer)]] = value
                return output
            if self.nodes:
                self.state.cur[self.net.node_id[("attn", layer)]] = value
        self.state.attn_out[layer] = value
        if self.attention and info["heads"] is not None:
            self.state.attn_weights[layer] = _attention_weights(output)
        return None

    def _mlp_post(self, layer, module, args, output):
        value = _activation(output)
        changed = None
        if layer in self.mlp_override:
            changed = self.mlp_override[layer].to(device=value.device, dtype=value.dtype)
            value = changed
        self.state.mlp_out[layer] = value
        if self.nodes:
            self.state.cur[self.net.node_id[("mlp", layer)]] = value
        return None if changed is None else _with_output(output, changed)

    def _proj_post(self, layer, kinds, module, args, output):
        info = self.net.layers[layer]
        full = _activation(output)
        if self.proj:
            for kind in kinds:
                _, offset, unit, count = info["proj"][kind]
                self.state.proj[(layer, kind)] = full[..., offset : offset + unit * count]
        edits = []
        for kind in kinds:
            _, offset, unit, count = info["proj"][kind]
            for index in range(count):
                key = (kind, layer, index)
                lo = offset + index * unit
                if key in self.record:
                    self.state.inputs[key] = full[..., lo : lo + unit].detach().clone()
                if key in self.replace:
                    edits.append((lo, unit, self.replace[key]))
        if self.engine is not None:
            return _with_output(output, self.engine.projection(self, layer, kinds, full))
        if not edits:
            return None
        full = full.clone()
        for lo, unit, value in edits:
            full[..., lo : lo + unit] = value.to(device=full.device, dtype=full.dtype)
        return _with_output(output, full)

    def _module_pre(self, key, norm, module, args, kwargs):
        """Pre-hook of a module that reads one residual: an MLP or a non-head attention block."""
        value, slot = _input_activation(args, kwargs)
        replaced = self._swap(key, value, value)
        if self.engine is not None:
            replaced = self.engine.module_input(self, key, norm)
            replaced = replaced.to(device=value.device, dtype=value.dtype)
        return None if replaced is None else _with_input(args, kwargs, slot, replaced)

    def _final_pre(self, module, args, kwargs):
        value, slot = _input_activation(args, kwargs)
        self.state.final_in = value
        replaced = self._swap(("logits",), value, value)
        if self.engine is not None:
            replaced = self.engine.final_input(self, value)
        return None if replaced is None else _with_input(args, kwargs, slot, replaced)

    def specs(self) -> list[tuple]:
        """Each hook as ``(module name, function, is_pre_hook)``."""
        net = self.net
        out: list[tuple] = [(net.embedding, self._emb_hook, False)]
        for layer in net.layer_ids:
            info = net.layers[layer]
            out.append((info["name"], partial(self._layer_pre, layer), True))
            out.append((info["attn"], partial(self._attn_post, layer), False))
            if info["heads"]:
                out.append((info["o_module"], partial(self._o_pre, layer), True))
                modules: dict[str, list[str]] = {}
                for kind, (name, *_rest) in info["proj"].items():
                    modules.setdefault(name, []).append(kind)
                for name, kinds in modules.items():
                    out.append((name, partial(self._proj_post, layer, tuple(kinds)), False))
            else:
                key = ("attn", layer)
                out.append((info["attn"], partial(self._module_pre, key, info["norm1"]), True))
            key = ("mlp", layer)
            out.append((info["mlp"], partial(self._module_pre, key, info["norm2"]), True))
            out.append((info["mlp"], partial(self._mlp_post, layer), False))
        out.append((net.final_norm, self._final_pre, True))
        return out

    def run(self, ids: Any) -> Any:
        """Run the model on ``ids``, with these hooks in place, and return all logits."""
        hooks: list[_TemporaryHook] = []
        try:
            for name, function, pre in self.specs():
                hooks.append(_TemporaryHook(self.net.model.get_submodule(name), function, pre=pre))
                hooks[-1].register()
            output = self.net.model(input_ids=ids, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        return output.logits if hasattr(output, "logits") else output[0]


class _Engine:
    """Gives every reader of the residual stream its own residual.

    A reader is a head's q input, a key/value head's k or v input, an MLP or non-head attention
    block, or the final norm. The residual a reader sees is the running residual plus, for each
    removed edge ``u -> reader``, ``corrupt output of u - current output of u``. The running
    residual is the sum of what the nodes upstream currently write, so this is the sum, over
    all upstream nodes, of each node's current output where the edge is kept and its corrupt
    output where it is removed. With nothing removed it is the model's own residual. With
    everything removed it is the corrupt residual, node by node.

    With ``grad`` each reader's residual is a fresh tensor in the autograd graph, kept in
    ``state.leaves``, so one backward pass gives the gradient at every reader.
    """

    def __init__(self, net: _Net, corrupt_nodes: Any = None, drop: Any = None, grad: bool = False):
        self.net, self.corrupt, self.drop, self.grad = net, corrupt_nodes, drop, grad

    def _residual(self, run: _Pass, key: tuple, base: Any, count: int, limit: int) -> Any:
        import torch

        stack = base.unsqueeze(0).repeat(count, 1, 1, 1)
        if self.drop is not None:
            row = self.net.row[key]
            mask = self.drop[row : row + count, :limit]
            if bool(mask.any()):
                current = torch.stack([run.state.cur[i].to(base.device) for i in range(limit)])
                corrupt = self.corrupt[:limit].to(device=base.device, dtype=base.dtype)
                change = torch.einsum("nu,ubth->nbth", mask.to(base), corrupt - current)
                stack = stack + change
        if self.grad:
            run.state.leaves[key] = stack
        return stack

    def projection(self, run: _Pass, layer: int, kinds: tuple, full: Any) -> Any:
        import torch

        info = self.net.layers[layer]
        weights = self.net.weights(layer)
        norm = self.net.model.get_submodule(info["norm1"])
        base = run.state.pre[layer]
        parts = []
        for kind in kinds:
            _, _, unit, count = info["proj"][kind]
            stack = self._residual(run, (kind, layer, 0), base, count, self.net.first[layer])
            normed = norm(stack)
            matrix = weights[kind].to(device=normed.device, dtype=normed.dtype)
            value = torch.einsum("nbth,nrh->nbtr", normed, matrix)
            bias = weights[f"{kind}_bias"]
            if bias is not None:
                value = value + bias.to(value)[:, None, None, :]
            batch, tokens = value.shape[1:3]
            parts.append(value.permute(1, 2, 0, 3).reshape(batch, tokens, count * unit))
        return torch.cat(parts, dim=-1).to(device=full.device, dtype=full.dtype)

    def module_input(self, run: _Pass, key: tuple, norm: str) -> Any:
        layer = key[1]
        state = run.state
        if key[0] == "mlp":
            base = state.pre[layer] + state.attn_out[layer]
            limit = self.net.mlp_node[layer]
        else:
            base = state.pre[layer]
            limit = self.net.first[layer]
        stack = self._residual(run, key, base, 1, limit)
        return self.net.model.get_submodule(norm)(stack)[0]

    def final_input(self, run: _Pass, value: Any) -> Any:
        stack = self._residual(run, ("logits",), value, 1, len(self.net.nodes))
        return stack[0]


# -- the mixin -----------------------------------------------------------------------------


def _metric_function(self: Any, metric: Any) -> Any:
    """A function from last-position logits ``[batch, vocab]`` to a scalar tensor."""
    import torch

    if callable(metric):
        function = metric
    elif (
        isinstance(metric, (tuple, list))
        and len(metric) == 2
        and all(isinstance(item, str) and item for item in metric)
    ):
        answer, other = (_first_token(self.tokenizer, item) for item in metric)
        if answer == other:
            raise ValueError("the answer and counterfactual of a metric pair share a first token")

        def function(logits):
            return (logits[:, answer] - logits[:, other]).mean()

    else:
        raise ValueError(
            "metric must be a callable over last-position logits or an (answer, counterfactual) "
            "pair of strings"
        )

    def checked(logits):
        value = function(logits)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError("metric must return a scalar tensor")
        return value.reshape(())

    return checked


def _first_token(tokenizer: Any, text: str) -> int:
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if not ids:
        raise ValueError(f"{text!r} has no tokens")
    return int(ids[0])


class AnalysesMixin:
    """Lab methods for direct logit attribution, path patching, and edge attribution patching."""

    # -- shared plumbing ---------------------------------------------------------------

    @contextmanager
    def _analysis(self, passes: int = 0):
        """A checked net with the model in eval mode; train/eval modes come back on exit.

        A call over a pass cap is rejected here, before any pass runs.
        """
        _eager(self.model)
        net = _Net(self.model)
        if passes:
            self._budget_state()
            for budget in (self.budget, *self._evaluation_budgets):
                budget.check(passes)
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            yield net
        finally:
            for module, training in modes:
                module.training = training

    def _ids(self, text: str) -> Any:
        if not isinstance(text, str) or not text:
            raise ValueError("prompts must be nonempty strings")
        ids = self.tokenizer(text, return_tensors="pt", return_token_type_ids=False)["input_ids"]
        return ids.to(self.device)

    def _groups(self, clean: Any, corrupt: Any) -> list[dict[str, Any]]:
        """Prompt pairs grouped by token count, each group one batch of clean and corrupt ids."""
        import torch

        for name, value in (("clean", clean), ("corrupt", corrupt)):
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"{name} must be a nonempty list of prompts")
        if len(clean) != len(corrupt):
            raise ValueError("clean and corrupt must hold the same number of prompts")
        by_length: dict[int, list[tuple[Any, Any]]] = {}
        for index, (first, second) in enumerate(zip(clean, corrupt, strict=True)):
            a, b = self._ids(first), self._ids(second)
            if a.shape[1] != b.shape[1]:
                raise ValueError(
                    f"pair {index} has different token counts ({a.shape[1]} and {b.shape[1]}); "
                    "path and edge patching need equal token counts"
                )
            by_length.setdefault(int(a.shape[1]), []).append((a, b))
        total = len(clean)
        return [
            {
                "clean": torch.cat([a for a, _ in rows]),
                "corrupt": torch.cat([b for _, b in rows]),
                "weight": len(rows) / total,
            }
            for _, rows in sorted(by_length.items())
        ]

    def _record(self, net: _Net, ids: Any, **options: Any) -> tuple[_Rec, _Pass]:
        import torch

        run = _Pass(net, **options)
        with torch.no_grad():
            logits = run.run(ids)
        return _Rec(run.state, logits, net, run.nodes), run

    def _verify(self, net: _Net, rec: _Rec) -> None:
        """Refuse a model whose forward is not what the analyses recompute."""
        import torch

        state_layers = net.layer_ids
        total = rec.pre0.clone()
        for layer in state_layers:
            total = total + rec.attn_out[layer] + rec.mlp_out[layer]
        scale = max(1.0, float(rec.final_in.abs().max()))
        tolerance = _tolerance(total.dtype, scale, 0.0, 32.0)
        gap = float((total - rec.final_in).abs().max())
        if gap > tolerance:
            raise _refuse(
                f"the residual stream is not additive: the embeddings plus each layer's attention "
                f"and MLP outputs differ from the final norm's input by {gap:.3g} (limit "
                f"{tolerance:.3g}). A layer scale, residual multiplier, or norm on a sublayer "
                "output can do this"
            )
        for layer in state_layers:
            info = net.layers[layer]
            if not info["heads"]:
                continue
            weights = net.weights(layer)
            written = _head_outputs(rec.o_in[layer], weights["o"]).sum(0)
            if weights["o_bias"] is not None:
                written = written + weights["o_bias"].to(written)
            gap = float((written - rec.attn_out[layer]).abs().max())
            limit = _tolerance(
                written.dtype, max(1.0, float(rec.attn_out[layer].abs().max())), 0.0, 32.0
            )
            if gap > limit:
                raise _refuse(
                    f"{info['attn']}'s output is not the sum of its heads' writes plus the "
                    f"output bias (differs by {gap:.3g})"
                )
        norm = self.model.get_submodule(net.final_norm)
        with torch.no_grad():
            # The weights, not the module: a hook on the head must not hide a rescaling.
            again = torch.nn.functional.linear(
                norm(rec.final_in), net.head.weight, getattr(net.head, "bias", None)
            )
        gap = float((again.to(rec.logits) - rec.logits).abs().max())
        if gap > _tolerance(rec.logits.dtype, max(1.0, float(rec.logits.abs().max())), 0.0, 8.0):
            raise _refuse(
                f"the model's logits are not the output head applied to the final norm "
                f"({net.final_norm}) of the residual stream (differs by {gap:.3g}); a scale or "
                "softcap on the logits can do this"
            )

    def _prepare(self, net: _Net, groups: list[dict[str, Any]]) -> None:
        """Run clean and corrupt once for every group (two passes per pair) and keep them."""
        for group in groups:
            group["c"], _ = self._record(net, group["clean"], nodes=True)
            group["k"], _ = self._record(net, group["corrupt"], nodes=True)
            self._verify(net, group["c"])

    @staticmethod
    def _score(function: Any, logits: Any) -> Any:
        return function(logits[:, -1].float())

    # -- direct logit attribution ------------------------------------------------------

    def logit_attribution(
        self,
        prompt: str,
        target: Any,
        *,
        by: Any = ("component",),
        positions: int = -1,
    ) -> dict[str, Any]:
        """Split one logit, or a logit difference, into what each component wrote.

        ``target`` is an answer string (the first token's logit) or an ``(answer,
        counterfactual)`` pair (the logit difference, answer minus counterfactual); there is no
        callable metric here. The quantity is read at ``positions``, one token index (default
        the last token), the position that predicts the next token.

        The final residual at that position is the sum of the embeddings, each head's write,
        each MLP output, each attention block that is not split into heads, and the
        output-projection biases. The final norm is linearized by freezing its scale at this
        pass's value, and for LayerNorm each node's mean over the hidden dimension is
        subtracted. The contributions plus the norm's bias and the output head's bias then sum
        to the model's own value, which the call checks and returns as ``total`` next to
        ``actual``. Attribution is a linear estimate of a node's effect: ablating a node also
        changes the norm scale and everything downstream.

        ``by`` is a list of "component" and "position". With "position", each head's entry has
        ``positions``, its contribution split over source tokens by attention weight times
        value (times the query gate for gated heads); the parts sum to the head's value.

        Returns ``{"total", "actual", "tokens", "embeddings", "heads": [{"layer", "head",
        "value", "positions"?}], "mlps": [{"layer", "value"}], "mixers": [{"layer",
        "value"}], "biases": {"attention_output": [{"layer", "value"}], "final_norm",
        "output_head"}}``. MLP values include any bias of the MLP's own output layer.

        Costs one forward pass. Raises NotImplementedError for models without an additive
        residual stream or where the recomputation does not match the model, as described in
        this module's docstring, and ValueError for a bad ``by``, ``target`` or position.
        """
        by = (by,) if isinstance(by, str) else tuple(by)
        if not by or any(item not in _BY for item in by):
            raise ValueError(f"by must be a nonempty list drawn from {list(_BY)}")
        if isinstance(target, str):
            tokens = [_first_token(self.tokenizer, target)]
        elif isinstance(target, (tuple, list)) and len(target) == 2:
            tokens = [_first_token(self.tokenizer, item) for item in target]
            if tokens[0] == tokens[1]:
                raise ValueError(
                    "the answer and counterfactual of a target pair share a first token"
                )
        else:
            raise ValueError("target must be an answer string or an (answer, counterfactual) pair")
        ids = self._ids(prompt)
        size = ids.shape[1]
        if type(positions) is not int or not -size <= positions < size:
            raise ValueError(f"position must be one integer token index in [-{size}, {size - 1}]")
        at = positions % size
        with self._analysis(passes=1) as net:
            rec, _ = self._record(net, ids, proj="position" in by, attention="position" in by)
            self._verify(net, rec)
            norm = self.model.get_submodule(net.final_norm)
            x = rec.final_in[0, at].double().cpu()
            lin = _LinearNorm(norm, x)
            head = net.head
            unembed = head.weight.detach().double().cpu()
            direction = (
                unembed[tokens[0]] if len(tokens) == 1 else unembed[tokens[0]] - unembed[tokens[1]]
            )
            head_bias = getattr(head, "bias", None)
            offset = 0.0
            if head_bias is not None:
                bias = head_bias.detach().double().cpu()
                offset = float(bias[tokens[0]] - (bias[tokens[1]] if len(tokens) == 2 else 0))

            def value(vector):
                return float(direction @ lin(vector.double().cpu()))

            logits = rec.logits[0, at].double().cpu()
            actual = float(logits[tokens[0]] - (logits[tokens[1]] if len(tokens) == 2 else 0))
            result: dict[str, Any] = {
                "tokens": [self.tokenizer.decode([int(i)]).strip() for i in ids[0]],
                "embeddings": value(rec.pre0[0, at]),
                "heads": [],
                "mlps": [],
                "mixers": [],
                "biases": {"attention_output": []},
            }
            for layer in net.layer_ids:
                info = net.layers[layer]
                if info["heads"]:
                    weights = net.weights(layer)
                    own = _head_outputs(rec.o_in[layer][:, at : at + 1], weights["o"])[:, 0, 0]
                    for head_index in range(info["nh"]):
                        entry = {
                            "layer": layer,
                            "head": head_index,
                            "value": value(own[head_index]),
                        }
                        if "position" in by:
                            entry["positions"] = self._by_position(
                                net, rec, layer, head_index, at, direction, lin
                            )
                        result["heads"].append(entry)
                    if weights["o_bias"] is not None:
                        result["biases"]["attention_output"].append(
                            {"layer": layer, "value": value(weights["o_bias"])}
                        )
                else:
                    result["mixers"].append(
                        {"layer": layer, "value": value(rec.attn_out[layer][0, at])}
                    )
                result["mlps"].append({"layer": layer, "value": value(rec.mlp_out[layer][0, at])})
            result["biases"]["final_norm"] = float(direction @ lin.beta)
            result["biases"]["output_head"] = offset
            total = (
                result["embeddings"]
                + sum(item["value"] for item in result["heads"])
                + sum(item["value"] for item in result["mlps"])
                + sum(item["value"] for item in result["mixers"])
                + sum(item["value"] for item in result["biases"]["attention_output"])
                + result["biases"]["final_norm"]
                + offset
            )
            # Each node's output is rounded to the model's dtype, so the sum may differ from
            # the model's value by rounding steps at the size of the parts, not of the total.
            parts = [abs(result["embeddings"])]
            parts += [
                abs(item["value"]) for key in ("heads", "mlps", "mixers") for item in result[key]
            ]
            parts += [abs(item["value"]) for item in result["biases"]["attention_output"]]
            size = 1 + abs(actual) + sum(parts)
            limit = max(1e-3 * (1 + abs(actual)), _tolerance(rec.logits.dtype, size, 0.0, 4.0))
            if abs(total - actual) > limit:
                raise _refuse(
                    f"the contributions sum to {total:.6g} but the model's own value is "
                    f"{actual:.6g} (limit {limit:.3g})"
                )
        return {"total": total, "actual": actual, **result}

    @staticmethod
    def _by_position(net, rec, layer, head, at, direction, lin):
        """One head's value split over source tokens: attention weight times value (times gate)."""
        import torch

        info = net.layers[layer]
        weights = net.weights(layer)
        kv = head // (info["nh"] // info["nkv"])
        hd = info["hd"]
        attention = rec.attn_weights[layer][0, head, at].double().cpu()  # [keys]
        v = rec.proj[(layer, "v")][0].double().cpu().reshape(-1, info["nkv"], hd)[:, kv]
        mixed = attention[:, None] * v[: len(attention)]
        if info["gated"]:
            q = rec.proj[(layer, "q")][0, at].double().cpu().reshape(info["nh"], 2 * hd)
            mixed = mixed * torch.sigmoid(q[head, hd:])
        rebuilt = mixed.sum(0)
        written = rec.o_in[layer][0, at, head * hd : (head + 1) * hd].double().cpu()
        scale = max(1.0, float(written.abs().max()))
        if float((rebuilt - written).abs().max()) > _tolerance(rec.o_in[layer].dtype, scale, 1e-4):
            raise _refuse(
                f"{info['attn']} head {head}: attention weights times value vectors do not "
                "rebuild the head's output, so it cannot be split over source positions"
            )
        wo = weights["o"][head].double().cpu()  # [hidden, hd]
        return [float(direction @ lin(row @ wo.T)) for row in mixed]

    # -- path patching -----------------------------------------------------------------

    def path_patch(
        self,
        clean: list[str],
        corrupt: list[str],
        senders: list[Any],
        receivers: list[Any],
        *,
        metric: Any,
        freeze_mlps: bool = False,
    ) -> dict[str, Any]:
        """Patch the corrupt value of some senders along paths into some receivers.

        ``clean`` and ``corrupt`` are equal-length lists of prompts, paired by position, each
        pair with the same token count. ``metric`` is a callable over last-position logits
        ``[batch, vocab]`` or an ``(answer, counterfactual)`` pair of strings (the mean logit
        difference); the result averages over pairs.

        Senders are heads ``(layer, head)``, MLPs ``("mlp", layer)``, ``"embed"`` for the
        embeddings, and ``("attn", layer)`` for a layer whose attention is not split into
        heads. Receivers are head inputs ``(layer, head, "q" | "k" | "v")``, MLP inputs
        ``("mlp", layer)``, ``("attn", layer)``, and ``"logits"`` (the residual entering the
        final norm). A ``k`` or ``v`` receiver is the key/value head that ``head`` reads, which
        in a grouped-query model is shared by the whole group. Senders are patched at every
        position. Receivers need at least one entry; senders may be empty.

        Following Wang et al. (2022): record clean and corrupt runs; run clean again with the
        senders at their corrupt outputs and every other head (and attention block) frozen at
        its clean output, and every other MLP too when ``freeze_mlps``, recording each
        receiver's input; then run clean with only those receiver inputs replaced. Without
        ``freeze_mlps`` the MLPs between a sender and a receiver respond, so the path includes
        them. What is frozen decides what the result means.

        Returns ``{"clean", "corrupt", "patched", "fraction"}`` where ``fraction`` is
        ``(patched - clean) / (corrupt - clean)``, or None when clean equals corrupt. The
        embeddings differ between a clean and a corrupt prompt, so the patched metric reaches
        the corrupt one only when "embed" is also a sender.

        Costs four forward passes per prompt pair. Raises ValueError for unequal token
        counts, unknown senders or receivers, and NotImplementedError as ``logit_attribution``
        does.
        """
        import torch

        function = _metric_function(self, metric)
        groups = self._groups(clean, corrupt)
        if not isinstance(receivers, (list, tuple)) or not receivers:
            raise ValueError("receivers must be a nonempty list")
        if not isinstance(senders, (list, tuple)):
            raise ValueError("senders must be a list")
        passes = 4 * len(clean)
        with self._analysis(passes=passes) as net:
            sender_keys = {net.node_key(spec) for spec in senders}
            receiver_keys = [net.input_key(spec) for spec in receivers]
            totals = {"clean": 0.0, "corrupt": 0.0, "patched": 0.0}
            for group in groups:
                c, _ = self._record(net, group["clean"])
                k, _ = self._record(net, group["corrupt"])
                self._verify(net, c)
                frozen = self._freeze(net, c, k, sender_keys, freeze_mlps)
                _, middle = self._record(net, group["clean"], record=tuple(receiver_keys), **frozen)
                with torch.no_grad():
                    patched = _Pass(net, replace=middle.state.inputs).run(group["clean"])
                for name, logits in (
                    ("clean", c.logits),
                    ("corrupt", k.logits),
                    ("patched", patched),
                ):
                    totals[name] += group["weight"] * float(self._score(function, logits))
        span = totals["corrupt"] - totals["clean"]
        totals["fraction"] = None if span == 0 else (totals["patched"] - totals["clean"]) / span
        return totals

    @staticmethod
    def _freeze(net: _Net, c: _Rec, k: _Rec, senders: set, freeze_mlps: bool) -> dict[str, Any]:
        """Hook options for the second run: senders corrupt, other heads and blocks frozen."""
        sender_heads: dict[int, list[int]] = {}
        for key in senders:
            if key[0] == "head":
                sender_heads.setdefault(key[1], []).append(key[2])

        def o_override(layer, value):
            new = c.o_in[layer].clone().to(device=value.device, dtype=value.dtype)
            hd = net.layers[layer]["hd"]
            for head in sender_heads.get(layer, []):
                new[..., head * hd : (head + 1) * hd] = k.o_in[layer][
                    ..., head * hd : (head + 1) * hd
                ]
            return new

        mlp: dict[int, Any] = {}
        attn: dict[int, Any] = {}
        for layer in net.layer_ids:
            if ("mlp", layer) in senders:
                mlp[layer] = k.mlp_out[layer]
            elif freeze_mlps:
                mlp[layer] = c.mlp_out[layer]
            if net.layers[layer]["heads"] is None:
                attn[layer] = (k if ("attn", layer) in senders else c).attn_out[layer]
        return {
            "o_override": o_override,
            "mlp_override": mlp,
            "attn_override": attn,
            "embed_override": k.pre0 if ("embed",) in senders else None,
        }

    # -- edge attribution patching -----------------------------------------------------

    def _edge_scores(self, net: _Net, groups: list[dict[str, Any]], function: Any, steps: int):
        """Integrated-gradient edge scores as a float64 matrix ``[inputs, nodes]``."""
        import torch

        scores = torch.zeros(len(net.inputs), len(net.nodes), dtype=torch.float64)
        for group in groups:
            c, k = group["c"], group["k"]
            device = c.nodes[0].device
            delta = torch.stack([n.to(device) for n in k.nodes]) - torch.stack(
                [n.to(device) for n in c.nodes]
            )
            for alpha in torch.linspace(0, 1, steps).tolist():
                point = (c.emb + alpha * (k.emb - c.emb)).detach().clone().requires_grad_(True)
                run = _Pass(net, emb_replace=point, engine=_Engine(net, grad=True))
                with torch.enable_grad():
                    logits = run.run(group["clean"])
                    value = self._score(function, logits)
                    keys = list(run.state.leaves)
                    grads = torch.autograd.grad(
                        value, [run.state.leaves[key] for key in keys], allow_unused=True
                    )
                for key, grad in zip(keys, grads, strict=True):
                    if grad is None:
                        continue
                    row = net.row[key]
                    count = grad.shape[0]
                    limit = net.limit[row]
                    part = torch.einsum(
                        "nbth,ubth->nu",
                        grad.detach().double().cpu(),
                        delta[:limit].double().cpu(),
                    )
                    scores[row : row + count, :limit] += part * group["weight"] / steps
        return scores

    @staticmethod
    def _edge_list(net: _Net, scores: Any) -> list[dict[str, Any]]:
        edges = []
        for row, key in enumerate(net.inputs):
            for node in range(net.limit[row]):
                edges.append(
                    {
                        "sender": net.label(net.nodes[node]),
                        "receiver": net.input_label(key),
                        "score": float(scores[row, node]),
                    }
                )
        edges.sort(key=lambda edge: abs(edge["score"]), reverse=True)
        return edges

    @staticmethod
    def _check_steps(steps: Any) -> None:
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be an integer of at least 1")

    def edge_attribution(
        self,
        clean: list[str],
        corrupt: list[str],
        *,
        metric: Any,
        steps: int = 5,
    ) -> list[dict[str, Any]]:
        """Score every edge of the model's computation graph by edge attribution patching.

        ``clean``, ``corrupt`` and ``metric`` are as for ``path_patch``. An upstream node is
        the embeddings (``"embed"``), a head ``(layer, head)``, an MLP ``("mlp", layer)``, or
        an attention block ``("attn", layer)`` that is not split into heads. A downstream input
        is a head's ``(layer, head, "q")``, a key/value head's ``(layer, kv_head, "k")`` or
        ``"v"`` (the key/value head's own index, which is the head's index only without
        grouped queries), ``("mlp", layer)``, ``("attn", layer)``, or ``"logits"``. An edge
        links a node to every input that comes later in the forward pass.

        The score of ``u -> v`` is ``(output of u on corrupt - output of u on clean)``, dotted
        with the gradient of the metric at the residual stream as ``v`` reads it, summed over
        positions and pairs and averaged over ``steps`` points that move the input token
        embeddings from clean (the first point) to corrupt (the last; one step uses only the
        clean point) with the clean prompt's other inputs (Hanna et al., 2024; one step is
        plain attribution patching). The gradient at each input comes from one backward pass
        per step, through that reader's own norm and projection slice. The score is a first-
        order estimate of what removing the edge does to the metric, so confirm a circuit
        built from it with ``find_circuit``, which patches for real.

        Returns every edge as ``{"sender", "receiver", "score"}``, sorted by absolute score.
        Costs ``2 + steps`` forward passes per prompt pair, plus a backward pass per step.
        Raises as ``path_patch`` does, and ValueError for a bad ``steps``.
        """
        function = _metric_function(self, metric)
        self._check_steps(steps)
        groups = self._groups(clean, corrupt)
        with self._analysis(passes=(2 + steps) * len(clean)) as net:
            self._prepare(net, groups)
            scores = self._edge_scores(net, groups, function, steps)
            return self._edge_list(net, scores)

    def find_circuit(
        self,
        clean: list[str],
        corrupt: list[str],
        *,
        metric: Any,
        edges: list[dict[str, Any]] | None = None,
        top: int = 50,
        steps: int = 5,
    ) -> dict[str, Any]:
        """Propose a circuit from edge scores, then test it by patching.

        Without ``edges`` the call runs ``edge_attribution`` and keeps the ``top`` edges by
        absolute score. With ``edges`` (dicts with ``sender`` and ``receiver`` as
        ``edge_attribution`` returns them, so a list from a saved result works) it tests those,
        and an empty list tests the empty circuit. Edges that are not on a path from the
        embeddings to the logits through other kept edges are dropped.

        The circuit run is the clean prompts with every edge outside the circuit replaced by
        its corrupt value: each reader gets the residual that sums, over earlier nodes, the
        node's current output where its edge is kept and its corrupt output where it is not,
        and sends that through its own norm and projection slice. Keeping every edge gives
        the clean metric and keeping none gives the corrupt one.

        Returns ``{"clean", "corrupt", "circuit", "faithfulness", "edges", "total_edges"}``.
        ``faithfulness`` is ``(circuit - corrupt) / (clean - corrupt)`` (None when they are
        equal) and ``edges`` are the kept edges, as given or scored. Faithfulness here is
        measured on the prompts the circuit was found on; test other prompts before trusting
        it.

        Costs ``3 + steps`` forward passes per prompt pair, or 3 when ``edges`` is given.
        Raises ValueError for a bad ``top``, ``steps`` or edge, and as ``path_patch`` does.
        """
        import torch

        function = _metric_function(self, metric)
        self._check_steps(steps)
        if type(top) is not int or top < 1:
            raise ValueError("top must be a positive integer")
        groups = self._groups(clean, corrupt)
        passes = (3 if edges is not None else 3 + steps) * len(clean)
        with self._analysis(passes=passes) as net:
            if edges is not None:
                if not isinstance(edges, (list, tuple)):
                    raise ValueError("edges must be a list of {'sender', 'receiver'} dicts")
                chosen = [self._edge_key(net, edge) for edge in edges]
                given = list(edges)
            else:
                chosen, given = [], []
            self._prepare(net, groups)
            if edges is None:
                scores = self._edge_scores(net, groups, function, steps)
                listed = self._edge_list(net, scores)[:top]
                chosen = [self._edge_key(net, edge) for edge in listed]
                given = listed
            kept_pairs = self._prune(net, set(chosen))
            kept_edges = [
                edge for edge, pair in zip(given, chosen, strict=True) if pair in kept_pairs
            ]
            drop = torch.zeros(len(net.inputs), len(net.nodes))
            for row in range(len(net.inputs)):
                drop[row, : net.limit[row]] = 1
            for node, row in kept_pairs:
                drop[row, node] = 0
            totals = {"clean": 0.0, "corrupt": 0.0, "circuit": 0.0}
            for group in groups:
                c, k = group["c"], group["k"]
                device = k.nodes[0].device
                engine = _Engine(
                    net, corrupt_nodes=torch.stack([n.to(device) for n in k.nodes]), drop=drop
                )
                run = _Pass(net, engine=engine)
                with torch.no_grad():
                    logits = run.run(group["clean"])
                for name, values in (
                    ("clean", c.logits),
                    ("corrupt", k.logits),
                    ("circuit", logits),
                ):
                    totals[name] += group["weight"] * float(self._score(function, values))
        span = totals["clean"] - totals["corrupt"]
        totals["faithfulness"] = (
            None if span == 0 else (totals["circuit"] - totals["corrupt"]) / span
        )
        totals["edges"] = kept_edges
        totals["total_edges"] = sum(net.limit)
        return totals

    @staticmethod
    def _edge_key(net: _Net, edge: Any) -> tuple[int, int]:
        if not isinstance(edge, dict) or "sender" not in edge or "receiver" not in edge:
            raise ValueError("each edge must be a dict with 'sender' and 'receiver'")
        node = net.node_id[net.node_key(edge["sender"])]
        row = net.row[net.input_key(edge["receiver"], kv=True)]
        if node >= net.limit[row]:
            raise ValueError(f"edge {edge['sender']!r} -> {edge['receiver']!r} does not go forward")
        return node, row

    @staticmethod
    def _prune(net: _Net, pairs: set) -> set:
        """The edges that lie on a path from the embeddings to the logits within ``pairs``."""
        while True:
            reached = {0}
            grew = True
            while grew:
                grew = False
                for node, row in pairs:
                    if node in reached:
                        for owner in net.owners(row):
                            if owner not in reached:
                                reached.add(owner)
                                grew = True
            forward = {(node, row) for node, row in pairs if node in reached}
            useful = {-1}
            grew = True
            while grew:
                grew = False
                for node, row in forward:
                    if node not in useful and any(owner in useful for owner in net.owners(row)):
                        useful.add(node)
                        grew = True
            both = {
                (node, row)
                for node, row in forward
                if any(owner in useful for owner in net.owners(row))
            }
            if both == pairs:
                return both
            pairs = both


class _LinearNorm:
    """A final norm with its scale frozen at one residual vector, so it is linear.

    ``norm(x) = centered(x) * scale * gain + beta`` for LayerNorm, with ``centered`` the
    mean-subtracted vector; RMS norms do not center and have no beta. The gain is read from
    the module, by probing it with a vector of ones for RMS-style norms, so ``1 + weight``
    variants need no special case. The constructor checks the formula against the module on
    ``x`` and refuses a norm it does not reproduce.
    """

    def __init__(self, module: Any, x: Any) -> None:
        import torch
        from torch import nn

        self.centered = isinstance(module, nn.LayerNorm)
        hidden = x.shape[0]
        if self.centered:
            eps = float(module.eps)
            self.gain = (
                module.weight.detach().double().cpu()
                if module.weight is not None
                else torch.ones(hidden, dtype=torch.float64)
            )
            self.beta = (
                module.bias.detach().double().cpu()
                if module.bias is not None
                else torch.zeros(hidden, dtype=torch.float64)
            )
        else:
            eps = getattr(module, "variance_epsilon", getattr(module, "eps", None))
            if eps is None:
                raise _refuse(f"{type(module).__name__} has no epsilon, so it is not a known norm")
            parameter = next(module.parameters(), None)
            ones = torch.ones(1, 1, hidden)
            if parameter is not None:
                ones = ones.to(device=parameter.device, dtype=parameter.dtype)
            with torch.no_grad():
                probe = module(ones)[0, 0].double().cpu()
            self.gain = probe * ((1.0 + eps) ** 0.5)
            self.beta = torch.zeros(hidden, dtype=torch.float64)
        center = x - x.mean() if self.centered else x
        spread = (center**2).mean()
        self.scale = float((spread + eps) ** -0.5)
        parameter = next(module.parameters(), None)
        shaped = x.float().reshape(1, 1, -1)
        if parameter is not None:
            shaped = shaped.to(device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad():
            actual = module(shaped)[0, 0].double().cpu()
        gap = float((actual - (self(x) + self.beta)).abs().max())
        dtype = parameter.dtype if parameter is not None else torch.float32
        limit = _tolerance(dtype, max(1.0, float(actual.abs().max())), 1e-3)
        if gap > limit:
            raise _refuse(
                f"the final norm {type(module).__name__} does not match LayerNorm or an RMS "
                f"norm on this residual (differs from the module by {gap:.3g}, limit "
                f"{limit:.3g}), so it cannot be linearized"
            )

    def __call__(self, vector: Any) -> Any:
        centered = vector - vector.mean() if self.centered else vector
        return centered * self.scale * self.gain
