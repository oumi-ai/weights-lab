"""Attribution graphs: which features drive which others on one prompt, from transcoders.

The graph describes a replacement model: the session's model with every MLP's output written
as its transcoder's features times their decoder rows plus an error term, with attention
patterns and norm denominators held at their real values. Each of those changes keeps every
forward value exactly as the real model computes it and changes only the gradients, so the
model is affine in its leaves (the token embeddings, the features, and the error terms). An
edge is a source's value times the gradient of a target with respect to it, so a target's
incoming edges plus its constant part add up to its value.
"""

from __future__ import annotations

import heapq
import json
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from .circuits import _blocks, _layer_index
from .inspection import (
    _activation,
    _attention_weights,
    _eager,
    _input_activation,
    _is_int,
    _TemporaryHook,
    _with_input,
    _with_output,
)
from .saes import LabSAE

# Where a transcoder may read and write for its layer to count as an MLP replacement.
_READS = ("ln_2", "post_attention_layernorm", "pre_feedforward_layernorm", "mlp")
_WRITES = ("mlp", "post_feedforward_layernorm")
_TOLERANCE = 1e-4


def _allowed(scale: float) -> float:
    """The error the exactness checks accept for values of this size."""
    return _TOLERANCE * max(1.0, abs(scale))


# Norms


def _norm_kind(module: Any) -> tuple[str, float] | None:
    """("layer" or "rms", epsilon) for a normalization module, None for any other module."""
    from torch import nn

    name = type(module).__name__
    if isinstance(module, nn.LayerNorm):
        if len(module.normalized_shape) != 1:
            raise ValueError(f"{name} normalizes over more than the last dimension")
        return "layer", float(module.eps)
    if name.endswith("RMSNorm"):
        eps = getattr(module, "eps", getattr(module, "variance_epsilon", None))
        if eps is None:
            raise ValueError(f"{name} has no epsilon, so its normalization cannot be frozen")
        return "rms", float(eps)
    if "norm" in name.lower():
        raise ValueError(f"cannot freeze the denominator of {name}; only LayerNorm and RMSNorm")
    return None


def _denominator(kind: str, eps: float, x: Any) -> Any:
    """The scalar a norm divides each vector by: its standard deviation or root mean square."""
    import torch

    if kind == "layer":
        return torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
    return torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _site_denominator(transcoder: LabSAE, x: Any) -> Any:
    """The denominator of the normalization in a transcoder's input transform, or None."""
    if transcoder.site.norm is None:
        return None
    kind, eps = transcoder.site.norm
    return _denominator(kind, eps, x.float())


def _encoder_input(transcoder: LabSAE, x: Any, denominator: Any) -> Any:
    """``LabSAE.transform`` with the normalization's denominator held fixed."""
    site = transcoder.site
    x = x.float()
    if site.norm is not None:
        if site.norm[0] == "layer":
            x = x - x.mean(-1, keepdim=True)
        x = x / denominator.to(x.device)
    else:
        if site.center:
            x = x - x.mean(-1, keepdim=True)
        if site.subtract is not None:
            x = x - site.subtract.to(device=x.device, dtype=x.dtype)
    return x.to(device=transcoder.device, dtype=transcoder.dtype)


# Leaves and evaluations


@dataclass
class _Leaves:
    """The values the replacement model is affine in.

    ``embed`` is the first layer's input ``[1, tokens, hidden]``, ``errors`` maps a layer to
    its error terms ``[tokens, hidden]``, and ``acts`` maps a layer to the activations of its
    chosen features ``[count]``.
    """

    embed: Any
    errors: dict[int, Any]
    acts: dict[int, Any]

    def sources(self) -> list[Any]:
        layers = sorted(self.errors)
        return [self.embed, *(self.errors[i] for i in layers), *(self.acts[i] for i in layers)]

    def map(self, function: Any) -> _Leaves:
        return _Leaves(
            function(self.embed),
            {i: function(v) for i, v in self.errors.items()},
            {i: function(v) for i, v in self.acts.items()},
        )


@dataclass
class _Evaluation:
    targets: Any  # every chosen feature's pre-activation, in node order
    logits: Any  # the last position's logits ``[vocab]``


def _check_float32(model: Any) -> None:
    import torch

    dtypes = {p.dtype for p in model.parameters()}
    if dtypes != {torch.float32}:
        names = ", ".join(sorted(str(d).removeprefix("torch.") for d in dtypes))
        raise ValueError(
            f"attribution graphs need a float32 model so the exactness checks can hold; this "
            f"session's parameters are {names}. Start a session with dtype float32"
        )


@contextmanager
def _frozen_attention(model: Any, patterns: dict[Any, Any]):
    """Run attention with the recorded patterns, for the duration of the context.

    Registers an attention function (and the mask function that goes with it) under a name of
    its own, points every config that names an attention implementation at it, and restores
    both the configs and the registries on exit, even after an error.
    """
    import torch
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

    name = f"weights_lab_frozen_{id(patterns)}"

    def attend(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        if module not in patterns:
            raise RuntimeError("an attention module ran that the pattern recording did not see")
        weights = patterns[module].to(device=value.device, dtype=value.dtype)
        repeats = weights.shape[1] // value.shape[1]
        if repeats > 1:
            value = value.repeat_interleave(repeats, dim=1)
        return torch.matmul(weights, value).transpose(1, 2), weights

    configs: dict[int, Any] = {}
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            configs.setdefault(id(config), config)
    previous = {key: config._attn_implementation for key, config in configs.items()}
    try:
        AttentionInterface.register(name, attend)
        AttentionMaskInterface.register(name, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
        for config in configs.values():
            config._attn_implementation = name
        yield
    finally:
        for key, config in configs.items():
            config._attn_implementation = previous[key]
        AttentionInterface._global_mapping.pop(name, None)
        AttentionMaskInterface._global_mapping.pop(name, None)


def _hook(stack: ExitStack, module: Any, function: Any, pre: bool = False) -> None:
    hook = _TemporaryHook(module, function, pre=pre)
    stack.callback(hook.remove)  # tracked first, so an interrupt cannot orphan the hook
    hook.register()


class _Replacement:
    """The replacement model on one prompt: one recording pass, then evaluations.

    Building one runs the real model once and records, with no gradient, what the replacement
    holds fixed: the attention patterns, each norm's denominator, and each transcoder's input
    and the MLP's real output. ``evaluate`` runs the model once more with those held fixed and
    the leaves set to the values given, so it returns the real model's numbers for the real
    leaves, the constant part for zeros, and a finite difference for a scaled leaf. Every
    forward call counts a pass; ``passes`` says how many this object has made.
    """

    def __init__(
        self, lab: Any, transcoders: dict[int, LabSAE], inputs: dict, max_features: int | None
    ) -> None:
        self.lab, self.model = lab, lab.model
        self.transcoders = dict(sorted(transcoders.items()))
        self.layers = list(self.transcoders)
        self.inputs = inputs
        self.positions = int(inputs["input_ids"].shape[1])
        self.passes = 0
        found = _blocks(self.model)
        self.attention = {_layer_index(name): self.model.get_submodule(name) for name, _ in found}
        first = min(self.attention)
        self.first_block = self.model.get_submodule(
            next(n for n, _ in found if _layer_index(n) == first).rsplit(".", 1)[0]
        )
        self.norms = []
        for module in self.model.modules():
            kind = _norm_kind(module)
            if kind is not None:
                self.norms.append((module, *kind))
        self.patterns: dict[Any, Any] = {}
        self.rho: dict[Any, Any] = {}
        self.denominators: dict[int, Any] = {}
        self._record(max_features)

    # Pass 1

    def _record(self, max_features: int | None) -> None:
        import torch

        reads: dict[int, Any] = {}
        writes: dict[int, Any] = {}
        seen: dict[Any, int] = {}
        start: list[Any] = []
        with ExitStack() as stack:

            def begin(_, args, kwargs):
                start.append(_input_activation(args, kwargs)[0].detach())

            _hook(stack, self.first_block, begin, pre=True)
            for layer, transcoder in self.transcoders.items():

                def read(_, args, kwargs, layer=layer):
                    reads[layer] = _input_activation(args, kwargs)[0].detach()

                def write(_, args, output, layer=layer):
                    writes[layer] = _activation(output).detach()

                _hook(stack, self.model.get_submodule(transcoder.module), read, pre=True)
                _hook(stack, self.model.get_submodule(transcoder.output_module), write)
            for module in self.attention.values():

                def keep(owner, args, output):
                    self.patterns[owner] = _attention_weights(output).detach()

                _hook(stack, module, keep)
            for module, kind, eps in self.norms:

                def measure(owner, args, output, kind=kind, eps=eps):
                    seen[owner] = seen.get(owner, 0) + 1
                    self.rho[owner] = _denominator(kind, eps, args[0].detach())

                _hook(stack, module, measure)
            with torch.no_grad():
                logits = self._call()
        repeated = [m for m, count in seen.items() if count > 1]
        if repeated:
            raise ValueError(
                f"{type(repeated[0]).__name__} ran more than once in a forward pass, so its "
                "denominator cannot be frozen"
            )
        self.real_logits = logits[0, -1].detach().float()
        self._embed = start[0]
        self._reads, self._writes = reads, writes
        self._choose(max_features)

    def _choose(self, max_features: int | None) -> None:
        """Find each layer's active features, keep the largest ``max_features``, set the leaves."""
        import torch

        candidates = []
        for layer, transcoder in self.transcoders.items():
            x = self._reads[layer][0]
            self.denominators[layer] = _site_denominator(transcoder, x)
            with torch.no_grad():
                shaped = _encoder_input(transcoder, x, self.denominators[layer])
                if not torch.allclose(shaped, transcoder.transform(x), atol=_TOLERANCE):
                    raise ValueError(
                        f"layer {layer}: the transcoder's input transform could not be "
                        "reproduced with a frozen normalization"
                    )
                activations = transcoder.encode(x)
                fired = activations > 0
                pre = shaped @ transcoder.sae.W_enc + transcoder.sae.b_enc
                if fired.any():
                    gap = float((pre - activations)[fired].abs().max())
                    if gap > _allowed(float(activations.max())):
                        raise ValueError(
                            f"layer {layer}: where a feature fires its activation differs "
                            f"from its encoder pre-activation by {gap:.3g}; only ReLU, "
                            "JumpReLU and top-k encoders are supported"
                        )
                positions, features = fired.nonzero(as_tuple=True)
                values = activations[fired].float()
                self._check_decoder(layer, transcoder, activations, positions, features, values)
            candidates += zip(
                [layer] * len(values),
                positions.tolist(),
                features.tolist(),
                values.tolist(),
                strict=True,
            )
        if max_features is not None and len(candidates) > max_features:
            candidates = sorted(candidates, key=lambda item: -item[3])[:max_features]
        candidates.sort(key=lambda item: item[:3])
        self.chosen: dict[int, list[tuple[int, int, float]]] = {i: [] for i in self.layers}
        for layer, position, feature, value in candidates:
            self.chosen[layer].append((position, feature, value))
        self._build_leaves()

    def _constant(self, transcoder: LabSAE) -> Any:
        import torch

        zero = torch.zeros(1, transcoder.d_sae, device=transcoder.device, dtype=transcoder.dtype)
        return transcoder.decode(zero)[0].detach().float().cpu()

    def _check_decoder(
        self,
        layer: int,
        transcoder: LabSAE,
        activations: Any,
        positions: Any,
        features: Any,
        values: Any,
    ) -> None:
        """The decoder rows and constant must reproduce ``decode``, or edges would be wrong."""
        built = self._constant(transcoder).expand(activations.shape[0], -1).clone()
        if features.numel():
            rows = transcoder.decoder_rows(features.tolist()).cpu()
            built = built.index_add(0, positions.cpu(), values.cpu()[:, None] * rows)
        made = transcoder.decode(activations).detach().float().cpu()
        gap = float((built - made).abs().max())
        if gap > _allowed(float(made.abs().max())):
            raise ValueError(
                f"layer {layer}: the decoder rows do not reproduce the transcoder's "
                f"reconstruction (difference {gap:.3g})"
            )

    def _build_leaves(self) -> None:
        import torch

        self.slots: dict[int, tuple[int, int]] = {}
        self.position_index: dict[int, Any] = {}
        self.rows: dict[int, Any] = {}
        self.constants: dict[int, Any] = {}
        self.encoders: dict[int, tuple[Any, Any]] = {}
        self.skips: dict[int, Any] = {}
        acts: dict[int, Any] = {}
        errors: dict[int, Any] = {}
        offset = 0
        for layer, transcoder in self.transcoders.items():
            entries = self.chosen[layer]
            self.slots[layer] = (offset, offset + len(entries))
            offset += len(entries)
            features = [f for _, f, _ in entries]
            positions = torch.tensor([p for p, _, _ in entries], dtype=torch.long)
            real = self._writes[layer][0].float()
            self.constants[layer] = self._constant(transcoder)
            self.rows[layer] = (
                transcoder.decoder_rows(features).cpu()
                if features
                else torch.zeros(0, real.shape[-1])
            )
            with torch.no_grad():
                index = torch.tensor(features, dtype=torch.long, device=transcoder.device)
                self.encoders[layer] = (
                    transcoder.sae.W_enc[:, index].T.detach().contiguous(),
                    transcoder.sae.b_enc[index].detach(),
                )
            skip = getattr(transcoder.sae, "W_skip", None)
            self.skips[layer] = None if skip is None else skip.T.detach()
            self.position_index[layer] = positions
            values = torch.tensor([v for _, _, v in entries], dtype=torch.float32)
            acts[layer] = values.to(real.device)
            made = self._reconstruction(layer, acts[layer], real.shape[0], real)
            with torch.no_grad():
                shaped = _encoder_input(transcoder, self._reads[layer][0], self.denominators[layer])
                made = made + self._skip(layer, shaped, real)
            errors[layer] = (real - made).detach()
        self._leaves = _Leaves(self._embed.detach().float(), errors, acts)
        self.real_targets = torch.cat([acts[i] for i in self.layers]).to(self._embed.device)

    def _reconstruction(self, layer: int, acts: Any, positions: int, like: Any) -> Any:
        """The transcoder's output for the chosen features: its constant plus their rows."""
        import torch

        device = like.device
        rows = self.rows[layer].to(device)
        spread = torch.zeros(positions, rows.shape[-1], device=device, dtype=like.dtype)
        spread = spread.index_add(
            0, self.position_index[layer].to(device), acts.to(device)[:, None] * rows
        )
        return self.constants[layer].to(device) + spread

    def _skip(self, layer: int, encoded: Any, like: Any) -> Any:
        """A skip transcoder's ``W_skip`` term: linear in its (frozen-norm) input, else zero."""
        skip = self.skips[layer]
        if skip is None:
            return 0.0
        return (encoded @ skip).float().to(device=like.device, dtype=like.dtype)

    def _call(self) -> Any:
        model = self.model
        flags = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            self.passes += 1
            return model(**self.inputs, use_cache=False).logits
        finally:
            for module, flag in flags:
                module.training = flag

    # Evaluations

    def real_leaves(self) -> _Leaves:
        return self._leaves.map(lambda tensor: tensor.detach().clone())

    def evaluate(self, leaves: _Leaves, *, grad: bool = False) -> _Evaluation:
        """Run the replacement model with these leaves; ``grad`` keeps the graph for backward."""
        import torch

        captured: dict[int, Any] = {}
        shaped: dict[int, Any] = {}
        with ExitStack() as stack:
            stack.enter_context(_frozen_attention(self.model, self.patterns))
            stack.enter_context(torch.enable_grad() if grad else torch.no_grad())

            def begin(_, args, kwargs):
                value, slot = _input_activation(args, kwargs)
                new = leaves.embed.to(device=value.device, dtype=value.dtype)
                return _with_input(args, kwargs, slot, new)

            _hook(stack, self.first_block, begin, pre=True)
            for module, kind, eps in self.norms:

                def freeze(owner, args, output, kind=kind, eps=eps):
                    ratio = _denominator(kind, eps, args[0]) / self.rho[owner].to(args[0].device)
                    bias = getattr(owner, "bias", None) if kind == "layer" else None
                    if bias is None:
                        return output * ratio
                    return bias + (output - bias) * ratio

                _hook(stack, module, freeze)
            for layer, transcoder in self.transcoders.items():

                def read(_, args, kwargs, layer=layer):
                    captured[layer] = _input_activation(args, kwargs)[0]

                def write(_, args, output, layer=layer, transcoder=transcoder):
                    y = _activation(output)
                    made = self._reconstruction(layer, leaves.acts[layer], y.shape[1], y[0])
                    shaped[layer] = _encoder_input(
                        transcoder, captured[layer][0], self.denominators[layer]
                    )
                    made = made + self._skip(layer, shaped[layer], y[0])
                    new = made + leaves.errors[layer].to(y.device)
                    return _with_output(output, new[None].to(y.dtype))

                _hook(stack, self.model.get_submodule(transcoder.module), read, pre=True)
                _hook(stack, self.model.get_submodule(transcoder.output_module), write)
            logits = self._call()
            pieces = []
            for layer in self.transcoders:
                if not self.chosen[layer]:
                    continue
                encoded = shaped[layer]
                weights, bias = self.encoders[layer]
                at = self.position_index[layer].to(encoded.device)
                pieces.append(((encoded[at] * weights).sum(-1) + bias).to(logits.device))
            targets = torch.cat(pieces) if pieces else torch.zeros(0, device=logits.device)
        return _Evaluation(targets, logits[0, -1].float())

    # Node tables

    def source_nodes(self, tokens: list[str], ids: list[int]) -> list[dict[str, Any]]:
        """Embedding, error and feature nodes in the order of ``_Leaves.sources`` (no values)."""
        nodes = []

        def make(kind, layer, position, feature, node_id):
            return {
                "id": node_id,
                "kind": kind,
                "layer": layer,
                "position": position,
                "token": tokens[position],
                "token_id": ids[position],
                "feature": feature,
            }

        for p in range(self.positions):
            nodes.append(make("embedding", None, p, None, f"emb.{p}"))
        for layer in self.layers:
            for p in range(self.positions):
                nodes.append(make("error", layer, p, None, f"err.{layer}.{p}"))
        for layer in self.layers:
            for position, feature, _ in self.chosen[layer]:
                node_id = f"feat.{layer}.{position}.{feature}"
                nodes.append(make("feature", layer, position, feature, node_id))
        return nodes

    def scaled(self, leaves: _Leaves, node_id: str, factor: float) -> _Leaves:
        """A copy of ``leaves`` with one source node's value multiplied by ``factor``."""
        parts = node_id.split(".")
        new = leaves.map(lambda tensor: tensor.detach().clone())
        if parts[0] == "emb":
            new.embed[0, int(parts[1])] *= factor
        elif parts[0] == "err":
            new.errors[int(parts[1])][int(parts[2])] *= factor
        elif parts[0] == "feat":
            layer, position, feature = int(parts[1]), int(parts[2]), int(parts[3])
            entries = self.chosen[layer]
            slot = next(i for i, (p, f, _) in enumerate(entries) if (p, f) == (position, feature))
            new.acts[layer][slot] *= factor
        else:
            raise ValueError(f"not a source node: {node_id!r}")
        return new

    def target_value(self, evaluation: _Evaluation, node_id: str) -> float:
        """A feature node's pre-activation, or a logit node's logit minus the mean logit."""
        parts = node_id.split(".")
        if parts[0] == "logit":
            logits = evaluation.logits
            return float(logits[int(parts[1])] - logits.mean())
        layer, position, feature = int(parts[1]), int(parts[2]), int(parts[3])
        start, _ = self.slots[layer]
        entries = self.chosen[layer]
        slot = next(i for i, (p, f, _) in enumerate(entries) if (p, f) == (position, feature))
        return float(evaluation.targets[start + slot])


# Building the graph


def _estimate_gb(targets: int, sources: int) -> float:
    """The peak memory of the adjacency work in GB: two float32 matrices of targets x sources.

    One holds the signed weights and one is scratch for the normalized absolute values, which
    pruning rewrites in place. A tenth more covers the batch buffers and vectors.
    """
    return 2 * 4 * targets * sources * 1.15 / 1e9


def _edges(engine: _Replacement, leaves: _Leaves, targets: Any, batch_size: int, width: int) -> Any:
    """Each target's gradient-times-value for every source, ``[targets, sources]`` on the CPU.

    Sources are in the order of ``_Leaves.sources``: embeddings by position, errors by layer
    then position, features by node order. One backward pass covers ``batch_size`` targets.
    """
    import torch

    count = targets.shape[0]
    sources = leaves.sources()
    matrix = torch.zeros(count, width)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        selector = torch.zeros(stop - start, count, device=targets.device)
        selector[torch.arange(stop - start), torch.arange(start, stop)] = 1.0
        try:
            found = torch.autograd.grad(
                targets,
                sources,
                grad_outputs=selector,
                is_grads_batched=True,
                retain_graph=True,
                allow_unused=True,
            )
        except RuntimeError:  # an operation without a batching rule: one backward per target
            rows = [
                torch.autograd.grad(
                    targets, sources, grad_outputs=row, retain_graph=True, allow_unused=True
                )
                for row in selector
            ]
            found = tuple(
                None
                if all(r[i] is None for r in rows)
                else torch.stack(
                    [torch.zeros_like(sources[i]) if r[i] is None else r[i] for r in rows]
                )
                for i in range(len(sources))
            )
        offset = 0
        for source, gradient in zip(sources, found, strict=True):
            if gradient is None:
                gradient = torch.zeros(stop - start, *source.shape, device=source.device)
            product = gradient * source.detach()
            if source.ndim >= 2:
                product = product.sum(-1)
            product = product.reshape(stop - start, -1)
            matrix[start:stop, offset : offset + product.shape[1]] = product.float().cpu()
            offset += product.shape[1]
    return matrix


def _row_abs_sums(matrix: Any, chunk: int = 4096) -> Any:
    import torch

    return torch.cat([matrix[i : i + chunk].abs().sum(1) for i in range(0, len(matrix), chunk)])


def _normalize_rows(matrix: Any, chunk: int = 4096) -> None:
    """Divide each row by its sum, in place; a row of zeros stays zero."""
    for i in range(0, len(matrix), chunk):
        part = matrix[i : i + chunk]
        part /= part.sum(1, keepdim=True).clamp_min(1e-30)


def _walk(shares: Any, start: Any, first_feature: int, steps: int) -> Any:
    """Each source's summed influence on the logit rows over every path.

    ``shares`` is ``[targets, sources]`` with rows summing to 1 (or 0), ``start`` weights the
    target rows (the logits). A feature source is also a target row, at its index minus
    ``first_feature``, so the influence on it is passed on.
    """
    import torch

    rows = shares.shape[0]
    total = torch.zeros(shares.shape[1])
    walk = start.clone()
    for _ in range(steps):
        reach = walk @ shares
        total += reach
        walk = torch.zeros(rows)
        walk[: shares.shape[1] - first_feature] = reach[first_feature:]
        if not bool(walk.any()):
            break
    return total


def _keep_through(scores: Any, threshold: float) -> Any:
    """A boolean mask of the highest scores whose running sum first reaches ``threshold``."""
    import torch

    mask = torch.zeros_like(scores, dtype=torch.bool)
    total = float(scores.sum())
    if threshold >= 1.0:
        mask[:] = True
        return mask
    if total <= 0:
        return mask
    order = torch.argsort(scores, descending=True)
    running = torch.cumsum(scores[order], 0) / total
    count = int(torch.searchsorted(running, torch.tensor(threshold, dtype=running.dtype))) + 1
    mask[order[:count]] = True
    return mask


def _cover_threshold(scores: Any, fraction: float, chunk: int = 1024) -> float:
    """The largest score ``t`` such that the scores at least ``t`` sum to ``fraction`` of all.

    Found by bisection over row chunks, so no sorted copy of the matrix is made.
    """
    import torch

    total = sum(
        float(scores[i : i + chunk].sum(dtype=torch.float64)) for i in range(0, len(scores), chunk)
    )
    if total <= 0:
        return float("inf")
    low, high = 0.0, float(scores.max())
    for _ in range(40):
        middle = (low + high) / 2
        covered = 0.0
        for i in range(0, len(scores), chunk):
            part = scores[i : i + chunk]
            covered += float(part[part >= middle].sum(dtype=torch.float64))
        if covered >= fraction * total:
            low = middle
        else:
            high = middle
    return low


def _analyze(
    weights: Any,
    probabilities: Any,
    engine: _Replacement,
    features: int,
    node_threshold: float,
    edge_threshold: float,
) -> dict[str, Any]:
    """Influence, the replacement score, and the pruned graph, as the paper describes.

    ``weights`` is the signed ``[targets, sources]`` matrix (feature rows, then logit rows),
    which is not changed. One scratch matrix of the same size is rewritten in place for the
    normalized absolute weights, first of the whole graph and then of the pruned one.
    Node indices are sources first (embeddings, errors, features) and then logits; a target
    row ``r`` is node ``first_feature + r``.
    """
    import torch

    positions = engine.positions
    first_feature = positions * (1 + len(engine.layers))
    rows, sources = weights.shape
    logits = rows - features
    steps = len(engine.layers) + 3
    start = torch.zeros(rows)
    start[features:] = probabilities / probabilities.sum()

    scratch = torch.empty_like(weights)
    torch.abs(weights, out=scratch)
    _normalize_rows(scratch)
    influence = _walk(scratch, start, first_feature, steps)
    score = _replacement_score(scratch, engine, positions, first_feature, features, start)

    keep = torch.zeros(sources + logits, dtype=torch.bool)
    keep[:sources] = _keep_through(influence, node_threshold)
    keep[:positions] = True  # tokens are never pruned
    keep[sources:] = True  # neither are the logits
    parents = []
    for k in range(logits):  # a logit keeps its strongest parent, so pruning never orphans it
        row = weights[features + k].abs()
        if bool(row.any()):
            parent = int(row.argmax())
            parents.append((features + k, parent))
            keep[parent] = True

    # Rescore the edges on the pruned graph: zero what was dropped, renormalize, walk again.
    torch.abs(weights, out=scratch)
    dropped_rows = ~keep[first_feature : first_feature + rows]
    scratch[dropped_rows] = 0
    scratch[:, ~keep[:sources]] = 0
    present, floor = None, None
    if edge_threshold >= 1.0:
        present = scratch > 0
    _normalize_rows(scratch)
    pruned_influence = _walk(scratch, start, first_feature, steps)
    row_influence = start.clone()
    row_influence[:features] += pruned_influence[first_feature:]
    scratch.mul_(row_influence[:, None])
    if present is None:
        floor = _cover_threshold(scratch, edge_threshold)
    pairs, scores = [], []
    for i in range(0, rows, 1024):
        part = scratch[i : i + 1024]
        chosen = present[i : i + 1024] if present is not None else (part >= floor) & (part > 0)
        found = chosen.nonzero()
        pairs.append(torch.stack([found[:, 0] + i, found[:, 1]], 1).to(torch.int32))
        scores.append(part[chosen])
    del scratch, present
    pairs = torch.cat(pairs) if pairs else torch.zeros(0, 2, dtype=torch.int32)
    scores = torch.cat(scores) if scores else torch.zeros(0)
    target = pairs[:, 0].long() + first_feature
    source = pairs[:, 1].long()

    # Features and errors need an outgoing edge, and features an incoming one; dropping a node
    # can strand others, so repeat until nothing changes.
    alive = keep.clone()
    size = sources + logits
    everything = node_threshold >= 1.0 and edge_threshold >= 1.0
    while not everything:
        edge = alive[source] & alive[target]
        source, target, scores, pairs = source[edge], target[edge], scores[edge], pairs[edge]
        outgoing = torch.bincount(source, minlength=size) > 0
        incoming = torch.bincount(target, minlength=size) > 0
        updated = alive.clone()
        updated[positions:sources] &= outgoing[positions:sources]
        updated[first_feature:sources] &= incoming[first_feature:sources]
        if bool((updated == alive).all()):
            break
        alive = updated
    have = {(int(r), int(c)) for r, c in pairs.tolist()}
    extra = [(r, c) for r, c in parents if (r, c) not in have]
    for _row, column in extra:
        alive[column] = True
    if extra:
        added = torch.tensor(extra, dtype=torch.int32)
        pairs = torch.cat([pairs, added])
        scores = torch.cat([scores, torch.zeros(len(extra))])
    order = torch.argsort(scores, descending=True)
    pairs = pairs[order]
    return {
        "influence": influence,
        "replacement_score": score,
        "alive": alive,
        "edges": pairs,
        "first_feature": first_feature,
    }


def _replacement_score(
    shares: Any, engine: _Replacement, positions: int, first_feature: int, features: int, start: Any
) -> float:
    """The probability that a walk from the logits, following edges by share, avoids all errors.

    ``shares`` rows sum to 1. A walk ends at a token (clean) or an error node (not clean); a
    node with no incoming edges counts as clean.
    """
    import torch

    sources = shares.shape[1]
    reach = torch.zeros(sources)
    reach[:positions] = 1.0
    silent = shares.sum(1) == 0
    for layer in engine.layers:
        low, high = engine.slots[layer]
        value = shares[low:high] @ reach
        reach[first_feature + low : first_feature + high] = torch.where(
            silent[low:high], torch.ones(high - low), value
        )
    logits = shares[features:] @ reach
    logits = torch.where(silent[features:], torch.ones(len(logits)), logits)
    return float((start[features:] * logits).sum())


class AttributionGraph:
    """A feature-level graph of how a model produced its answer on one prompt.

    ``nodes`` and ``edges`` are lists of plain dicts (see ``Lab.attribution_graph``);
    ``metadata`` records the prompt, settings, checks and pass count. ``to_dict`` and
    ``to_json`` give the whole graph as JSON. A graph built by ``Lab.attribution_graph`` can
    also run ``intervene`` and ``measure``; one read back from a file cannot, because it
    holds no model. ``inputs`` are the exact model inputs the graph was built on.
    """

    def __init__(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        metadata: dict[str, Any],
        *,
        lab: Any = None,
        transcoders: dict[int, LabSAE] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> None:
        self.nodes, self.edges, self.metadata = nodes, edges, metadata
        self._lab, self._transcoders, self._inputs = lab, transcoders, inputs
        self._by_id = {node["id"]: node for node in nodes}

    def __repr__(self) -> str:
        return (
            f"AttributionGraph({len(self.nodes)} nodes, {len(self.edges)} edges, "
            f"replacement_score={self.metadata.get('replacement_score')})"
        )

    @property
    def inputs(self) -> dict[str, Any]:
        """The model inputs the graph ran on: ``input_ids`` and, if the tokenizer made one,
        ``attention_mask``, as tensors ``[1, tokens]`` on the lab's device.

        These include a BOS token the transcoders asked for. Positions in the graph index this
        sequence, so any forward pass that measures a node must run on exactly these tokens;
        ``lab.score`` and ``lab.evaluate`` tokenize on their own and do not.
        """
        import torch

        if self._inputs is not None:
            return {key: value.clone() for key, value in self._inputs.items()}
        ids = torch.tensor([self.metadata["input_ids"]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def to_dict(self) -> dict[str, Any]:
        """The graph as ``{"metadata", "nodes", "edges"}``, all JSON types."""
        return {"metadata": self.metadata, "nodes": self.nodes, "edges": self.edges}

    def to_json(self, path: str | Path) -> Path:
        """Write the graph as JSON and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), allow_nan=False))
        return target

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttributionGraph:
        """Rebuild a graph from ``to_dict`` output. It cannot run ``intervene`` or ``measure``."""
        return cls(list(data["nodes"]), list(data["edges"]), dict(data["metadata"]))

    @classmethod
    def from_json(cls, path: str | Path) -> AttributionGraph:
        """Read a graph written by ``to_json``. It cannot run ``intervene`` or ``measure``."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    def node(self, node_id: str) -> dict[str, Any]:
        """The node with this id; raises ValueError for an unknown one."""
        if node_id not in self._by_id:
            raise ValueError(f"unknown node {node_id!r}")
        return self._by_id[node_id]

    def top_paths(self, node: str, k: int = 5) -> list[dict[str, Any]]:
        """The ``k`` strongest paths into ``node`` along the graph's edges, best first.

        A path runs from a node with no incoming edges (usually an embedding or an error
        term) to ``node``. Its score is the product, along its edges, of each edge's share of
        its target's incoming absolute weight, so it is at most 1 and a longer path is never
        stronger than its prefix. Each result has ``nodes`` (source first), ``weights`` (the
        edges' weights) and ``score``. Raises ValueError for an unknown node or a ``k`` below 1.
        """
        self.node(node)
        if not _is_int(k) or k < 1:
            raise ValueError("k must be a positive integer")
        incoming: dict[str, list[tuple[str, float]]] = {}
        for edge in self.edges:
            incoming.setdefault(edge["target"], []).append((edge["source"], edge["weight"]))
        totals = {t: sum(abs(w) for _, w in found) for t, found in incoming.items()}
        queue: list[tuple[float, int, tuple[str, ...], tuple[float, ...]]] = [
            (-1.0, 0, (node,), ())
        ]
        counter, results = 1, []
        while queue and len(results) < k:
            negative, _, path, weights = heapq.heappop(queue)
            parents = incoming.get(path[0])
            if not parents:
                results.append({"nodes": list(path), "weights": list(weights), "score": -negative})
                continue
            for source, weight in parents:
                share = abs(weight) / totals[path[0]]
                if share == 0:
                    continue
                heapq.heappush(
                    queue,
                    (negative * share, counter, (source, *path), (weight, *weights)),
                )
                counter += 1
        return results

    def _groups(self, node_ids: Any, value: Any) -> dict[tuple[int, int], dict[int, float]]:
        """The edits for a set of feature nodes, by (layer, position); validates both."""
        if self._lab is None or self._transcoders is None:
            raise ValueError(
                "this graph was loaded from a file and holds no model; rebuild it with "
                "lab.attribution_graph to intervene or measure"
            )
        if isinstance(node_ids, str):
            node_ids = [node_ids]
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError("value must be a number")
        groups: dict[tuple[int, int], dict[int, float]] = {}
        for node_id in list(node_ids):
            node = self.node(node_id)
            if node["kind"] != "feature":
                raise ValueError(
                    f"{node_id} is a {node['kind']} node; only feature nodes can be set"
                )
            groups.setdefault((node["layer"], node["position"]), {})[node["feature"]] = value
        return groups

    @contextmanager
    def intervene(self, node_ids: Any, value: float = 0.0):
        """Set feature nodes to ``value`` in the real model while the context is open.

        Each node's feature is set at its position through ``LabSAE.patch``, which keeps the
        reconstruction error and changes the transcoder's output by the difference times the
        decoder rows. The positions index ``graph.inputs``, so run the model on exactly those
        tokens inside the context, for example ``lab.model(**graph.inputs)``. ``lab.score``
        and ``lab.evaluate`` tokenize the prompt themselves (without the BOS token the graph
        has), so their positions are off by one: any forward pass in the context whose tokens
        differ from the graph's raises ValueError, naming ``graph.measure``, which does the
        run and returns the effect on each logit node. Raises ValueError, before any hook is
        added, for an unknown node, a node that is not a feature, a ``value`` that is not a
        number, and a graph read from a file.
        """
        groups = self._groups(node_ids, value)
        expected = self.metadata["input_ids"]

        def check(module, args, kwargs):
            import torch

            ids = kwargs.get("input_ids", args[0] if args else None)
            ok = (
                isinstance(ids, torch.Tensor)
                and ids.ndim == 2
                and ids.shape[1] == len(expected)
                and bool((ids.cpu() == torch.tensor(expected)).all())
            )
            if not ok:
                raise ValueError(
                    "this forward pass is not on the graph's tokens, so the intervention's "
                    "positions would patch other tokens. Use graph.measure(node_ids), or run "
                    "the model on graph.inputs inside the context"
                )

        with ExitStack() as stack:
            _hook(stack, self._lab.model, check, pre=True)
            for (layer, position), edits in sorted(groups.items()):
                stack.enter_context(self._transcoders[layer].patch(edits, positions=[position]))
            yield

    def measure(self, node_ids: Any, value: float = 0.0) -> dict[str, dict[str, float]]:
        """Run the real model on ``graph.inputs`` with feature nodes set, and measure the logits.

        Returns ``baseline`` and ``intervened`` (each logit node's id mapped to its logit minus
        the mean logit, before and after the nodes are set to ``value`` through
        ``LabSAE.patch``) and ``effect`` (their difference). The graph's own edge is an
        estimate in the replacement model; this is the real model's answer. It costs 2 forward
        passes (one for each), and the pass budget is checked for both first. Raises
        ValueError for the reasons ``intervene`` does, before any pass runs.
        """
        import torch

        self._groups(node_ids, value)
        lab = self._lab
        lab._budget.check(2)
        model = lab.model
        inputs = {key: tensor.to(lab.device) for key, tensor in self.inputs.items()}
        logits = [n for n in self.nodes if n["kind"] == "logit"]

        def run() -> dict[str, float]:
            flags = [(module, module.training) for module in model.modules()]
            try:
                model.eval()
                with torch.no_grad():
                    row = model(**inputs, use_cache=False).logits[0, -1].float()
            finally:
                for module, flag in flags:
                    module.training = flag
            centered = row - row.mean()
            return {n["id"]: float(centered[n["token_id"]]) for n in logits}

        baseline = run()
        with self.intervene(node_ids, value):
            intervened = run()
        return {
            "baseline": baseline,
            "intervened": intervened,
            "effect": {key: intervened[key] - baseline[key] for key in baseline},
        }


def _check_settings(settings: dict[str, Any]) -> None:
    for name in ("max_logits", "batch_size"):
        if not _is_int(settings[name]) or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    cap = settings["max_features"]
    if cap is not None and (not _is_int(cap) or cap < 1):
        raise ValueError("max_features must be None or a positive integer")
    for name in ("logit_probability", "node_threshold", "edge_threshold"):
        number = settings[name]
        if not isinstance(number, Real) or isinstance(number, bool) or not 0 < number <= 1:
            raise ValueError(f"{name} must be a number above 0 and at most 1")
    limit = settings["max_memory_gb"]
    if not isinstance(limit, Real) or isinstance(limit, bool) or not limit > 0:
        raise ValueError("max_memory_gb must be a positive number")
    if not isinstance(settings["labels"], bool):
        raise ValueError("labels must be True or False")


def _transcoder_map(lab: Any, transcoders: Any) -> dict[int, LabSAE]:
    """Check the transcoders and return them by layer."""
    if isinstance(transcoders, dict):
        given = list(transcoders.items())
    elif isinstance(transcoders, (list, tuple)):
        given = [(None, item) for item in transcoders]
    else:
        raise ValueError("transcoders must be a list of LabSAE transcoders or a dict by layer")
    if not given:
        raise ValueError("pass one transcoder for every MLP layer")
    found: dict[int, LabSAE] = {}
    for key, item in given:
        if not isinstance(item, LabSAE):
            raise ValueError(f"expected a LabSAE transcoder, got {type(item).__name__}")
        if item.lab is not lab:
            raise ValueError("a transcoder belongs to another session; load it with this lab")
        if item.kind != "transcoder":
            raise ValueError(
                f"{item.hook_name} is not a transcoder: an attribution graph replaces each "
                "MLP, which needs a transcoder that reads its input and writes its output"
            )
        reads, writes = item.module.rsplit(".", 1), item.output_module.rsplit(".", 1)
        if (
            item.at != "input"
            or item.output_at != "output"
            or reads[-1] not in _READS
            or writes[-1] not in _WRITES
            or reads[0] != writes[0]
        ):
            raise ValueError(
                f"{item.hook_name} -> {item.sae.cfg.metadata.hook_name_out} does not replace "
                "an MLP: a graph needs a transcoder that reads the MLP's input and writes the "
                "MLP's output"
            )
        if item.sae.cfg.normalize_activations != "none":
            raise ValueError(
                f"{item.hook_name} uses normalize_activations="
                f"{item.sae.cfg.normalize_activations!r}, which makes its encoder nonlinear in "
                "the residual stream; only transcoders without it are supported"
            )
        layer = _layer_index(item.module)
        if key is not None and key != layer:
            raise ValueError(f"the transcoder for layer {key} reads layer {layer}")
        if layer in found:
            raise ValueError(f"two transcoders for layer {layer}")
        found[layer] = item
    if len({item.prepend_bos for item in found.values()}) > 1:
        raise ValueError("the transcoders disagree on prepend_bos")
    try:
        layers = sorted({entry["layer"] for entry in lab.heads()})
    except NotImplementedError as error:
        raise ValueError(f"attribution graphs do not support this architecture: {error}") from error
    missing = [layer for layer in layers if layer not in found]
    if missing:
        raise ValueError(
            "the graph needs every MLP replaced; no transcoder for "
            + ", ".join(f"layer {layer}" for layer in missing)
        )
    extra = sorted(set(found) - set(layers))
    if extra:
        raise ValueError(f"the model has no layer {extra[0]}")
    mlps = {
        name.rsplit(".", 1)[0] for name, _ in lab.model.named_modules() if name.endswith(".mlp")
    }
    attention = {name.rsplit(".", 1)[0] for name, _ in _blocks(lab.model)}
    if mlps != attention:
        raise ValueError(
            "attribution graphs do not support this architecture: "
            + ", ".join(sorted(mlps - attention))
            + " has an MLP but no softmax attention block whose pattern can be frozen"
        )
    return found


class GraphMixin:
    """``Lab.attribution_graph``."""

    def attribution_graph(
        self,
        prompt: str,
        transcoders: Any,
        *,
        max_logits: int = 10,
        logit_probability: float = 0.95,
        max_features: int | None = 10_000,
        node_threshold: float = 0.8,
        edge_threshold: float = 0.98,
        batch_size: int = 16,
        max_memory_gb: float = 8,
        labels: bool = False,
    ) -> AttributionGraph:
        """Build a feature-level attribution graph for one prompt from per-layer transcoders.

        ``transcoders`` is a list of ``LabSAE`` transcoders, one per MLP layer, or a dict from
        layer to transcoder. The graph describes a replacement model: this session's model
        with each MLP's output written as its transcoder's features times their decoder rows
        plus an error term, attention patterns held at their real values, and each norm's
        denominator held fixed. Every forward value is exactly the real model's; only the
        gradients differ, so the model is affine in its inputs.

        Nodes are ``embedding`` (one per token position), ``feature`` (each transcoder feature
        that fires, at its layer and position), ``error`` (the real MLP output minus the
        features' reconstruction, one per layer and position), and ``logit`` (the top next
        tokens, until their probability reaches ``logit_probability``, at most
        ``max_logits``). Only the ``max_features`` features with the largest activations are
        nodes (``None`` keeps them all); the rest are absorbed into the error terms. A skip
        transcoder's skip connection is a linear path from the MLP's input, so it carries edges
        to later nodes like any other; the error is what remains after features and skip. An
        edge is the direct effect of a source on a target: the source's value times the
        gradient of the target with respect to it, where a target is a feature's encoder
        pre-activation or a logit minus the mean logit. A target's edges plus its ``bias``
        add up to its ``activation``. All of one target's edges come from one backward pass,
        and ``batch_size`` targets share a batched backward.

        Pruning follows Ameisen et al., 2025, and circuit-tracer's ``prune_graph``: each
        node's influence on the logit nodes is computed through the row-normalized absolute
        adjacency matrix, nodes are kept until their influence reaches ``node_threshold`` of
        the total, the edges are rescored on the pruned graph (renormalized, with influence
        recomputed there) and kept until they reach ``edge_threshold`` of the total, and then
        features and errors without an outgoing edge, and features without an incoming one,
        are dropped, repeatedly. Token nodes and logit nodes are never pruned. A threshold of
        1.0 keeps every node and every edge with a weight. Differences from circuit-tracer:
        each logit also keeps its strongest parent (and that edge), so no logit is left
        without a cause; the reported node ``influence`` is the share of the unpruned
        graph's total, not recomputed after pruning; and the edge cutoff is found by
        bisection, so ties at the cutoff are all kept.

        The ``replacement_score`` in the metadata is the share of backward paths from the
        logits to the tokens that go through features and never through an error node.

        This costs 3 forward passes, whatever the prompt's length or the number of features:
        one to record, one to run the replacement model, and one with all leaves at zero to
        measure each target's constant part. Backward passes do not count. The pass budget is
        checked for all 3 before the first runs. The adjacency work needs two float32
        matrices of targets x sources; before allocating them the peak is estimated, and a
        graph over ``max_memory_gb`` (default 8, which leaves room on a 16 GB machine for the
        model and the batch buffers, and fits about 30,000 nodes) is refused with a message
        naming ``max_features``. That check follows the recording pass, so it costs 1 pass.
        ``max_features=10000`` by default (the cap circuit-tracer uses) keeps a typical graph
        near 1 GB.

        ``labels=True`` asks Neuronpedia for each kept feature's explanation, one network
        request per feature without a cached label (``LabSAE.labels`` explains caching); it is
        off by default, and nothing is requested without it.

        The call checks, and raises ValueError rather than return a graph that fails: the
        replacement model's logits equal the model's, each feature's pre-activation equals its
        activation, and each target's edges plus bias equal its value (to 1e-4 of the size of
        the terms added). It refuses a model that is not float32, a quantized or adapter
        session, an architecture ``heads`` does not support or with a layer lacking softmax
        attention, transcoders that are not one per MLP layer reading the MLP's input and
        writing its output, a transcoder with run-time input normalization, a norm other than
        LayerNorm or RMSNorm, and out-of-range settings. Hooks, the attention setting,
        train/eval modes and RNG state are restored on exit, even after an error. Needs the
        ``saes`` extra for the transcoders. Edges describe the replacement model, so they
        cannot show effects that move through attention patterns. To test a node, use
        ``AttributionGraph.measure``.
        """
        import torch

        settings = {
            "max_logits": max_logits,
            "logit_probability": logit_probability,
            "max_features": max_features,
            "node_threshold": node_threshold,
            "edge_threshold": edge_threshold,
            "batch_size": batch_size,
            "max_memory_gb": max_memory_gb,
            "labels": labels,
        }
        _check_settings(settings)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        found = _transcoder_map(self, transcoders)
        model = self.model
        _eager(model)
        if getattr(self, "quantization", None) is not None:
            raise ValueError("attribution graphs do not support quantized sessions")
        if model.config._attn_implementation != "eager":
            raise ValueError("attribution graphs need eager attention, to read attention patterns")
        _check_float32(model)
        self._budget.check(3)
        inputs = next(iter(found.values()))._tokenized(prompt)
        if inputs["input_ids"].shape[0] != 1:
            raise ValueError("attribution graphs run one prompt")
        engine = _Replacement(self, found, inputs, max_features)

        features = int(engine.real_targets.numel())
        width = engine.positions * (1 + len(engine.layers)) + features
        estimate = _estimate_gb(features + max_logits, width)
        if estimate > max_memory_gb:
            raise ValueError(
                f"this graph has {features} features and would need about {estimate:.1f} GB "
                f"for its adjacency matrices, over max_memory_gb={max_memory_gb}; lower "
                f"max_features (now {max_features}) or raise max_memory_gb"
            )

        leaves = engine.real_leaves().map(lambda tensor: tensor.requires_grad_(True))
        evaluated = engine.evaluate(leaves, grad=True)
        gap = float((evaluated.logits.detach() - engine.real_logits).abs().max())
        if gap > _allowed(float(engine.real_logits.abs().max())):
            raise ValueError(
                f"the replacement model's logits differ from the model's by {gap:.3g}; this "
                "architecture is not supported"
            )
        feature_gap = (
            float((evaluated.targets.detach() - engine.real_targets).abs().max())
            if features
            else 0.0
        )
        if feature_gap > _allowed(float(engine.real_targets.abs().max()) if features else 1.0):
            raise ValueError(
                f"a feature's pre-activation in the replacement model differs from its "
                f"activation by {feature_gap:.3g}; this architecture is not supported"
            )

        probabilities = torch.softmax(engine.real_logits, -1)
        order = torch.argsort(probabilities, descending=True)
        chosen, covered = [], 0.0
        for token in order.tolist():
            chosen.append(token)
            covered += float(probabilities[token])
            if covered >= logit_probability or len(chosen) >= max_logits:
                break
        logit_values = evaluated.logits[chosen] - evaluated.logits.mean()
        targets = torch.cat([evaluated.targets, logit_values])
        matrix = _edges(engine, leaves, targets, batch_size, width)

        zero = engine.evaluate(engine.real_leaves().map(torch.zeros_like))
        zero_values = (
            torch.cat([zero.targets, zero.logits[chosen] - zero.logits.mean()]).detach().cpu()
        )
        values = targets.detach().cpu()
        residual = (matrix.sum(1) + zero_values - values).abs()
        edge_error = float(residual.max()) if residual.numel() else 0.0
        # Float32 sums lose digits to cancellation, so the bar is the size of the terms added.
        size = torch.stack([values.abs(), zero_values.abs(), _row_abs_sums(matrix)]).max(0).values
        if bool((residual > _TOLERANCE * size.clamp_min(1.0)).any()):
            raise ValueError(
                f"a target's edges plus its constant part differ from its value by "
                f"{edge_error:.3g}; this architecture is not supported"
            )

        ids = inputs["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        nodes = engine.source_nodes(tokens, ids)
        feature_start = engine.positions * (1 + len(engine.layers))
        for node, norm in zip(
            nodes[:feature_start],
            _vector_norms(leaves.sources()[: 1 + len(engine.layers)]),
            strict=True,
        ):
            node["activation"] = norm
        feature_values = engine.real_targets.cpu().tolist()
        for node, value in zip(nodes[feature_start:], feature_values, strict=True):
            node["activation"] = value
        bias = dict(
            zip(
                (n["id"] for n in nodes[feature_start:]),
                zero_values[:features].tolist(),
                strict=True,
            )
        )
        shares = probabilities[chosen].detach().cpu()
        for token, value, constant, share in zip(
            chosen,
            logit_values.detach().cpu().tolist(),
            zero_values[features:].tolist(),
            shares.tolist(),
            strict=True,
        ):
            node = {
                "id": f"logit.{token}",
                "kind": "logit",
                "layer": None,
                "position": engine.positions - 1,
                "token": self.tokenizer.convert_ids_to_tokens(token),
                "token_id": token,
                "feature": None,
                "activation": value,
                "probability": share,
            }
            bias[node["id"]] = constant
            nodes.append(node)

        analysis = _analyze(matrix, shares, engine, features, node_threshold, edge_threshold)
        influence = analysis["influence"]
        total_influence = float(influence.sum())
        for index, node in enumerate(nodes):
            node["influence"] = (
                float(influence[index]) / total_influence
                if total_influence > 0 and index < len(influence)
                else 0.0
            )
            if node["id"] in bias:
                node["bias"] = bias[node["id"]]
        alive = analysis["alive"]
        kept_nodes = [node for index, node in enumerate(nodes) if bool(alive[index])]
        if labels:
            self._label(kept_nodes, found)
        pairs = analysis["edges"].long()
        targets_index = (pairs[:, 0] + analysis["first_feature"]).tolist()
        sources_index = pairs[:, 1].tolist()
        edge_weights = matrix[pairs[:, 0], pairs[:, 1]].tolist()
        edges = [
            {"source": nodes[s]["id"], "target": nodes[t]["id"], "weight": w}
            for s, t, w in zip(sources_index, targets_index, edge_weights, strict=True)
        ]
        metadata = {
            "prompt": prompt,
            "tokens": tokens,
            "input_ids": ids,
            "model_id": self.model_id,
            "transcoders": [
                {
                    "layer": layer,
                    "release": item.release,
                    "sae_id": item.sae_id,
                    "hook_name": item.hook_name,
                    "hook_name_out": item.sae.cfg.metadata.hook_name_out,
                    "d_sae": item.d_sae,
                    "architecture": item.sae.cfg.architecture(),
                }
                for layer, item in sorted(found.items())
            ],
            "settings": settings,
            "replacement_score": float(analysis["replacement_score"]),
            "forward_passes": engine.passes,
            "logit_probability_covered": covered,
            "checks": {
                "replacement_logit_max_error": gap,
                "feature_value_max_error": feature_gap,
                "edge_sum_max_error": edge_error,
            },
        }
        return AttributionGraph(
            kept_nodes,
            edges,
            metadata,
            lab=self,
            transcoders=found,
            inputs={key: value.detach().clone() for key, value in inputs.items()},
        )

    def _label(self, nodes: list[dict[str, Any]], found: dict[int, LabSAE]) -> None:
        for layer, item in found.items():
            wanted = [n for n in nodes if n["kind"] == "feature" and n["layer"] == layer]
            if not wanted or item.neuronpedia_id is None:
                continue
            names = item.labels(sorted({n["feature"] for n in wanted}))
            for node in wanted:
                if names.get(node["feature"]):
                    node["label"] = names[node["feature"]]


def _vector_norms(vectors: list[Any]) -> list[float]:
    """The L2 norm of each position's vector: embeddings first, then each layer's errors."""
    norms = []
    for vector in vectors:
        norms += vector.detach().reshape(-1, vector.shape[-1]).norm(dim=-1).cpu().tolist()
    return norms
