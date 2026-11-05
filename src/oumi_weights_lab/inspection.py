"""Eager activation hooks, short-row gradients, and document-split probes."""

from __future__ import annotations

import math
import random
from contextlib import contextmanager

from .quantized import lookup


def _eager(model):
    for module in model.modules():
        if hasattr(module, "_orig_mod") or getattr(module, "_compiled_call_impl", None):
            raise ValueError("inspection requires an eager, uncompiled model")


def _activation(output):
    import torch

    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor) or value.ndim < 3:
        raise ValueError(
            "module output must be a tensor [batch, tokens, ...] or a tuple starting with one"
        )
    return value


_RECORD_AT = ("output", "input", "attention")
_PATCH_AT = ("output", "input")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _listed(positions):
    """Positions checked and normalized once, when a context opens.

    A slice or "all" is kept as given, an iterable is read once so every hook call sees the
    same list, and anything that cannot work on any call raises. Whether positions fit a
    given call is checked by ``_positions`` at the forward.
    """
    if positions is None or _is_int(positions):
        return positions
    if isinstance(positions, str):
        if positions != "all":
            raise ValueError('positions must be an integer, a list of integers, a slice, or "all"')
        return positions
    if isinstance(positions, slice):
        if not all(part is None or _is_int(part) for part in (positions.start, positions.stop)):
            raise ValueError("slice positions must use integers or None")
        if positions.step is not None and (not _is_int(positions.step) or positions.step < 1):
            raise ValueError("slice positions need a positive integer step")
        return positions
    try:
        values = list(positions)
    except TypeError:
        raise ValueError(
            'positions must be an integer, a list of integers, a slice, or "all"'
        ) from None
    if not values or not all(_is_int(index) for index in values):
        raise ValueError("positions must be an integer or a nonempty list of integers")
    if len(set(values)) != len(values):
        raise ValueError("positions must select distinct tokens")
    return values


def _padding_positions(positions):
    """The positions batched scoring must check against left padding.

    A negative position counts from the end and survives padding; a nonnegative one does not.
    "all" is safe. A slice is safe only when it ends at the end of every sequence: a negative
    start, a stop of None or a negative number, and a step of 1. Any other slice counts from
    the left, so it returns 0.
    """
    if positions is None:
        return [-1]
    if isinstance(positions, str):
        return []
    if isinstance(positions, slice):
        start, stop, step = positions.start, positions.stop, positions.step
        anchored = (
            start is not None and start < 0 and (stop is None or stop < 0) and step in (None, 1)
        )
        return [start] if anchored else [0]
    values = [positions] if _is_int(positions) else positions
    if not isinstance(values, list):
        return []
    return [index for index in values if _is_int(index)]


def _positions(value, positions, dim=1):
    """Indices along ``dim`` of ``value`` that ``positions`` selects on this call."""
    size = value.shape[dim]
    if isinstance(positions, str):
        return list(range(size))
    if isinstance(positions, slice):
        selected = list(range(*positions.indices(size)))
        if not selected:
            raise ValueError("positions select no tokens")
        return selected
    selected = [-1] if positions is None else [positions] if _is_int(positions) else list(positions)
    if not selected or any(not _is_int(index) for index in selected):
        raise ValueError("positions must be an integer or a nonempty list of integers")
    if any(index < -size or index >= size for index in selected):
        raise ValueError("token position is outside the module output")
    # Aliases such as [2, -1] on three tokens would silently drop one replacement.
    selected = [index % size for index in selected]
    if len(set(selected)) != len(selected):
        raise ValueError("positions must select distinct tokens")
    return selected


def _take(value, indices, dim=1):
    """A copy of ``value`` restricted to ``indices`` along ``dim``."""
    import torch

    return value.index_select(dim, torch.as_tensor(indices, device=value.device))


def _replaced(value, positions, values):
    """``value`` with the selected token positions replaced by ``values`` (a tensor or callable)."""
    import torch

    indices = _positions(value, positions)
    replacement = values(_take(value, indices)) if callable(values) else values
    replacement = torch.as_tensor(replacement, device=value.device, dtype=value.dtype)
    updated = value.clone()
    updated[:, indices] = replacement
    return updated


def _input_activation(args, kwargs):
    """The tensor a module is called with, and where it came from (0 or "hidden_states")."""
    import torch

    if args and isinstance(args[0], torch.Tensor):
        value, slot = args[0], 0
    elif isinstance(kwargs.get("hidden_states"), torch.Tensor):
        value, slot = kwargs["hidden_states"], "hidden_states"
    else:
        raise ValueError(
            "module has no tensor input: expected a tensor as the first argument "
            "or as hidden_states="
        )
    if value.ndim < 3:
        raise ValueError("module input must be a tensor [batch, tokens, ...]")
    return value, slot


def _with_input(args, kwargs, slot, updated):
    """``(args, kwargs)`` with ``updated`` in the slot ``_input_activation`` read it from."""
    if slot == 0:
        return (updated, *args[1:]), kwargs
    return args, {**kwargs, slot: updated}


def _with_output(output, updated):
    """``output`` with ``updated`` in place of the activation ``_activation`` read from it."""
    if isinstance(output, tuple):
        entries = (updated, *output[1:])
        return type(output)(*entries) if hasattr(output, "_fields") else entries
    return updated


def _attention_weights(output):
    """The ``[batch, heads, queries, keys]`` weights in an eager attention module's output."""
    import torch

    pair = isinstance(output, (tuple, list)) and len(output) > 1
    hidden, weights = (output[0], output[1]) if pair else (None, None)
    # Batch and query counts must match the first output, or any 4-D tensor would pass.
    if (
        not isinstance(weights, torch.Tensor)
        or weights.ndim != 4
        or not isinstance(hidden, torch.Tensor)
        or hidden.ndim < 2
        or weights.shape[0] != hidden.shape[0]
        or weights.shape[2] != hidden.shape[1]
    ):
        raise ValueError(
            "attention capture needs a module whose output's second element is the attention "
            "weights; the lab loads eager attention, so check the module name points at an "
            "attention block, not a layer or a linear-attention block"
        )
    return weights


def _check_at(at, allowed):
    if not isinstance(at, str) or at not in allowed:
        raise ValueError(f"at must be one of {', '.join(map(repr, allowed))}")


def _remove_by_identity(module, hook):
    """Remove ``hook`` from torch's private hook dicts when its handle was never returned.

    Covers forward hooks and forward pre-hooks. A torch without these dicts leaves the hook
    registered; it is inert once removed.
    """
    for family in (
        ("_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called"),
        ("_forward_pre_hooks", "_forward_pre_hooks_with_kwargs"),
    ):
        hooks = getattr(module, family[0], {})
        for key in [key for key, value in hooks.items() if value is hook]:
            for attribute in family:
                getattr(module, attribute, {}).pop(key, None)


class _TemporaryHook:
    """A forward hook, or with ``pre=True`` a forward pre-hook, inert once removed.

    A pre-hook is called as ``function(module, args, kwargs)``. Worker timeouts raise
    BaseException anywhere, including inside register_forward_hook after the hook is
    stored but before its handle returns.
    """

    def __init__(self, module, function, pre=False):
        self.module, self.function, self.pre = module, function, pre
        self.active = True
        self.handle = None

    def __call__(self, *args):
        return self.function(*args) if self.active else None

    def register(self, **kwargs):
        if self.pre:
            self.handle = self.module.register_forward_pre_hook(self, with_kwargs=True, **kwargs)
        else:
            self.handle = self.module.register_forward_hook(self, **kwargs)

    def remove(self):
        self.active = False
        if self.handle is None:
            _remove_by_identity(self.module, self)
        else:
            self.handle.remove()


def _accepts(model, name):
    """Whether the base model's forward takes a keyword argument called ``name``."""
    import inspect

    base = model.get_base_model() if hasattr(model, "peft_config") else model
    return name in inspect.signature(base.forward).parameters


def _accepts_position_ids(model):
    return _accepts(model, "position_ids")


class InspectionMixin:
    def lora_diagnostics(self, pattern="*", adapter=None, top_k=8):
        """Measure scaled LoRA updates without materializing dense update matrices.

        Return one JSON-compatible row per module and adapter. By default inspect
        active adapters; a name or list of names selects adapters explicitly.
        Multiple adapters are measured separately, never as their combined update.
        """
        from .diagnostics import lora_diagnostics

        return lora_diagnostics(self.model, pattern, adapter, top_k)

    @contextmanager
    def _track_positions(self, positions):
        """List an active hook's positions so batched scoring can refuse padding that moves them."""
        entry = (object(), _padding_positions(positions))
        if not hasattr(self, "_position_hooks"):
            self._position_hooks = []
        try:
            # Inside try: a timeout right after appending still removes the entry.
            self._position_hooks.append(entry)
            yield
        finally:
            self._position_hooks = [item for item in self._position_hooks if item is not entry]

    def patches(self):
        """Describe active activation interventions without retaining tensor copies."""
        return [dict(item) for item in getattr(self, "_patches", [])]

    @contextmanager
    def record(self, modules, positions=None, at="output"):
        """Capture selected positions as detached CPU tensors for each call.

        ``at`` picks what to capture. "output" is a module's output, "input" its first
        tensor argument (or ``hidden_states=``), and "attention" the attention weights
        ``[batch, heads, selected, keys]`` from an attention block, where positions select
        queries. Positions are None (the last token), an int, a list of distinct ints,
        "all", or a slice. Attention capture needs eager attention and refuses a module
        whose output has no 4-D second element. A bad ``at`` or position raises before any
        hook is registered.
        """
        _eager(self.model)
        _check_at(at, _RECORD_AT)
        positions = _listed(positions)
        names = [modules] if isinstance(modules, str) else list(modules)
        if not names or len(set(names)) != len(names):
            raise ValueError("choose a nonempty list of distinct module names")
        targets = [(name, self.model.get_submodule(name)) for name in names]
        captured = {name: [] for name in names}
        hooks = []

        def read(args, last):
            if at == "input":
                return _input_activation(args, last)[0], 1
            if at == "attention":
                return _attention_weights(last), 2
            return _activation(last), 1

        try:
            for name, target in targets:

                def capture(module, args, last, name=name):
                    value, dim = read(args, last)
                    chosen = _take(value, _positions(value, positions, dim), dim)
                    captured[name].append(chosen.detach().cpu().clone())

                # Track before registering so an interrupt cannot orphan the hook.
                hooks.append(_TemporaryHook(target, capture, pre=at == "input"))
                hooks[-1].register()
            with self._track_positions(positions):
                yield captured
        finally:
            for hook in hooks:
                hook.remove()

    @contextmanager
    def patch(self, module, values, positions=None, at="output"):
        """Replace selected activation positions for the duration of the context.

        ``at`` is "output" or "input"; an input patch replaces the module's first tensor
        argument (or ``hidden_states=``) before it runs. Positions are as for ``record``.
        A callable receives the selected tensor and must return its replacement.
        Tensor replacements may broadcast to the selected activation shape and take the
        incoming tensor's device and dtype. Attention weights cannot be patched; use record.
        """
        _eager(self.model)
        if at == "attention":
            raise ValueError("attention weights cannot be patched; use record")
        _check_at(at, _PATCH_AT)
        positions = _listed(positions)
        target = self.model.get_submodule(module)

        def replace(hooked, inputs, output):
            return _with_output(output, _replaced(_activation(output), positions, values))

        def replace_input(hooked, args, kwargs):
            value, slot = _input_activation(args, kwargs)
            return _with_input(args, kwargs, slot, _replaced(value, positions, values))

        self._patch_revision = getattr(self, "_patch_revision", 0) + 1
        if not hasattr(self, "_patches"):
            self._patches = []
        metadata = {
            "id": f"patch-{self._patch_revision}",
            "module": module,
            "positions": repr(positions),
            "at": at,
            "kind": "callable" if callable(values) else "tensor",
        }
        hook = _TemporaryHook(
            target, replace_input if at == "input" else replace, pre=at == "input"
        )
        # Worker timeouts raise BaseException anywhere, so register inside cleanup.
        try:
            hook.register()
            self._patches.append(metadata)
            with self._track_positions(positions):
                yield
        finally:
            hook.remove()
            for index, item in enumerate(self._patches):
                if item is metadata:
                    del self._patches[index]
                    break
            self._patch_revision += 1

    def gradient(
        self,
        prompt,
        target=None,
        parameters=None,
        max_tokens=512,
        *,
        modules=None,
        positions="all",
        at="output",
    ):
        """Return each row's mean causal loss gradient, stacked on dimension zero.

        A string target masks the prompt so only the answer contributes to loss.
        Dict inputs may contain their own labels, with -100 marking ignored tokens.
        Padding in attention_mask is never scored or used to predict a real token;
        without explicit position_ids, padded rows count positions from their first
        real token. Tied aliases (such as lm_head.weight) are accepted, and aliases
        requested together share one returned tensor. parameters=None returns each
        tensor once, under its first name.
        Existing parameter gradients, train/eval flags, and requires_grad flags survive.

        ``modules`` (a name or a nonempty list of distinct names) adds gradients of the same
        loss with respect to those modules' activations. ``at`` is "output" or "input", as for
        ``patch``. Each entry is ``[rows, selected tokens, ...]``, detached on CPU in the
        activation's dtype. ``positions`` selects tokens as for ``record``, but defaults to
        "all", not the last token: in a causal model the last position predicts nothing that
        is scored, so its output gradient is zero. With ``modules`` given, ``parameters=None``
        returns no parameter gradients; pass both to get both. A module that did not run, or
        ran more than once in one forward, raises, because its gradient would be missing or
        would mix calls. An input gradient is through that module's use of its input only,
        not other uses of the same tensor, such as a residual connection. Where the activation
        has no gradient (every parameter frozen), the gradient is taken at a detached copy
        that replaces it, so forward values are unchanged. ``positions`` and ``at`` need
        ``modules``; ``at`` "attention" is refused.
        """
        import torch
        import torch.nn.functional as functional

        _eager(self.model)
        _check_at(at, _PATCH_AT)
        if modules is None:
            if not (isinstance(positions, str) and positions == "all") or at != "output":
                raise ValueError("positions and at select activation gradients; pass modules")
            module_names = []
        else:
            positions = _listed(positions)
            module_names = [modules] if isinstance(modules, str) else list(modules)
            if not module_names or len(set(module_names)) != len(module_names):
                raise ValueError("choose a nonempty list of distinct module names")
        targets = [(name, self.model.get_submodule(name)) for name in module_names]
        if not isinstance(max_tokens, int) or max_tokens < 2:
            raise ValueError("max_tokens must be an integer of at least two")
        if torch.is_inference_mode_enabled():
            raise ValueError("gradient cannot run inside torch.inference_mode()")
        if isinstance(prompt, str):
            if target is None:
                inputs = dict(self.tokenizer(prompt, return_tensors="pt"))
            else:
                if not isinstance(target, str) or not target:
                    raise ValueError("target must be a nonempty string")
                prefix = list(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
                answer = list(self.tokenizer(target, add_special_tokens=False)["input_ids"])
                if not prefix or not answer:
                    raise ValueError("prompt and target must each contain tokens")
                inputs = {
                    "input_ids": torch.tensor([prefix + answer]),
                    "labels": torch.tensor([[-100] * len(prefix) + answer]),
                }
        elif isinstance(prompt, dict):
            if target is not None:
                raise ValueError("use labels in dict inputs instead of a target string")
            inputs = dict(prompt)
        else:
            raise ValueError("prompt must be text or a dict of model inputs")
        inputs = {name: torch.as_tensor(value).to(self.device) for name, value in inputs.items()}
        ids = inputs.get("input_ids")
        if ids is None or ids.ndim != 2 or ids.shape[0] == 0:
            raise ValueError("input_ids must have shape [batch, tokens]")
        if not 2 <= ids.shape[1] <= max_tokens:
            raise ValueError(f"gradient rows must contain 2 to {max_tokens} tokens")
        labels = inputs.pop("labels", ids.clone()).clone()
        if labels.shape != ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        mask = inputs.get("attention_mask")
        if mask is not None:
            if mask.shape != ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            labels[mask == 0] = -100
            # After the causal shift, a pad position would predict the next real token.
            labels[:, 1:][mask[:, :-1] == 0] = -100
            # Absolute position embeddings must not count left padding.
            if (
                "position_ids" not in inputs
                and (mask == 0).any()
                and _accepts_position_ids(self.model)
            ):
                inputs["position_ids"] = (mask.long().cumsum(-1) - 1).clamp(min=0)
        if any(not (row[1:] != -100).any().item() for row in labels):
            raise ValueError("each row must have at least one scored token after its first token")
        if modules is not None:
            # Refuse positions this length cannot satisfy before any pass is spent.
            _positions(torch.empty(1, ids.shape[1], 0), positions, 1)
        # Tied weights have several names; parts() lists every alias.
        named = dict(self.model.named_parameters(remove_duplicate=False))
        if parameters is not None:
            names = list(parameters)
        elif modules is None:
            # Quantized weights have no gradient to take; the float parameters keep theirs.
            names = [
                name
                for name, _ in self.model.named_parameters()
                if lookup(self.model, name) is None
            ]
        else:
            names = []
        if parameters is not None or modules is None:
            if (
                not names
                or len(set(names)) != len(names)
                or any(name not in named for name in names)
            ):
                raise ValueError("parameters must name distinct existing model parameters")
        for name in names if parameters is not None else []:
            packed = lookup(self.model, name)
            if packed is not None:
                raise ValueError(
                    f"cannot take a gradient with respect to {name}: it is a "
                    f"{packed.method} quantized weight"
                )
        # Aliases of one tensor share one gradient, computed once.
        selected = list({id(named[name]): named[name] for name in names}.values())
        if any(not parameter.is_floating_point() for parameter in selected):
            raise ValueError("gradients require floating point parameters")
        flags = [parameter.requires_grad for parameter in selected]
        modes = [(module, module.training) for module in self.model.modules()]
        gradients = {id(parameter): [] for parameter in selected}
        activation_gradients = {name: [] for name in module_names}
        try:
            for parameter in selected:
                parameter.requires_grad_(True)
            self.model.eval()
            with torch.enable_grad():
                for row in range(ids.shape[0]):
                    row_inputs = {name: value[row : row + 1] for name, value in inputs.items()}
                    seen = {}
                    hooks = []
                    try:
                        for name, module in targets:
                            hooks.append(
                                _TemporaryHook(
                                    module, self._gradient_hook(name, at, seen), pre=at == "input"
                                )
                            )
                            hooks[-1].register()
                        output = self.model(**row_inputs)
                    finally:
                        for hook in hooks:
                            hook.remove()
                    for name in module_names:
                        if name not in seen:
                            raise ValueError(f"module {name!r} did not run in the forward")
                    logits = output.logits if hasattr(output, "logits") else output[0]
                    loss = functional.cross_entropy(
                        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                        labels[row : row + 1, 1:].reshape(-1),
                        ignore_index=-100,
                    )
                    activations = [seen[name] for name in module_names]
                    values = (
                        torch.autograd.grad(loss, activations + selected, allow_unused=True)
                        if loss.requires_grad
                        else [None] * (len(activations) + len(selected))
                    )
                    for name, activation, value in zip(
                        module_names, activations, values[: len(activations)], strict=True
                    ):
                        value = torch.zeros_like(activation) if value is None else value
                        chosen = _take(value, _positions(value, positions, 1), 1)
                        activation_gradients[name].append(chosen[0].detach().cpu())
                    for parameter, value in zip(selected, values[len(activations) :], strict=True):
                        gradients[id(parameter)].append(
                            torch.zeros_like(parameter, device="cpu")
                            if value is None
                            else value.detach().cpu()
                        )
        finally:
            for parameter, flag in zip(selected, flags, strict=True):
                parameter.requires_grad_(flag)
            for module, training in modes:
                module.training = training
        stacked = {key: torch.stack(rows) for key, rows in gradients.items()}
        result = {name: torch.stack(rows) for name, rows in activation_gradients.items()}
        result.update({name: stacked[id(named[name])] for name in names})
        return result

    @staticmethod
    def _gradient_hook(name, at, seen):
        """A hook that records one module's activation for a row, ready for autograd.grad."""

        def keep(value, always):
            if name in seen:
                raise ValueError(
                    f"module {name!r} ran more than once in one forward, "
                    "so its gradient would mix calls"
                )
            # A stand-in only this module sees. A view keeps the graph, so upstream gradients
            # still flow; a frozen model's activation has none, so it gets a detached leaf.
            if value.requires_grad:
                seen[name] = value.view_as(value) if always else value
            else:
                seen[name] = value.detach().requires_grad_(True)
            # The model gets a copy, so an in-place op downstream cannot modify a leaf or
            # rebase the tensor the gradient is taken at.
            return seen[name].clone()

        def after(module, args, output):
            return _with_output(output, keep(_activation(output), False))

        def before(module, args, kwargs):
            value, slot = _input_activation(args, kwargs)
            return _with_input(args, kwargs, slot, keep(value, True))

        return before if at == "input" else after

    def probe(
        self, features, labels, groups, train_groups=None, seed=0, test_fraction=0.25, ridge=1e-3
    ):
        return train_probe(features, labels, groups, train_groups, seed, test_fraction, ridge)


def train_probe(
    features, labels, groups, train_groups=None, seed=0, test_fraction=0.25, ridge=1e-3
):
    """Fit a ridge classifier with an intercept and split whole source documents."""
    import torch

    # MPS has no float64: move to CPU before widening.
    features = torch.as_tensor(features).detach().cpu().to(torch.float64)
    labels = torch.as_tensor(labels).detach().cpu()

    def document_ids(values):
        normalized = []
        for value in values:
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError("each document group must be a scalar ID")
                value = value.item()
            try:
                hash(value)
            except TypeError as exc:
                raise ValueError("document group IDs must be hashable") from exc
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("document group IDs must be finite")
            normalized.append(value)
        return normalized

    groups = document_ids(groups)
    if features.ndim != 2 or features.shape[1] == 0 or not torch.isfinite(features).all():
        raise ValueError("features must be a finite matrix with at least one feature")
    if labels.ndim != 1 or len(labels) != len(features) or len(groups) != len(features):
        raise ValueError("features, labels, and groups must contain the same number of rows")
    if labels.is_floating_point() and not torch.isfinite(labels).all():
        raise ValueError("labels must be finite")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    unique_groups = list(dict.fromkeys(groups))
    if len(unique_groups) < 2:
        raise ValueError("at least two document groups are required")
    if train_groups is None:
        shuffled = unique_groups.copy()
        random.Random(seed).shuffle(shuffled)
        ntest = min(len(shuffled) - 1, max(1, math.ceil(len(shuffled) * test_fraction)))
        train_set = set(shuffled[ntest:])
    else:
        train_set = set(document_ids(train_groups))
        if not train_set or not train_set < set(unique_groups):
            raise ValueError("train_groups must be a nonempty proper subset of known groups")
    train_mask = torch.tensor([group in train_set for group in groups])
    classes, encoded = labels.unique(sorted=True, return_inverse=True)
    if len(classes) < 2 or len(encoded[train_mask].unique()) != len(classes):
        raise ValueError("at least two classes are required and training must contain every class")
    x = features[train_mask]
    y = torch.nn.functional.one_hot(encoded[train_mask], len(classes)).to(torch.float64)
    xmean, ymean = x.mean(0), y.mean(0)
    x, y = x - xmean, y - ymean
    if x.shape[0] < x.shape[1]:
        coefficients = x.T @ torch.linalg.solve(x @ x.T + ridge * torch.eye(len(x)), y)
    else:
        coefficients = torch.linalg.solve(x.T @ x + ridge * torch.eye(x.shape[1]), x.T @ y)
    intercept = ymean - xmean @ coefficients
    prediction = (features @ coefficients + intercept).argmax(1)
    return {
        "accuracy": (prediction[~train_mask] == encoded[~train_mask]).double().mean().item(),
        "train_accuracy": (prediction[train_mask] == encoded[train_mask]).double().mean().item(),
        "coefficients": coefficients,
        "intercept": intercept,
        "classes": classes,
        "train_groups": [group for group in unique_groups if group in train_set],
        "test_groups": [group for group in unique_groups if group not in train_set],
        "train_size": int(train_mask.sum()),
        "test_size": int((~train_mask).sum()),
    }
