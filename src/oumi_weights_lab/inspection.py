"""Eager activation hooks, short-row gradients, and document-split probes."""

from __future__ import annotations

import math
import random
from contextlib import contextmanager


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


def _positions(value, positions):
    selected = (
        [-1]
        if positions is None
        else [positions]
        if isinstance(positions, int)
        else list(positions)
    )
    if not selected or any(not isinstance(index, int) for index in selected):
        raise ValueError("positions must be an integer or a nonempty list of integers")
    if any(index < -value.shape[1] or index >= value.shape[1] for index in selected):
        raise ValueError("token position is outside the module output")
    return selected


class InspectionMixin:
    def lora_diagnostics(self, pattern="*", adapter=None, top_k=8):
        """Measure scaled LoRA updates without materializing dense update matrices.

        Return one JSON-compatible row per module and adapter. By default inspect
        active adapters; a name or list of names selects adapters explicitly.
        Multiple adapters are measured separately, never as their combined update.
        """
        from .diagnostics import lora_diagnostics

        return lora_diagnostics(self.model, pattern, adapter, top_k)

    def patches(self):
        """Describe active activation interventions without retaining tensor copies."""
        return [dict(item) for item in getattr(self, "_patches", [])]

    @contextmanager
    def record(self, modules, positions=None):
        """Capture selected output positions as detached CPU tensors for each call."""
        _eager(self.model)
        names = [modules] if isinstance(modules, str) else list(modules)
        if not names or len(set(names)) != len(names):
            raise ValueError("choose a nonempty list of distinct module names")
        targets = [(name, self.model.get_submodule(name)) for name in names]
        captured = {name: [] for name in names}
        handles = []
        try:
            for name, target in targets:

                def capture(module, inputs, output, name=name):
                    value = _activation(output)
                    captured[name].append(
                        value[:, _positions(value, positions)].detach().cpu().clone()
                    )

                handles.append(target.register_forward_hook(capture))
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def patch(self, module, values, positions=None):
        """Replace selected activation positions for the duration of the context.

        A callable receives the selected tensor and must return its replacement.
        Tensor replacements may broadcast to the selected activation shape.
        """
        import torch

        _eager(self.model)
        target = self.model.get_submodule(module)

        def replace(hooked, inputs, output):
            value = _activation(output)
            indices = _positions(value, positions)
            chosen = value[:, indices].clone()
            replacement = values(chosen) if callable(values) else values
            replacement = torch.as_tensor(replacement, device=value.device, dtype=value.dtype)
            updated = value.clone()
            updated[:, indices] = replacement
            if isinstance(output, tuple):
                entries = (updated, *output[1:])
                return type(output)(*entries) if hasattr(output, "_fields") else entries
            return updated

        handle = target.register_forward_hook(replace)
        self._patch_revision = getattr(self, "_patch_revision", 0) + 1
        if not hasattr(self, "_patches"):
            self._patches = []
        metadata = {
            "id": f"patch-{self._patch_revision}",
            "module": module,
            "positions": repr(positions),
            "kind": "callable" if callable(values) else "tensor",
        }
        self._patches.append(metadata)
        try:
            yield
        finally:
            handle.remove()
            self._patches.remove(metadata)
            self._patch_revision += 1

    def gradient(self, prompt, target=None, parameters=None, max_tokens=512):
        """Return each row's mean causal loss gradient, stacked on dimension zero.

        A string target masks the prompt so only the answer contributes to loss.
        Dict inputs may contain their own labels, with -100 marking ignored tokens.
        Existing parameter gradients, train/eval flags, and requires_grad flags survive.
        """
        import torch
        import torch.nn.functional as functional

        _eager(self.model)
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
        if "attention_mask" in inputs:
            if inputs["attention_mask"].shape != ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            labels[inputs["attention_mask"] == 0] = -100
        if any(not (row[1:] != -100).any().item() for row in labels):
            raise ValueError("each row must have at least one scored token after its first token")
        named = dict(self.model.named_parameters())
        names = list(named) if parameters is None else list(parameters)
        if not names or len(set(names)) != len(names) or any(name not in named for name in names):
            raise ValueError("parameters must name distinct existing model parameters")
        selected = [named[name] for name in names]
        if any(not parameter.is_floating_point() for parameter in selected):
            raise ValueError("gradients require floating point parameters")
        flags = [parameter.requires_grad for parameter in selected]
        modes = [(module, module.training) for module in self.model.modules()]
        gradients = {name: [] for name in names}
        try:
            for parameter in selected:
                parameter.requires_grad_(True)
            self.model.eval()
            with torch.enable_grad():
                for row in range(ids.shape[0]):
                    row_inputs = {name: value[row : row + 1] for name, value in inputs.items()}
                    output = self.model(**row_inputs)
                    logits = output.logits if hasattr(output, "logits") else output[0]
                    loss = functional.cross_entropy(
                        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                        labels[row : row + 1, 1:].reshape(-1),
                        ignore_index=-100,
                    )
                    values = (
                        torch.autograd.grad(loss, selected, allow_unused=True)
                        if loss.requires_grad
                        else [None] * len(selected)
                    )
                    for name, parameter, value in zip(names, selected, values, strict=True):
                        gradients[name].append(
                            torch.zeros_like(parameter, device="cpu")
                            if value is None
                            else value.detach().cpu()
                        )
        finally:
            for parameter, flag in zip(selected, flags, strict=True):
                parameter.requires_grad_(flag)
            for module, training in modes:
                module.training = training
        return {name: torch.stack(rows) for name, rows in gradients.items()}

    def probe(
        self, features, labels, groups, train_groups=None, seed=0, test_fraction=0.25, ridge=1e-3
    ):
        return train_probe(features, labels, groups, train_groups, seed, test_fraction, ridge)


def train_probe(
    features, labels, groups, train_groups=None, seed=0, test_fraction=0.25, ridge=1e-3
):
    """Fit a ridge classifier with an intercept and split whole source documents."""
    import torch

    features = torch.as_tensor(features).detach().to(device="cpu", dtype=torch.float64)
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
