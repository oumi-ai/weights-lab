"""Building blocks for writing a new method as one short function.

``run_with`` runs the model once with activations or weights replaced, ``jacobian``, ``jvp``
and ``vjp`` differentiate between two recordable points, and ``sweep`` tries many
interventions in one batched pass. All three reuse the hooks of ``inspection``, count passes
through the lab's normal counter, and leave nothing attached afterward.
"""

from __future__ import annotations

from contextlib import ExitStack

from .inspection import (
    _PATCH_AT,
    _RECORD_AT,
    _accepts,
    _check_at,
    _eager,
    _listed,
    _patch_hook,
    _positions,
    _record_hook,
    _replaced,
    _site_hook,
)
from .quantized import lookup
from .scoring import _pad_token

_INTERVENTION_KEYS = {"module", "at", "positions", "value"}
_RECORD_KEYS = {"module", "at", "positions"}
_LOGITS = ("last", "all", None)


def _normalize_intervention(model, spec):
    """One intervention checked and resolved: its module, ``at``, positions and value."""
    if not isinstance(spec, dict):
        raise ValueError("an intervention is a dict with module, value, and optional at, positions")
    extra = set(spec) - _INTERVENTION_KEYS
    if extra:
        raise ValueError(f"unknown intervention keys: {', '.join(sorted(map(str, extra)))}")
    if "module" not in spec or not isinstance(spec["module"], str):
        raise ValueError("an intervention needs a module name")
    if "value" not in spec:
        raise ValueError("an intervention needs a value: a tensor or a callable")
    at = spec.get("at", "output")
    if at == "attention":
        raise ValueError("attention weights cannot be replaced; at must be 'output' or 'input'")
    _check_at(at, _PATCH_AT)
    return {
        "name": spec["module"],
        "module": model.get_submodule(spec["module"]),
        "at": at,
        "positions": _listed(spec.get("positions")),
        "value": spec["value"],
    }


def _normalize_record(model, spec):
    if isinstance(spec, str):
        spec = {"module": spec}
    if not isinstance(spec, dict) or not isinstance(spec.get("module"), str):
        raise ValueError("record entries are module names or dicts with a module name")
    extra = set(spec) - _RECORD_KEYS
    if extra:
        raise ValueError(f"unknown record keys: {', '.join(sorted(map(str, extra)))}")
    at = spec.get("at", "output")
    _check_at(at, _RECORD_AT)
    return {
        "name": spec["module"],
        "module": model.get_submodule(spec["module"]),
        "at": at,
        "positions": _listed(spec.get("positions")),
    }


def _normalize_records(model, record):
    entries = [record] if isinstance(record, (str, dict)) else list(record)
    entries = [_normalize_record(model, spec) for spec in entries]
    names = [entry["name"] for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError("record each module once; records are keyed by module name")
    return entries


def _check_logits(logits):
    if logits not in _LOGITS:
        raise ValueError('logits must be "last", "all", or None')


def _check_weights(model, weights):
    """Weight replacements checked against the model; returns them on the parameters' devices."""
    if not isinstance(weights, dict):
        raise ValueError("weights must map parameter names to tensors")
    import torch

    named = dict(model.named_parameters(remove_duplicate=False))
    chosen, owners, given = {}, {}, {}
    for name, value in weights.items():
        if name not in named:
            raise ValueError(f"{name!r} is not a parameter of this model")
        parameter = named[name]
        packed = lookup(model, name)
        if packed is not None:
            raise ValueError(
                f"{name} is a {packed.method} quantized weight; run_with cannot replace it. "
                "Use lab.edit and lab.undo for quantized weights"
            )
        value = torch.as_tensor(value)
        if value.shape != parameter.shape:
            raise ValueError(
                f"replacement for {name} has shape {tuple(value.shape)}, "
                f"but the parameter has shape {tuple(parameter.shape)}"
            )
        first = owners.get(id(parameter))
        if first is not None and given[first] is not value:
            raise ValueError(
                f"{first} and {name} are tied to one parameter; "
                "give one replacement, which both uses will share"
            )
        # Compare the tensors as given: a cast makes a new copy for each name.
        owners.setdefault(id(parameter), name)
        given[name] = value
        chosen[name] = (
            chosen[first]
            if first is not None
            else value.to(device=parameter.device, dtype=parameter.dtype)
        )
    return chosen


class MethodsMixin:
    """Run with replacements, differentiate between two points, and sweep interventions."""

    # Inputs --------------------------------------------------------------------------

    def _method_inputs(self, inputs):
        """Model kwargs on ``lab.device``, and the length of the shortest real sequence.

        A string is one sequence. A list of strings is left-padded with an attention mask
        and position ids, as scoring batches are. Text is prepared as ``score`` prepares a
        prompt: prefix attachments and any chat template apply, and special tokens follow
        scoring's rule. A dict of tensors passes through, with
        tensors moved to ``lab.device``. The second value is None when nothing is padded.
        """
        import torch

        tokenizer = self.tokenizer
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(inputs, (list, tuple)):
            if not inputs or not all(isinstance(text, str) for text in inputs):
                raise ValueError("inputs must be text, a nonempty list of text, or a dict")
            with self._prompts_prepared_once():
                rows = [self._prompt_ids(self._scoring_prompt(text)) for text in inputs]
            length = max(map(len, rows))
            pad = _pad_token(tokenizer)
            ids = [[pad] * (length - len(row)) + row for row in rows]
            mask = [[0] * (length - len(row)) + [1] * len(row) for row in rows]
            built = {
                "input_ids": torch.tensor(ids, device=self.device),
                "attention_mask": torch.tensor(mask, device=self.device),
            }
            if len(rows) > 1 and _accepts(self.model, "position_ids"):
                built["position_ids"] = (built["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            return built, (min(map(len, rows)) if len(set(map(len, rows))) > 1 else None)
        if not isinstance(inputs, dict):
            raise ValueError("inputs must be text, a nonempty list of text, or a dict")
        built = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        mask = built.get("attention_mask")
        if isinstance(mask, torch.Tensor) and mask.ndim == 2 and (mask == 0).any():
            return built, int(mask.sum(-1).min())
        return built, None

    @staticmethod
    def _sequences(inputs):
        for key in ("input_ids", "inputs_embeds"):
            value = inputs.get(key)
            if hasattr(value, "shape") and len(value.shape) >= 2:
                return int(value.shape[0])
        return 1

    def _single_sequence(self, inputs):
        built, _ = self._method_inputs(inputs)
        if self._sequences(built) != 1:
            raise ValueError("derivatives take one sequence: a string or inputs with batch size 1")
        return built

    def _check_budget(self, count):
        """Raise BudgetExceeded when ``count`` more passes would cross any active cap."""
        self._budget_state()
        with self._budget_guard:
            for budget in [self.budget, *self._evaluation_budgets]:
                budget.check(count)

    # Running -------------------------------------------------------------------------

    def _forward(self, inputs, weights):
        """The model's output for ``inputs``, with ``weights`` swapped in for this call only."""
        if not weights:
            return self.model(**inputs)
        from torch.func import functional_call

        return functional_call(self.model, weights, args=(), kwargs=inputs)

    @staticmethod
    def _logits_of(output, logits):
        if logits is None:
            return None
        value = output.logits if hasattr(output, "logits") else output[0]
        return value[:, -1] if logits == "last" else value

    def run_with(
        self, inputs, *, activations=(), weights=None, record=(), logits="last", grad=False
    ):
        """Run the model once with activations or whole parameters replaced for this call only.

        ``inputs`` is a string, a list of strings (left-padded with an attention mask and
        position ids, the way scoring batches are), or a dict of tensors passed to the model.
        Text is prepared as ``score`` prepares a prompt (prefix attachments, any chat
        template, scoring's special-token rule), so numbers line up with ``score`` and
        ``evaluate`` on the same text; a dict is passed through unchanged. Active logit
        attachments are model hooks, so they do apply to the returned logits (and padded
        batches refuse them, as scoring does).

        ``activations`` is a list of interventions, each
        ``{"module": name, "at": "output" | "input", "positions": None, "value": tensor or
        callable}`` as for ``patch``: positions default to the last token, and a callable
        receives the selected tensor and returns its replacement. ``weights`` maps parameter
        names to full replacement tensors through ``torch.func.functional_call``, so nothing
        is written into the model. Shapes must match. A tied parameter has several names and
        one value: replacing ``lm_head.weight`` on a tied model also replaces the input
        embedding for this call. Two names for one parameter with different tensors are
        refused, and so is a quantized weight (use ``lab.edit`` and ``lab.undo``).

        ``record`` lists module names or dicts ``{"module", "at", "positions"}``, captured
        as ``record`` captures them, each module once. ``logits`` is "last", "all", or None.
        Interventions apply before records read, so a record sees the replaced value.

        With ``grad=False`` the call runs under ``torch.no_grad()`` and logits and records
        come back detached on the CPU, logits in float32. With ``grad=True`` the graph is
        kept and logits and records stay on their device, attached, so a method can
        differentiate through the call. requires_grad flags and modes are not changed.

        Returns ``{"logits": ..., "records": {module: [tensor per call]},
        "forward_passes": n}``, where n is the sequences this call ran. Hooks are removed,
        parameters are unchanged, and ``lab.patches()`` is untouched, even after an error.
        Absolute (nonnegative) positions raise on a padded batch; negative ones and "all"
        do not.
        """
        import torch

        _eager(self.model)
        _check_logits(logits)
        specs = [_normalize_intervention(self.model, spec) for spec in activations]
        records = _normalize_records(self.model, record)
        replacements = _check_weights(self.model, weights) if weights else None
        if grad and torch.is_inference_mode_enabled():
            raise ValueError("grad=True cannot run inside torch.inference_mode()")
        built, shortest = self._method_inputs(inputs)
        captured = {entry["name"]: [] for entry in records}
        keep = (lambda chosen: chosen) if grad else (lambda chosen: chosen.detach().cpu().clone())
        hooks = []
        before = self._forward_passes
        try:
            with ExitStack() as stack:
                for entry in [*specs, *records]:
                    stack.enter_context(self._track_positions(entry["positions"]))
                if shortest is not None:
                    self._check_padded(shortest, self._sequences(built))
                for spec in specs:
                    hooks.append(
                        _patch_hook(spec["module"], spec["at"], spec["positions"], spec["value"])
                    )
                    hooks[-1].register()
                for entry in records:
                    hooks.append(
                        _record_hook(
                            entry["module"],
                            entry["at"],
                            entry["positions"],
                            captured[entry["name"]],
                            keep,
                        )
                    )
                    hooks[-1].register()
                with torch.enable_grad() if grad else torch.no_grad():
                    output = self._forward(built, replacements)
                    chosen = self._logits_of(output, logits)
        finally:
            for hook in hooks:
                hook.remove()
        if chosen is not None and not grad:
            chosen = chosen.detach().float().cpu()
        return {
            "logits": chosen,
            "records": captured,
            "forward_passes": self._forward_passes - before,
        }

    def _check_padded(self, shortest, count):
        """Refuse a padded batch where left padding would move an active hook's positions."""
        self._check_batching(count)
        if any(-index > shortest for index in self._hook_positions() if index < 0):
            raise ValueError("token position is outside the module output")

    # Derivatives ---------------------------------------------------------------------

    def _trace(
        self, inputs, source, target, source_at, target_at, source_positions, target_positions
    ):
        """Run once with the source selection swapped for a fresh leaf; return (leaf, target).

        The leaf is the source selection as ``[1, selected, ...]`` and the target is the
        target selection with its graph back to the leaf. Parameters do not track gradients
        during the run, and every flag is restored.
        """
        import torch

        _eager(self.model)
        for at in (source_at, target_at):
            _check_at(at, _PATCH_AT)
        source_positions, target_positions = _listed(source_positions), _listed(target_positions)
        source_module = self.model.get_submodule(source)
        target_module = self.model.get_submodule(target)
        if torch.is_inference_mode_enabled():
            raise ValueError("derivatives cannot run inside torch.inference_mode()")
        built = self._single_sequence(inputs)
        leaves, targets = [], []

        def swap(value):
            indices = _positions(value, source_positions)
            leaf = value.detach().index_select(1, torch.as_tensor(indices, device=value.device))
            leaf = leaf.clone().requires_grad_(True)
            leaves.append(leaf)
            updated = value.clone()
            updated[:, indices] = leaf
            return updated

        hooks = []
        parameters = list(self.model.parameters())
        flags = [parameter.requires_grad for parameter in parameters]
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            hooks.append(_site_hook(source_module, source_at, swap))
            hooks[-1].register()
            hooks.append(
                _record_hook(target_module, target_at, target_positions, targets, lambda x: x)
            )
            hooks[-1].register()
            with torch.enable_grad():
                self.model(**built)
        finally:
            for hook in hooks:
                hook.remove()
            for parameter, flag in zip(parameters, flags, strict=True):
                parameter.requires_grad_(flag)
        for name, seen in ((source, leaves), (target, targets)):
            if len(seen) != 1:
                what = "did not run" if not seen else f"ran {len(seen)} times"
                raise ValueError(
                    f"module {name!r} {what} in the forward; a derivative needs it to run once"
                )
        return leaves[0], targets[0]

    @staticmethod
    def _grad_of(target, leaf, cotangent):
        """The gradient of ``(target * cotangent).sum()`` at ``leaf``; zeros if unconnected."""
        import torch

        if not target.requires_grad:
            return torch.zeros_like(leaf)
        (gradient,) = torch.autograd.grad(target, leaf, cotangent, allow_unused=True)
        return torch.zeros_like(leaf) if gradient is None else gradient

    @staticmethod
    def _check_direction(value, like, label):
        import torch

        value = torch.as_tensor(value)
        if value.shape != like.shape:
            raise ValueError(
                f"{label} has shape {tuple(value.shape)}, but the selection has shape "
                f"{tuple(like.shape)} without the batch dimension"
            )
        return value.to(device=like.device, dtype=like.dtype)

    def jacobian(
        self,
        inputs,
        source,
        target,
        *,
        source_at="output",
        target_at="output",
        source_positions=None,
        target_positions=None,
    ):
        """The derivative of one point's selected activation with respect to another's.

        A point is a module, ``at`` ("output" or "input") and positions, as for ``patch``;
        positions default to the last token. ``inputs`` is one sequence: a string or a
        dict with batch size 1. The model runs with the source selection replaced through a
        hook, so the derivative is the effect of changing that selection with everything
        else held fixed. Returns a float32 CPU tensor shaped
        ``[*target_selected_shape, *source_selected_shape]``, both without the batch
        dimension.

        The source and the target must each run exactly once in the forward pass; otherwise
        this raises, naming the module. A target that runs before the source gives a zero
        derivative. Parameters do not accumulate ``.grad``; requires_grad flags and the
        train/eval mode are left as they were. This costs one forward pass.
        """
        import torch

        leaf, out = self._trace(
            inputs, source, target, source_at, target_at, source_positions, target_positions
        )
        shape = out[0].shape
        if not out.requires_grad:
            rows = torch.zeros(out[0].numel(), *leaf[0].shape)
        else:
            basis = torch.eye(out[0].numel(), dtype=out.dtype, device=out.device)
            basis = basis.reshape(-1, *out.shape)
            try:
                (rows,) = torch.autograd.grad(
                    out, leaf, basis, is_grads_batched=True, allow_unused=True
                )
            except RuntimeError:
                # Some operations have no batching rule; one backward per element works anywhere.
                rows = [
                    torch.autograd.grad(out, leaf, row, retain_graph=True, allow_unused=True)[0]
                    for row in basis
                ]
                rows = torch.stack([torch.zeros_like(leaf) if r is None else r for r in rows])
            if rows is None:
                rows = torch.zeros(out[0].numel(), *leaf.shape)
            rows = rows[:, 0]
        return rows.detach().float().cpu().reshape(*shape, *leaf[0].shape)

    def jvp(
        self,
        inputs,
        source,
        target,
        tangent,
        *,
        source_at="output",
        target_at="output",
        source_positions=None,
        target_positions=None,
    ):
        """Return ``(target_value, derivative)``: the target and its change along ``tangent``.

        The points and their rules are those of ``jacobian``. ``tangent`` has the source
        selection's shape without the batch dimension. Both results are float32 CPU
        tensors shaped like the target selection without the batch dimension. This costs
        one forward pass. The product is computed with a double backward, so it needs no
        Jacobian.
        """
        import torch

        leaf, out = self._trace(
            inputs, source, target, source_at, target_at, source_positions, target_positions
        )
        tangent = self._check_direction(tangent, leaf[0], "tangent")
        value = out[0].detach().float().cpu()
        if not out.requires_grad:
            return value, torch.zeros_like(value)
        dummy = torch.zeros_like(out, requires_grad=True)
        (gradient,) = torch.autograd.grad(out, leaf, dummy, create_graph=True, allow_unused=True)
        if gradient is None or not gradient.requires_grad:
            return value, torch.zeros_like(value)
        (derivative,) = torch.autograd.grad(gradient, dummy, tangent[None], allow_unused=True)
        if derivative is None:
            return value, torch.zeros_like(value)
        return value, derivative[0].detach().float().cpu()

    def vjp(
        self,
        inputs,
        source,
        target,
        cotangent,
        *,
        source_at="output",
        target_at="output",
        source_positions=None,
        target_positions=None,
    ):
        """Return ``(target_value, gradient)``: the target selection and ``cotangent`` pulled back.

        The points and their rules are those of ``jacobian``. ``cotangent`` has the target
        selection's shape without the batch dimension. ``target_value`` is shaped like it,
        and ``gradient`` like the source selection, both without the batch dimension and as
        float32 CPU tensors. This costs one forward pass.
        """
        leaf, out = self._trace(
            inputs, source, target, source_at, target_at, source_positions, target_positions
        )
        cotangent = self._check_direction(cotangent, out[0], "cotangent")
        gradient = self._grad_of(out, leaf, cotangent[None])
        return out[0].detach().float().cpu(), gradient[0].detach().float().cpu()

    # Sweeps --------------------------------------------------------------------------

    def sweep(self, inputs, interventions, *, record=(), logits="last", batch_size=8):
        """Try many intervention sets on one prompt in batched passes, one set per row.

        ``inputs`` is one prompt: a string, or a dict with batch size 1. It is repeated once
        per entry of ``interventions``, a list whose entries are None (a clean row) or a list
        of interventions in the format of ``run_with``. Row i gets only entry i's
        interventions, so rows may target the same module with different values. A callable
        receives that row's selected tensor, shaped ``[1, selected, ...]``.

        Rows run in chunks of ``batch_size`` sequences, with one hook per distinct
        (module, at) site. Each sequence counts one pass. The budget is checked before each
        chunk, so a sweep that would pass the cap fails before starting that chunk; chunks
        before it have run and been counted, and their results are not returned.

        Returns one dict per entry, ``{"logits": ..., "records": ...}``, shaped as
        ``run_with`` returns for that entry alone. Weight replacements are not supported
        here; call ``run_with`` in a loop for those. Nothing stays attached afterward.
        """

        _eager(self.model)
        _check_logits(logits)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(interventions, (list, tuple)) or not interventions:
            raise ValueError("interventions must be a nonempty list of None or intervention lists")
        rows = []
        for entry in interventions:
            if entry is not None and (
                not isinstance(entry, (list, tuple)) or isinstance(entry, dict)
            ):
                raise ValueError("each entry of interventions is None or a list of interventions")
            rows.append([_normalize_intervention(self.model, spec) for spec in entry or ()])
        records = _normalize_records(self.model, record)
        if isinstance(inputs, (list, tuple)):
            raise ValueError("sweep takes one prompt; a list of prompts is not supported")
        built, _ = self._method_inputs(inputs)
        if self._sequences(built) != 1:
            raise ValueError("sweep takes one prompt: a string or inputs with batch size 1")
        results = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            self._check_budget(len(chunk))
            results.extend(self._sweep_chunk(built, chunk, records, logits))
        return results

    def _sweep_chunk(self, built, chunk, records, logits):
        import torch

        count = len(chunk)
        batch = {
            key: value.expand(count, *value.shape[1:]).contiguous()
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == 1
            else value
            for key, value in built.items()
        }
        sites = {}
        for row, specs in enumerate(chunk):
            for spec in specs:
                site = sites.setdefault((spec["name"], spec["at"]), (spec["module"], {}))
                site[1].setdefault(row, []).append(spec)

        def applier(assigned):
            def apply(value):
                updated = value.clone()
                for row, specs in assigned.items():
                    piece = updated[row : row + 1]
                    for spec in specs:
                        piece = _replaced(piece, spec["positions"], spec["value"])
                    updated[row : row + 1] = piece
                return updated

            return apply

        captured = {entry["name"]: [] for entry in records}
        hooks = []
        try:
            with ExitStack() as stack:
                for entry in records:
                    stack.enter_context(self._track_positions(entry["positions"]))
                for (_, at), (module, assigned) in sites.items():
                    hooks.append(_site_hook(module, at, applier(assigned)))
                    hooks[-1].register()
                for entry in records:
                    hooks.append(
                        _record_hook(
                            entry["module"],
                            entry["at"],
                            entry["positions"],
                            captured[entry["name"]],
                            lambda chosen: chosen.detach().cpu().clone(),
                        )
                    )
                    hooks[-1].register()
                with torch.no_grad():
                    output = self.model(**batch)
                    chosen = self._logits_of(output, logits)
        finally:
            for hook in hooks:
                hook.remove()
        if chosen is not None:
            chosen = chosen.detach().float().cpu()
        return [
            {
                "logits": None if chosen is None else chosen[row : row + 1].clone(),
                "records": {
                    name: [value[row : row + 1].clone() for value in values]
                    for name, values in captured.items()
                },
            }
            for row in range(count)
        ]
