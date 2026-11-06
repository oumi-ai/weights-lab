"""Sparse autoencoders and transcoders on a session's model, through SAELens.

SAELens does the loading and the encode and decode math. This module finds the lab's own
module for each TransformerLens hook name, applies the transform that makes a Hugging Face
activation match what the dictionary was trained on, and routes reads and edits through
``Lab.record`` and ``Lab.patch``.
"""

from __future__ import annotations

import json
import re
import urllib.error
import warnings
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from .circuits import _blocks, _layer_index
from .inspection import _is_int, _positions

_INSTALL = "pip install 'oumi-weights-lab[saes]'"

# SAELens architectures whose decoder is affine, so an edit is a sum of decoder rows.
_ARCHITECTURES = {
    "standard",
    "gated",
    "jumprelu",
    "topk",
    "abstopk",
    "transcoder",
    "skip_transcoder",
    "jumprelu_transcoder",
    "jumprelu_skip_transcoder",
}

# SAELens model names and the Hugging Face ids that are the same model.
_MODEL_IDS = {
    "gpt2": {"gpt2", "openai-community/gpt2"},
    "gpt2-small": {"gpt2", "openai-community/gpt2"},
    "gemma-2-2b": {"google/gemma-2-2b"},
    "gemma-2-2b-it": {"google/gemma-2-2b-it"},
    "gemma-2-9b": {"google/gemma-2-9b"},
    "gemma-2-9b-it": {"google/gemma-2-9b-it"},
    "gemma-2-27b": {"google/gemma-2-27b"},
    "gemma-2-27b-it": {"google/gemma-2-27b-it"},
}

_HOOKS = (
    "hook_resid_pre",
    "hook_resid_mid",
    "hook_resid_post",
    "hook_attn_out",
    "hook_mlp_out",
    "hook_mlp_in",
    "attn.hook_z",
    "ln2.hook_normalized",
    "mlp.hook_in",
)

_RESID_MID_NOTE = (
    "{name} is the residual stream between attention and the MLP, which no module of a "
    "Hugging Face block takes as its output; a patch there would change the MLP's path but "
    "not the skip connection"
)
_NORMALIZED_NOTE = (
    "{name} is a normalized activation, so a feature edit has no exact place in the "
    "Hugging Face module's output (a normalized site can be read, not edited)"
)


def _real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


@dataclass(eq=False)
class Site:
    """Where a TransformerLens hook reads in the lab's modules, and how to match it.

    ``module`` and ``at`` are arguments to ``Lab.record`` and ``Lab.patch``. ``apply`` turns
    the activation recorded there into the one the dictionary was trained on:
    ``center`` subtracts the mean over the hidden dimension, ``norm`` is a
    ``(kind, eps)`` pair for a normalization without its scale, and ``subtract`` removes a
    fixed vector. ``read_only`` says why a feature edit has no exact place here, or None.
    """

    module: str
    at: str
    width: int
    center: bool = False
    norm: tuple[str, float] | None = None
    subtract: Any = None
    read_only: str | None = None

    @property
    def writable(self) -> bool:
        return self.read_only is None

    def apply(self, activations: Any) -> Any:
        import torch

        x = torch.as_tensor(activations)
        if not x.dtype.is_floating_point or x.dtype.itemsize < 4:
            x = x.float()
        if self.norm is not None:
            kind, eps = self.norm
            if kind == "layer":
                x = x - x.mean(-1, keepdim=True)
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        if self.center:
            x = x - x.mean(-1, keepdim=True)
        if self.subtract is not None:
            x = x - self.subtract.to(device=x.device, dtype=x.dtype)
        return x


def _hidden_size(model: Any) -> int:
    config = model.config
    config = config.get_text_config() if hasattr(config, "get_text_config") else config
    return int(config.hidden_size)


def _norm_settings(norm: Any, name: str) -> tuple[str, float]:
    from torch import nn

    kind = "layer" if isinstance(norm, nn.LayerNorm) else "rms"
    eps = getattr(norm, "eps", getattr(norm, "variance_epsilon", None))
    if eps is None:
        raise ValueError(f"{name} has no epsilon, so its normalization cannot be reproduced")
    return kind, float(eps)


def _value_bias(lab: Any, layer: int, children: dict[str, Any], fold_ln: bool) -> Any:
    """What TransformerLens folds out of ``hook_z``: the value bias, and the norm's bias."""
    import torch

    norm = children.get("ln_1", children.get("input_layernorm"))
    norm_bias = getattr(norm, "bias", None) if fold_ln else None
    total = []
    for head in range(len(lab.heads(layers=[layer]))):
        weights = lab.head_weights(layer, head)
        bias = torch.zeros(weights["v"].shape[0])
        if "v_bias" in weights:
            bias = bias + weights["v_bias"]
        if norm_bias is not None:
            bias = bias + weights["v"] @ norm_bias.detach().float().cpu()
        total.append(bias)
    vector = torch.cat(total)
    return vector if vector.any() else None


def hook_site(lab: Any, hook_name: str, model_kwargs: Mapping | None = None) -> Site:
    """The lab's site for a TransformerLens hook name, with the transform it needs.

    Supported are ``blocks.L.`` followed by ``hook_resid_pre`` (the block's input),
    ``hook_resid_post`` (its output), ``hook_resid_mid``, ``hook_mlp_in`` (the same
    residual, read before the norm), ``hook_attn_out``, ``hook_mlp_out``, ``attn.hook_z``
    (the output projection's input), ``ln2.hook_normalized`` (the MLP's normalized input),
    and ``mlp.hook_in``. Gemma 2's attention and MLP outputs are read after their
    extra post-norms, since that is what its residual stream receives.

    ``model_kwargs`` are the ``model_from_pretrained_kwargs`` an SAE records. SAELens
    loads a model without TransformerLens's weight processing and then applies those
    keywords, so only what they name is applied here: ``center_writing_weights`` centers
    the residual stream, the attention output, and the MLP output of a LayerNorm model;
    ``fold_value_biases`` (with ``fold_ln``) removes the value and norm biases from
    ``hook_z``; ``fold_ln`` makes ``mlp.hook_in`` the normalized activation.

    Raises ValueError for a hook this does not map, a layer the model lacks, and an
    architecture whose blocks it cannot read.
    """
    settings = dict(model_kwargs or {})
    match = re.fullmatch(r"blocks\.(\d+)\.(.+)", hook_name) if isinstance(hook_name, str) else None
    if match is None or match.group(2) not in _HOOKS:
        raise ValueError(
            f"hook {hook_name!r} is not mapped; the mapped hooks are blocks.L. followed by "
            + ", ".join(_HOOKS)
        )
    layer, part = int(match.group(1)), match.group(2)
    try:
        blocks = {_layer_index(name): (name, layout) for name, layout in _blocks(lab.model)}
    except NotImplementedError as error:
        raise ValueError(f"cannot map {hook_name!r}: {error}") from error
    # A hybrid model lists attention only for its full-attention layers, but every decoder
    # layer in the same stack has a residual stream.
    stacks = {name.rsplit(".", 2)[0] for name, _ in (v for v in blocks.values())}
    stack = next(iter(stacks))
    layers = sorted(
        int(child)
        for child, _ in lab.model.get_submodule(stack).named_children()
        if child.isdigit()
    )
    if layer not in layers:
        raise ValueError(
            f"the model has no layer {layer} for {hook_name!r}; its layers are "
            f"{layers[0]} to {layers[-1]}"
        )
    block = f"{stack}.{layer}"
    attention, layout = blocks.get(layer, (None, None))
    if attention is None and part in ("hook_attn_out", "attn.hook_z"):
        raise ValueError(
            f"layer {layer} has no full-attention block, which {hook_name!r} needs; "
            "its residual hooks still work"
        )
    children = dict(lab.model.get_submodule(block).named_children())
    sandwich = "pre_feedforward_layernorm" in children
    if sandwich:
        mid_norm, attn_post, mlp_post = (
            "pre_feedforward_layernorm",
            children.get("post_attention_layernorm"),
            children.get("post_feedforward_layernorm"),
        )
        if attn_post is None or mlp_post is None:
            raise ValueError(f"{block} has a pre-MLP norm but not both post-norms")
    elif "post_feedforward_layernorm" in children or (
        "post_attention_layernorm" in children and "input_layernorm" not in children
    ):
        raise ValueError(
            f"{block} normalizes after its sublayers and has no norm before its MLP "
            "(a post-norm block such as OLMo 2); its hooks are not mapped"
        )
    elif "ln_2" in children:
        mid_norm = "ln_2"
    elif "post_attention_layernorm" in children:
        mid_norm = "post_attention_layernorm"
    else:
        raise ValueError(f"{block} has no norm before its MLP that this can read")
    if "mlp" not in children:
        raise ValueError(f"{block} has no mlp child")
    norm_module = f"{block}.{mid_norm}"
    norm = _norm_settings(children[mid_norm], norm_module)
    width = _hidden_size(lab.model)
    # TransformerLens centers a residual write only in LayerNorm models.
    center = bool(settings.get("center_writing_weights", False)) and norm[0] == "layer"

    mlp = f"{block}.mlp"
    if part == "hook_resid_pre":
        return Site(block, "input", width, center)
    if part == "hook_resid_post":
        return Site(block, "output", width, center)
    if part in ("hook_resid_mid", "hook_mlp_in"):
        return Site(
            norm_module, "input", width, center, read_only=_RESID_MID_NOTE.format(name=part)
        )
    if part == "hook_attn_out":
        module = f"{block}.post_attention_layernorm" if sandwich else attention
        return Site(module, "output", width, center)
    if part == "hook_mlp_out":
        module = f"{block}.post_feedforward_layernorm" if sandwich else mlp
        return Site(module, "output", width, center)
    if part == "attn.hook_z":
        o_input = lab.heads(layers=[layer])[0]["o_input"]
        subtract = None
        if settings.get("fold_value_biases"):
            subtract = _value_bias(
                lab,
                layer,
                dict(lab.model.get_submodule(block).named_children()),
                bool(settings.get("fold_ln")),
            )
        return Site(
            o_input["module"], "input", layout["heads"] * layout["head_dim"], subtract=subtract
        )
    if part == "ln2.hook_normalized" or (part == "mlp.hook_in" and settings.get("fold_ln")):
        return Site(
            norm_module, "input", width, norm=norm, read_only=_NORMALIZED_NOTE.format(name=part)
        )
    return Site(mlp, "input", width)  # mlp.hook_in without folded norms: the MLP's own input


def _matches_session(lab: Any, model_name: str | None) -> str:
    """ "matched" or a "not checked" reason; raises ValueError for a known mismatch."""
    if not model_name:
        return "not checked: the SAE records no model name"
    key = model_name.lower()
    names = {key} | _MODEL_IDS.get(key, set())
    config_name = str(getattr(getattr(lab.model, "config", None), "_name_or_path", "") or "")
    spelled = {lab.model_id, config_name} - {""}
    if {name.lower() for name in spelled} & names:
        return "matched"
    # Look paths up as spelled: lowercasing first would miss them on a case-sensitive system.
    local = all(Path(name).exists() for name in spelled)
    if key not in _MODEL_IDS and "/" not in key:
        return f"not checked: no known mapping for the SAE's model name {model_name!r}"
    if local:
        return f"not checked: the session's model {lab.model_id!r} is a local path"
    raise ValueError(
        f"the SAE was trained on {model_name!r} but this session's model is "
        f"{lab.model_id!r}; pass allow_model_mismatch=True to use it on a fine-tune or copy"
    )


def _fetch_label(neuronpedia_id: str, index: int, timeout: float) -> str | None:
    import urllib.parse
    import urllib.request

    model, _, source = neuronpedia_id.partition("/")
    url = "https://www.neuronpedia.org/api/feature/" + "/".join(
        urllib.parse.quote(part, safe="") for part in (model, source, str(index))
    )
    request = urllib.request.Request(url, headers={"User-Agent": "oumi-weights-lab"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return _label_from(json.load(response))


def _label_from(data: Any) -> str | None:
    explanations = data.get("explanations") if isinstance(data, dict) else None
    if not isinstance(explanations, list):
        return None
    for entry in explanations:
        text = entry.get("description") if isinstance(entry, dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


class LabSAE:
    """A SAELens sparse autoencoder or transcoder tied to one session's model.

    ``module`` and ``at`` are where its input is read, as ``Lab.record`` takes them. A
    transcoder also has ``output_module`` and ``output_at``, where its output is written.
    ``encode`` takes an activation as the lab records it and applies the site's transform
    first; ``decode`` returns the reconstruction in the coordinates the dictionary was
    trained in. SAELens does its own ``b_dec`` handling inside both, so neither is applied
    twice. ``model_check`` says whether the SAE's model name matched the session's.
    """

    def __init__(
        self,
        lab: Any,
        sae: Any,
        site: Site,
        output_site: Site | None,
        *,
        release: str | None,
        sae_id: str | None,
        model_check: str,
        hook_name: str | None = None,
    ) -> None:
        metadata = sae.cfg.metadata
        self.lab = lab
        self.sae = sae
        self.site = site
        self.output_site = output_site
        self.release, self.sae_id, self.model_check = release, sae_id, model_check
        self.recorded_hook_name = metadata.hook_name
        self._label_misses: set[int] = set()
        self.hook_name = hook_name or metadata.hook_name
        self.module, self.at = site.module, site.at
        self.d_in, self.d_sae = int(sae.cfg.d_in), int(sae.cfg.d_sae)
        self.kind = "transcoder" if output_site is not None else "sae"
        self.output_module = output_site.module if output_site else None
        self.output_at = output_site.at if output_site else None
        self.model_name = metadata.model_name
        self.neuronpedia_id = metadata.neuronpedia_id
        self.prepend_bos = metadata.prepend_bos
        self.device = sae.device
        self.dtype = sae.dtype

    def __repr__(self) -> str:
        recorded = (
            f" (recorded {self.recorded_hook_name})"
            if self.hook_name != self.recorded_hook_name
            else ""
        )
        return (
            f"LabSAE({self.kind}, {self.hook_name}{recorded}, d_in={self.d_in}, "
            f"d_sae={self.d_sae}, architecture={self.sae.cfg.architecture()})"
        )

    # Tensors

    def _in_sae(self, x: Any) -> Any:
        return x.to(device=self.device, dtype=self.dtype)

    def transform(self, activations: Any) -> Any:
        """An activation as the lab records it, in the coordinates the SAE reads."""
        return self._in_sae(self.site.apply(activations))

    def transform_output(self, activations: Any) -> Any:
        """A transcoder's target as the lab records it, in the coordinates it predicts."""
        if self.output_site is None:
            raise ValueError("only a transcoder has an output site")
        return self._in_sae(self.output_site.apply(activations))

    def encode(self, activations: Any) -> Any:
        """Feature activations ``[..., d_sae]`` for activations ``[..., d_in]``."""
        return self.sae.encode(self.transform(activations))

    def decode(self, features: Any) -> Any:
        """The reconstruction from features, without a skip connection's term.

        A transcoder with a skip connection adds ``W_skip`` times its input to this;
        ``features`` includes it in the reconstruction it measures. An SAE trained with
        ``layer_norm`` or ``constant_norm_rescale`` undoes the scaling of the activation
        it last encoded, so call ``decode`` right after the matching ``encode``.
        """
        import torch

        return self.sae.decode(self._in_sae(torch.as_tensor(features)))

    def decoder_rows(self, indices: Any) -> Any:
        """The vectors a feature adds to the output when it rises by one, ``[n, d_out]``.

        These are ``W_dec`` rows as stored. They are not assumed to have unit norm: check
        ``.norm(dim=-1)``. A top-k SAE that rescales by decoder norm decodes with unit-norm
        rows, and those are returned.
        """
        import torch

        values = list(indices)
        if not values or not all(_is_int(i) and 0 <= i < self.d_sae for i in values):
            raise ValueError(f"feature indices must be integers in 0..{self.d_sae - 1}")
        rows = self.sae.W_dec.detach()[torch.as_tensor(values, device=self.sae.W_dec.device)]
        rows = rows.float()
        if getattr(self.sae.cfg, "rescale_acts_by_decoder_norm", False):
            rows = rows / rows.norm(dim=-1, keepdim=True)
        return rows

    def _output_scale(self, transformed: Any) -> Any:
        """What SAELens multiplies a decoded SAE output by to undo its input normalization.

        None when it does nothing. A transcoder's output is never rescaled.
        """
        mode = self.sae.cfg.normalize_activations
        if self.output_site is not None or mode == "none":
            return None
        if mode == "layer_norm":
            return (transformed - transformed.mean(-1, keepdim=True)).std(-1, keepdim=True)
        return transformed.norm(dim=-1, keepdim=True) / self.d_in**0.5

    def _skip(self, transformed: Any) -> Any:
        skip = getattr(self.sae, "W_skip", None)
        return 0 if skip is None else transformed @ skip.T

    # Reading

    def _batches(self, inputs: Any) -> list[dict[str, Any]]:
        import torch

        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(inputs, (list, tuple)) and inputs and all(isinstance(t, str) for t in inputs):
            return [self._tokenized(text) for text in inputs]
        if isinstance(inputs, dict):
            batch = {k: torch.as_tensor(v).to(self.lab.device) for k, v in inputs.items()}
            ids = batch.get("input_ids")
            if ids is None or ids.ndim != 2 or ids.shape[0] == 0:
                raise ValueError("input_ids must have shape [batch, tokens]")
            return [batch]
        raise ValueError(
            "inputs must be a string, a nonempty list of strings, or a dict of model inputs"
        )

    def _tokenized(self, text: str) -> dict[str, Any]:
        import torch

        encoded = dict(self.lab.tokenizer(text, return_tensors="pt"))
        ids = encoded["input_ids"]
        bos = getattr(self.lab.tokenizer, "bos_token_id", None)
        if self.prepend_bos and bos is not None and ids.shape[1] and int(ids[0, 0]) != bos:
            encoded["input_ids"] = torch.cat([torch.full_like(ids[:, :1], bos), ids], dim=1)
            if "attention_mask" in encoded:
                mask = encoded["attention_mask"]
                encoded["attention_mask"] = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
        return {k: v.to(self.lab.device) for k, v in encoded.items() if k != "token_type_ids"}

    def features(self, inputs: Any, positions: Any = None) -> dict[str, Any]:
        """Run the model once per sequence and read features at the chosen positions.

        ``inputs`` is a string, a list of strings, or a dict of model inputs with
        ``input_ids`` ``[batch, tokens]``. Text gets the tokenizer's BOS token first when
        the SAE was trained with one and the tokenizer does not add it. ``positions`` is as
        for ``Lab.record``: None is the last token. Padding named by ``attention_mask`` is
        left out. Every model forward counts as a pass per sequence.

        Returns ``features`` (``[rows, d_sae]``, float32 on the CPU), ``sequence`` and
        ``position`` (each row's sequence and token index), ``rows``, and how well the
        dictionary reproduces what it is meant to: ``mse``, ``fvu`` (the fraction of
        variance unexplained, with variance taken across the rows), and ``l0`` (mean number
        of active features per row). A transcoder is scored on the output it predicts. Check
        ``fvu`` before trusting features; a large value usually means the wrong site or
        transform. With one row, ``fvu`` is NaN. A BOS token's activations are often far
        larger than the rest; ``positions=slice(1, None)`` leaves position 0 out.
        """
        import torch

        batches = self._batches(inputs)
        # Refuse the whole call before any pass, not partway through it.
        wanted_passes = sum(batch["input_ids"].shape[0] for batch in batches)
        self.lab._budget_state()
        for budget in [self.lab.budget, *self.lab._evaluation_budgets]:
            budget.check(wanted_passes)
        model = self.lab.model
        was_training = model.training
        sources, targets, sequences, places = [], [], [], []
        try:
            model.eval()
            with ExitStack() as stack, torch.no_grad():
                read = stack.enter_context(
                    self.lab.record(self.module, positions=positions, at=self.at)
                )
                write = None
                if self.output_site is not None:
                    write = stack.enter_context(
                        self.lab.record(self.output_module, positions=positions, at=self.output_at)
                    )
                offset = 0
                for batch in batches:
                    model(**batch)
                    ids = batch["input_ids"]
                    chosen = _positions(ids, positions, 1)
                    mask = batch.get("attention_mask")
                    keep = (
                        torch.ones(ids.shape[0], len(chosen), dtype=torch.bool, device=ids.device)
                        if mask is None
                        else mask[:, chosen].bool()
                    ).cpu()
                    sources.append(read[self.module][-1][keep])
                    if write is not None:
                        targets.append(write[self.output_module][-1][keep])
                    rows, columns = keep.nonzero(as_tuple=True)
                    sequences.append(rows + offset)
                    places.append(torch.as_tensor(chosen)[columns])
                    offset += ids.shape[0]
        finally:
            model.train(was_training)
        source = torch.cat(sources)
        if self.output_site is None:
            wanted = self.transform(source)
        else:
            wanted = self.transform_output(torch.cat(targets))
        with torch.no_grad():
            pieces, errors = [], []
            for start in range(0, source.shape[0], 4096):
                chunk = self.transform(source[start : start + 4096])
                features = self.sae.encode(chunk)
                made = self.sae.decode(features) + self._skip(chunk)
                pieces.append(features.float().cpu())
                errors.append((wanted[start : start + 4096] - made).float())
            error = torch.cat(errors)
            active = torch.cat(pieces)
            spread = (wanted.float() - wanted.float().mean(0)).pow(2).sum()
        fvu = (
            float(error.pow(2).sum() / spread)
            if source.shape[0] > 1 and spread > 0
            else float("nan")
        )
        return {
            "features": active,
            "sequence": torch.cat(sequences),
            "position": torch.cat(places),
            "rows": int(active.shape[0]),
            "mse": float(error.pow(2).mean()),
            "fvu": fvu,
            "l0": float((active != 0).sum(-1).float().mean()),
        }

    # Editing

    def _checked(self, mapping: Any, what: str, *, callables: bool) -> dict[int, Any]:
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{what} must map feature indices to values")
        checked = {}
        for index, value in mapping.items():
            if not _is_int(index) or not 0 <= index < self.d_sae:
                raise ValueError(f"feature indices must be integers in 0..{self.d_sae - 1}")
            if not (_real(value) or (callables and callable(value))):
                raise ValueError(
                    f"feature {index}: values must be numbers"
                    + (" or callables" if callables else "")
                )
            checked[index] = value
        return checked

    def _writable(self) -> Site:
        target = self.output_site or self.site
        if not target.writable:
            raise ValueError(target.read_only)
        return target

    @contextmanager
    def patch(self, edits: Mapping, positions: Any = None):
        """Set features while the context is open, keeping the reconstruction error.

        ``edits`` maps a feature index to a new value or to a callable of the current value
        (a tensor ``[batch, positions]``). The site's activation changes by
        ``(f_new - f_old)[changed] @ W_dec[changed]`` and nothing else, so a patch that
        changes no feature leaves the activation unchanged to the bit. A transcoder
        changes its output the same way, from features read at its input. ``positions`` is
        as for ``Lab.patch``. The hook is removed on exit, even after an error.

        Raises ValueError, before any hook is added, for a bad index or value, and for a
        site with no exact place for an edit (a normalized activation, or the residual
        between attention and the MLP).
        """
        import torch

        changes = self._checked(edits, "edits", callables=True)
        target = self._writable()
        if not changes:
            yield
            return
        indices = list(changes)
        rows = self.decoder_rows(indices)
        with ExitStack() as stack:
            seen = None
            if self.output_site is not None:
                seen = stack.enter_context(
                    self.lab.record(self.module, positions=positions, at=self.at)
                )

            def replace(selected: Any) -> Any:
                with torch.no_grad():
                    source = selected if seen is None else seen[self.module][-1]
                    old = self.encode(source.to(selected.device))
                    old_values = old[..., indices]
                    new_values = torch.stack(
                        [
                            torch.as_tensor(
                                changes[i](old[..., i]) if callable(changes[i]) else changes[i],
                                dtype=old.dtype,
                                device=old.device,
                            ).expand_as(old[..., i])
                            for i in indices
                        ],
                        dim=-1,
                    )
                    difference = new_values - old_values
                    moved = difference.ne(0).any(-1, keepdim=True).to(selected.device)
                    delta = difference @ rows.to(difference.dtype)
                    scale = self._output_scale(self.transform(source.to(selected.device)))
                    if scale is not None:
                        delta = delta * scale
                    delta = delta.to(selected.device)
                    updated = (selected.float() + delta).to(selected.dtype)
                    return torch.where(moved, updated, selected)

            stack.enter_context(
                self.lab.patch(target.module, replace, positions=positions, at=target.at)
            )
            yield

    @contextmanager
    def steer(self, features: Mapping, positions: Any = None):
        """Add ``scale * decoder row`` for each ``{index: scale}`` while the context is open.

        The decoder rows are as ``decoder_rows`` returns them: they have unit norm only
        for a top-k SAE that rescales by decoder norm, so divide a scale by the row's norm
        to add a fixed length. For an SAE trained with ``layer_norm`` or
        ``constant_norm_rescale`` the row is scaled by each token's own normalization, so
        the push is in the model's units. A transcoder adds to its output. ``positions`` is as for
        ``Lab.patch``. Raises ValueError, before any hook is added, for a bad index or
        scale and for a site with no exact place for an edit.
        """
        scales = self._checked(features, "features", callables=False)
        target = self._writable()
        if not scales:
            yield
            return
        import torch

        indices = list(scales)
        weights = torch.as_tensor([float(scales[i]) for i in indices], device=self.device)
        vector = (weights[:, None] * self.decoder_rows(indices).to(weights.device)).sum(0)

        def add(selected: Any) -> Any:
            push = vector.to(selected.device)
            with torch.no_grad():
                scale = self._output_scale(self.transform(selected))
            if scale is not None:
                push = push * scale.to(selected.device)
            return (selected.float() + push).to(selected.dtype)

        with self.lab.patch(target.module, add, positions=positions, at=target.at):
            yield

    # Labels

    def _label_file(self) -> Path:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", self.neuronpedia_id)
        return self.lab.artifacts / "saes" / f"{name}.json"

    def labels(self, indices: Any, timeout: float = 5.0) -> dict[int, str | None]:
        """Neuronpedia's explanation for each feature, or None where there is none.

        None means no label was found, the SAE has no Neuronpedia id, or the network did
        not answer within ``timeout`` seconds; nothing is raised. After the first network
        failure the rest are not requested; an HTTP error or unreadable reply for one
        feature leaves only that feature as None. Found labels are cached under the session's
        ``artifacts/saes`` directory, and a feature Neuronpedia has no label for is not
        asked about again by this object. A label is a machine-written lead, not a fact: read
        the feature's top activating text before relying on it.
        """
        values = list(indices)
        if not all(_is_int(i) and 0 <= i < self.d_sae for i in values):
            raise ValueError(f"feature indices must be integers in 0..{self.d_sae - 1}")
        found: dict[int, str | None] = dict.fromkeys(values)
        if self.neuronpedia_id is None:
            return found
        path = self._label_file()
        try:
            cache = json.loads(path.read_text())
        except (OSError, ValueError):
            cache = {}
        if not isinstance(cache, dict):
            cache = {}
        fresh = False
        for index in values:
            if isinstance(cache.get(str(index)), str):
                found[index] = cache[str(index)]
                continue
            if index in self._label_misses:
                continue
            try:
                label = _fetch_label(self.neuronpedia_id, index, timeout)
            except (urllib.error.HTTPError, ValueError):
                continue  # this feature has no usable answer; the next may
            except Exception:
                break  # the network did not answer; the rest would wait too
            if label is None:
                self._label_misses.add(index)
            else:
                found[index] = cache[str(index)] = label
                fresh = True
        if fresh:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(cache))
            except OSError:
                pass
        return found


class SAEMixin:
    """``Lab.load_sae``."""

    def load_sae(
        self,
        release: str | None = None,
        sae_id: str | None = None,
        *,
        path: str | Path | None = None,
        device: Any = None,
        allow_model_mismatch: bool = False,
        model_kwargs: Mapping | None = None,
        hook_name: str | None = None,
    ) -> LabSAE:
        """Load a SAELens sparse autoencoder or transcoder for this session's model.

        Name a SAELens ``release`` and ``sae_id`` (downloaded from the Hugging Face Hub),
        or give ``path``, a directory in SAELens's format. ``device`` defaults to the
        session's device. Needs the ``saes`` extra.

        The SAE's recorded ``model_from_pretrained_kwargs`` say what TransformerLens
        processing it was trained under (see ``hook_site``). Many releases record nothing;
        ``model_kwargs`` replaces what is recorded, for example
        ``{"center_writing_weights": True}``. Check ``fvu`` from ``features`` either way.

        ``hook_name`` replaces the input hook the SAE records, for a release whose recorded
        hook is not where it reads (Gemma Scope 2 transcoders record ``blocks.L.hook_mlp_in``
        but read the MLP's own input, ``blocks.L.mlp.hook_in``). The override is mapped and
        its width checked like any hook name. A transcoder's output site still comes from its
        recorded ``hook_name_out``. The LabSAE keeps both names: ``hook_name`` is the one used
        and ``recorded_hook_name`` the one in the config.

        Raises ValueError, before any forward pass, for a hook name the mapping does not
        cover, a site whose width is not ``d_in`` (or a transcoder's output width not
        ``d_out``), a model that does not match the session's (compared where the SAE's
        model name is known; otherwise the SAE loads and ``model_check`` says it was not
        checked, with a warning; ``allow_model_mismatch=True`` accepts a fine-tune or a
        copy under another name), an architecture without an affine decoder,
        ``normalize_activations`` other than none, layer_norm or constant_norm_rescale, and
        an adapter session, whose sites are defined by base weights. Raises
        ImportError naming the extra when SAELens is missing.
        """
        if (path is None) == (release is None or sae_id is None):
            raise ValueError("pass release and sae_id, or path, not both or neither")
        try:
            from sae_lens import SAE
        except ImportError as error:
            raise ImportError(
                f"load_sae needs SAELens, which the 'saes' extra installs: {_INSTALL}"
            ) from error
        if hasattr(self.model, "peft_config"):
            raise ValueError(
                "load_sae does not support an adapter (PEFT) session: sites are defined by "
                "the base model's modules and an SAE was trained without the adapter; "
                "export the merged model and start a session on it"
            )
        target = str(device if device is not None else self.device)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*model_from_pretrained_kwargs.*")
            if path is not None:
                sae = SAE.load_from_disk(str(path), device=target)
            else:
                sae = SAE.from_pretrained(release, sae_id, device=target)
        architecture = sae.cfg.architecture()
        if architecture not in _ARCHITECTURES:
            raise ValueError(
                f"SAE architecture {architecture!r} is not supported; supported: "
                + ", ".join(sorted(_ARCHITECTURES))
            )
        if sae.cfg.normalize_activations not in ("none", "layer_norm", "constant_norm_rescale"):
            raise ValueError(
                f"normalize_activations={sae.cfg.normalize_activations!r} is not supported "
                "(an expected_average_only_in factor that could not be folded into the weights, "
                "or covariance whitening)"
            )
        metadata = sae.cfg.metadata
        if hook_name is not None and not isinstance(hook_name, str):
            raise ValueError("hook_name must be a string")
        used = hook_name or metadata.hook_name
        if not used:
            raise ValueError("the SAE records no hook name, so its site is unknown; pass hook_name")
        if metadata.hook_head_index is not None:
            raise ValueError("a per-head SAE (hook_head_index) is not supported")
        kwargs = dict(metadata.model_from_pretrained_kwargs or {})
        if model_kwargs is not None:
            kwargs = dict(model_kwargs)
        site = hook_site(self, used, kwargs)
        output_site = None
        if "transcoder" in architecture:
            if not metadata.hook_name_out:
                raise ValueError("the transcoder records no output hook (hook_name_out)")
            output_site = hook_site(self, metadata.hook_name_out, kwargs)
        if site.width != sae.cfg.d_in:
            raise ValueError(
                f"{used} has width {site.width} in this model but the SAE has d_in {sae.cfg.d_in}"
            )
        if output_site is not None and output_site.width != sae.cfg.d_out:
            raise ValueError(
                f"{metadata.hook_name_out} has width {output_site.width} in this model but the "
                f"transcoder has d_out {sae.cfg.d_out}"
            )
        if allow_model_mismatch:
            check = "not checked: allow_model_mismatch=True"
        else:
            check = _matches_session(self, metadata.model_name)
            if check != "matched":
                warnings.warn(
                    f"the SAE's model could not be checked ({check}); only the shape was checked",
                    UserWarning,
                    stacklevel=2,
                )
        if sae.cfg.reshape_activations == "hook_z":
            # The lab reads hook_z flat, as the output projection's input.
            sae.turn_off_forward_pass_hook_z_reshaping()
        sae.eval().requires_grad_(False)
        return LabSAE(
            self,
            sae,
            site,
            output_site,
            release=release,
            sae_id=sae_id,
            model_check=check,
            hook_name=used,
        )
