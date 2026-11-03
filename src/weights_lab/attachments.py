"""Temporary side modules and prompt transformations for controlled experiments."""

from __future__ import annotations

import copy
from typing import Any


class Attachment:
    def __init__(self, lab, identifier, name, mode, remove):
        self.lab, self.id, self.name, self.mode = lab, identifier, name, mode
        self._remove = remove
        self.active = True

    def remove(self):
        if self.active:
            self._remove()
            self.active = False
            self.component = None
            self.lab._attachment_revision += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


class AttachmentMixin:
    def attachments(self) -> list[dict[str, Any]]:
        return [
            {"id": item.id, "name": item.name, "mode": item.mode}
            for item in getattr(self, "_attachments", [])
            if item.active
        ]

    def attach(self, component, *, mode="logits", name=None) -> Attachment:
        """Attach a logit delta callback or a prompt-to-prompt transformation.

        Logit callbacks receive (model, args, kwargs, output) and return a tensor
        broadcastable to output.logits. Prefix callbacks receive and return text.
        The returned handle can be removed explicitly or used as a context manager.
        These experimental components are not included in checkpoint exports.
        """
        if mode not in {"logits", "prefix"} or not callable(component):
            raise ValueError("choose a callable component and mode logits or prefix")
        if not hasattr(self, "_attachments"):
            self._attachments = []
            self._attachment_revision = 0
        name = name or type(component).__name__
        identifier = f"attachment-{len(self._attachments) + 1:06d}"
        if mode == "logits":

            def apply(module, args, kwargs, output):
                import torch

                logits = output.logits if hasattr(output, "logits") else output[0]
                delta = torch.as_tensor(
                    component(module, args, kwargs, output),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                if not torch.isfinite(delta).all():
                    raise ValueError("attachment produced a nonfinite logit delta")
                updated = logits + delta
                if updated.shape != logits.shape:
                    raise ValueError("attachment must preserve the logit shape")
                if hasattr(output, "logits"):
                    result = copy.copy(output)
                    result.logits = updated
                    return result
                return (updated, *output[1:])

            target = (
                self.model.get_base_model() if hasattr(self.model, "peft_config") else self.model
            )
            hook = target.register_forward_hook(apply, with_kwargs=True)
            remove = hook.remove
        else:

            def remove():
                pass

        handle = Attachment(self, identifier, name, mode, remove)
        handle.component = component
        self._attachments.append(handle)
        self._attachment_revision += 1
        return handle

    def _prepare_prompt(self, prompt: str) -> str:
        for item in getattr(self, "_attachments", []):
            if item.active and item.mode == "prefix":
                prompt = item.component(prompt)
                if not isinstance(prompt, str):
                    raise ValueError("prefix components must return text")
        return prompt

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """Generate with active prompt transformations and return only new text."""
        import torch

        inputs = self.tokenizer(self._prepare_prompt(prompt), return_tensors="pt").to(self.device)
        with torch.no_grad():
            tokens = self.model.generate(**inputs, **generation_kwargs)
        return self.tokenizer.decode(
            tokens[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
