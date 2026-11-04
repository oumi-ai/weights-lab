"""A merged LoRA checkpoint's tensors, read from the live model one layer at a time.

PEFT's safe merge writes each merged weight to a new tensor and points the base
layer at it. Holding the original tensors and pointing the layer back at them
restores the model exactly, so no layer stays merged and the model is never copied.
"""

from __future__ import annotations

from typing import Any


def layerwise_blocker(model: Any) -> str | None:
    """Why ``model`` (a PeftModel) cannot be merged layer by layer, or None when it can."""
    from peft.tuners.lora import LoraLayer
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils import ModulesToSaveWrapper

    if any(config.peft_type != "LORA" for config in model.peft_config.values()):
        return "only LoRA adapters merge layer by layer"
    base = model.get_base_model()
    tied = getattr(getattr(base, "config", None), "tie_word_embeddings", False)
    for embedding in (base.get_input_embeddings(), base.get_output_embeddings()):
        # Merging into a tied embedding unties it, which PEFT's own unload records.
        if tied and isinstance(embedding, (BaseTunerLayer, ModulesToSaveWrapper)):
            return "an adapter changes tied embeddings"
    for name, module in base.named_modules():
        if isinstance(module, BaseTunerLayer):
            if not isinstance(module, LoraLayer) or hasattr(
                module, "unload_and_optionally_merge_module"
            ):
                return f"{name} is a {type(module).__name__} layer"
            variants = [
                *getattr(module, "use_dora", {}).values(),
                *getattr(module, "lora_variant", {}).values(),
            ]
            if any(variants):
                return f"{name} uses a LoRA variant such as DoRA"
        elif isinstance(module, ModulesToSaveWrapper):
            if any(isinstance(inner, BaseTunerLayer) for inner in _saved(module).modules()):
                return f"{name} saves a module that has its own adapter"
        elif hasattr(module, "unload_and_optionally_merge_module"):
            return f"{name} is a {type(module).__name__}"
    return None


def merged_state_dict(model: Any) -> dict[str, Any]:
    """The state dict ``merge_and_unload(safe_merge=True)`` would give, without changing ``model``.

    Merged weights are copied to CPU; every other tensor is the live one, so the
    caller must not modify the model while it holds the result.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils import ModulesToSaveWrapper

    state: dict[str, Any] = {}

    def visit(module: Any, prefix: str) -> None:
        if isinstance(module, BaseTunerLayer):
            state.update(_merged_layer(module, prefix))
            return
        if isinstance(module, ModulesToSaveWrapper):
            visit(_saved(module), prefix)
            return
        for name, value in module._parameters.items():
            if value is not None:
                state[prefix + name] = value.detach()
        for name, value in module._buffers.items():
            if value is not None and name not in module._non_persistent_buffers_set:
                state[prefix + name] = value.detach()
        for name, child in module._modules.items():
            if child is not None:
                visit(child, f"{prefix}{name}.")

    visit(model.get_base_model(), "")
    return state


def _saved(wrapper: Any) -> Any:
    """The module a ModulesToSaveWrapper unloads to, as PEFT's own unload picks it."""
    if hasattr(wrapper, "unload_and_optionally_merge_module"):
        return wrapper.unload_and_optionally_merge_module(
            merge=False, safe_merge=True, adapter_names=None
        )
    return wrapper.modules_to_save[wrapper.active_adapter]


def _merged_layer(layer: Any, prefix: str) -> dict[str, Any]:
    """Merge one LoRA layer, copy its base layer's tensors to CPU, and restore it exactly."""
    base = layer.get_base_layer()
    originals = {name: value.data for name, value in base.named_parameters(recurse=False)}
    merged = list(layer.merged_adapters)
    try:
        layer.merge(safe_merge=True)
        return {
            prefix + name: value.detach().to("cpu", copy=True)
            for name, value in base.state_dict(keep_vars=True).items()
        }
    finally:
        for name, value in base.named_parameters(recurse=False):
            value.data = originals[name]
        layer.merged_adapters[:] = merged
