"""The model and files available to snippets in a worker session."""

from pathlib import Path
from typing import Any

from .attachments import AttachmentMixin
from .budget import PassBudget
from .editing import EditingMixin
from .inspection import InspectionMixin
from .scoring import ScoringMixin


class Lab(EditingMixin, ScoringMixin, InspectionMixin, AttachmentMixin):
    """A loaded model, tokenizer, and the session's artifact directory.

    Forward passes count calls to the base language model, including generation
    steps. A batched call counts once; a call that fails also counts. Snippets
    retain control of gradient tracking and should use ``torch.no_grad()`` for
    inference when they do not need gradients.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        session_path: Path,
        model_id: str = "",
        pass_limit: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.session_path = Path(session_path)
        self.artifacts = self.session_path / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self._forward_passes = 0
        self._budget = PassBudget(pass_limit)
        # PEFT generation calls the underlying language model directly.
        counted_model = model.get_base_model() if hasattr(model, "peft_config") else model
        self._counter_hook = counted_model.register_forward_pre_hook(self._count_forward)

    @property
    def device(self) -> Any:
        """The model's current device, suitable for moving tokenizer outputs."""
        return self.model.device

    @property
    def forward_passes(self) -> int:
        """Number of base-model forward calls since loading."""
        return self._forward_passes

    @property
    def passes_used(self) -> int:
        return self._forward_passes

    @property
    def revision(self) -> int:
        return (
            self.editbook.revision
            + getattr(self, "_attachment_revision", 0)
            + getattr(self, "_patch_revision", 0)
        )

    def _count_forward(self, module: Any, inputs: Any) -> None:
        self._before_forward()
        self._forward_passes += 1


def load_lab(config: dict[str, Any], session_path: Path) -> Lab:
    """Load one eager Hugging Face causal language model, optionally with PEFT.

    Configuration keys are ``model``, ``adapter``, ``device``, ``dtype``, and
    ``seed``. The worker creates the session and artifact directories first.
    Custom model code from the Hub is never enabled.
    """
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        set_seed,
    )

    model_id = config["model"]
    device = _resolve_device(torch, config.get("device", "auto"))
    dtype_name = config.get("dtype", "auto")
    dtypes = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtypes:
        raise ValueError(f"Unsupported dtype {dtype_name!r}; choose {', '.join(dtypes)}")

    set_seed(config.get("seed", 0))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    reference = {"revision": config.get("revision"), "trust_remote_code": False}
    model_config = AutoConfig.from_pretrained(model_id, **reference)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **reference)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Preserve a checkpoint's declared multimodal architecture when it has one.
    # Nested text_config alone is also used by causal-only model families.
    architectures = model_config.architectures or []

    def declares(factory):
        if not architectures or type(model_config) not in factory._model_mapping:
            return False
        classes = factory._model_mapping[type(model_config)]
        classes = classes if isinstance(classes, tuple) else (classes,)
        return any(cls.__name__ in architectures for cls in classes)

    causal_supported = type(model_config) in AutoModelForCausalLM._model_mapping
    model_class = AutoModelForCausalLM
    if not causal_supported or (
        declares(AutoModelForImageTextToText) and not declares(AutoModelForCausalLM)
    ):
        model_class = AutoModelForImageTextToText
    model = model_class.from_pretrained(
        model_id,
        config=model_config,
        dtype=dtypes[dtype_name],
        attn_implementation="eager",
        **reference,
    )
    if config.get("adapter"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, config["adapter"], revision=config.get("adapter_revision"), is_trainable=False
        )
    model.to(device)
    model.eval()
    lab = Lab(
        model, tokenizer, session_path, model_id=model_id, pass_limit=config.get("pass_limit")
    )
    lab.model_commit = getattr(model_config, "_commit_hash", None) or config.get("revision")
    return lab


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda:0"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid device {requested!r}") from exc
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Choose auto, cpu, mps, cuda, or cuda:N for device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device
