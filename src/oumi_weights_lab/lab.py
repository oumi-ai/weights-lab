"""The model and files available to snippets in a worker session."""

import sys
from pathlib import Path
from typing import Any

from .attachments import AttachmentMixin
from .budget import PassBudget
from .circuits import CircuitsMixin
from .editing import EditingMixin
from .hf_data import hf_rows
from .inspection import InspectionMixin
from .persistence import PersistenceMixin
from .scoring import ScoringMixin
from .session import DTYPES, QUANTIZE


class Lab(
    EditingMixin, ScoringMixin, InspectionMixin, CircuitsMixin, AttachmentMixin, PersistenceMixin
):
    """A loaded model, tokenizer, and the session's artifact directory.

    Forward passes count sequences run through the base language model, including
    generation steps. A batched call counts once per sequence; a call that fails
    also counts. Snippets retain control of gradient tracking and should use
    ``torch.no_grad()`` for inference when they do not need gradients.
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
        # load_lab fills these in for quantized models and for --allow-hub-kernels.
        self.quantization: dict[str, Any] | None = None
        self.allow_hub_kernels = False
        if hasattr(model, "peft_config"):
            from peft.tuners.tuners_utils import BaseTuner

            # Older PEFT calls model.forward directly and skips all Module hooks.
            # Normalize only the plain delegation, preserving custom tuner logic.
            for module in model.modules():
                if isinstance(module, BaseTuner) and (
                    getattr(module.forward, "__func__", None) is BaseTuner.forward
                ):
                    module.forward = module.model.__call__
        # PEFT generation calls the underlying language model directly.
        counted_model = model.get_base_model() if hasattr(model, "peft_config") else model
        self._counter_hook = counted_model.register_forward_pre_hook(
            self._count_forward, with_kwargs=True
        )
        self._count_beams_once(counted_model)

    @property
    def device(self) -> Any:
        """The input embeddings' device, where tokenizer outputs belong."""
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, NotImplementedError):
            return self.model.device

    @property
    def device_map(self) -> dict[str, Any] | None:
        """A copy of the module-to-device map the model was dispatched with, if any."""
        model = self.model.get_base_model() if hasattr(self.model, "peft_config") else self.model
        device_map = getattr(model, "hf_device_map", None)
        return dict(device_map) if device_map else None

    @property
    def device_memory(self) -> dict[str, float]:
        """GiB of parameters on each device, as ``peek`` reports them."""
        totals: dict[str, int] = {}
        for value in self.model.parameters():
            key = str(value.device)
            totals[key] = totals.get(key, 0) + value.numel() * value.element_size()
        return {key: round(size / 2**30, 2) for key, size in sorted(totals.items())}

    @property
    def forward_passes(self) -> int:
        """Sequences run through the base model since loading, including generation steps."""
        return self._forward_passes

    # Evaluation rows from a Hugging Face dataset split; needs the datasets extra.
    hf_rows = staticmethod(hf_rows)


def block_hub_kernels(allow: bool) -> None:
    """Keep the ``kernels`` package from fetching kernel code from the Hub, unless ``allow``.

    bitsandbytes and Transformers download and run Hub kernels only through that package,
    inside ``try`` blocks that fall back to built-in code. Setting its ``sys.modules`` entry
    to None makes ``import kernels`` raise ImportError and ``find_spec`` return None. Raises
    RuntimeError when the real package is already imported, since that is too late.
    """
    if allow:
        # An earlier guarded load in this process may have left a block behind.
        if "kernels" in sys.modules and sys.modules["kernels"] is None:
            del sys.modules["kernels"]
        return
    loaded = sys.modules.get("kernels")
    if loaded is not None:
        raise RuntimeError(
            "The kernels package is already imported, so it could run kernel code from the "
            "Hugging Face Hub. Start the session without importing it, or pass "
            "--allow-hub-kernels to accept Hub kernels."
        )
    sys.modules["kernels"] = None


def load_lab(config: dict[str, Any], session_path: Path) -> Lab:
    """Load one eager Hugging Face causal language model, optionally with PEFT.

    Configuration keys are ``model``, ``adapter``, ``device``, ``dtype``, and
    ``seed``. ``device_map`` (balanced or sequential) spreads the model over every
    visible CUDA GPU, within ``max_memory`` per GPU when given, and refuses before
    loading any weights when the model would need CPU or disk offload. ``quantize``
    (nf4, fp4 or int8) loads the model with bitsandbytes on one cpu, mps or cuda device;
    it needs the ``quantized`` extra and refuses ``device_map`` and a checkpoint that
    another method already quantized. A checkpoint that bitsandbytes already quantized
    loads as it is. Quantized weights are read-only. ``allow_hub_kernels`` lets kernel
    code from the Hub run; without it none can. The worker creates the session and
    artifact directories first. Custom model code from the Hub is never enabled.
    """
    block_hub_kernels(bool(config.get("allow_hub_kernels", False)))
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
    strategy = config.get("device_map")
    if strategy and device.type != "cuda":
        raise ValueError("--device-map needs CUDA GPUs")
    quantize = config.get("quantize")
    if quantize is not None and quantize not in QUANTIZE:
        raise ValueError(f"Unsupported quantize {quantize!r}; choose {', '.join(QUANTIZE)}")
    if quantize and strategy:
        raise ValueError("--quantize cannot be combined with --device-map yet")
    dtype_name = config.get("dtype", "auto")
    if dtype_name not in DTYPES:
        raise ValueError(f"Unsupported dtype {dtype_name!r}; choose {', '.join(DTYPES)}")
    dtypes = {name: "auto" if name == "auto" else getattr(torch, name) for name in DTYPES}

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
    declared = _declared_quantization(model_config)
    prequantized = (
        declared if declared is not None and declared["quant_method"] == "bitsandbytes" else None
    )
    if declared is not None and prequantized is None and quantize:
        raise ValueError(
            f"The checkpoint is already quantized with {declared['quant_method']}; "
            f"--quantize {quantize} cannot be applied to it"
        )
    if prequantized is not None:
        prequantized = _bitsandbytes_settings(prequantized)
        if quantize and quantize != prequantized["method"]:
            raise ValueError(
                f"The checkpoint is already quantized with {prequantized['method']}; "
                f"--quantize {quantize} disagrees. Omit --quantize to load it as it is"
            )
        if strategy:
            raise ValueError("A prequantized checkpoint cannot be combined with --device-map yet")
    quantized = bool(quantize) or prequantized is not None
    if quantized:
        version = _require_bitsandbytes()
        _check_quantized_device(torch, device)
    placement = {}
    if quantized:
        placement["device_map"] = {"": str(device)}
        if prequantized is None:
            placement["quantization_config"] = _quantization_config(
                quantize, _compute_dtype(torch, dtypes[dtype_name], model_config)
            )
    if strategy:
        placement["device_map"] = _gpu_device_map(
            model_class, model_config, dtypes[dtype_name], strategy, config.get("max_memory")
        )
    model = model_class.from_pretrained(
        model_id,
        config=model_config,
        dtype=dtypes[dtype_name],
        attn_implementation="eager",
        **placement,
        **reference,
    )
    if config.get("adapter"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, config["adapter"], revision=config.get("adapter_revision"), is_trainable=False
        )
    if strategy:
        stray = sorted({str(value.device) for value in model.parameters()} - _cuda_devices(model))
        if stray:
            raise RuntimeError(
                f"--device-map left parameters on {', '.join(stray)}; "
                "CPU and disk offload are not supported"
            )
    elif not quantized:
        model.to(device)
    model.eval()
    if quantized:
        _stop_cpu_repack(model)
    lab = Lab(
        model, tokenizer, session_path, model_id=model_id, pass_limit=config.get("pass_limit")
    )
    lab.allow_hub_kernels = bool(config.get("allow_hub_kernels", False))
    if quantized:
        lab.quantization = _quantization_summary(
            prequantized,
            quantize,
            _compute_dtype(torch, dtypes[dtype_name], model_config),
            version,
        )
    lab.model_commit = getattr(model_config, "_commit_hash", None) or config.get("revision")
    if config.get("resume"):
        lab.load_edits(config["resume"])
    return lab


def _cuda_devices(model: Any) -> set[str]:
    return {str(value.device) for value in model.parameters() if value.device.type == "cuda"}


def _gpu_memory(max_memory: dict[str, str] | None) -> dict[int, int]:
    """Bytes to use on each GPU: ``max_memory``, or 90% of each visible GPU's free memory.

    The rest of each GPU is left for activations and the generation cache.
    """
    import torch
    from accelerate.utils import convert_file_size_to_int

    count = torch.cuda.device_count()
    if max_memory is None:
        return {index: torch.cuda.mem_get_info(index)[0] * 9 // 10 for index in range(count)}
    limits = {}
    for index, size in sorted(max_memory.items(), key=lambda item: int(item[0])):
        if int(index) >= count:
            raise ValueError(f"GPU {index} is not available; {count} GPUs are visible")
        limit, free = convert_file_size_to_int(size), torch.cuda.mem_get_info(int(index))[0]
        if limit > free:
            raise ValueError(
                f"--max-memory asks for {limit / 2**30:.1f} GiB on GPU {index}, but GPU "
                f"{index} has {free / 2**30:.1f} GiB free"
            )
        limits[int(index)] = limit
    return limits


def _checkpoint_dtype(torch: Any, model_config: Any) -> Any:
    """The dtype ``dtype="auto"`` loads, from the config or its text config; None if unstated."""
    for source in (model_config, getattr(model_config, "text_config", None)):
        for name in ("dtype", "torch_dtype"):
            value = getattr(source, name, None)
            if isinstance(value, str):
                value = getattr(torch, value, None)
            if isinstance(value, torch.dtype):
                return value
    return None


def _compute_dtype(torch: Any, dtype: Any, model_config: Any) -> Any:
    """The dtype quantized layers compute in: ``--dtype``, or the checkpoint's float dtype.

    With ``auto``, a checkpoint that states no dtype, or a non-float one, gives float32.
    """
    if dtype != "auto":
        return dtype
    found = _checkpoint_dtype(torch, model_config)
    return found if found is not None and found.is_floating_point else torch.float32


def _declared_quantization(model_config: Any) -> dict[str, Any] | None:
    """The checkpoint config's ``quantization_config`` as a dict with a string ``quant_method``."""
    declared = getattr(model_config, "quantization_config", None)
    if declared is not None and not isinstance(declared, dict):
        declared = declared.to_dict() if hasattr(declared, "to_dict") else None
    if not declared or not declared.get("quant_method"):
        return None
    method = declared["quant_method"]
    return {**declared, "quant_method": str(getattr(method, "value", method))}


def _bitsandbytes_settings(declared: dict[str, Any]) -> dict[str, Any]:
    """Read the method, scale format and compute dtype of a bitsandbytes checkpoint config."""
    from transformers import BitsAndBytesConfig

    parsed = BitsAndBytesConfig.from_dict(dict(declared))
    if parsed.load_in_8bit:
        return {"method": "int8", "double_quant": False, "threshold": parsed.llm_int8_threshold}
    if parsed.load_in_4bit:
        return {
            "method": parsed.bnb_4bit_quant_type,
            "double_quant": bool(parsed.bnb_4bit_use_double_quant),
            "compute_dtype": parsed.bnb_4bit_compute_dtype,
            "blocksize": 64,
        }
    raise ValueError("The checkpoint's bitsandbytes settings are neither 4-bit nor 8-bit")


def _require_bitsandbytes() -> str:
    """Return the bitsandbytes version, or refuse before any weights load."""
    try:
        import bitsandbytes
    except Exception as error:
        raise ImportError(
            "Quantized loading needs bitsandbytes; install oumi-weights-lab[quantized], for "
            "example `uv tool install 'oumi-weights-lab[quantized] @ "
            "git+https://github.com/oumi-ai/weights-lab'`. bitsandbytes has no wheel for "
            "Intel Macs or macOS before 14."
        ) from error
    return bitsandbytes.__version__


def _check_quantized_device(torch: Any, device: Any) -> None:
    """Refuse a quantized model on Apple MPS with a torch too old for bitsandbytes there.

    On torch 2.4, bitsandbytes' MPS kernels need an operator that torch does not implement
    for MPS, so loading fails partway through with an unrelated error.
    """
    if str(device).split(":")[0] != "mps":
        return
    release = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if release < (2, 5):
        raise RuntimeError(
            f"Quantized models on Apple MPS need torch 2.5 or later; this is torch "
            f"{torch.__version__}. Upgrade torch, or load with --device cpu."
        )


def _quantization_config(method: str, compute_dtype: Any) -> Any:
    from transformers import BitsAndBytesConfig

    if method == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=method,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _quantization_summary(
    prequantized: dict[str, Any] | None, quantize: str | None, compute: Any, version: str
) -> dict[str, Any]:
    """What ``lab.quantization`` reports; ``prequantized`` holds a checkpoint's own settings."""
    from transformers import BitsAndBytesConfig

    settings = prequantized or {
        "method": quantize,
        "double_quant": False,
        **(
            {"threshold": BitsAndBytesConfig().llm_int8_threshold}
            if quantize == "int8"
            else {"blocksize": 64}
        ),
    }
    dtype = settings.get("compute_dtype") or compute
    return {
        "method": settings["method"],
        "source": "checkpoint" if prequantized else "load",
        "compute_dtype": str(dtype).removeprefix("torch."),
        "double_quant": settings["double_quant"],
        **{key: settings[key] for key in ("blocksize", "threshold") if key in settings},
        "bitsandbytes": version,
    }


def _stop_cpu_repack(model: Any) -> None:
    """Keep bitsandbytes from repacking 4-bit weights in place on the first CPU forward.

    On CPUs with AVX512-BF16 it would, and the repack loses precision in the scales.
    """
    for module in model.modules():
        if hasattr(module, "support_avx512bf16_for_cpu"):
            module.support_avx512bf16_for_cpu = False


def _gpu_device_map(
    model_class: Any, model_config: Any, dtype: Any, strategy: str, max_memory: Any
) -> dict[str, int]:
    """Plan a GPU-only device map on an empty copy of the model, before any weights load.

    The copy is built in the dtype that will load, so it is sized as the real weights.
    """
    import copy

    import torch

    try:
        from accelerate import init_empty_weights
    except ImportError as error:
        raise ImportError(
            "--device-map needs accelerate; install oumi-weights-lab[multi-gpu]"
        ) from error
    if dtype == "auto":
        # An unstated checkpoint dtype is sized as float32, the most it could need.
        dtype = _checkpoint_dtype(torch, model_config) or torch.float32
    with init_empty_weights():
        # from_config rewrites the dtypes of the config it is given.
        skeleton = model_class.from_config(
            copy.deepcopy(model_config), dtype=dtype, attn_implementation="eager"
        )
    # Building on the meta device unties tied weights; tie them again so the plan
    # keeps an output head on the same GPU as the embedding it shares.
    skeleton.tie_weights()
    return _plan_device_map(skeleton, strategy, _gpu_memory(max_memory), None)


def _plan_device_map(
    skeleton: Any, strategy: str, gpu_memory: dict[int, int], dtype: Any
) -> dict[str, Any]:
    """Place each module of an empty model on a GPU, as Transformers would for ``strategy``.

    Refuses, with the memory needed and available, when anything would be offloaded
    to CPU or disk. ``dtype`` None sizes weights as the skeleton holds them.
    """
    from accelerate.utils import compute_module_sizes, get_balanced_memory, infer_auto_device_map
    from accelerate.utils.modeling import get_max_layer_size

    no_split = getattr(skeleton, "_get_no_split_modules", None)
    options = {
        "no_split_module_classes": no_split(strategy)
        if callable(no_split)
        else list(getattr(skeleton, "_no_split_modules", None) or []),
        "dtype": dtype,
    }
    limits = dict(gpu_memory)
    if strategy == "balanced":
        limits = get_balanced_memory(skeleton, max_memory=limits, low_zero=False, **options)
    placement = infer_auto_device_map(skeleton, max_memory=limits, **options)
    offloaded = [name for name, device in placement.items() if device in ("cpu", "disk")]
    if offloaded:
        sizes = compute_module_sizes(skeleton, dtype=dtype)
        needed, offered = sizes[""], sum(gpu_memory.values())
        gpus = f"{len(gpu_memory)} GPU{'' if len(gpu_memory) == 1 else 's'}"
        weights = dtype or next(skeleton.parameters()).dtype
        sized = f" in {str(weights).removeprefix('torch.')}"
        reserve = ""
        if needed <= offered:
            # accelerate, like Transformers, keeps the first GPU's room for the largest layer.
            layer, _ = get_max_layer_size(
                list(skeleton.named_children()), sizes, options["no_split_module_classes"]
            )
            reserve = (
                f" The weights would fit in total, but GPU {min(gpu_memory)} keeps "
                f"{layer / 2**30:.1f} GiB free for the largest layer, as Transformers does."
            )
        raise ValueError(
            f"The model needs about {needed / 2**30:.1f} GiB for its weights{sized}, but its "
            f"layers do not fit: the {gpus} offer {offered / 2**30:.1f} GiB, "
            f"and {', '.join(offloaded[:3])} would be offloaded.{reserve} CPU and disk "
            "offload are not supported; use more or larger GPUs, a smaller --dtype, or a "
            "larger --max-memory."
        )
    return placement


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
