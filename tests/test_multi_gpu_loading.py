"""Loading one model across several GPUs: flags, placement, and refusing CPU or disk offload.

CI has no GPUs, so the loader runs against patched CUDA queries; accelerate plans a
map of GPU indices without touching a GPU. The live multi-GPU validator loads for real.
"""

from unittest.mock import Mock

import pytest

from oumi_weights_lab import cli, session
from oumi_weights_lab.launchers import SkyPilotLauncher

GIB = 2**30


def _llama_config():
    transformers = pytest.importorskip("transformers")
    return transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        eos_token_id=1,
        bos_token_id=1,
    )


def _skeleton():
    pytest.importorskip("accelerate")
    transformers = pytest.importorskip("transformers")
    from accelerate import init_empty_weights

    with init_empty_weights():
        return transformers.LlamaForCausalLM(_llama_config())


def _size(skeleton):
    return sum(value.numel() * 4 for value in skeleton.parameters())


def test_max_memory_reads_gpu_size_pairs():
    assert session.parse_max_memory("0=70GiB, 1=69.5GB") == {"0": "70GiB", "1": "69.5GB"}
    for text in ["0:70GiB", "x=1GiB", "0=70", "0=1GiB,0=2GiB", "", "-1=1GiB"]:
        with pytest.raises(ValueError, match="GPU=SIZE"):
            session.parse_max_memory(text)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"device_map": "auto"}, "balanced or sequential"),
        ({"device_map": "balanced", "device": "cpu"}, "--device auto or cuda"),
        ({"device_map": "balanced", "device": "cuda:1"}, "--device auto or cuda"),
        ({"max_memory": {"0": "1GiB"}}, "--max-memory needs --device-map"),
        ({"device_map": "balanced", "max_memory": {"zero": "1GiB"}}, "GPU=SIZE"),
    ],
)
def test_placement_settings_are_checked_before_a_session_exists(config, message, tmp_path):
    with pytest.raises(ValueError, match=message):
        session.create_session(tmp_path / "session", {"model": "m", **config})
    assert not (tmp_path / "session").exists()
    runner = Mock(side_effect=AssertionError("invalid config reached an external command"))
    with pytest.raises(ValueError, match=message):
        SkyPilotLauncher("weights-test", runner=runner).start(
            "m", "/remote/session", source=tmp_path, **config
        )
    runner.assert_not_called()


def test_start_parses_device_map_flags():
    start = ["start", "--session", "s", "--model", "m"]
    args = cli._parser().parse_args(start)
    assert args.device_map is None and args.max_memory is None
    args = cli._parser().parse_args(
        [*start, "--device-map", "sequential", "--max-memory", "0=70GiB,1=70GiB"]
    )
    assert args.device_map == "sequential"
    assert args.max_memory == {"0": "70GiB", "1": "70GiB"}
    for bad in (["--device-map", "auto"], ["--max-memory", "0:70GiB"]):
        with pytest.raises(SystemExit):
            cli._parser().parse_args([*start, *bad])


def test_balanced_placement_spreads_layers_across_gpus():
    from oumi_weights_lab.lab import _plan_device_map

    skeleton = _skeleton()
    placement = _plan_device_map(skeleton, "balanced", {0: 10 * GIB, 1: 10 * GIB}, None)
    assert set(placement.values()) == {0, 1}
    assert _plan_device_map(skeleton, "sequential", {0: 10 * GIB, 1: 10 * GIB}, None) == {"": 0}


def test_placement_that_would_offload_is_refused_with_the_memory_numbers():
    from oumi_weights_lab.lab import _plan_device_map

    skeleton = _skeleton()
    half = _size(skeleton) // 2
    with pytest.raises(ValueError, match=r"needs about [0-9.]+ GiB.*2 GPUs offer [0-9.]+ GiB"):
        _plan_device_map(skeleton, "sequential", {0: half, 1: half // 2}, None)


def test_gpu_memory_defaults_to_most_of_each_gpus_free_memory(monkeypatch):
    torch = pytest.importorskip("torch")
    from oumi_weights_lab.lab import _gpu_memory

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: ((index + 1) * 10 * GIB, 0))
    assert _gpu_memory(None) == {0: 9 * GIB, 1: 18 * GIB}
    assert _gpu_memory({"1": "1GiB", "0": "512MiB"}) == {0: GIB // 2, 1: GIB}
    with pytest.raises(ValueError, match="GPU 2 is not available"):
        _gpu_memory({"2": "1GiB"})


@pytest.fixture
def fake_gpus(tiny_lab, monkeypatch):
    """Two 'GPUs' for the loader, a tiny Llama checkpoint config, and a recording model loader."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (80 * GIB, 80 * GIB))
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", Mock(return_value=_llama_config())
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", Mock(return_value=tiny_lab.tokenizer)
    )
    model = Mock(wraps=tiny_lab.model)
    loader = Mock(return_value=model)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", loader)
    return loader, model


def test_loader_passes_its_exact_gpu_map_and_checks_where_weights_landed(fake_gpus, tmp_path):
    from oumi_weights_lab.lab import load_lab

    loader, model = fake_gpus
    size = _size(_skeleton())
    # GPU 0 holds about half the model; sequential fills it, then moves on to GPU 1.
    config = {
        "model": "test/llama",
        "device": "cuda",
        "device_map": "sequential",
        "max_memory": {"0": f"{size * 0.6 / GIB:.6f}GiB", "1": f"{size * 2 / GIB:.6f}GiB"},
    }
    # The stand-in model's weights are on CPU, as an offloading load would leave them.
    with pytest.raises(RuntimeError, match="parameters on cpu"):
        load_lab(config, tmp_path / "session")
    placement = loader.call_args.kwargs["device_map"]
    assert set(placement.values()) == {0, 1}
    model.to.assert_not_called()


def test_a_model_too_big_for_the_gpus_is_refused_before_loading(fake_gpus, tmp_path):
    from oumi_weights_lab.lab import load_lab

    loader, _ = fake_gpus
    config = {"model": "test/llama", "device": "auto", "device_map": "balanced"}
    config["max_memory"] = {"0": "1KiB", "1": "1KiB"}
    with pytest.raises(ValueError, match="CPU and disk offload are not supported"):
        load_lab(config, tmp_path / "session")
    loader.assert_not_called()


def test_device_map_needs_cuda(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from oumi_weights_lab.lab import load_lab

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="needs CUDA GPUs"):
        load_lab({"model": "test/llama", "device_map": "balanced"}, tmp_path / "session")


def test_device_memory_sums_parameters_per_device(tiny_lab):
    size = sum(value.numel() * value.element_size() for value in tiny_lab.model.parameters())
    assert tiny_lab.device_memory == {"cpu": round(size / GIB, 2)}
