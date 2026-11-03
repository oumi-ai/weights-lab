from unittest.mock import Mock

import pytest

from weights_lab.lab import load_lab
from weights_lab.launchers import SkyPilotLauncher


@pytest.mark.parametrize(
    "config_name,expected_factory",
    [
        ("Gemma4UnifiedAssistantConfig", "causal"),
        ("Qwen3_5Config", "causal"),
        ("Qwen3VLConfig", "image_text"),
    ],
)
def test_loader_uses_registered_factory_for_nested_text_config(
    config_name, expected_factory, tiny_lab, tmp_path, monkeypatch
):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    config_class = getattr(transformers, config_name, None)
    if config_class is None:
        pytest.skip(f"installed Transformers does not provide {config_name}")
    config = config_class()
    assert config.text_config is not None
    config._commit_hash = "resolved-commit"
    causal = transformers.AutoModelForCausalLM
    image_text = transformers.AutoModelForImageTextToText
    if expected_factory == "causal":
        assert config_class in causal._model_mapping
        if config_name == "Gemma4UnifiedAssistantConfig":
            assert config_class not in image_text._model_mapping
    else:
        assert config_class not in causal._model_mapping
        assert config_class in image_text._model_mapping

    config_loader = Mock(return_value=config)
    tokenizer_loader = Mock(return_value=tiny_lab.tokenizer)
    selected = Mock(return_value=tiny_lab.model)
    rejected = Mock(side_effect=AssertionError("wrong model factory selected"))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config_loader)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(
        causal, "from_pretrained", selected if expected_factory == "causal" else rejected
    )
    monkeypatch.setattr(
        image_text, "from_pretrained", selected if expected_factory == "image_text" else rejected
    )

    loaded = load_lab(
        {
            "model": "test/model",
            "revision": "requested-revision",
            "device": "cpu",
            "dtype": "float32",
            "pass_limit": 2,
        },
        tmp_path / "loaded-session",
    )
    reference = {"revision": "requested-revision", "trust_remote_code": False}
    config_loader.assert_called_once_with("test/model", **reference)
    tokenizer_loader.assert_called_once_with("test/model", **reference)
    selected.assert_called_once_with(
        "test/model", config=config, dtype=torch.float32, attn_implementation="eager", **reference
    )
    rejected.assert_not_called()
    assert loaded.model is tiny_lab.model
    assert not loaded.model.training
    assert loaded.model_commit == "resolved-commit"
    assert loaded.budget.limit == 2


@pytest.mark.parametrize(
    "model,config",
    [
        ("", {}),
        ("   ", {}),
        (None, {}),
        ("model", {"dtype": "typo"}),
        ("model", {"device": "gpu"}),
        ("model", {"device": "cuda:-1"}),
        ("model", {"pass_limit": -1}),
        ("model", {"snippet_timeout": float("nan")}),
        ("model", {"poll_interval": 0}),
    ],
)
def test_cloud_start_validates_static_config_before_external_work(model, config, tmp_path):
    runner = Mock(side_effect=AssertionError("invalid config reached an external command"))
    launcher = SkyPilotLauncher("weights-test", runner=runner)
    with pytest.raises(ValueError):
        launcher.start(model, "/remote/session", source=tmp_path, **config)
    runner.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_loader_preserves_declared_qwen_multimodal_architecture(tiny_lab, tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    config_class = getattr(transformers, "Qwen3_5Config", None)
    if config_class is None:
        pytest.skip("installed Transformers does not provide Qwen3_5Config")
    config = config_class(architectures=["Qwen3_5ForConditionalGeneration"])
    assert config_class in transformers.AutoModelForCausalLM._model_mapping
    assert config_class in transformers.AutoModelForImageTextToText._model_mapping
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", Mock(return_value=config))
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", Mock(return_value=tiny_lab.tokenizer)
    )
    causal = Mock(side_effect=AssertionError("declared multimodal architecture was discarded"))
    image_text = Mock(return_value=tiny_lab.model)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", causal)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", image_text)

    loaded = load_lab(
        {"model": "test/qwen-checkpoint", "device": "cpu", "revision": "checkpoint-revision"},
        tmp_path / "multimodal-session",
    )
    image_text.assert_called_once()
    assert image_text.call_args.args == ("test/qwen-checkpoint",)
    assert image_text.call_args.kwargs["config"] is config
    assert image_text.call_args.kwargs["revision"] == "checkpoint-revision"
    assert loaded.model is tiny_lab.model
    causal.assert_not_called()
