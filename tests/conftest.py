import pytest


@pytest.fixture
def tiny_tokenizer():
    import tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3, "yes": 4, "no": 5},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )


def _tiny_config(kind, overrides):
    common = {"vocab_size": 6, "bos_token_id": 1, "eos_token_id": 1, "pad_token_id": 1}
    if kind == "gpt2":
        from transformers import GPT2Config

        return GPT2Config(
            **{**common, "n_positions": 64, "n_embd": 16, "n_layer": 2, "n_head": 2, **overrides}
        )
    if kind == "llama_gqa":
        from transformers import LlamaConfig

        return LlamaConfig(
            **{
                **common,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
                "max_position_embeddings": 64,
                **overrides,
            }
        )
    if kind == "qwen3_5":
        import transformers

        config_class = getattr(transformers, "Qwen3_5TextConfig", None)
        if config_class is None:
            pytest.skip("installed Transformers does not provide Qwen3_5TextConfig")
        return config_class(
            **{
                **common,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "full_attention"],
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "intermediate_size": 32,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "max_position_embeddings": 64,
                **overrides,
            }
        )
    raise ValueError(f"unknown tiny model kind: {kind}")


@pytest.fixture
def tiny_model(tiny_tokenizer):
    """A factory: ``tiny_model(kind, **overrides)`` returns ``(model, tokenizer)``.

    ``kind`` is "gpt2", "llama_gqa" or "qwen3_5" (a two-layer hybrid with one linear-attention
    layer). Weights are seeded (``seed=0`` unless overridden) and attention is eager.
    Overrides replace config fields.
    """

    def build(kind, seed=0, **overrides):
        import torch
        from transformers import AutoModelForCausalLM

        config = _tiny_config(kind, overrides)
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
        return model, tiny_tokenizer

    return build


@pytest.fixture
def make_lab(tmp_path):
    """A factory: ``make_lab(model, tokenizer)`` returns a Lab in a fresh session directory."""
    count = 0

    def build(model, tokenizer, path=None):
        nonlocal count
        from oumi_weights_lab import Lab

        count += 1
        return Lab(model, tokenizer, path or tmp_path / f"lab-{count}", model_id="test-model")

    return build


@pytest.fixture
def tiny_lab(tmp_path, tiny_tokenizer):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    from oumi_weights_lab import Lab

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=6,
            n_positions=64,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=1,
        )
    ).eval()
    return Lab(model, tiny_tokenizer, tmp_path / "lab", model_id="test-model")
