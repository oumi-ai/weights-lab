import pytest


@pytest.fixture
def tiny_lab(tmp_path):
    import tokenizers
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from oumi_weights_lab import Lab

    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3, "yes": 4, "no": 5},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )
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
    return Lab(model, tokenizer, tmp_path / "lab", model_id="test-model")
