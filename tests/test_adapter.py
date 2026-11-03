import pytest

from weights_lab.lab import load_lab


def test_local_lora_adapter_loads_in_eval_mode_and_counts_base_model_calls(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    peft = pytest.importorskip("peft")
    model_path = tmp_path / "base-model"
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3}, unk_token="<unk>"
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    ).save_pretrained(model_path)
    torch.manual_seed(0)
    transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=4,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            eos_token_id=1,
            bos_token_id=1,
            pad_token_id=1,
        )
    ).save_pretrained(model_path)
    adapter = peft.get_peft_model(
        transformers.AutoModelForCausalLM.from_pretrained(model_path),
        peft.LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=["c_attn"],
            r=2,
            lora_alpha=4,
            fan_in_fan_out=True,
        ),
    )
    with torch.no_grad():
        for name, parameter in adapter.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.125)
    adapter_path = tmp_path / "adapter"
    adapter.save_pretrained(adapter_path)

    lab = load_lab(
        {
            "model": str(model_path),
            "adapter": str(adapter_path),
            "device": "cpu",
            "dtype": "float32",
        },
        tmp_path / "session",
    )
    assert isinstance(lab.model, peft.PeftModel)
    assert not lab.model.training
    assert lab.device.type == "cpu"
    loaded_lora = [value for name, value in lab.model.named_parameters() if "lora_B" in name]
    assert loaded_lora
    assert all(torch.all(value == 0.125) for value in loaded_lora)
    inputs = lab.tokenizer(["hello world", "world hello"], return_tensors="pt").to(lab.device)
    with torch.no_grad():
        output = lab.model(**inputs)
    assert tuple(output.logits.shape) == (2, 2, 4)
    assert lab.forward_passes == 1
    with torch.no_grad():
        generated = lab.model.generate(
            **inputs, max_new_tokens=2, min_new_tokens=2, do_sample=False
        )
    assert tuple(generated.shape) == (2, 4)
    assert lab.forward_passes == 3
