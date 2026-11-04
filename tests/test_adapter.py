import pytest

from oumi_weights_lab.budget import BudgetExceeded
from oumi_weights_lab.lab import load_lab


@pytest.fixture
def adapter_lab(tmp_path):
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
    return lab


def test_local_lora_adapter_loads_in_eval_mode_and_counts_base_model_calls(adapter_lab):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    lab = adapter_lab
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


def test_adapter_failed_base_forward_counts_once(adapter_lab):
    torch = pytest.importorskip("torch")
    lab = adapter_lab
    with pytest.raises(IndexError):
        lab.model(input_ids=torch.tensor([[999]]))
    assert lab.forward_passes == lab.budget.used == 1
    inputs = lab.tokenizer("hello", return_tensors="pt")
    lab.model(**inputs)
    assert lab.forward_passes == lab.budget.used == 2


def test_adapter_budget_blocks_forward_and_partial_generation_before_execution(adapter_lab):
    lab = adapter_lab
    lab.budget.limit = 2
    calls = []
    handle = lab.model.get_base_model().transformer.register_forward_pre_hook(
        lambda module, inputs: calls.append(1)
    )
    inputs = lab.tokenizer("hello", return_tensors="pt")
    try:
        lab.model(**inputs)
        with pytest.raises(BudgetExceeded):
            lab.model.generate(**inputs, min_new_tokens=3, max_new_tokens=3, do_sample=False)
        assert lab.forward_passes == lab.budget.used == len(calls) == 2
        with pytest.raises(BudgetExceeded):
            lab.model(**inputs)
        assert lab.forward_passes == lab.budget.used == len(calls) == 2
    finally:
        handle.remove()


def test_adapter_local_budget_uses_shared_guard(adapter_lab):
    lab = adapter_lab
    inputs = lab.tokenizer("hello", return_tensors="pt")
    with lab.limit_passes(1) as local:
        lab.model(**inputs)
        with pytest.raises(BudgetExceeded):
            lab.model(**inputs)
        assert local.used == lab.budget.used == lab.forward_passes == 1
    lab.model(**inputs)
    assert lab.budget.used == lab.forward_passes == 2


def test_adapter_inspection_gradients_and_export_preserve_live_model(adapter_lab):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    lab = adapter_lab
    inputs = lab.tokenizer("hello world", return_tensors="pt")
    block = "base_model.model.transformer.h.0"
    with torch.no_grad(), lab.record(block) as recorded:
        baseline = lab.model(**inputs).logits.clone()
    assert len(recorded[block]) == 1
    with (
        torch.no_grad(),
        lab.patch(block, lambda value: value.zero_()),
        lab.record(block) as patched,
    ):
        lab.model(**inputs)
    assert torch.count_nonzero(patched[block][0]) == 0
    parameter = next(name for name, _ in lab.model.named_parameters() if "lora_B" in name)
    gradients = lab.gradient(dict(inputs), parameters=[parameter])[parameter]
    assert gradients.shape[0] == 1
    assert torch.isfinite(gradients).all()
    assert torch.count_nonzero(gradients) > 0
    used = lab.forward_passes
    destination = lab.export("merged")
    restored = transformers.AutoModelForCausalLM.from_pretrained(destination).eval()
    with torch.no_grad():
        assert torch.allclose(restored(**inputs).logits, baseline, atol=1e-6)
        assert torch.allclose(lab.model(**inputs).logits, baseline, atol=1e-6)
    # Export's copied model must not retain a counter bound to the live Lab.
    assert lab.forward_passes == lab.budget.used == used + 1
