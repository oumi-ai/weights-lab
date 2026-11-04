import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from oumi_weights_lab.budget import BudgetExceeded
from oumi_weights_lab.lab import Lab


def workspace(path, state=None, limit=20):
    model = torch.nn.Linear(4, 3).to(dtype=torch.bfloat16)
    if state is not None:
        model.load_state_dict(state)
    path.mkdir()
    return Lab(model, SimpleNamespace(), path, model_id="tiny", pass_limit=limit)


def test_bundle_replays_overlapping_slices_and_preserves_exact_undo(tmp_path):
    lab = workspace(tmp_path / "first")
    base = {k: v.clone() for k, v in lab.model.state_dict().items()}
    mask = torch.tensor([True, False, True])
    first = lab.edit("weight", lambda v: v + 0.1, index=mask, note="rows")
    removed = lab.zero("bias")
    lab.undo(removed)
    second = lab.zero("weight", index=(slice(None), 0))
    expected = lab.model.weight.detach().clone()
    lab.model(torch.ones(1, 4, dtype=torch.bfloat16))
    bundle = lab.save_edits("candidate")
    assert lab.forward_passes == 1
    fresh = workspace(tmp_path / "second", base)
    fresh.load_edits(bundle)
    assert torch.equal(fresh.model.weight, expected)
    assert fresh.active_edit_ids == [first, second]
    assert fresh.edits() == lab.edits()
    assert fresh.revision == lab.revision
    assert fresh.budget.used == fresh.forward_passes == 1
    fresh.undo(second)
    fresh.undo(first)
    assert torch.equal(fresh.model.weight, base["weight"])
    assert fresh.zero("bias") == "edit-000004"


def test_resume_preserves_budget_and_cannot_raise_limit(tmp_path):
    lab = workspace(tmp_path / "old", limit=1)
    base = lab.model.state_dict()
    lab.model(torch.ones(1, 4, dtype=torch.bfloat16))
    bundle = lab.save_edits("spent")
    fresh = workspace(tmp_path / "new", base, limit=None)
    fresh.load_edits(bundle)
    assert fresh.budget.limit == 1
    with pytest.raises(BudgetExceeded):
        fresh.model(torch.ones(1, 4, dtype=torch.bfloat16))
    assert fresh.forward_passes == 1
    with pytest.raises(ValueError, match="raise"):
        workspace(tmp_path / "raised", base, limit=2).load_edits(bundle)
    with pytest.raises(BudgetExceeded):
        workspace(tmp_path / "lower", base, limit=0).load_edits(bundle)


def test_bundle_rejects_different_unedited_weight_and_payload_corruption(tmp_path):
    lab = workspace(tmp_path / "old")
    base = {k: v.clone() for k, v in lab.model.state_dict().items()}
    lab.zero("weight")
    bundle = lab.save_edits("candidate")
    wrong = workspace(tmp_path / "wrong", base)
    with torch.no_grad():
        wrong.model.bias.add_(1)
    before = wrong.model.weight.detach().clone()
    with pytest.raises(ValueError, match="content"):
        wrong.load_edits(bundle)
    assert torch.equal(wrong.model.weight, before)
    payload = bundle / "edits.safetensors"
    payload.write_bytes(payload.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="checksum"):
        workspace(tmp_path / "fresh", base).load_edits(bundle)


def test_resume_carries_dev_and_failed_heldout_reservations(tmp_path):
    lab = workspace(tmp_path / "old")
    base = {k: v.clone() for k, v in lab.model.state_dict().items()}
    dev = [{"prompt": "dev", "options": [" a", " b"], "answer": 0}]
    held = [{"prompt": "held", "options": [" a", " b"], "answer": 0}]
    lab.score = lambda *args, **kwargs: {"A": 0.0, "B": -1.0}
    lab.evaluate(dev, name="dev")

    def fail(*args, **kwargs):
        raise RuntimeError("grader failed")

    lab.score = fail
    with pytest.raises(RuntimeError, match="grader"):
        lab.evaluate(held, name="held", held_out=True, max_passes=2)
    bundle = lab.save_edits("candidate")
    fresh = workspace(tmp_path / "new", base)
    fresh.load_edits(bundle)
    for rows, role in ((held, False), (held, True), (dev, True)):
        with pytest.raises(ValueError, match="reserved"):
            fresh.evaluate(rows, held_out=role, max_passes=2)
    assert json.loads((fresh.artifacts / "evaluations/_registry.json").read_text()) == json.loads(
        (lab.artifacts / "evaluations/_registry.json").read_text()
    )


def test_bundle_refuses_nonfresh_or_active_components_and_overwrite(tmp_path):
    lab = workspace(tmp_path / "old")
    bundle = lab.save_edits("candidate")
    with pytest.raises(FileExistsError):
        lab.save_edits("candidate")
    lab.zero("weight")
    with pytest.raises(ValueError, match="fresh"):
        lab.load_edits(bundle)
    with lab.attach(lambda prompt: prompt, mode="prefix"):
        with pytest.raises(ValueError, match="attachments"):
            lab.save_edits("unsafe")


def test_tied_parameters_and_chained_resume(tmp_path):
    def tied(path):
        model = torch.nn.Module()
        model.a = torch.nn.Parameter(torch.arange(4.0))
        model.b = model.a
        path.mkdir()
        return Lab(model, SimpleNamespace(), path, model_id="tied")

    first = tied(tmp_path / "first")
    first.zero("a", index=slice(0, 2))
    first.edit("b", lambda v: v + 3)
    second = tied(tmp_path / "second")
    second.load_edits(first.save_edits("save"))
    third = tied(tmp_path / "third")
    third.load_edits(second.save_edits("save"))
    third.undo()
    third.undo()
    assert torch.equal(third.model.a, torch.arange(4.0))
    assert third.model.a is third.model.b


def test_restore_failure_rolls_back_weights_registry_and_budget(tmp_path, monkeypatch):
    import oumi_weights_lab.persistence as persistence

    old = workspace(tmp_path / "old")
    base = {k: v.clone() for k, v in old.model.state_dict().items()}
    old.zero("weight")
    registry = old.artifacts / "evaluations/_registry.json"
    registry.parent.mkdir()
    registry.write_text(json.dumps({"session_id": "test", "datasets": {}}))
    bundle = old.save_edits("save")
    fresh = workspace(tmp_path / "new", base)
    real_write = persistence.write_json

    def interrupted(path, value):
        real_write(path, value)
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(persistence, "write_json", interrupted)
        with pytest.raises(KeyboardInterrupt):
            fresh.load_edits(bundle)
    assert torch.equal(fresh.model.weight, base["weight"])
    assert fresh.edits() == [] and fresh.forward_passes == 0
    assert not (fresh.artifacts / "evaluations/_registry.json").exists()
    fresh.load_edits(bundle)
    assert fresh.model.weight.count_nonzero() == 0


def test_changed_tokenizer_and_adapter_runtime_state_are_rejected(tmp_path):
    old = workspace(tmp_path / "old")
    base = {k: v.clone() for k, v in old.model.state_dict().items()}
    old.tokenizer = SimpleNamespace(get_vocab=lambda: {"word": 0})
    old.model.lora_A = {}
    old.model.scaling = {"default": 2.0}
    old.model.active_adapters = ["default"]
    old.model.disable_adapters = True
    bundle = old.save_edits("save")
    fresh = workspace(tmp_path / "new", base)
    fresh.tokenizer = SimpleNamespace(get_vocab=lambda: {"word": 0})
    fresh.model.lora_A = {}
    fresh.model.scaling = {"default": 2.0}
    fresh.model.active_adapters = ["default"]
    fresh.model.disable_adapters = False
    with pytest.raises(ValueError, match="content"):
        fresh.load_edits(bundle)
    fresh.model.disable_adapters = True
    fresh.tokenizer = SimpleNamespace(get_vocab=lambda: {"different": 0})
    with pytest.raises(ValueError, match="content"):
        fresh.load_edits(bundle)
    fresh.tokenizer = old.tokenizer
    fresh.load_edits(bundle)


def test_tokenizer_padding_and_truncation_calls_do_not_change_identity(tmp_path):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    from oumi_weights_lab.persistence import _fingerprint

    lab = workspace(tmp_path / "old")
    lab.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"?": 0, "pad": 1}, unk_token="?")),
        unk_token="?",
        pad_token="pad",
    )
    expected = _fingerprint(lab)
    lab.tokenizer("unknown", padding=True, truncation=True, max_length=4)
    assert _fingerprint(lab) == expected


def test_manifest_corruption_and_monotonic_component_revision(tmp_path):
    old = workspace(tmp_path / "old")
    base = {k: v.clone() for k, v in old.model.state_dict().items()}
    with old.attach(lambda prompt: prompt, mode="prefix"):
        pass
    assert old.revision == 2
    bundle = old.save_edits("saved")
    fresh = workspace(tmp_path / "new", base)
    fresh.load_edits(bundle)
    assert fresh.revision == 2
    with fresh.attach(lambda prompt: prompt, mode="prefix"):
        assert fresh.revision == 3
    assert fresh.revision == 4
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["registry"] = {"datasets": {}}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest checksum"):
        workspace(tmp_path / "corrupt", base).load_edits(bundle)


def test_real_peft_bundle_preserves_base_adapter_edits_and_logits(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=16, n_embd=8, n_layer=1, n_head=1)),
        peft.LoraConfig(
            task_type="CAUSAL_LM", target_modules=["c_attn"], r=2, lora_alpha=4, fan_in_fan_out=True
        ),
    ).eval()
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                parameter.fill_(0.1)
    fresh_model = copy.deepcopy(model)
    old = Lab(model, SimpleNamespace(), tmp_path / "old", model_id="peft", pass_limit=20)
    name = "base_model.model.transformer.h.0.attn.c_attn.lora_B.default.weight"
    old.zero(name)
    old.zero("base_model.model.transformer.h.0.ln_1.weight")
    with torch.no_grad():
        expected = model(torch.tensor([[2, 3]])).logits
    fresh = Lab(fresh_model, SimpleNamespace(), tmp_path / "new", model_id="peft")
    fresh.load_edits(old.save_edits("saved"))
    with torch.no_grad():
        assert torch.equal(fresh_model(torch.tensor([[2, 3]])).logits, expected)
    fresh.undo()
    fresh.undo()
    assert fresh_model.get_parameter(name).count_nonzero() > 0


def test_cli_saves_stops_and_resumes_a_local_checkpoint(tmp_path):
    from tokenizers import Tokenizer, models
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    checkpoint = tmp_path / "model"
    model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=16, n_embd=8, n_layer=1, n_head=1))
    model.save_pretrained(checkpoint)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"?": 0, "eos": 1, "hello": 2, "x": 3}, unk_token="?")
        ),
        unk_token="?",
        eos_token="eos",
        pad_token="eos",
    )
    tokenizer.save_pretrained(checkpoint)
    first, second = tmp_path / "first", tmp_path / "second"

    def cli(*args):
        output = subprocess.run(
            [sys.executable, "-m", "oumi_weights_lab", *map(str, args)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert output.returncode == 0, output.stderr + output.stdout
        return json.loads(output.stdout)

    try:
        cli(
            "start",
            "--session",
            first,
            "--model",
            checkpoint,
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--pass-limit",
            "4",
        )
        cli(
            "send",
            "--session",
            first,
            "--wait",
            "--code",
            "import torch\n"
            "lab.zero('transformer.h.0.mlp.c_fc.weight')\n"
            "model(torch.tensor([[2]]))\n"
            "lab.save_edits('saved')",
        )
        cli("stop", "--session", first)
        cli(
            "start",
            "--session",
            second,
            "--model",
            checkpoint,
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--resume",
            first / "artifacts/saved",
        )
        cli(
            "send",
            "--session",
            second,
            "--wait",
            "--code",
            "assert lab.forward_passes == lab.budget.used == 1\n"
            "assert lab.budget.limit == 4\n"
            "assert model.transformer.h[0].mlp.c_fc.weight.count_nonzero() == 0\n"
            "lab.undo()\n"
            "assert model.transformer.h[0].mlp.c_fc.weight.count_nonzero() > 0",
        )
    finally:
        for session in (first, second):
            if (session / "status.json").exists():
                cli("stop", "--session", session)
