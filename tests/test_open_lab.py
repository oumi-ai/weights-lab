"""open_lab: the lab in a plain script, without a worker."""

import pytest


def _save_tiny_model(path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3, "yes": 4, "no": 5},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    ).save_pretrained(path)
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=6,
        n_positions=16,
        n_embd=8,
        n_layer=2,
        n_head=2,
        eos_token_id=1,
        bos_token_id=1,
        pad_token_id=1,
    )
    transformers.GPT2LMHeadModel(config).save_pretrained(path)


def test_open_lab_loads_in_process_and_evaluates(tmp_path):
    from oumi_weights_lab import Lab, open_lab

    _save_tiny_model(tmp_path / "model")
    lab = open_lab(tmp_path / "model", session=tmp_path / "s", device="cpu", dtype="float32")
    assert isinstance(lab, Lab)
    assert lab.session_path == tmp_path / "s" and lab.artifacts.is_dir()
    rows = [{"prompt": "hello world", "options": [" yes", " no"], "answer": 0}] * 3
    summary = lab.evaluate(rows, name="baseline", batch_size=1)
    assert summary["total"] == 3 and summary["forward_passes"] == 6
    assert lab.heads()  # the analysis calls work as in a worker session
    edit = lab.zero("transformer.h.0.attn.c_proj.weight")
    lab.undo(edit)


def test_open_lab_defaults_to_a_new_session_under_sessions(tmp_path, monkeypatch):
    from oumi_weights_lab import open_lab

    _save_tiny_model(tmp_path / "model")
    monkeypatch.chdir(tmp_path)
    first = open_lab(tmp_path / "model", device="cpu")
    second = open_lab(tmp_path / "model", device="cpu")
    assert first.session_path.parent == tmp_path / "sessions"
    assert first.session_path != second.session_path


def test_open_lab_applies_the_pass_limit(tmp_path):
    from oumi_weights_lab import BudgetExceeded, open_lab

    _save_tiny_model(tmp_path / "model")
    lab = open_lab(tmp_path / "model", session=tmp_path / "s", device="cpu", pass_limit=1)
    rows = [{"prompt": "hello", "options": [" yes", " no"], "answer": 0}]
    with pytest.raises(BudgetExceeded):
        lab.evaluate(rows, name="over", batch_size=1)


def test_open_lab_refuses_an_existing_nonempty_session(tmp_path):
    from oumi_weights_lab import open_lab

    _save_tiny_model(tmp_path / "model")
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="new session directory"):
        open_lab(tmp_path / "model", session=tmp_path / "s", device="cpu")
