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


@pytest.mark.parametrize(
    "options",
    [
        {"pass_limit": -1},
        {"pass_limit": True},
        {"pass_limit": 1.5},
        {"dtype": "float8"},
        {"quantize": "int3"},
        {"device": "tpu"},
        {"device_map": "balanced", "quantize": "nf4"},
        {"max_memory": {"0": "1GiB"}},
    ],
)
def test_open_lab_rejects_bad_options_before_loading_or_creating_anything(
    tmp_path, monkeypatch, options
):
    from oumi_weights_lab import lab as lab_module
    from oumi_weights_lab import open_lab

    def fail(*args, **kwargs):
        raise AssertionError("the model must not load")

    monkeypatch.setattr(lab_module, "load_lab", fail)
    monkeypatch.chdir(tmp_path)
    options = {"device": "cpu", **options}
    with pytest.raises(ValueError):
        open_lab(tmp_path / "model", **options)
    assert not (tmp_path / "sessions").exists()
    with pytest.raises(ValueError):
        open_lab(tmp_path / "model", session=tmp_path / "s", **options)
    assert not (tmp_path / "s").exists()


def test_open_lab_removes_the_session_it_created_when_loading_fails(tmp_path, monkeypatch):
    from oumi_weights_lab import open_lab

    monkeypatch.chdir(tmp_path)
    with pytest.raises((OSError, ValueError)):
        open_lab(tmp_path / "no-such-model", device="cpu")
    assert list((tmp_path / "sessions").iterdir()) == []
    with pytest.raises((OSError, ValueError)):
        open_lab(tmp_path / "no-such-model", session=tmp_path / "s", device="cpu")
    assert not (tmp_path / "s").exists()
    (tmp_path / "empty").mkdir()
    with pytest.raises((OSError, ValueError)):
        open_lab(tmp_path / "no-such-model", session=tmp_path / "empty", device="cpu")
    assert (tmp_path / "empty").is_dir() and list((tmp_path / "empty").iterdir()) == []


def test_open_lab_resolves_a_relative_session(tmp_path, monkeypatch):
    from oumi_weights_lab import open_lab

    _save_tiny_model(tmp_path / "model")
    monkeypatch.chdir(tmp_path)
    lab = open_lab("model", session="rel", device="cpu")
    assert lab.session_path == (tmp_path / "rel").resolve()
    assert lab.session_path.is_absolute()


def test_open_lab_passes_every_load_option_to_the_loader(tmp_path, monkeypatch):
    from oumi_weights_lab import lab as lab_module
    from oumi_weights_lab import open_lab

    seen = {}

    def capture(config, session_path):
        seen["config"], seen["session"] = config, session_path
        return "lab"

    monkeypatch.setattr(lab_module, "load_lab", capture)
    monkeypatch.chdir(tmp_path)
    result = open_lab(
        "some/model",
        session="s",
        device="cpu",
        dtype="float32",
        adapter="some/adapter",
        seed=7,
        pass_limit=11,
        allow_hub_kernels=True,
        revision="rev1",
        adapter_revision="rev2",
        resume="edits.pt",
    )
    assert result == "lab"
    assert seen["config"] == {
        "model": "some/model",
        "adapter": "some/adapter",
        "device": "cpu",
        "dtype": "float32",
        "seed": 7,
        "pass_limit": 11,
        "allow_hub_kernels": True,
        "revision": "rev1",
        "adapter_revision": "rev2",
        "resume": str((tmp_path / "edits.pt").resolve()),
    }
    assert seen["session"] == (tmp_path / "s").resolve()


def test_open_lab_passes_placement_options_to_the_loader(tmp_path, monkeypatch):
    from oumi_weights_lab import lab as lab_module
    from oumi_weights_lab import open_lab

    seen = {}
    monkeypatch.setattr(
        lab_module, "load_lab", lambda config, session_path: seen.update(config) or "lab"
    )
    open_lab("m", session=tmp_path / "a", device_map="balanced", max_memory="0=1GiB,1=2GiB")
    assert seen["device_map"] == "balanced"
    assert seen["max_memory"] == {"0": "1GiB", "1": "2GiB"}
    open_lab("m", session=tmp_path / "b", device_map="balanced", max_memory={"0": "1GiB"})
    assert seen["max_memory"] == {"0": "1GiB"}
    open_lab("m", session=tmp_path / "c", device="cpu", quantize="nf4")
    assert seen["quantize"] == "nf4"
