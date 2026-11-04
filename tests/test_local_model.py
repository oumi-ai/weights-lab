import json
import subprocess
import sys

import pytest


def cli(*args, timeout=60):
    return subprocess.run(
        [sys.executable, "-m", "oumi_weights_lab", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_local_cpu_model_session_runs_multiple_snippets_and_counts_passes(tmp_path):
    """Exercise loading and the detached CLI without downloading model weights."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    model_path = tmp_path / "tiny-model"
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
    model = transformers.GPT2LMHeadModel(
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
    )
    model.save_pretrained(model_path)
    session_path = tmp_path / "session"
    started = cli(
        "start",
        "--session",
        session_path,
        "--model",
        model_path,
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--snippet-timeout",
        "10",
    )
    assert started.returncode == 0, started.stderr
    try:
        first = cli(
            "send",
            "--session",
            session_path,
            "--wait",
            "--timeout",
            "60",
            "--code",
            "import torch\n"
            "inputs = tokenizer('hello world', return_tensors='pt').to(lab.device)\n"
            "with torch.no_grad():\n    logits = model(**inputs).logits\n"
            "print(tuple(logits.shape))\n",
        )
        assert first.returncode == 0, first.stderr + first.stdout
        first_result = json.loads(first.stdout)
        assert first_result["state"] == "succeeded"
        assert first_result["forward_passes"] == 1
        second = cli(
            "send",
            "--session",
            session_path,
            "--wait",
            "--timeout",
            "10",
            "--code",
            "with torch.no_grad():\n    again = model(**inputs).logits\n"
            "assert torch.equal(logits, again)\nassert lab.forward_passes == 2\n"
            "(artifacts / 'shape.txt').write_text(str(tuple(again.shape)))\n",
        )
        assert second.returncode == 0, second.stderr + second.stdout
        second_result = json.loads(second.stdout)
        assert second_result["state"] == "succeeded"
        assert second_result["forward_passes"] == 1
        assert (session_path / "artifacts" / "shape.txt").read_text() == "(1, 2, 4)"
        generated = cli(
            "send",
            "--session",
            session_path,
            "--wait",
            "--timeout",
            "10",
            "--code",
            "with torch.no_grad():\n"
            "    tokens = model.generate(**inputs, max_new_tokens=2, "
            "min_new_tokens=2, do_sample=False)\n"
            "assert tokens.shape[-1] == 4\nassert lab.forward_passes == 4\n",
        )
        assert generated.returncode == 0, generated.stderr + generated.stdout
        assert json.loads(generated.stdout)["forward_passes"] == 2
    finally:
        stopped = cli("stop", "--session", session_path, "--timeout", "10")
        assert stopped.returncode == 0, stopped.stderr
    status = json.loads((session_path / "status.json").read_text())
    assert status["state"] == "stopped"
