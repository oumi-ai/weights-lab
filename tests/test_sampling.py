import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from oumi_weights_lab import sampling  # noqa: E402
from oumi_weights_lab.sampling import (  # noqa: E402
    FinishTracker,
    PerSequenceSampler,
    effective_config,
    stop_token_ids,
    warpers,
)


def _draws(seeds, scores, steps=5, processors=()):
    sampler = PerSequenceSampler(seeds, list(processors))
    picks = []
    for _ in range(steps):
        picks.append(sampler(None, scores.clone()).argmax(-1).tolist())
    return [list(column) for column in zip(*picks, strict=True)]


def test_a_rows_draws_do_not_depend_on_the_rest_of_its_batch():
    torch.manual_seed(0)
    scores = torch.randn(3, 11)
    batched = _draws([7, 8, 9], scores)
    for row, seed in enumerate([7, 8, 9]):
        assert _draws([seed], scores[row : row + 1]) == [batched[row]]


def test_sampler_returns_one_hot_scores():
    output = PerSequenceSampler([0, 1], [])(None, torch.randn(2, 5))
    assert ((output == 0).sum(-1) == 1).all()
    assert torch.isinf(output).sum() == 2 * 4


def test_zero_probability_tokens_are_never_drawn():
    scores = torch.full((1, 6), float("-inf"))
    scores[0, 3] = 0.0
    assert _draws([1], scores, steps=50) == [[3] * 50]


def test_warpers_apply_before_the_draw():
    config = transformers.GenerationConfig(do_sample=True, top_k=1)
    scores = torch.tensor([[0.0, 2.0, 1.0, -1.0]])
    assert _draws([3], scores, steps=20, processors=warpers(config)) == [[1] * 20]


def test_warpers_follow_generate_defaults_and_settings(tiny_lab):
    config, unused = effective_config(tiny_lab.model, {"do_sample": True})
    assert [type(w).__name__ for w in warpers(config)] == ["TopKLogitsWarper"]  # top_k=50
    assert config.top_k == 50 and unused == {}
    config, _ = effective_config(
        tiny_lab.model, {"do_sample": True, "temperature": 0.5, "top_k": 0, "top_p": 0.9}
    )
    assert [type(w).__name__ for w in warpers(config)] == [
        "TemperatureLogitsWarper",
        "TopPLogitsWarper",
    ]
    assert tiny_lab.model.generation_config.temperature != 0.5  # The model's config is untouched.


class LastTokenIs:
    """Stands in for StopStringCriteria: the word-level test vocabulary has no BPE pieces."""

    def __init__(self, tokenizer, stop_strings):
        self.token = tokenizer.convert_tokens_to_ids(stop_strings[0])

    def __call__(self, input_ids, scores):
        return input_ids[:, -1] == self.token


def test_tracker_records_why_and_when_each_row_finished(tiny_lab, monkeypatch):
    monkeypatch.setattr(sampling, "StopStringCriteria", LastTokenIs)
    prompt = [[2, 3], [2, 3], [2, 3]]  # "hello world"
    tracker = FinishTracker(3, 2, eos_token_id=1, stop_strings=["no"], tokenizer=tiny_lab.tokenizer)
    # One list of row tokens per step: row 1 hits EOS at step 1, row 0 the stop string
    # ("no") at step 2, and row 2 runs to the limit.
    steps = [[4, 1, 4], [5, 4, 4], [4, 4, 4]]
    ids = torch.tensor(prompt)
    for step, tokens in enumerate(steps, start=1):
        ids = torch.cat([ids, torch.tensor(tokens)[:, None]], dim=-1)
        assert not tracker(ids, None).any()
        if step == 2:
            assert tracker.unfinished == 1
    assert tracker.results(len(steps)) == [(2, "stop_string"), (1, "eos"), (3, "max_tokens")]


@pytest.mark.parametrize(
    ("model_eos", "tokenizer_eos", "settings", "expected"),
    [
        (5, 1, {}, [5, 1]),  # Qwen3.5: the model's EOS omits the end of the turn.
        ([5, 6], 1, {}, [5, 6, 1]),
        (None, 1, {}, [1]),
        (1, 1, {}, None),  # Already stops there, so nothing changes.
        ([5, 1], 1, {}, None),
        (5, None, {}, None),  # A tokenizer without an EOS adds nothing.
        (5, 1, {"eos_token_id": 5}, None),  # The caller's choice wins.
        (5, 1, {"eos_token_id": None}, None),
        (5, 1, {"generation_config": transformers.GenerationConfig(eos_token_id=5)}, None),
        (5, 1, {"generation_config": transformers.GenerationConfig()}, [5, 1]),
    ],
)
def test_stop_token_ids_add_the_tokenizers_eos_unless_the_caller_chose(
    model_eos, tokenizer_eos, settings, expected
):
    from types import SimpleNamespace

    model = SimpleNamespace(generation_config=transformers.GenerationConfig(eos_token_id=model_eos))
    tokenizer = SimpleNamespace(eos_token_id=tokenizer_eos)
    assert stop_token_ids(model, tokenizer, settings) == expected
