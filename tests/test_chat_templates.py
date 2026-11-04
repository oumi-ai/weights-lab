import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Uses only words in the tiny vocabulary: a system message adds "no", the user turn is
# wrapped as "hello <content>", and the generation prompt adds "world".
TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}no {% endif %}"
    "{% if flag %}yes {% endif %}"
    "hello {{ messages[-1]['content'] }}"
    "{% if add_generation_prompt %} world{% endif %}"
)
ROWS = [{"prompt": "yes", "options": [" yes", " no"], "answer": 0}]


@pytest.fixture
def chat_lab(tiny_lab):
    tiny_lab.tokenizer.chat_template = TEMPLATE
    return tiny_lab


def _items(summary):
    lines = (Path(summary["run_dir"]) / "items.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_chat_template_renders_the_user_turn_before_scoring(chat_lab):
    templated = chat_lab.score("yes", [" yes", " no"], chat_template=True)
    assert templated == chat_lab.score("hello yes world", [" yes", " no"])


def test_system_message_and_template_kwargs(chat_lab):
    settings = {"system": "rules", "kwargs": {"flag": True}}
    templated = chat_lab.score("yes", [" yes", " no"], chat_template=settings)
    assert templated == chat_lab.score("no yes hello yes world", [" yes", " no"])


def test_prefix_attachments_transform_the_text_inside_the_template(chat_lab):
    with chat_lab.attach(lambda prompt: "no " + prompt, mode="prefix"):
        templated = chat_lab.score("yes", [" yes", " no"], chat_template=True)
    assert templated == chat_lab.score("hello no yes world", [" yes", " no"])


def test_a_templates_own_bos_is_not_doubled(chat_lab):
    chat_lab.tokenizer.chat_template = "{{ bos_token }}" + TEMPLATE
    chat_lab.tokenizer.bos_token = "<eos>"
    calls = []
    handle = chat_lab.model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs["input_ids"][0].tolist()),
        with_kwargs=True,
    )
    try:
        chat_lab.score("yes", {"A": " yes"}, chat_template=True, batch_size=1)
    finally:
        handle.remove()
    assert calls == [[1, 2, 4, 3, 4]]  # <eos> hello yes world | yes


def test_evaluate_records_the_prompt_format_and_refuses_mixed_pairs(chat_lab):
    templated = chat_lab.evaluate(ROWS, name="templated", chat_template={"system": "rules"})
    raw = chat_lab.evaluate(ROWS, name="raw")
    digest = hashlib.sha256(TEMPLATE.encode()).hexdigest()
    assert templated["prompt_format"] == {
        "chat_template_sha256": digest,
        "system": "rules",
        "kwargs": {},
    }
    assert raw["prompt_format"] is None
    with pytest.raises(ValueError, match="prompt_format"):
        chat_lab.compare_runs(templated, raw)
    # Datasets are identified by their raw prompts, whatever format renders them.
    assert templated["dataset_hash"] == raw["dataset_hash"]


def test_sampled_mode_generates_from_the_rendered_prompt(chat_lab):
    config = {"max_new_tokens": 2}
    rows = [{"prompt": "yes", "answer": "x"}]
    templated = chat_lab.evaluate(
        rows, name="t", mode="sampled", generation_config=config, chat_template=True
    )
    raw = chat_lab.evaluate(
        [{"prompt": "hello yes world", "answer": "x"}],
        name="r",
        mode="sampled",
        generation_config=config,
    )
    assert _items(templated)[0]["generated"] == _items(raw)[0]["generated"]


def test_generate_renders_the_template_without_adding_special_tokens(chat_lab):
    assert chat_lab.generate("yes", chat_template=True, max_new_tokens=2, do_sample=False) == (
        chat_lab.generate("hello yes world", max_new_tokens=2, do_sample=False)
    )


def test_a_tokenizer_without_a_template_is_refused_before_reserving(tiny_lab):
    tiny_lab.tokenizer.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        tiny_lab.evaluate(ROWS, chat_template=True, held_out=True, max_passes=10)
    tiny_lab.evaluate(ROWS, held_out=True, max_passes=10)


@pytest.mark.parametrize(
    "setting",
    [{"system": 3}, {"kwargs": {"flag": {1, 2}}}, {"other": "x"}, "yes"],
    ids=["system", "kwargs", "unknown-key", "string"],
)
def test_invalid_chat_template_settings_are_refused(chat_lab, setting):
    with pytest.raises(ValueError, match="chat_template"):
        chat_lab.score("yes", [" yes"], chat_template=setting)
