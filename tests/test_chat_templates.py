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
    prompt_format = dict(templated["prompt_format"])
    assert len(prompt_format.pop("rendered_sha256")) == 64
    assert prompt_format == {"chat_template_sha256": digest, "system": "rules", "kwargs": {}}
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


OTHER = "hello no {{ messages[-1]['content'] }}{% if add_generation_prompt %} world{% endif %}"


def test_named_templates_are_resolved_before_reserving(chat_lab):
    chat_lab.tokenizer.chat_template = {"default": TEMPLATE, "other": OTHER}
    default = chat_lab.evaluate(
        ROWS, name="named", chat_template=True, held_out=True, max_passes=10
    )
    assert (
        default["prompt_format"]["chat_template_sha256"]
        == hashlib.sha256(TEMPLATE.encode()).hexdigest()
    )
    other = chat_lab.evaluate(
        [{**ROWS[0], "prompt": "no"}],
        name="other",
        chat_template={"kwargs": {"chat_template": "other"}},
    )
    assert (
        other["prompt_format"]["chat_template_sha256"] == hashlib.sha256(OTHER.encode()).hexdigest()
    )


def test_prompt_format_fingerprints_what_the_template_renders(chat_lab):
    plain = chat_lab.evaluate(ROWS, name="plain", chat_template=True)
    flagged = chat_lab.evaluate(ROWS, name="flagged", chat_template={"kwargs": {"flag": True}})
    again = chat_lab.evaluate(ROWS, name="again", chat_template=True)
    assert plain["prompt_format"]["rendered_sha256"] == again["prompt_format"]["rendered_sha256"]
    assert plain["prompt_format"]["rendered_sha256"] != flagged["prompt_format"]["rendered_sha256"]


def test_an_empty_system_message_is_rendered(chat_lab):
    templated = chat_lab.score("yes", [" yes", " no"], chat_template={"system": ""})
    assert templated == chat_lab.score("no hello yes world", [" yes", " no"])


def test_a_checker_scoring_raw_prompts_mid_evaluation_gets_raw_text(chat_lab):
    raw = chat_lab.score("yes", [" yes"])
    seen = []

    def checker(generated, answer, row):
        seen.append(chat_lab.score(row["prompt"], [" yes"]))
        return True

    chat_lab.evaluate(
        [{"prompt": "yes", "answer": "x"}],
        mode="sampled",
        checker=checker,
        checker_id="nested",
        chat_template=True,
        generation_config={"max_new_tokens": 1},
    )
    assert seen == [raw]


@pytest.mark.parametrize("key", ["tokenize", "add_generation_prompt", "conversation"])
def test_template_kwargs_the_lab_controls_are_refused(chat_lab, key):
    with pytest.raises(ValueError, match="chat_template"):
        chat_lab.score("yes", [" yes"], chat_template={"kwargs": {key: True}})


def test_generate_validates_the_template_before_prefix_transforms(tiny_lab):
    tiny_lab.tokenizer.chat_template = None
    calls = []
    with tiny_lab.attach(lambda prompt: calls.append(prompt) or prompt, mode="prefix"):
        with pytest.raises(ValueError, match="chat template"):
            tiny_lab.generate("yes", chat_template=True, max_new_tokens=1)
    assert calls == []
