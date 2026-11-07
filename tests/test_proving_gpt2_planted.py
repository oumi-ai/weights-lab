"""The GPT-2 planted families, built on a tiny GPT-2 through the same code as GPT-2 small."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from oumi_weights_lab.proving import (  # noqa: E402
    build_task,
    gpt2_planted,
    grade_claim,
    load_key,
    planted,
    tasks,
)

FAMILIES = list(gpt2_planted.GPT2_FAMILIES)
CHANCE = gpt2_planted.CHANCE
WIDTH = 256  # four heads of 64 dimensions, as in GPT-2 small


def files_under(root):
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


@pytest.fixture(scope="session")
def tiny_gpt2(tmp_path_factory):
    """A random 6-layer, 4-head GPT-2 whose tokenizer knows the word list.

    The weights are scaled the way a trained GPT-2 behaves where the build relies on it:
    token embeddings large next to what heads and MLPs write, so a word is still readable
    in later layers, and query-key weights halved, so attention mixes words less.
    """
    import tokenizers
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = ["<unk>", "<eos>", ",", *gpt2_planted.WORDS]
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=64,
        n_embd=WIDTH,
        n_layer=6,
        n_head=4,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=1,
        initializer_range=0.2,
        attn_implementation="eager",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = GPT2LMHeadModel(config).eval()
        with torch.no_grad():
            model.transformer.wte.weight.normal_(0, 1.0)
            model.transformer.wpe.weight.mul_(0.1)
            for block in model.transformer.h:
                block.attn.c_attn.weight[:, : 2 * WIDTH].mul_(0.5)
                block.attn.c_proj.weight.mul_(0.2)
                block.mlp.c_proj.weight.mul_(0.2)
    path = tmp_path_factory.mktemp("tiny-gpt2") / "model"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def gpt2_tasks(tiny_gpt2, tmp_path_factory):
    """Each GPT-2 family's dev task 0 on the tiny GPT-2: ``{family: {"dir", "key"}}``."""
    root = tmp_path_factory.mktemp("gpt2-proving")
    built = {}
    for family in FAMILIES:
        info = build_task(family, "dev", 0, root / "tasks", root / "keys", model_id=str(tiny_gpt2))
        built[family] = {"dir": Path(info["task_dir"]), "key": load_key(info["key_path"])}
    return built


@pytest.mark.parametrize("family", FAMILIES)
def test_each_family_builds_and_passes_its_validation(family, gpt2_tasks):
    key = gpt2_tasks[family]["key"]
    check = key["validation"]
    assert key["task"] == tasks.task_id(family, "dev", 0)
    assert (key["family"], key["split"], key["chance"]) == (family, "dev", CHANCE)
    assert check["rows"] == 300 and 0 <= check["attempt"] < tasks.ATTEMPTS
    assert check["native_accuracy"] <= CHANCE + gpt2_planted.NATIVE_SLACK
    assert check["baseline_accuracy"] >= 0.9
    assert check["key_ablated_accuracy"] <= CHANCE + 0.15
    assert check["only_key_accuracy"] <= CHANCE + 0.15
    assert min(check["key_head_ablated_alone"].values()) >= 0.85
    assert check["largest_other_effect"] <= 0.1
    assert all(abs(effect) <= 0.1 for effect in check["decoy_effect"].values())
    layers = [part["layer"] for part in key["components"]]
    assert layers == sorted(layers) and len(set(layers)) == len(layers)
    assert min(layers) >= gpt2_planted.FIRST_LAYER
    decoy = key["decoys"][0]
    assert decoy["kind"] == "head" and decoy not in key["components"]
    for name, norms in check["norms"].items():
        for value, low, high in norms.values():
            assert low * (1 - 1e-3) <= value <= high * (1 + 1e-3), name
    assert set(check["norms"]) == {f"{p['layer']}.{p['head']}" for p in key["components"]} | {
        f"{decoy['layer']}.{decoy['head']}"
    }
    spec = key["row_spec"]
    assert len(spec["flags"]) == 8 and len(spec["fillers"]) == 24
    assert len({*spec["flags"], *spec["fillers"], spec["marker"]}) == 33
    if family == "gpt2-copy-redundant":
        assert len(key["components"]) == 3
        assert len(check["key_pair_ablated"]) == 3
        assert min(check["key_pair_ablated"].values()) >= 0.85
    else:
        primary, backup = key["components"]
        assert primary["layer"] < backup["layer"]
        assert check["key_head_ablated_alone"][f"{backup['layer']}.{backup['head']}"] >= 0.9
        attention = check["backup_attention_to_flag"]
        assert attention["primary_present"] <= 0.1 and attention["primary_zeroed"] >= 0.5
        assert attention["marker_removed"] >= 0.5
        assert key["marker"]["after_layer"] == primary["layer"]
        assert len(key["marker"]["vector"]) == WIDTH


def test_only_the_planted_heads_change(gpt2_tasks, tiny_gpt2):
    from transformers import AutoModelForCausalLM

    original = AutoModelForCausalLM.from_pretrained(tiny_gpt2).state_dict()
    for built in gpt2_tasks.values():
        key = built["key"]
        changed = AutoModelForCausalLM.from_pretrained(built["dir"] / "model").state_dict()
        assert set(changed) == set(original)
        width, size = WIDTH, WIDTH // 4
        planted_heads = {(p["layer"], p["head"]) for p in key["components"] + key["decoys"]}
        for name, value in changed.items():
            before = original[name]
            if ".attn.c_attn." in name or ".attn.c_proj.weight" in name:
                layer = int(name.split(".")[2])
                mask = before.new_zeros(before.shape, dtype=bool)
                for head in (h for (at, h) in planted_heads if at == layer):
                    if "c_proj" in name:
                        mask[head * size : (head + 1) * size] = True
                    else:
                        for offset in (0, width, 2 * width):
                            start = offset + head * size
                            mask[..., start : start + size] = True
                assert (value[~mask] == before[~mask]).all(), name
                if mask.any() and name.endswith("weight"):
                    assert not (value[mask] == before[mask]).all(), name
            else:
                assert (value == before).all(), name


def test_validation_fails_when_the_named_key_is_not_the_mechanism(gpt2_tasks):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    built = gpt2_tasks["gpt2-copy-redundant"]
    key, root = built["key"], built["dir"]
    check = key["validation"]
    norms = check["norms"]
    facts = {
        "native_accuracy": check["native_accuracy"],
        "word_draw": check["word_draw"],
        "write_size": check["write_size"],
        "norms": {head: {p: v[0] for p, v in parts.items()} for head, parts in norms.items()},
        "layer_norm_ranges": {
            head: {p: (v[1], v[2]) for p, v in parts.items()} for head, parts in norms.items()
        },
    }
    model = AutoModelForCausalLM.from_pretrained(root / "model", attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(root / "model")
    rows = gpt2_planted.make_rows(300, 11, key["row_spec"])

    def plant(components, decoys):
        return gpt2_planted.Gpt2Plant(
            "gpt2-copy-redundant", model, tokenizer, components, decoys, key["row_spec"], facts
        )

    assert gpt2_planted.validate(plant(key["components"], key["decoys"]), rows)["passed"]
    # Naming the decoy and two of the key heads as the key: zeroing them leaves the third.
    wrong = gpt2_planted.validate(plant([key["decoys"][0], *key["components"][:2]], []), rows)
    assert not wrong["passed"]
    assert any("zeroing the key heads" in failure for failure in wrong["failures"])


def claim_of(key, components):
    check = key["validation"]
    return {
        "version": 1,
        "task": key["task"],
        "components": components,
        "ablation": {
            "accuracy_before": check["baseline_accuracy"],
            "accuracy_after": check["key_ablated_accuracy"],
            "rows": 300,
        },
    }


def test_the_key_as_a_claim_scores_one_and_the_decoy_scores_zero(gpt2_tasks, tmp_path):
    for family, built in gpt2_tasks.items():
        key = built["key"]
        result = grade_claim(built["dir"], key, claim_of(key, key["components"]), tmp_path / family)
        assert result["component_f1"] == 1.0 and result["ablation_reproduced"] == 1, result
        assert result["faithful"] is True and result["solved"] is True
        assert result["score"] == 1.0
        assert result["measured"]["baseline_accuracy"] >= 0.9
    built = gpt2_tasks["gpt2-copy-redundant"]
    key = built["key"]
    result = grade_claim(built["dir"], key, claim_of(key, key["decoys"]), tmp_path / "decoy")
    assert result["score"] == 0.0 and result["decoys_named"] == key["decoys"]
    assert result["faithful"] is False


def test_both_families_share_one_task_shape_and_name_neither(gpt2_tasks):
    infos = []
    for built in gpt2_tasks.values():
        info = json.loads((built["dir"] / "task.json").read_text())
        assert info["question"] == gpt2_planted.QUESTION and info["chance"] == CHANCE
        info.pop("id")
        infos.append(info)
    assert infos[0] == infos[1]
    names = [n.encode() for n in [*FAMILIES, *planted.PLANTED_FAMILIES]]
    names += [b"redundant", b"selfrepair", b"backup", b"copy-", b"decoy", b"eval_seed"]
    for family, built in gpt2_tasks.items():
        root = built["dir"]
        rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
        assert len(rows) == 200
        assert all(
            row["options"] == [f" {w}" for w in built["key"]["row_spec"]["flags"]] for row in rows
        )
        for path in files_under(root):
            data = path.read_bytes()
            for name in names:
                assert name not in data, (family, path.name, name)


def test_rows_list_words_and_end_with_the_closing_word():
    spec = {"flags": list("abcdefgh"), "fillers": [f"w{i}" for i in range(24)], "marker": "m"}
    rows = gpt2_planted.make_rows(60, 3, spec)
    assert rows == gpt2_planted.make_rows(60, 3, spec)
    assert rows != gpt2_planted.make_rows(60, 4, spec)
    for row in rows:
        words = row["prompt"].split(", ")
        assert row["prompt"].startswith(" ") and row["prompt"].endswith(", m")
        assert len(words) == gpt2_planted.PROMPT_WORDS + 1
        flags = [i for i, w in enumerate(words) if w.strip() in spec["flags"]]
        assert len(flags) == 1 and flags[0] >= 1
        assert row["answer"] == f" {words[flags[0]].strip()}"
        assert row["options"] == [f" {f}" for f in spec["flags"]]
    assert tasks.make_rows("gpt2-copy-redundant", 60, 3, row_spec=spec) == rows


def test_vocabulary_keeps_single_token_words_only(tiny_gpt2):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_gpt2)
    assert gpt2_planted.vocabulary(tokenizer) == list(gpt2_planted.WORDS)
    assert len(gpt2_planted.WORDS) == len(set(gpt2_planted.WORDS))


def test_plant_refuses_an_unknown_family_and_an_unsuitable_model():
    from transformers import GPT2Config, LlamaConfig

    class Fake:
        def __init__(self, config):
            self.config = config

    with pytest.raises(ValueError, match="unknown GPT-2 planted family"):
        gpt2_planted.plant("copy-head", 0)
    llama = LlamaConfig(hidden_size=16, num_attention_heads=2, num_hidden_layers=6)
    with pytest.raises(ValueError, match="need a GPT-2 model"):
        gpt2_planted._check_model(Fake(llama))
    with pytest.raises(ValueError, match="at least 5 layers"):
        gpt2_planted._check_model(Fake(GPT2Config(n_layer=4, n_embd=64, n_head=4)))
    with pytest.raises(ValueError, match="at least 13 dims"):
        gpt2_planted._check_model(Fake(GPT2Config(n_layer=6, n_embd=64, n_head=8)))


def test_gpt2_families_are_planted_tasks_with_dev_and_heldout_splits(
    tmp_path, tiny_gpt2, gpt2_tasks
):
    assert set(FAMILIES) <= set(tasks.ALL_FAMILIES)
    with pytest.raises(ValueError, match="planted family"):
        build_task("gpt2-copy-redundant", "calibration", 0, tmp_path / "t", tmp_path / "k")
    secret = b"a-held-out-secret-that-is-long-enough-0123456789"
    secret_file = tmp_path / "outside" / "secret.txt"
    secret_file.parent.mkdir()
    secret_file.write_bytes(secret)
    info = build_task(
        "gpt2-copy-selfrepair",
        "heldout",
        2,
        tmp_path / "tasks",
        tmp_path / "keys",
        secret_file=secret_file,
        model_id=str(tiny_gpt2),
    )
    assert info["task"] == tasks.task_id("gpt2-copy-selfrepair", "heldout", 2, secret)
    seed = tasks.task_seed("gpt2-copy-selfrepair", "heldout", 2, secret)
    key = load_key(info["key_path"])
    forbidden = [secret, str(seed).encode(), str(key["eval_seed"]).encode()]
    forbidden.append(json.dumps(key["row_spec"]).encode())
    for path in files_under(Path(info["task_dir"])):
        data = path.read_bytes()
        for needle in forbidden:
            assert needle not in data, (path.name, needle[:8])
    assert secret.decode() not in Path(info["key_path"]).read_text()
    dev = gpt2_tasks["gpt2-copy-selfrepair"]["key"]
    assert (key["components"], key["row_spec"]) != (dev["components"], dev["row_spec"])


def test_plant_is_deterministic_and_leaves_the_callers_rng_alone(gpt2_tasks, tiny_gpt2):
    import torch
    from transformers import AutoModelForCausalLM

    key = gpt2_tasks["gpt2-copy-selfrepair"]["key"]
    seed = tasks.task_seed("gpt2-copy-selfrepair", "dev", 0)
    torch.manual_seed(123)
    state = torch.get_rng_state()
    again = gpt2_planted.plant(
        "gpt2-copy-selfrepair", seed, key["validation"]["attempt"], model_id=str(tiny_gpt2)
    )
    assert torch.equal(torch.get_rng_state(), state)
    assert again.components == key["components"] and again.decoys == key["decoys"]
    assert again.row_spec == key["row_spec"]
    saved = AutoModelForCausalLM.from_pretrained(
        gpt2_tasks["gpt2-copy-selfrepair"]["dir"] / "model"
    )
    pairs = zip(again.model.state_dict().items(), saved.state_dict().values(), strict=True)
    for (name, a), b in pairs:
        assert torch.equal(a, b), name
