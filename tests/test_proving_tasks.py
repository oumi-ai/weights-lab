import hashlib
import hmac
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from oumi_weights_lab.proving import build_task, calibration, load_key, planted, tasks  # noqa: E402

FAMILIES = list(planted.PLANTED_FAMILIES)


def files_under(root):
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("index", [0, 1])
def test_planted_family_builds_and_passes_its_own_validation(family, index, tmp_path):
    info = build_task(family, "dev", index, tmp_path / "tasks", tmp_path / "keys")
    key = load_key(info["key_path"])
    check = key["validation"]
    chance = planted.FAMILIES[family].chance
    assert key["task"] == tasks.task_id(family, "dev", index) == info["task"]
    assert (key["family"], key["split"], key["index"]) == (family, "dev", index)
    assert key["chance"] == chance
    assert check["baseline_accuracy"] >= 0.95
    assert check["key_ablated_accuracy"] <= chance + 0.15
    assert check["largest_other_effect"] <= 0.05
    assert all(abs(effect) <= 0.05 for effect in check["decoy_effect"].values())
    assert key["decoys"] and key["components"]
    assert check["rows"] == 400
    assert 0 <= check["attempt"] < tasks.ATTEMPTS
    if family in ("copy-backup", "copy-redundant", "copy-selfrepair"):
        wanted = 3 if family == "copy-redundant" else 2
        assert len(key["components"]) == wanted
        assert len({c["layer"] for c in key["components"]}) == wanted
        alone = check["key_head_ablated_alone"]
        assert min(alone.values()) >= 0.85
    if family == "copy-redundant":
        assert len(check["key_pair_ablated"]) == 3
        assert min(check["key_pair_ablated"].values()) >= 0.85
    if family == "copy-selfrepair":
        primary, backup = key["components"]
        assert primary["layer"] < backup["layer"]
        assert check["key_head_ablated_alone"][f"{backup['layer']}.{backup['head']}"] >= 0.95
    if family in ("lookup-mlp", "feature-direction"):
        assert key["neurons"][0]["layer"] == key["components"][0]["layer"]
    if family == "feature-direction":
        assert check["direction_ablated_accuracy"] <= 0.65
        vector = key["directions"][0]["vector"]
        assert len(vector) == 64
        assert sum(x * x for x in vector) == pytest.approx(1.0)
        assert key["directions"][0]["after_layer"] == key["components"][0]["layer"]
        assert key["decoy_directions"][0]["after_layer"] == key["decoys"][0]["layer"]


def test_task_directory_layout_and_public_contents(proving_tasks):
    for family, built in proving_tasks.items():
        root = built["dir"]
        info = json.loads((root / "task.json").read_text())
        assert set(info) == {
            "id",
            "split",
            "kind",
            "question",
            "rows",
            "options_note",
            "metric",
            "chance",
            "claim_version",
        }
        assert info["id"] == tasks.task_id(family, "dev", 0)
        assert info["kind"] == "claim" and info["metric"] == "accuracy"
        assert info["claim_version"] == 1 and info["rows"] == "rows.jsonl"
        rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
        assert len(rows) == 200
        assert all(set(row) == {"prompt", "options", "answer"} for row in rows)
        assert all(row["answer"] in row["options"] for row in rows)
        assert (root / "model" / "config.json").exists()
        assert (root / "model" / "tokenizer.json").exists()
        # Nothing but data in the public directory: no key file, no seed.
        assert not [p for p in files_under(root) if "key" in p.name]
        for path in files_under(root):
            if path.suffix in {".json", ".jsonl"}:
                text = path.read_text()
                assert "eval_seed" not in text and "decoy" not in text


def test_same_seed_builds_identical_files_and_different_seeds_plant_elsewhere(tmp_path):
    first = build_task("copy-head", "dev", 3, tmp_path / "a", tmp_path / "ka")
    second = build_task("copy-head", "dev", 3, tmp_path / "b", tmp_path / "kb")
    a, b = Path(first["task_dir"]), Path(second["task_dir"])
    assert [p.relative_to(a) for p in files_under(a)] == [p.relative_to(b) for p in files_under(b)]
    for path in files_under(a):
        assert path.read_bytes() == (b / path.relative_to(a)).read_bytes(), path.name
    assert Path(first["key_path"]).read_bytes() == Path(second["key_path"]).read_bytes()
    for family in FAMILIES:
        places = {json.dumps(planted.plant(family, seed).components) for seed in range(4)}
        assert len(places) > 1, family


def test_builders_leave_the_callers_rng_alone(tmp_path):
    import torch

    torch.manual_seed(123)
    state = torch.get_rng_state()
    planted.plant("copy-head", 5)
    assert torch.equal(torch.get_rng_state(), state)


def test_seeds_follow_the_published_formulas():
    expected = int.from_bytes(hashlib.sha256(b"copy-head:dev:2").digest()[:8], "big")
    assert tasks.task_seed("copy-head", "dev", 2) == expected
    secret = b"s" * 40
    digest = hmac.new(secret, b"lookup-mlp:heldout:7", hashlib.sha256).digest()
    assert tasks.task_seed("lookup-mlp", "heldout", 7, secret) == int.from_bytes(digest[:8], "big")
    assert tasks.task_seed("lookup-mlp", "heldout", 7, secret) != tasks.task_seed(
        "lookup-mlp", "dev", 7
    )
    digest = hashlib.sha256(b"id:copy-head:dev:3").hexdigest()[:10]
    assert tasks.task_id("copy-head", "dev", 3) == f"dev-{digest}"
    mac = hmac.new(secret, b"id:copy-head:heldout:3", hashlib.sha256).hexdigest()[:10]
    assert tasks.task_id("copy-head", "heldout", 3, secret) == f"heldout-{mac}"
    assert tasks.task_id("copy-head", "heldout", 3, b"t" * 40) != f"heldout-{mac}"
    assert tasks.task_id("gpt2-ioi", "calibration", 0).startswith("calibration-")
    ids = {tasks.task_id(f, "dev", i) for f in FAMILIES for i in range(3)}
    assert len(ids) == 3 * len(FAMILIES)
    with pytest.raises(ValueError, match="needs the secret"):
        tasks.task_id("copy-head", "heldout", 3)
    with pytest.raises(ValueError, match="no seed"):
        tasks.task_seed("gpt2-ioi", "calibration", 0)
    with pytest.raises(ValueError, match="needs the secret"):
        tasks.task_seed("copy-head", "heldout", 0)
    with pytest.raises(ValueError, match="needs the secret"):
        tasks.task_seed("copy-head", "heldout", 0, b"short")


def test_heldout_build_never_writes_the_secret_the_seed_or_the_key(tmp_path):
    secret = b"a-held-out-secret-that-is-long-enough-0123456789\n"
    secret_file = tmp_path / "outside" / "secret.txt"
    secret_file.parent.mkdir()
    secret_file.write_bytes(secret)
    info = build_task(
        "lookup-mlp",
        "heldout",
        3,
        tmp_path / "tasks",
        tmp_path / "keys",
        secret_file=secret_file,
    )
    assert info["task"] == tasks.task_id("lookup-mlp", "heldout", 3, secret.strip())
    seed = tasks.task_seed("lookup-mlp", "heldout", 3, secret.strip())
    key = load_key(info["key_path"])
    forbidden = [
        secret.strip(),
        str(seed).encode(),
        hex(seed)[2:].encode(),
        seed.to_bytes(8, "big"),
        str(key["eval_seed"]).encode(),
        json.dumps(key["row_spec"]).encode(),
    ]
    scanned = files_under(Path(info["task_dir"]))
    assert len(scanned) >= 4
    for path in scanned:
        data = path.read_bytes()
        for needle in forbidden:
            assert needle not in data, (path.name, needle[:8])
    # The key and the result do not hold the secret or the seed either.
    for text in (Path(info["key_path"]).read_text(), json.dumps(info)):
        assert secret.strip().decode() not in text and str(seed) not in text


def test_build_task_refuses_unsafe_or_unsupported_requests(tmp_path):
    good_secret = tmp_path / "secret"
    good_secret.write_bytes(b"x" * 32 + b"\n")
    short_secret = tmp_path / "short"
    short_secret.write_bytes(b"y" * 31 + b"\n\n")
    out, keys = tmp_path / "out", tmp_path / "keys"
    with pytest.raises(ValueError, match="at least 32 bytes"):
        build_task("copy-head", "heldout", 0, out, keys, secret_file=short_secret)
    with pytest.raises(ValueError, match="needs a secret file"):
        build_task("copy-head", "heldout", 0, out, keys)
    with pytest.raises(ValueError, match="public"):
        build_task("copy-head", "dev", 0, out, keys, secret_file=good_secret)
    with pytest.raises(ValueError, match="inside the task directory"):
        build_task("copy-head", "dev", 0, out, out / tasks.task_id("copy-head", "dev", 0) / "keys")
    with pytest.raises(ValueError, match="inside the task directory"):
        build_task("copy-head", "dev", 0, out, out / tasks.task_id("copy-head", "dev", 0))
    secret = b"z" * 40
    inside = out / tasks.task_id("copy-head", "heldout", 0, secret) / "secret"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(secret)
    with pytest.raises(ValueError, match="secret file must not be inside"):
        build_task("copy-head", "heldout", 0, out, keys, secret_file=inside)
    with pytest.raises(ValueError, match="calibration family"):
        build_task("gpt2-ioi", "dev", 0, out, keys)
    with pytest.raises(ValueError, match="planted family"):
        build_task("copy-head", "calibration", 0, out, keys)
    with pytest.raises(ValueError, match="unknown split"):
        build_task("copy-head", "train", 0, out, keys)
    with pytest.raises(ValueError, match="unknown family"):
        build_task("nope", "dev", 0, out, keys)
    with pytest.raises(ValueError, match="non-negative"):
        build_task("copy-head", "dev", -1, out, keys)
    assert not (out / tasks.task_id("copy-head", "dev", 0)).exists() and not keys.exists()
    build_task("copy-head", "dev", 0, out, keys)
    with pytest.raises(ValueError, match="already exists"):
        build_task("copy-head", "dev", 0, out, keys)


def test_a_model_that_never_validates_raises_after_twenty_attempts(tmp_path, monkeypatch):
    calls = []
    real = planted.validate_plant

    def failing(result, rows):
        calls.append(1)
        return {**real(result, rows), "passed": False, "failures": ["forced"]}

    monkeypatch.setattr(planted, "validate_plant", failing)
    with pytest.raises(RuntimeError, match="20 attempts"):
        build_task("copy-head", "dev", 0, tmp_path / "out", tmp_path / "keys")
    assert len(calls) == 20
    assert not (tmp_path / "out" / tasks.task_id("copy-head", "dev", 0)).exists()


def test_saved_task_loads_in_a_lab_and_reaches_the_validated_baseline(proving_tasks, tmp_path):
    from oumi_weights_lab.lab import load_lab

    for family, built in proving_tasks.items():
        lab = load_lab(
            {"model": str(built["dir"] / "model"), "device": "cpu", "dtype": "float32"},
            tmp_path / family,
        )
        run = lab.evaluate(str(built["dir"] / "rows.jsonl"), name="baseline")
        rows = [json.loads(line) for line in (built["dir"] / "rows.jsonl").read_text().splitlines()]
        plain = planted.option_accuracy(lab.model, lab.tokenizer, rows)["accuracy"]
        assert run["accuracy"] == plain
        assert run["accuracy"] >= built["key"]["validation"]["baseline_accuracy"] - 0.05, family
        assert lab.model.config._attn_implementation == "eager"


def test_rows_can_be_regenerated_from_the_key(proving_tasks):
    for family, built in proving_tasks.items():
        key = built["key"]
        one = tasks.make_rows(family, 50, key["eval_seed"], row_spec=key["row_spec"])
        two = tasks.make_rows(family, 50, key["eval_seed"], row_spec=key["row_spec"])
        public = [json.loads(x) for x in (built["dir"] / "rows.jsonl").read_text().splitlines()]
        assert one == two
        assert one != public[:50]


def test_lookup_and_feature_rows_follow_the_hidden_maps(proving_tasks):
    lookup = proving_tasks["lookup-mlp"]["key"]["row_spec"]["attributes"]
    assert sorted(lookup) == sorted([i % 6 for i in range(12)])
    rows = planted.make_rows("lookup-mlp", 30, 1, {"attributes": lookup})
    for row in rows:
        subject = int(row["prompt"].split()[-1][1:])
        assert row["answer"] == f" a{lookup[subject]}"
    feature = proving_tasks["feature-direction"]["key"]["row_spec"]["yes_words"]
    assert len(feature) == 8
    for row in planted.make_rows("feature-direction", 30, 1, {"yes_words": feature}):
        word = int(row["prompt"].split()[-1][1:])
        assert row["answer"] == (" yes" if word in feature else " no")


# Calibration families: row generators and the key procedure, on a tiny GPT-2.

NAMES = calibration._NAMES[:10]
WORDS = [f"{a}{b}" for a in "xy" for b in "abcdefghijklmnopqrstuvwxyz"]


@pytest.fixture
def tiny_gpt2(tmp_path):
    import tokenizers
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = ["<unk>", "<eos>", "Then", ",", ".", "and", "went", "to", "the", "gave", "a"]
    vocab += list(calibration._PLACES) + list(calibration._OBJECTS) + NAMES + WORDS
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=64,
            n_embd=16,
            n_layer=2,
            n_head=2,
            eos_token_id=1,
            bos_token_id=1,
            pad_token_id=1,
            attn_implementation="eager",
        )
    ).eval()
    path = tmp_path / "tiny-gpt2"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return model, tokenizer, path


def test_ioi_rows_use_both_templates_and_name_the_indirect_object(tiny_gpt2):
    _, tokenizer, _ = tiny_gpt2
    rows = calibration.ioi_rows(tokenizer, 40, seed=5)
    assert rows == calibration.ioi_rows(tokenizer, 40, seed=5)
    assert rows != calibration.ioi_rows(tokenizer, 40, seed=6)
    orders = set()
    for row in rows:
        words = row["prompt"].replace(",", " ,").replace(".", " .").split()
        names = [w for w in words if w in NAMES]
        assert row["prompt"].endswith(" to") and len(names) == 3
        io = row["answer"].strip()
        assert row["options"] == [row["answer"], row["options"][1]]
        subject = row["options"][1].strip()
        assert io != subject and names.count(subject) == 2 and names.count(io) == 1
        assert names[2] == subject  # the giver is the repeated name
        orders.add(names[0] == io)  # ABBA starts with the indirect object, BABA with the subject
    assert orders == {True, False}
    # A name that is not one token is skipped, never used.
    assert {n for r in rows for n in (r["answer"].strip(),)} <= set(NAMES)


def test_ioi_rows_refuse_a_tokenizer_with_no_names(tiny_gpt2, tiny_tokenizer):
    with pytest.raises(ValueError, match="fewer than two single-token names"):
        calibration.ioi_rows(tiny_tokenizer, 5, seed=0)


def test_induction_rows_repeat_a_run_and_ask_for_the_next_word(tiny_gpt2):
    _, tokenizer, _ = tiny_gpt2
    rows = calibration.induction_rows(tokenizer, 30, seed=2)
    assert rows == calibration.induction_rows(tokenizer, 30, seed=2)
    for row in rows:
        words = row["prompt"].split()
        run, repeat = words[:20], words[20:]
        assert len(set(run)) == 20 and 5 <= len(repeat) <= 15
        assert repeat == run[: len(repeat)]
        assert row["answer"].strip() == run[len(repeat)]
        assert len(row["options"]) == 4 and len(set(row["options"])) == 4
        assert row["answer"] in row["options"]
        assert all(option.strip() in run for option in row["options"])
        for option in row["options"]:
            assert len(tokenizer.encode(option, add_special_tokens=False)) == 1


def test_induction_scores_match_a_direct_computation(tiny_gpt2):
    import torch

    model, _, _ = tiny_gpt2
    length, count = 12, 6
    scores = calibration.induction_scores(
        model, sequences=count, length=length, seed=4, batch_size=4
    )
    assert len(scores) == 2 and all(len(row) == 2 for row in scores)
    generator = torch.Generator().manual_seed(4)
    tokens = torch.randint(0, model.config.vocab_size, (count, length), generator=generator)
    tokens = torch.cat([tokens, tokens], dim=1)
    with torch.no_grad():
        attentions = model(tokens, output_attentions=True).attentions
    for layer in range(2):
        for head in range(2):
            total = 0.0
            for b in range(count):
                for p in range(length, 2 * length - 1):
                    total += float(attentions[layer][b, head, p, p - length + 1])
            expected = total / (count * (length - 1))
            assert scores[layer][head] == pytest.approx(expected, abs=1e-5)
    assert all(0 <= s <= 1 for row in scores for s in row)


def test_induction_components_pick_heads_at_or_above_the_threshold():
    scores = [[0.1, 0.4], [0.39, 0.9]]
    assert calibration.induction_components(scores) == [
        {"kind": "head", "layer": 0, "head": 1},
        {"kind": "head", "layer": 1, "head": 1},
    ]
    assert calibration.induction_components(scores, 0.95) == []


def test_ioi_key_lists_the_26_published_heads():
    heads = calibration.ioi_components()
    assert len(heads) == 26 and len({(h["layer"], h["head"]) for h in heads}) == 26
    assert {"kind": "head", "layer": 9, "head": 9} in heads
    assert {"kind": "head", "layer": 11, "head": 10} in heads
    assert all(0 <= h["layer"] < 12 and 0 <= h["head"] < 12 for h in heads)


def test_calibration_task_builds_from_a_local_model_without_a_seed(
    tiny_gpt2, tmp_path, monkeypatch
):
    _, _, path = tiny_gpt2
    monkeypatch.setattr(calibration, "INDUCTION_THRESHOLD", 0.0)
    monkeypatch.setattr(calibration, "INDUCTION_LENGTH", 12)
    monkeypatch.setattr(calibration, "INDUCTION_SEQUENCES", 8)
    info = build_task(
        "gpt2-induction", "calibration", 0, tmp_path / "t", tmp_path / "k", model_id=str(path)
    )
    key = load_key(info["key_path"])
    assert key["task"] == tasks.task_id("gpt2-induction", "calibration", 0)
    assert key["chance"] == 0.25 and key["split"] == "calibration"
    assert len(key["components"]) == 4  # threshold 0 keeps every head of the 2x2 model
    assert key["validation"]["rows"] == 400 and "procedure" in key["validation"]
    root = tmp_path / "t" / key["task"]
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 200 and (root / "model" / "config.json").exists()
    with pytest.raises(ValueError, match="GPT-2 small"):
        build_task("gpt2-ioi", "calibration", 0, tmp_path / "t", tmp_path / "k", model_id=str(path))
    assert not (tmp_path / "t" / tasks.task_id("gpt2-ioi", "calibration", 0)).exists()


def test_task_json_and_files_do_not_reveal_the_family(tmp_path):
    built = [
        build_task(family, "dev", 0, tmp_path / family, tmp_path / f"k-{family}")
        for family in planted.COPY_FAMILIES
    ]
    infos = [json.loads((Path(b["task_dir"]) / "task.json").read_text()) for b in built]
    assert len({info["id"] for info in infos}) == len(infos)
    for info in infos:
        info.pop("id")
    assert all(info == infos[0] for info in infos) and "family" not in infos[0]
    assert all(b["task"].startswith("dev-") and len(b["task"]) == len("dev-") + 10 for b in built)
    for built_one in built:  # the same model shape too, so the config does not hint
        config = json.loads((Path(built_one["task_dir"]) / "model" / "config.json").read_text())
        assert (config["n_layer"], config["n_head"], config["n_embd"], config["n_inner"]) == (
            6,
            4,
            64,
            256,
        )


def test_no_file_in_a_task_directory_contains_a_family_name(proving_tasks):
    names = [
        name.encode() for name in [*planted.PLANTED_FAMILIES, *calibration.CALIBRATION_FAMILIES]
    ]
    names += [b"backup", b"copy-", b"lookup"]
    for built in proving_tasks.values():
        for path in files_under(built["dir"]):
            data = path.read_bytes()
            for name in names:
                assert name not in data, (path.name, name)
        assert not any(name.decode() in built["dir"].name for name in names)
