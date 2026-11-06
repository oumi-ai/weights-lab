import pytest
import torch

from oumi_weights_lab import BudgetExceeded

KINDS = ["gpt2", "llama_gqa", "qwen3_5"]
BLOCKS = {
    "gpt2": "transformer.h.{}",
    "llama_gqa": "model.layers.{}",
    "qwen3_5": "model.layers.{}",
}
PAIRS = [
    ("hello yes", "hello no"),
    ("world yes", "world no"),
    ("yes yes", "yes no"),
    ("hello world yes", "hello world no"),
    ("world hello yes", "world hello no"),
    ("no no yes", "no no no"),
    ("hello hello hello yes", "hello hello hello no"),
    ("world world yes", "world world no"),
]


def lab_for(kind, tiny_model, make_lab, plant=None, **overrides):
    model, tokenizer = tiny_model(kind, **overrides)
    if plant is not None:
        with torch.no_grad():
            model.get_input_embeddings().weight[tokenizer.convert_tokens_to_ids("yes")] += plant
    return make_lab(model, tokenizer), model, tokenizer


def last_token_vector(lab, module, text, at="output"):
    ids = torch.tensor([lab.tokenizer.encode(text, add_special_tokens=False)])
    with torch.no_grad(), lab.record(module, at=at) as captured:
        lab.model(input_ids=ids)
    return captured[module][0][0, 0].float()


@pytest.mark.parametrize("kind", KINDS)
def test_mean_diff_matches_manual_computation(kind, tiny_model, make_lab):
    lab, _, _ = lab_for(kind, tiny_model, make_lab)
    rows = lab.contrast_directions(PAIRS, held_out=0.25, seed=3)
    assert [row["layer"] for row in rows] == [0, 1]
    for row in rows:
        module = BLOCKS[kind].format(row["layer"])
        assert row["module"] == module
        assert (row["at"], row["method"]) == ("output", "mean_diff")
        diffs = torch.stack(
            [
                last_token_vector(lab, module, PAIRS[i][0])
                - last_token_vector(lab, module, PAIRS[i][1])
                for i in row["train_pairs"]
            ]
        )
        mean = diffs.mean(0)
        assert row["direction"].dtype == torch.float32 and row["direction"].device.type == "cpu"
        assert abs(row["norm"] - mean.norm().item()) < 1e-6
        assert torch.allclose(row["direction"], mean / mean.norm(), atol=1e-6)
        assert abs(row["direction"].norm().item() - 1) < 1e-6
        assert row["explained_variance"] is None


def test_input_site_and_layer_subset(tiny_model, make_lab):
    lab, _, _ = lab_for("llama_gqa", tiny_model, make_lab)
    rows = lab.contrast_directions(PAIRS, layers=[1], at="input", held_out=0)
    assert [row["layer"] for row in rows] == [1] and rows[0]["at"] == "input"
    diffs = torch.stack(
        [
            last_token_vector(lab, "model.layers.1", p, at="input")
            - last_token_vector(lab, "model.layers.1", n, at="input")
            for p, n in PAIRS
        ]
    )
    assert torch.allclose(rows[0]["direction"] * rows[0]["norm"], diffs.mean(0), atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_pca_direction_is_unit_signed_and_reports_variance(kind, tiny_model, make_lab):
    lab, _, _ = lab_for(kind, tiny_model, make_lab)
    mean_rows = lab.contrast_directions(PAIRS, seed=1)
    pca_rows = lab.contrast_directions(PAIRS, method="pca", seed=1)
    for mean_row, row in zip(mean_rows, pca_rows, strict=True):
        assert row["method"] == "pca"
        assert row["train_pairs"] == mean_row["train_pairs"]
        assert abs(row["direction"].norm().item() - 1) < 1e-5
        assert 0 <= row["explained_variance"] <= 1
        assert row["norm"] == pytest.approx(mean_row["norm"], abs=1e-6)
        assert torch.dot(row["direction"], mean_row["direction"]).item() >= 0


def test_pca_matches_a_manual_decomposition(tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    row = lab.contrast_directions(PAIRS, layers=[1], method="pca", held_out=0)[0]
    module = "transformer.h.1"
    diffs = torch.stack(
        [last_token_vector(lab, module, p) - last_token_vector(lab, module, n) for p, n in PAIRS]
    ).double()
    _, singular, vt = torch.linalg.svd(diffs, full_matrices=False)
    expected = vt[0] if torch.dot(vt[0], diffs.mean(0)) >= 0 else -vt[0]
    assert torch.allclose(row["direction"].double(), expected, atol=1e-4)
    # The share of the uncentered second moment, not of the variance around the mean.
    share = (singular[0] ** 2 / (singular**2).sum()).item()
    assert row["explained_variance"] == pytest.approx(share, abs=1e-5)
    assert share == pytest.approx(
        (diffs @ expected).pow(2).sum().item() / diffs.pow(2).sum().item()
    )


def test_pca_of_all_zero_differences_is_the_zero_direction():
    import oumi_weights_lab.directions as directions

    direction, norm, share = directions._direction(torch.zeros(4, 16), "pca")
    assert torch.equal(direction, torch.zeros(16)) and norm == 0 and share is None


def test_split_is_by_pair_deterministic_and_disjoint(tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    first = lab.contrast_directions(PAIRS, layers=[0], held_out=0.25, seed=5)[0]
    again = lab.contrast_directions(PAIRS, layers=[0], held_out=0.25, seed=5)[0]
    assert first["train_pairs"] == again["train_pairs"]
    assert first["heldout_pairs"] == again["heldout_pairs"]
    assert torch.equal(first["direction"], again["direction"])
    train, held = set(first["train_pairs"]), set(first["heldout_pairs"])
    assert not train & held and train | held == set(range(len(PAIRS)))
    assert len(held) == 2
    splits = {
        tuple(lab.contrast_directions(PAIRS, layers=[0], seed=seed)[0]["heldout_pairs"])
        for seed in range(6)
    }
    assert len(splits) > 1


def test_no_held_out_pairs_gives_no_held_out_metrics(tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    row = lab.contrast_directions(PAIRS, layers=[0], held_out=0)[0]
    assert row["heldout_pairs"] == [] and row["train_pairs"] == list(range(len(PAIRS)))
    for key in ("heldout_pair_accuracy", "heldout_auc", "heldout_effect_size"):
        assert row[key] is None


def test_single_held_out_pair_has_no_effect_size(tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    row = lab.contrast_directions(PAIRS[:4], layers=[0], held_out=0.25)[0]
    assert len(row["heldout_pairs"]) == 1
    assert row["heldout_pair_accuracy"] in (0.0, 1.0)
    assert row["heldout_auc"] in (0.0, 0.5, 1.0)
    assert row["heldout_effect_size"] is None


@pytest.mark.parametrize("method", ["mean_diff", "pca"])
@pytest.mark.parametrize("kind", KINDS)
def test_planted_direction_is_recovered_and_separates_held_out_pairs(
    kind, method, tiny_model, make_lab
):
    generator = torch.Generator().manual_seed(7)
    planted = torch.randn(16, generator=generator)
    planted = planted / planted.norm()
    lab, _, _ = lab_for(kind, tiny_model, make_lab, plant=planted * 30)
    row = lab.contrast_directions(PAIRS, layers=[0], method=method, seed=2)[0]
    assert torch.dot(row["direction"], planted).item() > 0.9
    assert row["heldout_pair_accuracy"] == 1.0
    assert row["heldout_auc"] == 1.0
    assert row["heldout_effect_size"] > 3


def test_scrambled_labels_do_not_separate_held_out_pairs(tiny_model, make_lab):
    generator = torch.Generator().manual_seed(1)
    planted = torch.randn(16, generator=generator)
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab, plant=planted * 30)
    # Half the pairs are flipped, so "yes" is positive in some pairs and negative in others.
    pairs = [pair if index % 2 else pair[::-1] for index, pair in enumerate(PAIRS)]
    rows = [
        lab.contrast_directions(pairs, layers=[0], held_out=0.5, seed=seed)[0] for seed in range(4)
    ]
    assert min(row["heldout_pair_accuracy"] for row in rows) < 1.0


@pytest.mark.parametrize("kind", KINDS)
def test_passes_count_two_per_pair_and_padding_changes_nothing(kind, tiny_model, make_lab):
    lab, _, _ = lab_for(kind, tiny_model, make_lab)
    before = lab.forward_passes
    wide = lab.contrast_directions(PAIRS, batch_size=8, method="pca", seed=4)
    assert lab.forward_passes - before == 2 * len(PAIRS)
    narrow = lab.contrast_directions(PAIRS, batch_size=1, method="pca", seed=4)
    assert lab.forward_passes - before == 4 * len(PAIRS)
    for a, b in zip(wide, narrow, strict=True):
        assert torch.allclose(a["direction"], b["direction"], atol=1e-5)
        assert a["norm"] == pytest.approx(b["norm"], abs=1e-5)
        assert a["heldout_pairs"] == b["heldout_pairs"]
        assert a["heldout_auc"] == b["heldout_auc"]


def test_pass_cap_is_checked_before_any_pass_runs(tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    before = lab.forward_passes
    with lab.limit_passes(2 * len(PAIRS) - 1), pytest.raises(BudgetExceeded):
        lab.contrast_directions(PAIRS)
    assert lab.forward_passes == before
    with lab.limit_passes(2 * len(PAIRS)):
        lab.contrast_directions(PAIRS)


def test_hooks_modes_and_rng_are_restored_even_after_an_error(tiny_model, make_lab):
    lab, model, _ = lab_for("gpt2", tiny_model, make_lab)
    counter = len(model._forward_pre_hooks)
    model.train()
    state = torch.get_rng_state()
    lab.contrast_directions(PAIRS)
    assert model.training and torch.equal(torch.get_rng_state(), state)
    # One-token prompts cannot supply position -2; it would be left padding.
    with pytest.raises(ValueError, match="at least 2 tokens"):
        lab.contrast_directions(
            [("yes", "no"), ("hello", "world"), ("world", "yes"), ("no", "hello")], positions=-2
        )

    def fail(*args, **kwargs):
        raise RuntimeError("forward failed")

    handle = model.transformer.h[1].register_forward_hook(fail)
    with pytest.raises(RuntimeError, match="forward failed"):
        lab.contrast_directions(PAIRS)
    handle.remove()
    assert model.training
    assert len(model._forward_pre_hooks) == counter
    assert not any(module._forward_hooks for module in model.modules())
    assert not any(module._forward_pre_hooks for module in model.transformer.modules())
    assert lab.patches() == []


def test_dict_pairs_work(tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    as_dicts = [{"positive": p, "negative": n} for p, n in PAIRS]
    a = lab.contrast_directions(as_dicts, layers=[0])[0]
    b = lab.contrast_directions(PAIRS, layers=[0])[0]
    assert torch.equal(a["direction"], b["direction"])


def test_second_to_last_position(tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    rows = lab.contrast_directions(PAIRS, layers=[0], positions=-2, held_out=0)
    assert rows[0]["direction"].shape == (16,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pairs": PAIRS[:3]}, "at least 4"),
        ({"pairs": PAIRS[:3] + [PAIRS[0]]}, "duplicate"),
        ({"pairs": PAIRS[:3] + [("", "no")]}, "empty"),
        ({"pairs": PAIRS[:3] + [("   ", "no")]}, "empty"),
        ({"pairs": PAIRS[:3] + [("yes", "yes")]}, "differ"),
        ({"pairs": PAIRS[:3] + [("yes", 4)]}, "text"),
        ({"pairs": PAIRS[:3] + [("yes",)]}, "pair"),
        ({"pairs": PAIRS[:3] + [{"positive": "yes"}]}, "positive"),
        ({"positions": 0}, "padding"),
        ({"positions": [-1]}, "padding"),
        ({"positions": "all"}, "padding"),
        ({"positions": True}, "padding"),
        ({"layers": [5]}, "layer"),
        ({"layers": [0, 0]}, "layer"),
        ({"layers": []}, "layer"),
        ({"held_out": 0.6}, "held_out"),
        ({"held_out": -0.1}, "held_out"),
        ({"held_out": True}, "held_out"),
        ({"method": "svd"}, "method"),
        ({"at": "attention"}, "at must"),
        ({"batch_size": 0}, "batch_size"),
        ({"seed": 1.5}, "seed"),
    ],
)
def test_refusals(kwargs, message, tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    before = lab.forward_passes
    arguments = {"pairs": PAIRS, **kwargs}
    with pytest.raises(ValueError, match=message):
        lab.contrast_directions(arguments.pop("pairs"), **arguments)
    assert lab.forward_passes == before


def test_a_model_without_numbered_layers_is_refused(make_lab, tiny_tokenizer):
    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(6, 4)

        def forward(self, input_ids, attention_mask=None):
            return self.embed(input_ids)

    lab = make_lab(Plain(), tiny_tokenizer)
    with pytest.raises(ValueError, match="decoder layers"):
        lab.contrast_directions(PAIRS, batch_size=1)


def test_direction_feeds_project_out_and_steering(tiny_model, make_lab):
    planted = torch.zeros(16)
    planted[3] = 1
    lab, model, tokenizer = lab_for("llama_gqa", tiny_model, make_lab, plant=planted * 30)
    row = lab.contrast_directions(PAIRS, layers=[0])[0]
    direction = row["direction"]
    name = "model.layers.0.mlp.down_proj.weight"
    before = model.get_parameter(name).detach().clone()
    lab.project_out(name, direction)
    after = model.get_parameter(name).detach()
    assert (direction @ after).abs().max() < 1e-5
    lab.undo()
    assert torch.equal(model.get_parameter(name), before)
    ids = torch.tensor([tokenizer.encode("hello world", add_special_tokens=False)])
    with torch.no_grad():
        plain = model(input_ids=ids).logits
        with lab.patch(
            row["module"], lambda v: v + 4 * row["norm"] * direction.to(v.device), positions="all"
        ):
            steered = model(input_ids=ids).logits
    assert not torch.allclose(plain, steered)


SHARED_NEGATIVE = [
    ("hello yes", "no"),
    ("world yes", "no"),
    ("yes yes", "no"),
    ("hello world yes", "no"),
    ("world hello yes", "world hello no"),
    ("no no yes", "no no no"),
    ("hello hello hello yes", "hello hello hello no"),
    ("world world yes", "world world no"),
]


@pytest.mark.parametrize("seed", range(8))
def test_pairs_that_share_a_prompt_stay_on_one_side_of_the_split(seed, tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    row = lab.contrast_directions(SHARED_NEGATIVE, layers=[0], held_out=0.25, seed=seed)[0]
    group = {0, 1, 2, 3}
    assert group <= set(row["train_pairs"]) or group <= set(row["heldout_pairs"])
    assert row["pair_groups"] == 5
    assert row["heldout_pairs"]


def test_a_chain_of_shared_prompts_is_one_group(tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    # Pair 1 reuses pair 0's positive and pair 2 reuses pair 1's negative.
    chain = [
        ("hello yes", "hello no"),
        ("hello yes", "world no"),
        ("world yes", "world no"),
        ("yes yes", "yes no"),
        ("no no yes", "no no no"),
    ]
    for seed in range(8):
        row = lab.contrast_directions(chain, layers=[0], held_out=0.4, seed=seed)[0]
        assert row["pair_groups"] == 3
        sides = [set(row["train_pairs"]), set(row["heldout_pairs"])]
        assert any({0, 1, 2} <= side for side in sides)


def test_a_pair_that_also_appears_reversed_is_refused(tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    before = lab.forward_passes
    with pytest.raises(ValueError, match="reversed"):
        lab.contrast_directions(PAIRS[:3] + [(PAIRS[0][1], PAIRS[0][0])])
    assert lab.forward_passes == before


def test_one_group_cannot_be_split_but_can_train_whole(tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    pairs = [(f"{word} yes", "no") for word in ("hello", "world", "yes", "no")]
    before = lab.forward_passes
    with pytest.raises(ValueError, match="share prompts|one group"):
        lab.contrast_directions(pairs)
    assert lab.forward_passes == before
    row = lab.contrast_directions(pairs, layers=[0], held_out=0)[0]
    assert row["pair_groups"] == 1 and row["heldout_pairs"] == []


def test_at_least_half_the_pairs_train(tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    for seed in range(6):
        row = lab.contrast_directions(PAIRS[:5], layers=[0], held_out=0.5, seed=seed)[0]
        assert len(row["heldout_pairs"]) == 2 and len(row["train_pairs"]) == 3
    # Groups of 3 and 2 pairs: only the smaller fits in the cap of 2.
    pairs = [("hello yes", "hello no"), ("hello yes", "world no"), ("world yes", "world no")] + [
        ("yes yes", "yes no"),
        ("no no yes", "no no no"),
    ]
    for seed in range(6):
        row = lab.contrast_directions(pairs, layers=[0], held_out=0.5, seed=seed)[0]
        assert len(row["train_pairs"]) >= 3 and row["heldout_pairs"]


def test_rows_do_not_share_split_lists(tiny_model, make_lab):
    lab, _, _ = lab_for("gpt2", tiny_model, make_lab)
    first, second = lab.contrast_directions(PAIRS)
    assert first["train_pairs"] == second["train_pairs"]
    assert first["train_pairs"] is not second["train_pairs"]
    assert first["heldout_pairs"] is not second["heldout_pairs"]
    first["train_pairs"].clear()
    assert second["train_pairs"]
