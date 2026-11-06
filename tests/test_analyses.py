"""Circuit analyses: the identities each analysis must satisfy, on every supported layout."""

import math

import pytest
import torch

KINDS = ["gpt2", "llama_gqa", "qwen3_5"]
CLEAN = ["hello world yes no hello", "world hello no yes world"]
CORRUPT = ["hello hello yes no hello", "world world no yes world"]
PAIR = ("yes", "no")
YES, NO = 4, 5

# Names of the norms around a block, by fixture, for hand computations.
NORMS = {
    "gpt2": {
        "final": "transformer.ln_f",
        "mlp": "transformer.h.{}.ln_2",
        "attn": "transformer.h.{}.ln_1",
    },
    "llama_gqa": {
        "final": "model.norm",
        "mlp": "model.layers.{}.post_attention_layernorm",
        "attn": "model.layers.{}.input_layernorm",
    },
}
LAYERS = {"gpt2": "transformer.h.{}", "llama_gqa": "model.layers.{}"}


def margin(logits):
    return (logits[:, YES] - logits[:, NO]).mean()


def strengthen(model):
    """Larger weights than the default init, so that metrics differ by far more than 1e-5."""
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.mul_(6)
    return model


def make(kind, tiny_model, make_lab, seed=0, **overrides):
    model, tokenizer = tiny_model(kind, seed=seed, **overrides)
    return make_lab(strengthen(model), tokenizer)


def tokens(lab, text):
    return lab.tokenizer(text, return_tensors="pt", return_token_type_ids=False)["input_ids"]


def hook_count(model):
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules()
    )


def run_logits(lab, text):
    with torch.no_grad():
        return lab.model(input_ids=tokens(lab, text).to(lab.device)).logits


# --- direct logit attribution -------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("target", ["yes", PAIR])
def test_attribution_sums_to_the_actual_logit(kind, target, tiny_model, make_lab):
    lab = make(kind, tiny_model, make_lab)
    result = lab.logit_attribution("hello world yes no hello", target)
    logits = run_logits(lab, "hello world yes no hello")[0, -1]
    expected = logits[YES] if target == "yes" else logits[YES] - logits[NO]
    assert result["actual"] == pytest.approx(float(expected), abs=1e-5)
    assert result["total"] == pytest.approx(result["actual"], abs=1e-4)
    parts = result["embeddings"]
    parts += sum(head["value"] for head in result["heads"])
    parts += sum(mlp["value"] for mlp in result["mlps"])
    parts += sum(item["value"] for item in result["biases"]["attention_output"])
    parts += result["biases"]["final_norm"] + result["biases"]["output_head"]
    parts += sum(item["value"] for item in result.get("mixers", []))
    assert parts == pytest.approx(result["total"], abs=1e-5)
    assert len(result["heads"]) == len(lab.heads())
    assert [m["layer"] for m in result["mlps"]] == [0, 1]


def test_attribution_counts_one_pass(tiny_model, make_lab):
    lab = make("gpt2", tiny_model, make_lab)
    before = lab.forward_passes
    lab.logit_attribution("hello world yes no hello", PAIR, by=("component", "position"))
    assert lab.forward_passes - before == 1


@pytest.mark.parametrize("kind", KINDS)
def test_position_parts_sum_to_each_head(kind, tiny_model, make_lab):
    lab = make(kind, tiny_model, make_lab)
    result = lab.logit_attribution("hello world yes no hello", PAIR, by=("component", "position"))
    assert result["tokens"] == ["hello", "world", "yes", "no", "hello"]
    for head in result["heads"]:
        assert len(head["positions"]) == 5
        assert sum(head["positions"]) == pytest.approx(head["value"], abs=1e-5)
    # Without "position" the split is not computed.
    plain = lab.logit_attribution("hello world yes no hello", PAIR)
    assert all("positions" not in head for head in plain["heads"])


def test_attribution_at_an_earlier_position(tiny_model, make_lab):
    lab = make("llama_gqa", tiny_model, make_lab)
    text = "hello world yes no hello"
    result = lab.logit_attribution(text, PAIR, positions=2)
    logits = run_logits(lab, text)[0, 2]
    assert result["actual"] == pytest.approx(float(logits[YES] - logits[NO]), abs=1e-5)
    assert result["total"] == pytest.approx(result["actual"], abs=1e-4)


@pytest.mark.parametrize("kind", KINDS)
def test_zeroing_a_last_layer_head_changes_the_logit_by_its_attribution(kind, tiny_model, make_lab):
    # Large embeddings make the final residual much longer than one head's output, so
    # removing the head barely moves the final norm's scale. The linear estimate then
    # holds to about 5% of the head's value; it is exact only for a frozen scale.
    model, tokenizer = tiny_model(kind)
    with torch.no_grad():
        model.get_input_embeddings().weight.mul_(30)
    lab = make_lab_from(make_lab, model, tokenizer)
    text = "hello world yes no hello"
    result = lab.logit_attribution(text, PAIR)
    before = float((lambda x: x[0, -1, YES] - x[0, -1, NO])(run_logits(lab, text)))
    last = max(entry["layer"] for entry in lab.heads())
    checked = 0
    for entry in lab.heads(layers=[last]):
        value = next(
            h["value"] for h in result["heads"] if (h["layer"], h["head"]) == (last, entry["head"])
        )
        ids = [lab.zero(**part) for part in entry["o"]]
        after = float((lambda x: x[0, -1, YES] - x[0, -1, NO])(run_logits(lab, text)))
        for _ in ids:
            lab.undo()
        assert after - before == pytest.approx(-value, abs=0.05 * abs(value) + 1e-4)
        checked += 1
    assert checked >= 2


def make_lab_from(make_lab, model, tokenizer):
    return make_lab(model, tokenizer)


def test_attribution_refuses_bad_arguments(tiny_model, make_lab):
    lab = make("gpt2", tiny_model, make_lab)
    with pytest.raises(ValueError, match="by"):
        lab.logit_attribution("hello world", PAIR, by=("layer",))
    with pytest.raises(ValueError, match="target"):
        lab.logit_attribution("hello world", ("yes", "yes"))
    with pytest.raises(ValueError, match="position"):
        lab.logit_attribution("hello world", PAIR, positions=9)


# --- path patching ------------------------------------------------------------------------


def all_senders(lab, mixers=()):
    layers = sorted({entry["layer"] for entry in lab.heads()})
    senders = ["embed"] + [(entry["layer"], entry["head"]) for entry in lab.heads()]
    senders += [("mlp", layer) for layer in range(2)]
    senders += [("attn", layer) for layer in mixers]
    assert layers
    return senders


def mixers_of(kind):
    return [0] if kind == "qwen3_5" else []


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("freeze", [False, True])
def test_path_patch_end_cases(kind, freeze, tiny_model, make_lab):
    lab = make(kind, tiny_model, make_lab)
    senders = all_senders(lab, mixers_of(kind))
    full = lab.path_patch(CLEAN, CORRUPT, senders, ["logits"], metric=margin, freeze_mlps=freeze)
    empty = lab.path_patch(CLEAN, CORRUPT, [], ["logits"], metric=margin, freeze_mlps=freeze)
    assert full["clean"] != pytest.approx(full["corrupt"], abs=1e-3)
    assert full["patched"] == pytest.approx(full["corrupt"], abs=1e-5)
    assert full["fraction"] == pytest.approx(1.0, abs=1e-4)
    assert empty["patched"] == pytest.approx(empty["clean"], abs=1e-5)
    assert empty["fraction"] == pytest.approx(0.0, abs=1e-4)
    assert full["clean"] == pytest.approx(empty["clean"], abs=1e-6)
    # The clean metric is the model's own.
    by_hand = sum(float(margin(run_logits(lab, text)[:, -1])) for text in CLEAN) / 2
    assert full["clean"] == pytest.approx(by_hand, abs=1e-5)


def test_leaving_out_the_embeddings_leaves_their_effect(tiny_model, make_lab):
    lab = make("gpt2", tiny_model, make_lab)
    # The prompts differ at the last token, which the logits read directly.
    clean, corrupt = ["hello world yes no hello"], ["hello world yes no world"]
    senders = all_senders(lab)
    without = lab.path_patch(
        clean, corrupt, [s for s in senders if s != "embed"], ["logits"], metric=margin
    )
    with_embed = lab.path_patch(clean, corrupt, senders, ["logits"], metric=margin)
    assert without["patched"] != pytest.approx(without["corrupt"], abs=1e-3)
    assert with_embed["patched"] == pytest.approx(with_embed["corrupt"], abs=1e-5)


def test_path_patch_direct_path_matches_a_hand_computation(tiny_model, make_lab):
    lab = make("gpt2", tiny_model, make_lab)
    clean, corrupt = CLEAN[0], CORRUPT[0]
    o_module, final = "transformer.h.0.attn.c_proj", "transformer.ln_f"
    with (
        lab.record([o_module], positions="all", at="input") as c_o,
        lab.record([final], positions="all", at="input") as c_f,
        torch.no_grad(),
    ):
        lab.model(input_ids=tokens(lab, clean))
    with lab.record([o_module], positions="all", at="input") as k_o, torch.no_grad():
        lab.model(input_ids=tokens(lab, corrupt))
    head = lab.head_weights(0, 1)["o"]  # [hidden, head_dim]
    delta = (k_o[o_module][0][..., 8:16] - c_o[o_module][0][..., 8:16]) @ head.T
    model = lab.model
    with torch.no_grad():
        logits = model.lm_head(model.transformer.ln_f(c_f[final][0] + delta))[:, -1]
    predicted = float(margin(logits))
    # Only the direct head -> logits path, with every MLP held at its clean output.
    result = lab.path_patch(
        [clean], [corrupt], [(0, 1)], ["logits"], metric=margin, freeze_mlps=True
    )
    assert result["patched"] == pytest.approx(predicted, abs=1e-5)
    assert result["patched"] != pytest.approx(result["clean"], abs=1e-4)


def test_a_sender_after_its_receiver_changes_nothing(tiny_model, make_lab):
    lab = make("llama_gqa", tiny_model, make_lab)
    result = lab.path_patch(CLEAN, CORRUPT, [(1, 0)], [(0, 1, "q"), ("mlp", 0)], metric=margin)
    assert result["patched"] == pytest.approx(result["clean"], abs=1e-6)


@pytest.mark.parametrize("kind", ["gpt2", "llama_gqa"])
def test_every_head_input_receiver_runs(kind, tiny_model, make_lab):
    lab = make(kind, tiny_model, make_lab)
    for part in "qkv":
        result = lab.path_patch(CLEAN, CORRUPT, [(0, 0), ("mlp", 0)], [(1, 1, part)], metric=margin)
        assert math.isfinite(result["patched"])
    # Patching all of layer 1's inputs from every earlier sender is the layer's total effect.
    earlier = ["embed", *[(0, e["head"]) for e in lab.heads(layers=[0])], ("mlp", 0)]
    receivers = [(1, h, p) for h in range(len(lab.heads(layers=[1]))) for p in "qkv"]
    receivers.append(("mlp", 1))
    receivers.append("logits")
    result = lab.path_patch(CLEAN, CORRUPT, earlier, receivers, metric=margin)
    assert math.isfinite(result["fraction"])


def test_path_patch_pass_cost_and_cleanup(tiny_model, make_lab):
    lab = make("gpt2", tiny_model, make_lab)
    hooks = hook_count(lab.model)
    before = lab.forward_passes
    lab.path_patch(CLEAN, CORRUPT, [(0, 0)], ["logits"], metric=margin)
    assert lab.forward_passes - before == 4 * len(CLEAN)
    assert hook_count(lab.model) == hooks

    def broken(logits):
        raise RuntimeError("metric failed")

    with pytest.raises(RuntimeError, match="metric failed"):
        lab.path_patch(CLEAN, CORRUPT, [(0, 0)], ["logits"], metric=broken)
    assert hook_count(lab.model) == hooks


def test_path_patch_refuses_bad_input(tiny_model, make_lab):
    lab = make("gpt2", tiny_model, make_lab)
    with pytest.raises(ValueError, match="token count"):
        lab.path_patch(["hello world"], ["hello world yes"], [], ["logits"], metric=margin)
    with pytest.raises(ValueError, match="same number"):
        lab.path_patch(CLEAN, CORRUPT[:1], [], ["logits"], metric=margin)
    with pytest.raises(ValueError, match="sender"):
        lab.path_patch(CLEAN, CORRUPT, [(7, 0)], ["logits"], metric=margin)
    with pytest.raises(ValueError, match="receiver"):
        lab.path_patch(CLEAN, CORRUPT, [(0, 0)], [(0, 0, "x")], metric=margin)
    with pytest.raises(ValueError, match="receiver"):
        lab.path_patch(CLEAN, CORRUPT, [(0, 0)], [], metric=margin)
    with pytest.raises(ValueError, match="metric"):
        lab.path_patch(CLEAN, CORRUPT, [(0, 0)], ["logits"], metric="loss")


def test_pair_metric_is_the_mean_logit_difference(tiny_model, make_lab):
    lab = make("gpt2", tiny_model, make_lab)
    by_pair = lab.path_patch(CLEAN, CORRUPT, [], ["logits"], metric=PAIR)
    by_call = lab.path_patch(CLEAN, CORRUPT, [], ["logits"], metric=margin)
    assert by_pair["clean"] == pytest.approx(by_call["clean"], abs=1e-6)
    assert by_pair["corrupt"] == pytest.approx(by_call["corrupt"], abs=1e-6)


# --- edge attribution patching ------------------------------------------------------------


EDGE_COUNTS = {"gpt2": 46, "llama_gqa": 82}


@pytest.mark.parametrize("kind", ["gpt2", "llama_gqa"])
def test_edge_scores_have_the_documented_shape(kind, tiny_model, make_lab):
    lab = make(kind, tiny_model, make_lab)
    edges = lab.edge_attribution(CLEAN, CORRUPT, metric=margin, steps=2)
    assert len(edges) == EDGE_COUNTS[kind]
    assert all(set(edge) == {"sender", "receiver", "score"} for edge in edges)
    scores = [abs(edge["score"]) for edge in edges]
    assert scores == sorted(scores, reverse=True)
    assert len({(str(e["sender"]), str(e["receiver"])) for e in edges}) == len(edges)
    senders = {str(e["sender"]) for e in edges}
    assert "embed" in senders and "(0, 0)" in senders and "('mlp', 1)" in senders
    receivers = {str(e["receiver"]) for e in edges}
    assert "logits" in receivers and "(1, 0, 'q')" in receivers and "('mlp', 0)" in receivers

    # Every edge goes forward in the model: a sender's place comes before its receiver's.
    def sender_place(sender):
        if sender == "embed":
            return (-1, 0)
        return (sender[1], 1) if sender[0] == "mlp" else (sender[0], 0)

    def receiver_place(receiver):
        if receiver == "logits":
            return (99, 0)
        return (receiver[1], 1) if receiver[0] == "mlp" else (receiver[0], 0)

    assert all(sender_place(e["sender"]) < receiver_place(e["receiver"]) for e in edges)
    # k and v receivers are named by key/value head, so a grouped-query model has fewer.
    heads = len(lab.heads(layers=[0]))
    kv = len({e["kv_head"] for e in lab.heads(layers=[0])})
    k_receivers = {e["receiver"] for e in edges if e["receiver"][-1:] == ("k",)}
    assert {r[1] for r in k_receivers if r[0] == 1} == set(range(kv))
    assert heads >= kv


def plain_attribution(lab, kind, clean, corrupt, where):
    """Attribution patching by hand: a gradient at the clean run, cut at one norm's input."""
    layers = LAYERS[kind]
    model = lab.model
    norm = {
        "logits": NORMS[kind]["final"],
        "mlp0": NORMS[kind]["mlp"].format(0),
        "mlp1": NORMS[kind]["mlp"].format(1),
        "attn1": NORMS[kind]["attn"].format(1),
    }[where]
    o_proj = {
        "gpt2": "transformer.h.{}.attn.c_proj",
        "llama_gqa": "model.layers.{}.self_attn.o_proj",
    }[kind]
    mlp_out = {"gpt2": "transformer.h.{}.mlp", "llama_gqa": "model.layers.{}.mlp"}[kind]
    names = [layers.format(0), o_proj.format(0), o_proj.format(1), mlp_out.format(0)]
    names += [mlp_out.format(1)]
    recorded = {}
    for label, text in (("clean", clean), ("corrupt", corrupt)):
        with (
            lab.record(names[:3], positions="all", at="input") as ins,
            lab.record(names[3:], positions="all") as outs,
            torch.no_grad(),
        ):
            model(input_ids=tokens(lab, text))
        recorded[label] = ({**ins}, {**outs})
    leaf = {}

    def cut(module, args, kwargs):
        value = args[0] if args else kwargs["hidden_states"]
        leaf["x"] = value.detach().clone().requires_grad_(True)
        return (
            ((leaf["x"], *args[1:]), kwargs)
            if args
            else (args, {**kwargs, "hidden_states": leaf["x"]})
        )

    handle = model.get_submodule(norm).register_forward_pre_hook(cut, with_kwargs=True)
    try:
        with torch.enable_grad():
            logits = model(input_ids=tokens(lab, clean)).logits
            margin(logits[:, -1]).backward()
    finally:
        handle.remove()
    grad = leaf["x"].grad

    def node(label):
        if label == "embed":
            return (
                recorded["corrupt"][0][layers.format(0)][0]
                - recorded["clean"][0][layers.format(0)][0]
            )
        if label[0] == "mlp":
            name = mlp_out.format(label[1])
            return recorded["corrupt"][1][name][0] - recorded["clean"][1][name][0]
        name = o_proj.format(label[0])
        h = label[1]
        hd = recorded["clean"][0][name][0].shape[-1] // len(
            [e for e in lab.heads(layers=[label[0]])]
        )
        diff = recorded["corrupt"][0][name][0] - recorded["clean"][0][name][0]
        return diff[..., h * hd : (h + 1) * hd] @ lab.head_weights(*label)["o"].T

    return grad, node


@pytest.mark.parametrize("kind", ["gpt2", "llama_gqa"])
def test_one_step_equals_plain_attribution_patching(kind, tiny_model, make_lab):
    lab = make(kind, tiny_model, make_lab)
    clean, corrupt = CLEAN[0], CORRUPT[0]
    edges = lab.edge_attribution([clean], [corrupt], metric=margin, steps=1)
    found = {(str(e["sender"]), str(e["receiver"])): e["score"] for e in edges}
    checks = [
        ("logits", "embed", "logits"),
        ("logits", (0, 0), "logits"),
        ("logits", ("mlp", 0), "logits"),
        ("mlp0", "embed", ("mlp", 0)),
        ("mlp0", (0, 1), ("mlp", 0)),
        ("mlp1", (0, 1), ("mlp", 1)),
        ("mlp1", ("mlp", 0), ("mlp", 1)),
    ]
    for where, sender, receiver in checks:
        grad, node = plain_attribution(lab, kind, clean, corrupt, where)
        expected = float((node(sender) * grad).sum())
        assert found[(str(sender), str(receiver))] == pytest.approx(expected, rel=1e-4, abs=1e-6)
    # All of layer 1's attention inputs read one normed residual, so their scores add up to
    # the gradient through that norm.
    grad, node = plain_attribution(lab, kind, clean, corrupt, "attn1")
    total = sum(
        score
        for (sender, receiver), score in found.items()
        if sender == "embed" and receiver.startswith("(1, ")
    )
    assert total == pytest.approx(float((node("embed") * grad).sum()), rel=1e-4, abs=1e-6)


def test_more_steps_change_the_scores(tiny_model, make_lab):
    lab = make("gpt2", tiny_model, make_lab)
    one = {
        str(e["sender"]) + str(e["receiver"]): e["score"]
        for e in lab.edge_attribution(CLEAN, CORRUPT, metric=margin, steps=1)
    }
    five = {
        str(e["sender"]) + str(e["receiver"]): e["score"]
        for e in lab.edge_attribution(CLEAN, CORRUPT, metric=margin, steps=5)
    }
    assert one.keys() == five.keys()
    assert any(abs(one[key] - five[key]) > 1e-6 for key in one)


def test_edge_attribution_cost_and_cleanup(tiny_model, make_lab):
    lab = make("gpt2", tiny_model, make_lab)
    hooks = hook_count(lab.model)
    model = lab.model
    model.train()
    before = lab.forward_passes
    flags = [p.requires_grad for p in model.parameters()]
    lab.edge_attribution(CLEAN, CORRUPT, metric=margin, steps=3)
    assert lab.forward_passes - before == (2 + 3) * len(CLEAN)
    assert hook_count(model) == hooks
    assert model.training
    assert [p.requires_grad for p in model.parameters()] == flags
    assert all(p.grad is None for p in model.parameters())

    def broken(logits):
        raise RuntimeError("metric failed")

    with pytest.raises(RuntimeError, match="metric failed"):
        lab.edge_attribution(CLEAN, CORRUPT, metric=broken)
    assert hook_count(model) == hooks and model.training


@pytest.mark.parametrize("kind", KINDS)
def test_full_and_empty_circuits_reproduce_clean_and_corrupt(kind, tiny_model, make_lab):
    lab = make(kind, tiny_model, make_lab)
    everything = lab.edge_attribution(CLEAN, CORRUPT, metric=margin, steps=1)
    full = lab.find_circuit(CLEAN, CORRUPT, metric=margin, edges=everything)
    empty = lab.find_circuit(CLEAN, CORRUPT, metric=margin, edges=[])
    top = lab.find_circuit(CLEAN, CORRUPT, metric=margin, top=30, steps=2)
    assert full["clean"] != pytest.approx(full["corrupt"], abs=1e-3)
    assert full["circuit"] == pytest.approx(full["clean"], abs=1e-5)
    assert full["faithfulness"] == pytest.approx(1.0, abs=1e-4)
    assert len(full["edges"]) == len(everything) == full["total_edges"]
    assert empty["circuit"] == pytest.approx(empty["corrupt"], abs=1e-5)
    assert empty["faithfulness"] == pytest.approx(0.0, abs=1e-4)
    assert empty["edges"] == []
    assert top["clean"] == pytest.approx(full["clean"], abs=1e-6)
    assert top["corrupt"] == pytest.approx(full["corrupt"], abs=1e-6)
    assert 0 < len(top["edges"]) <= 30
    expected = (top["circuit"] - top["corrupt"]) / (top["clean"] - top["corrupt"])
    assert top["faithfulness"] == pytest.approx(expected, abs=1e-6)


def test_find_circuit_keeps_only_edges_on_a_path(tiny_model, make_lab):
    lab = make("gpt2", tiny_model, make_lab)
    # Two edges that never meet: embed -> layer 0 MLP, and head (1, 0) -> layer 1 MLP.
    edges = [
        {"sender": "embed", "receiver": ("mlp", 0), "score": 1.0},
        {"sender": (1, 0), "receiver": ("mlp", 1), "score": 0.5},
    ]
    result = lab.find_circuit(CLEAN, CORRUPT, metric=margin, edges=edges)
    assert result["edges"] == []
    edges.append({"sender": ("mlp", 0), "receiver": "logits", "score": 0.1})
    assert [
        e["receiver"] for e in lab.find_circuit(CLEAN, CORRUPT, metric=margin, edges=edges)["edges"]
    ] == [
        ("mlp", 0),
        "logits",
    ]


def test_find_circuit_cost_and_cleanup(tiny_model, make_lab):
    lab = make("llama_gqa", tiny_model, make_lab)
    hooks = hook_count(lab.model)
    before = lab.forward_passes
    result = lab.find_circuit(CLEAN, CORRUPT, metric=margin, top=5, steps=2)
    assert lab.forward_passes - before == (3 + 2) * len(CLEAN)
    before = lab.forward_passes
    lab.find_circuit(CLEAN, CORRUPT, metric=margin, edges=result["edges"])
    assert lab.forward_passes - before == 3 * len(CLEAN)
    assert hook_count(lab.model) == hooks
    with pytest.raises(ValueError, match="top"):
        lab.find_circuit(CLEAN, CORRUPT, metric=margin, top=0)
    with pytest.raises(ValueError, match="edge"):
        lab.find_circuit(
            CLEAN, CORRUPT, metric=margin, edges=[{"sender": (1, 0), "receiver": (0, 0, "q")}]
        )
    with pytest.raises(ValueError, match="steps"):
        lab.edge_attribution(CLEAN, CORRUPT, metric=margin, steps=0)


def test_circuit_survives_json(tiny_model, make_lab):
    import json

    lab = make("llama_gqa", tiny_model, make_lab)
    first = lab.find_circuit(CLEAN, CORRUPT, metric=PAIR, top=8, steps=1)
    again = lab.find_circuit(
        CLEAN, CORRUPT, metric=PAIR, edges=json.loads(json.dumps(first["edges"]))
    )
    assert again["circuit"] == pytest.approx(first["circuit"], abs=1e-6)


def test_pass_limit_rejects_before_running(tiny_model, make_lab):
    from oumi_weights_lab.budget import BudgetExceeded

    lab = make("gpt2", tiny_model, make_lab)
    with lab.limit_passes(3):
        with pytest.raises(BudgetExceeded):
            lab.path_patch(CLEAN, CORRUPT, [(0, 0)], ["logits"], metric=margin)
    assert lab.forward_passes == 0


# --- refusals -----------------------------------------------------------------------------


def test_post_sublayer_norms_are_refused(make_lab, tiny_tokenizer):
    import transformers

    if not hasattr(transformers, "Gemma2Config"):
        pytest.skip("installed Transformers does not provide Gemma2Config")
    config = transformers.Gemma2Config(
        vocab_size=6, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4, max_position_embeddings=64,
        pad_token_id=1, bos_token_id=1, eos_token_id=1,
    )  # fmt: skip
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    lab = make_lab(model.eval(), tiny_tokenizer)
    assert lab.heads()  # heads() itself works on this layout
    for call in (
        lambda: lab.logit_attribution("hello world", PAIR),
        lambda: lab.path_patch(CLEAN, CORRUPT, [], ["logits"], metric=margin),
        lambda: lab.edge_attribution(CLEAN, CORRUPT, metric=margin),
        lambda: lab.find_circuit(CLEAN, CORRUPT, metric=margin),
    ):
        with pytest.raises(NotImplementedError, match="layers.0.post_attention_layernorm|layers.0"):
            call()
    assert lab.forward_passes == 0


def test_parallel_blocks_are_refused(make_lab, tiny_tokenizer):
    import transformers

    if not hasattr(transformers, "GPTJConfig"):
        pytest.skip("installed Transformers does not provide GPTJConfig")
    config = transformers.GPTJConfig(
        vocab_size=6, n_embd=16, n_layer=2, n_head=2, n_positions=64, rotary_dim=4,
        bos_token_id=1, eos_token_id=1, pad_token_id=1,
    )  # fmt: skip
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    lab = make_lab(model.eval(), tiny_tokenizer)
    with pytest.raises(NotImplementedError, match="h.0"):
        lab.logit_attribution("hello world", PAIR)


def test_an_unrecognized_output_scale_is_refused(tiny_model, make_lab):
    model, tokenizer = tiny_model("llama_gqa")
    # A scale after the output head breaks the decomposition, so the call must say so.
    model.lm_head.register_forward_hook(lambda module, args, output: output * 0.5)
    lab = make_lab(model, tokenizer)
    hooks = hook_count(model)
    with pytest.raises(NotImplementedError, match="logits"):
        lab.logit_attribution("hello world yes", PAIR)
    assert hook_count(model) == hooks


def test_a_scaled_residual_is_refused(tiny_model, make_lab):
    model, tokenizer = tiny_model("llama_gqa")

    def scale(module, args, output):
        if isinstance(output, tuple):
            return (output[0] * 1.5, *output[1:])
        return output * 1.5

    # A multiplier on a layer's output breaks the additive residual stream.
    model.model.layers[0].register_forward_hook(scale)
    lab = make_lab(model, tokenizer)
    with pytest.raises(NotImplementedError, match="additive"):
        lab.logit_attribution("hello world yes", PAIR)
    with pytest.raises(NotImplementedError, match="additive"):
        lab.edge_attribution(CLEAN, CORRUPT, metric=margin)


def test_an_attention_block_that_is_not_its_heads_is_refused(tiny_model, make_lab):
    model, tokenizer = tiny_model("llama_gqa")
    original = model.model.layers[0].self_attn.forward

    def scaled(*args, **kwargs):
        output = original(*args, **kwargs)
        return (output[0] * 2, *output[1:])

    model.model.layers[0].self_attn.forward = scaled
    lab = make_lab(model, tokenizer)
    with pytest.raises(NotImplementedError, match="heads"):
        lab.logit_attribution("hello world yes", PAIR)


def test_modes_and_grads_survive_every_call(tiny_model, make_lab):
    lab = make("llama_gqa", tiny_model, make_lab)
    model = lab.model
    model.train()
    state = torch.get_rng_state()
    lab.logit_attribution("hello world yes", PAIR, by=("component", "position"))
    lab.path_patch(CLEAN, CORRUPT, [(0, 0)], ["logits"], metric=margin)
    lab.find_circuit(CLEAN, CORRUPT, metric=margin, top=5, steps=1)
    assert model.training and all(module.training for module in model.modules())
    assert torch.equal(torch.get_rng_state(), state)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_group_receivers_share_their_key_head(tiny_model, make_lab):
    lab = make("llama_gqa", tiny_model, make_lab)
    # Heads 0 and 1 read one key/value head, so patching it through either is the same.
    senders = ["embed", ("mlp", 0)]
    first = lab.path_patch(CLEAN, CORRUPT, senders, [(1, 0, "k")], metric=margin)
    second = lab.path_patch(CLEAN, CORRUPT, senders, [(1, 1, "k")], metric=margin)
    other = lab.path_patch(CLEAN, CORRUPT, senders, [(1, 2, "k")], metric=margin)
    assert first["patched"] == pytest.approx(second["patched"], abs=1e-7)
    assert first["patched"] != pytest.approx(other["patched"], abs=1e-6)


def test_prompts_of_different_lengths_average_over_pairs(tiny_model, make_lab):
    lab = make("llama_gqa", tiny_model, make_lab)
    clean = ["hello world yes no hello", "hello world yes"]
    corrupt = ["hello hello yes no hello", "hello hello yes"]
    receivers = [(1, 0, "v"), "logits"]
    both = lab.path_patch(clean, corrupt, [(0, 1)], receivers, metric=margin)
    each = [
        lab.path_patch([c], [k], [(0, 1)], receivers, metric=margin)
        for c, k in zip(clean, corrupt, strict=True)
    ]
    for name in ("clean", "corrupt", "patched"):
        assert both[name] == pytest.approx(sum(r[name] for r in each) / 2, abs=1e-6)
    scores = {
        (str(e["sender"]), str(e["receiver"])): e["score"]
        for e in lab.edge_attribution(clean, corrupt, metric=margin, steps=2)
    }
    for index in range(2):
        single = lab.edge_attribution([clean[index]], [corrupt[index]], metric=margin, steps=2)
        for e in single:
            scores[(str(e["sender"]), str(e["receiver"]))] -= e["score"] / 2
    assert max(abs(value) for value in scores.values()) < 1e-6
    circuit = lab.find_circuit(clean, corrupt, metric=margin, edges=[])
    assert circuit["circuit"] == pytest.approx(circuit["corrupt"], abs=1e-5)


def test_quantized_and_adapter_models_are_refused(tiny_model, make_lab, monkeypatch):
    lab = make("gpt2", tiny_model, make_lab)
    monkeypatch.setattr("oumi_weights_lab.analyses.is_quantized", lambda module: True)
    with pytest.raises(NotImplementedError, match="bitsandbytes"):
        lab.logit_attribution("hello world", PAIR)
    monkeypatch.undo()
    lab.model.peft_config = {}
    try:
        with pytest.raises(NotImplementedError, match="merged model"):
            lab.path_patch(CLEAN, CORRUPT, [], ["logits"], metric=margin)
    finally:
        del lab.model.peft_config
    assert lab.forward_passes == 0


# --- half precision -----------------------------------------------------------------------


@pytest.mark.parametrize("dtype,tolerance", [(torch.bfloat16, 0.05), (torch.float16, 0.01)])
@pytest.mark.parametrize("kind", ["gpt2", "llama_gqa"])
def test_half_precision_models_work_to_the_dtypes_precision(
    kind, dtype, tolerance, tiny_model, make_lab
):
    model, tokenizer = tiny_model(kind)
    lab = make_lab(strengthen(model).to(dtype), tokenizer)
    for target in ("yes", PAIR):
        result = lab.logit_attribution(
            "hello world yes no hello", target, by=("component", "position")
        )
        assert result["total"] == pytest.approx(result["actual"], abs=tolerance)
        for head in result["heads"]:
            assert sum(head["positions"]) == pytest.approx(head["value"], abs=tolerance)
    patched = lab.path_patch(CLEAN, CORRUPT, [(0, 0)], ["logits"], metric=margin)
    assert math.isfinite(patched["patched"])
    circuit = lab.find_circuit(CLEAN, CORRUPT, metric=margin, top=10, steps=1)
    assert math.isfinite(circuit["circuit"])


def test_half_precision_still_refuses_a_different_model(tiny_model, make_lab):
    model, tokenizer = tiny_model("llama_gqa")
    model.lm_head.register_forward_hook(lambda module, args, output: output * 0.5)
    lab = make_lab(strengthen(model).to(torch.bfloat16), tokenizer)
    with pytest.raises(NotImplementedError, match="logits"):
        lab.logit_attribution("hello world yes", PAIR)


# --- routing: hand patches that a mis-routed edge or slice would not match ------------------


def grab_input(state, layer):
    def hook(module, args):
        if not state.get("busy"):
            state[f"in{layer}"] = args[0]

    return hook


def replace_slice(state, model, layer, unit, width, resid_for):
    """Replace one head's slice of a projection with one computed from another residual."""
    block = model.model.layers[layer]

    def hook(module, args, out):
        resid = resid_for(state[f"in{layer}"])
        state["busy"] = True
        normed = block.input_layernorm(resid)
        state["busy"] = False
        alt = torch.nn.functional.linear(normed, module.weight, module.bias)
        out = out.clone()
        out[..., unit * width : (unit + 1) * width] = alt[..., unit * width : (unit + 1) * width]
        return out

    return hook


def run_with(model, ids, hooks):
    handles = []
    for module, kind, function in hooks:
        register = (
            module.register_forward_hook if kind == "post" else module.register_forward_pre_hook
        )
        handles.append(register(function))
    try:
        with torch.no_grad():
            return model(input_ids=ids).logits[:, -1]
    finally:
        for handle in handles:
            handle.remove()


def test_removing_specific_edges_matches_a_hand_patch(tiny_model, make_lab):
    lab = make("llama_gqa", tiny_model, make_lab)
    model, layers, hd = lab.model, lab.model.model.layers, 4
    clean, corrupt = CLEAN[0], CORRUPT[0]
    c_ids, k_ids = tokens(lab, clean), tokens(lab, corrupt)
    with torch.no_grad():
        d_emb = model.model.embed_tokens(k_ids) - model.model.embed_tokens(c_ids)

    def o_input(ids, layer):
        store = {}
        module = layers[layer].self_attn.o_proj
        handle = module.register_forward_pre_hook(lambda m, a: store.update(x=a[0].detach()))
        with torch.no_grad():
            model(input_ids=ids)
        handle.remove()
        return store["x"]

    def head_write(o, layer, head):
        weight = layers[layer].self_attn.o_proj.weight
        return o[..., head * hd : (head + 1) * hd] @ weight[:, head * hd : (head + 1) * hd].T

    everything = lab.edge_attribution([clean], [corrupt], metric=margin, steps=1)
    full = lab.find_circuit([clean], [corrupt], metric=margin, edges=everything)

    def circuit_without(removed):
        kept = [e for e in everything if (e["sender"], e["receiver"]) not in removed]
        return lab.find_circuit([clean], [corrupt], metric=margin, edges=kept)["circuit"]

    k_o0, c_o0 = o_input(k_ids, 0), o_input(c_ids, 0)
    d_head01 = head_write(k_o0, 0, 1) - head_write(c_o0, 0, 1)
    corrupt_head02 = head_write(k_o0, 0, 2)
    state = {}

    def final_fix(module, args):
        return (args[0] + corrupt_head02 - head_write(state["o0"], 0, 2),)

    def at_layer(layer, proj, unit, shift):
        return [
            (layers[layer].input_layernorm, "pre", grab_input(state, layer)),
            (
                getattr(layers[layer].self_attn, f"{proj}_proj"),
                "post",
                replace_slice(state, model, layer, unit, hd, lambda r: r + shift),
            ),
        ]

    cases = [
        # embed -> key/value head 0 of layer 1 (k)
        ({("embed", (1, 0, "k"))}, at_layer(1, "k", 0, d_emb)),
        # embed -> query head 2 of layer 0
        ({("embed", (0, 2, "q"))}, at_layer(0, "q", 2, d_emb)),
        # head (0, 1) -> value head 1 of layer 1
        ({((0, 1), (1, 1, "v"))}, at_layer(1, "v", 1, d_head01)),
        # two edges at once: embed -> (0, 2, q) and (0, 2) -> logits
        (
            {("embed", (0, 2, "q")), ((0, 2), "logits")},
            at_layer(0, "q", 2, d_emb)
            + [
                (layers[0].self_attn.o_proj, "pre", lambda m, a: state.update(o0=a[0])),
                (model.model.norm, "pre", final_fix),
            ],
        ),
    ]
    for removed, hooks in cases:
        by_hand = float(margin(run_with(model, c_ids, hooks)))
        assert circuit_without(removed) == pytest.approx(by_hand, abs=1e-5)
        # The edge matters, so the check cannot pass by routing nothing.
        assert abs(by_hand - full["clean"]) > 1e-4


def per_head_gradient(lab, kind, clean, corrupt, proj, unit, alpha):
    """The metric's gradient at one head's input residual in layer 1, at an interpolated point."""
    model, layer = lab.model, lab.model.model.layers[1]
    c_ids, k_ids = tokens(lab, clean), tokens(lab, corrupt)
    hd = 4
    with torch.no_grad():
        emb_c = model.model.embed_tokens(c_ids)
        emb_k = model.model.embed_tokens(k_ids)
    width = 2 * hd if kind == "qwen3_5" and proj == "q_proj" else hd
    state = {}

    def replace(module, args, out):
        leaf = state["in1"].detach().clone().requires_grad_(True)
        state["leaf"] = leaf
        state["busy"] = True
        normed = layer.input_layernorm(leaf)
        state["busy"] = False
        alt = torch.nn.functional.linear(normed, module.weight, module.bias)
        lo = unit * width
        return torch.cat([out[..., :lo], alt[..., lo : lo + width], out[..., lo + width :]], -1)

    hooks = [
        model.model.embed_tokens.register_forward_hook(
            lambda m, a, o: emb_c + alpha * (emb_k - emb_c)
        ),
        layer.input_layernorm.register_forward_pre_hook(grab_input(state, 1)),
        getattr(layer.self_attn, proj).register_forward_hook(replace),
    ]
    try:
        with torch.enable_grad():
            value = margin(model(input_ids=c_ids).logits[:, -1])
            (grad,) = torch.autograd.grad(value, [state["leaf"]])
    finally:
        for handle in hooks:
            handle.remove()
    return grad, emb_k - emb_c


@pytest.mark.parametrize("kind", ["llama_gqa", "qwen3_5"])
@pytest.mark.parametrize("steps", [1, 3])
def test_edge_scores_into_each_head_match_hand_gradients(kind, steps, tiny_model, make_lab):
    lab = make(kind, tiny_model, make_lab)
    clean, corrupt = "hello world yes no hello", "hello world yes no world"
    scores = {
        (e["sender"], e["receiver"]): e["score"]
        for e in lab.edge_attribution([clean], [corrupt], metric=margin, steps=steps)
    }
    alphas = torch.linspace(0, 1, steps).tolist()
    for proj, part, count in (("q_proj", "q", 4), ("k_proj", "k", 2), ("v_proj", "v", 2)):
        found = []
        for unit in range(count):
            total = 0
            for alpha in alphas:
                grad, delta = per_head_gradient(lab, kind, clean, corrupt, proj, unit, alpha)
                total = total + grad / steps
            by_hand = float((delta * total).sum())
            assert scores[("embed", (1, unit, part))] == pytest.approx(by_hand, rel=1e-4, abs=1e-6)
            found.append(by_hand)
        # Heads differ, so a score handed to the wrong head would not match.
        assert max(found) - min(found) > 1e-4


@pytest.mark.parametrize("kind", ["gpt2", "llama_gqa"])
@pytest.mark.parametrize("freeze", [False, True])
def test_head_to_head_path_patch_matches_a_hand_patch(kind, freeze, tiny_model, make_lab):
    lab = make(kind, tiny_model, make_lab)
    model = lab.model
    if kind == "gpt2":
        blocks, hd, nh, nkv = model.transformer.h, 8, 2, 2

        def o_module(i):
            return blocks[i].attn.c_proj

        def projection(layer, part, unit):
            base = {"q": 0, "k": nh * hd, "v": 2 * nh * hd}[part]
            return blocks[layer].attn.c_attn, base + unit * hd, base + (unit + 1) * hd
    else:
        blocks, hd, nh, nkv = model.model.layers, 4, 4, 2

        def o_module(i):
            return blocks[i].self_attn.o_proj

        def projection(layer, part, unit):
            return getattr(blocks[layer].self_attn, f"{part}_proj"), unit * hd, (unit + 1) * hd

    def record(text, module, pre):
        store = {}
        if pre:
            hook = (module, "pre", lambda m, a: store.update(x=a[0].detach().clone()))
        else:
            hook = (module, "post", lambda m, a, o: store.update(x=o.detach().clone()))
        run_with(model, tokens(lab, text), [hook])
        return store["x"]

    differences = []
    for sender, (layer, head, part) in ((1, (1, 1, "q")), (0, (1, 1, "k")), (1, (1, 0, "v"))):
        totals = []
        for clean, corrupt in zip(CLEAN, CORRUPT, strict=True):
            k_o = record(corrupt, o_module(0), True)
            c_mlp = record(clean, blocks[0].mlp, False)
            unit = head if part == "q" else head // (nh // nkv)
            module, lo, hi = projection(layer, part, unit)
            seen = {}

            def swap(m, args, k_o=k_o, sender=sender):
                x = args[0].clone()
                x[..., sender * hd : (sender + 1) * hd] = k_o[..., sender * hd : (sender + 1) * hd]
                return (x,)

            hooks = [
                (o_module(0), "pre", swap),
                (
                    module,
                    "post",
                    lambda m, a, o, seen=seen, lo=lo, hi=hi: seen.update(r=o[..., lo:hi].clone()),
                ),
            ]
            if freeze:
                hooks.append((blocks[0].mlp, "post", lambda m, a, o, c_mlp=c_mlp: c_mlp))
            run_with(model, tokens(lab, clean), hooks)

            def put(m, a, o, seen=seen, lo=lo, hi=hi):
                o = o.clone()
                o[..., lo:hi] = seen["r"]
                return o

            patched = run_with(model, tokens(lab, clean), [(module, "post", put)])
            totals.append(float(margin(patched)))
        by_hand = sum(totals) / len(totals)
        result = lab.path_patch(
            CLEAN, CORRUPT, [(0, sender)], [(layer, head, part)], metric=margin, freeze_mlps=freeze
        )
        assert result["patched"] == pytest.approx(by_hand, abs=2e-6)
        differences.append(abs(result["patched"] - result["clean"]))
    # Each path moves the metric, so a patch routed to the wrong place would not match.
    assert min(differences) > 2e-5


# --- error types ---------------------------------------------------------------------------


def test_receivers_that_do_not_exist_raise_value_errors(tiny_model, make_lab):
    lab = make("qwen3_5", tiny_model, make_lab)
    with pytest.raises(ValueError, match="no attention heads"):
        lab.path_patch(CLEAN, CORRUPT, [], [(0, 0, "q")], metric=margin)
    with pytest.raises(ValueError, match="no layer 7"):
        lab.path_patch(CLEAN, CORRUPT, [], [(7, 0, "k")], metric=margin)
    edge = {"sender": "embed", "receiver": (7, 0, "k")}
    with pytest.raises(ValueError, match="no layer 7"):
        lab.find_circuit(CLEAN, CORRUPT, metric=margin, edges=[edge])
    edge = {"sender": "embed", "receiver": (0, 0, "q")}
    with pytest.raises(ValueError, match="no attention heads"):
        lab.find_circuit(CLEAN, CORRUPT, metric=margin, edges=[edge])
