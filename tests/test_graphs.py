import json

import pytest

pytest.importorskip("sae_lens")

import torch  # noqa: E402
from test_saes import D, build_gemma2, make_sae  # noqa: E402

from oumi_weights_lab import graphs  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:(?s).*model_from_pretrained_kwargs")

PROMPT = "hello world yes no hello"
EVERYTHING = {"node_threshold": 1.0, "edge_threshold": 1.0}


@pytest.fixture
def lab_for(tiny_model, make_lab, tiny_tokenizer):
    def build(kind):
        if kind == "gemma2":
            return make_lab(build_gemma2(tiny_tokenizer), tiny_tokenizer)
        return make_lab(*tiny_model(kind))

    return build


def transcoders_for(lab, tmp_path, architecture="transcoder", layers=(0, 1), seed=0):
    """One random transcoder per layer, reading the MLP's normalized input."""
    found = {}
    for layer in layers:
        path = tmp_path / f"{architecture}-{layer}"
        make_sae(
            path,
            f"blocks.{layer}.ln2.hook_normalized",
            architecture=architecture,
            hook_name_out=f"blocks.{layer}.hook_mlp_out",
            d_out=D,
            seed=seed + layer,
        )
        found[layer] = lab.load_sae(path=path)
    return found


@pytest.fixture(params=["gpt2", "llama_gqa", "gemma2"])
def setup(request, lab_for, tmp_path):
    lab = lab_for(request.param)
    return lab, transcoders_for(lab, tmp_path)


def snapshot(lab):
    """What a call must leave as it found it."""
    model = lab.model
    return {
        "hooks": {
            name: (
                dict(module._forward_hooks),
                dict(module._forward_pre_hooks),
                dict(module._backward_hooks),
            )
            for name, module in model.named_modules()
        },
        "attention": model.config._attn_implementation,
        "training": [module.training for module in model.modules()],
        "requires_grad": {name: p.requires_grad for name, p in model.named_parameters()},
        "weights": {name: p.detach().clone() for name, p in model.named_parameters()},
        "rng": torch.get_rng_state(),
    }


def assert_restored(lab, before):
    after = snapshot(lab)
    assert after["hooks"] == before["hooks"]
    assert after["attention"] == before["attention"]
    assert after["training"] == before["training"]
    assert after["requires_grad"] == before["requires_grad"]
    assert torch.equal(after["rng"], before["rng"])
    for name, weight in before["weights"].items():
        assert torch.equal(after["weights"][name], weight), name
    assert not any(key.startswith("weights_lab") for key in _registered())


def _registered():
    from transformers import AttentionInterface

    return list(AttentionInterface._global_mapping)


def engine_for(lab, transcoders, max_features=None):
    inputs = next(iter(transcoders.values()))._tokenized(PROMPT)
    return graphs._Replacement(lab, transcoders, inputs, max_features)


def real_logits(lab, transcoders):
    inputs = next(iter(transcoders.values()))._tokenized(PROMPT)
    with torch.no_grad():
        return lab.model(**inputs).logits[0, -1]


# Exactness: the replacement model reproduces the real model


@pytest.mark.parametrize("architecture", ["transcoder", "skip_transcoder"])
def test_replacement_logits_equal_the_real_logits(lab_for, tmp_path, architecture):
    for kind in ["gpt2", "llama_gqa", "gemma2"]:
        lab = lab_for(kind)
        transcoders = transcoders_for(lab, tmp_path / kind, architecture)
        engine = engine_for(lab, transcoders)
        evaluated = engine.evaluate(engine.real_leaves())
        assert torch.allclose(evaluated.logits, real_logits(lab, transcoders), atol=1e-4), kind
        # Feature targets take the value the transcoder's encoder gave them.
        assert torch.allclose(evaluated.targets, engine.real_targets, atol=1e-4), kind


def test_replacement_logits_hold_with_capped_features(setup):
    lab, transcoders = setup
    engine = engine_for(lab, transcoders, max_features=5)
    assert sum(len(a) for a in engine.real_leaves().acts.values()) == 5
    evaluated = engine.evaluate(engine.real_leaves())
    assert torch.allclose(evaluated.logits, real_logits(lab, transcoders), atol=1e-4)


def test_edges_and_bias_sum_to_each_targets_value(setup):
    lab, transcoders = setup
    graph = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING)
    data = graph.to_dict()
    incoming = {}
    for edge in data["edges"]:
        incoming[edge["target"]] = incoming.get(edge["target"], 0.0) + edge["weight"]
    targets = [n for n in data["nodes"] if n["kind"] in ("feature", "logit")]
    assert targets
    for node in targets:
        total = incoming.get(node["id"], 0.0) + node["bias"]
        assert abs(total - node["activation"]) < 1e-4, node["id"]
    assert data["metadata"]["checks"]["edge_sum_max_error"] < 1e-4


def test_graph_values_are_the_models_own(setup):
    lab, transcoders = setup
    graph = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING)
    logits = real_logits(lab, transcoders)
    centered = logits - logits.mean()
    for node in graph.nodes:
        if node["kind"] == "logit":
            assert node["activation"] == pytest.approx(float(centered[node["token_id"]]), abs=1e-4)
    inputs = next(iter(transcoders.values()))._tokenized(PROMPT)
    for layer, transcoder in transcoders.items():
        features = transcoder.features(inputs, positions="all")["features"]
        for node in graph.nodes:
            if node["kind"] == "feature" and node["layer"] == layer:
                value = float(features[node["position"], node["feature"]])
                assert node["activation"] == pytest.approx(value, abs=1e-4)


def test_a_direct_edge_matches_finite_differences_in_the_frozen_model(setup):
    lab, transcoders = setup
    graph = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING)
    engine = engine_for(lab, transcoders)
    edges = sorted(graph.to_dict()["edges"], key=lambda e: -abs(e["weight"]))
    kinds = {n["id"]: n["kind"] for n in graph.nodes}
    values = {n["id"]: n["activation"] for n in graph.nodes}
    chosen = []
    for want in ("logit", "feature"):
        chosen += [e for e in edges if kinds[e["target"]] == want][:3]
    assert chosen
    delta = 0.05
    base = engine.evaluate(engine.real_leaves())
    for edge in chosen:
        value = {}
        for sign in (1, -1):
            leaves = engine.scaled(engine.real_leaves(), edge["source"], 1 + sign * delta)
            value[sign] = engine.target_value(engine.evaluate(leaves), edge["target"])
        slope = (value[1] - value[-1]) / (2 * delta)
        assert slope == pytest.approx(edge["weight"], abs=1e-3, rel=1e-3), edge
        assert engine.target_value(base, edge["target"]) == pytest.approx(
            values[edge["target"]], abs=1e-3
        )


# The graph


def test_nodes_and_metadata(setup):
    lab, transcoders = setup
    before = lab.forward_passes
    graph = lab.attribution_graph(PROMPT, transcoders, max_logits=3, **EVERYTHING)
    assert lab.forward_passes - before == 3
    data = graph.to_dict()
    meta = data["metadata"]
    assert meta["prompt"] == PROMPT
    assert meta["tokens"] == [n["token"] for n in data["nodes"] if n["kind"] == "embedding"]
    assert meta["forward_passes"] == 3
    assert meta["model_id"] == "test-model"
    assert len(meta["transcoders"]) == 2
    assert 0.0 <= meta["replacement_score"] <= 1.0
    assert meta["settings"]["max_logits"] == 3
    kinds = [n["kind"] for n in data["nodes"]]
    positions = len(meta["tokens"])
    assert kinds.count("embedding") == positions
    assert kinds.count("error") == 2 * positions
    assert 1 <= kinds.count("logit") <= 3
    ids = [n["id"] for n in data["nodes"]]
    assert len(set(ids)) == len(ids)
    for node in data["nodes"]:
        assert {"id", "kind", "layer", "position", "token", "feature", "activation"} <= set(node)
        assert node["influence"] >= 0
    # Everything kept: every active feature is a node.
    for layer, transcoder in transcoders.items():
        inputs = transcoder._tokenized(PROMPT)
        active = int((transcoder.features(inputs, positions="all")["features"] > 0).sum())
        assert sum(1 for n in data["nodes"] if n["kind"] == "feature" and n["layer"] == layer) == (
            active
        )


def test_logit_nodes_cover_the_requested_probability(lab_for, tmp_path):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)
    probabilities = torch.softmax(real_logits(lab, transcoders), -1)
    graph = lab.attribution_graph(PROMPT, transcoders, max_logits=10, logit_probability=0.5)
    nodes = [n for n in graph.nodes if n["kind"] == "logit"]
    covered = sum(float(probabilities[n["token_id"]]) for n in nodes)
    assert covered >= 0.5 or len(nodes) == min(10, probabilities.numel())
    assert nodes[0]["token_id"] == int(probabilities.argmax())
    if len(nodes) > 1:
        assert covered - float(probabilities[nodes[-1]["token_id"]]) < 0.5


def test_pruning_keeps_logits_and_their_strongest_parents(lab_for, tmp_path):
    lab = lab_for("llama_gqa")
    transcoders = transcoders_for(lab, tmp_path)
    full = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING)
    pruned = lab.attribution_graph(PROMPT, transcoders, node_threshold=0.5, edge_threshold=0.5)
    full_ids = {n["id"] for n in full.nodes}
    pruned_ids = {n["id"] for n in pruned.nodes}
    assert pruned_ids < full_ids
    assert {n["id"] for n in full.nodes if n["kind"] == "logit"} <= pruned_ids
    edges = pruned.to_dict()["edges"]
    assert all(e["source"] in pruned_ids and e["target"] in pruned_ids for e in edges)
    assert len(edges) < len(full.to_dict()["edges"])
    top = next(n["id"] for n in full.nodes if n["kind"] == "logit")
    parents = [e for e in full.to_dict()["edges"] if e["target"] == top]
    strongest = max(parents, key=lambda e: abs(e["weight"]))
    assert strongest["source"] in pruned_ids
    assert any(e["target"] == top and e["source"] == strongest["source"] for e in edges)
    # Edge weights are not rescaled by pruning.
    weights = {(e["source"], e["target"]): e["weight"] for e in full.to_dict()["edges"]}
    for e in edges:
        assert e["weight"] == pytest.approx(weights[(e["source"], e["target"])])


def test_thresholds_of_one_keep_every_edge_with_a_weight(lab_for, tmp_path):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)
    graph = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING)
    data = graph.to_dict()
    assert all(e["weight"] != 0 for e in data["edges"])
    positions = len(data["metadata"]["tokens"])
    feature_nodes = [n for n in data["nodes"] if n["kind"] == "feature"]
    # A layer-0 feature at the last position reads only embeddings, through frozen attention.
    embeddings = {n["position"]: n["id"] for n in data["nodes"] if n["kind"] == "embedding"}
    first = next(n for n in feature_nodes if n["layer"] == 0 and n["position"] == positions - 1)
    sources = {e["source"] for e in data["edges"] if e["target"] == first["id"]}
    assert sources and sources <= set(embeddings.values())
    assert embeddings[positions - 1] in sources


def test_invalid_settings_are_refused(lab_for, tmp_path):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)
    for bad in [
        {"max_logits": 0},
        {"logit_probability": 0.0},
        {"logit_probability": 1.5},
        {"max_features": 0},
        {"node_threshold": 0.0},
        {"edge_threshold": 1.2},
        {"batch_size": 0},
    ]:
        with pytest.raises(ValueError, match=next(iter(bad))):
            lab.attribution_graph(PROMPT, transcoders, **bad)
    with pytest.raises(ValueError, match="prompt"):
        lab.attribution_graph("", transcoders)


def test_batch_size_does_not_change_the_edges(lab_for, tmp_path):
    lab = lab_for("llama_gqa")
    transcoders = transcoders_for(lab, tmp_path)
    one = lab.attribution_graph(PROMPT, transcoders, batch_size=1, **EVERYTHING).to_dict()
    many = lab.attribution_graph(PROMPT, transcoders, batch_size=7, **EVERYTHING).to_dict()
    key = lambda e: (e["source"], e["target"])  # noqa: E731
    a = {key(e): e["weight"] for e in one["edges"]}
    b = {key(e): e["weight"] for e in many["edges"]}
    assert a.keys() == b.keys()
    for name in a:
        assert a[name] == pytest.approx(b[name], abs=1e-5, rel=1e-5)


def test_list_and_dict_transcoders_agree(lab_for, tmp_path):
    lab = lab_for("gpt2")
    by_layer = transcoders_for(lab, tmp_path)
    one = lab.attribution_graph(PROMPT, by_layer, **EVERYTHING).to_dict()
    two = lab.attribution_graph(PROMPT, [by_layer[1], by_layer[0]], **EVERYTHING).to_dict()
    assert one["nodes"] == two["nodes"]


# Export


def test_to_json_round_trips(setup, tmp_path):
    lab, transcoders = setup
    graph = lab.attribution_graph(PROMPT, transcoders)
    path = graph.to_json(tmp_path / "graph.json")
    assert json.loads(path.read_text()) == graph.to_dict()
    again = graphs.AttributionGraph.from_json(path)
    assert again.to_dict() == graph.to_dict()
    with pytest.raises(ValueError, match="loaded from a file"):
        with again.intervene([again.nodes[0]["id"]]):
            pass


def test_top_paths_follow_the_strongest_edges(lab_for, tmp_path):
    lab = lab_for("llama_gqa")
    transcoders = transcoders_for(lab, tmp_path)
    graph = lab.attribution_graph(PROMPT, transcoders, node_threshold=0.5, edge_threshold=0.6)
    logit = next(n["id"] for n in graph.nodes if n["kind"] == "logit")
    paths = graph.top_paths(logit, 4)
    assert 1 <= len(paths) <= 4
    scores = [p["score"] for p in paths]
    assert scores == sorted(scores, reverse=True)
    edges = {(e["source"], e["target"]): e["weight"] for e in graph.to_dict()["edges"]}
    kinds = {n["id"]: n["kind"] for n in graph.nodes}
    for path in paths:
        assert path["nodes"][-1] == logit
        assert kinds[path["nodes"][0]] in ("embedding", "error")
        pairs = list(zip(path["nodes"], path["nodes"][1:], strict=False))
        assert path["weights"] == [pytest.approx(edges[pair]) for pair in pairs]
    # Each score is the product of the edges' shares of their targets' incoming weight.
    totals = {}
    for (_source, target), weight in edges.items():
        totals[target] = totals.get(target, 0.0) + abs(weight)
    for path in paths:
        product = 1.0
        for pair in zip(path["nodes"], path["nodes"][1:], strict=False):
            product *= abs(edges[pair]) / totals[pair[1]]
        assert path["score"] == pytest.approx(product, rel=1e-4)
    # And they are the best there are: search every path.
    incoming = {}
    for (source, target), weight in edges.items():
        incoming.setdefault(target, []).append((source, abs(weight) / totals[target]))
    every = []

    def walk(node, score):
        if node not in incoming:
            every.append(score)
            return
        for source, share in incoming[node]:
            walk(source, score * share)

    walk(logit, 1.0)
    assert [p["score"] for p in paths] == pytest.approx(sorted(every, reverse=True)[:4], rel=1e-4)
    assert len({tuple(p["nodes"]) for p in paths}) == len(paths)
    with pytest.raises(ValueError, match="unknown node"):
        graph.top_paths("nope", 2)


# Interventions


def strongest_last_layer_parent(graph, transcoders):
    """A last-layer feature at the last position with the strongest edge to the top logit."""
    nodes = {n["id"]: n for n in graph.nodes}
    logit = next(n for n in graph.nodes if n["kind"] == "logit")
    last, final = max(transcoders), len(graph.metadata["tokens"]) - 1
    candidates = [
        e
        for e in graph.edges
        if e["target"] == logit["id"]
        and nodes[e["source"]]["kind"] == "feature"
        and nodes[e["source"]]["layer"] == last
        and nodes[e["source"]]["position"] == final
    ]
    return logit, max(candidates, key=lambda e: abs(e["weight"])), nodes


def test_measure_moves_the_logit_the_way_the_edge_says(lab_for, tmp_path):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)
    graph = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING)
    logit, edge, nodes = strongest_last_layer_parent(graph, transcoders)
    # A last-layer feature at the last position reaches the logit only through the final
    # norm, so the graph's edge predicts the effect of removing it.
    before, passes = snapshot(lab), lab.forward_passes
    result = graph.measure([edge["source"]])
    assert lab.forward_passes - passes == 2
    assert_restored(lab, before)
    assert result["baseline"][logit["id"]] == pytest.approx(logit["activation"], abs=1e-4)
    effect = result["intervened"][logit["id"]] - result["baseline"][logit["id"]]
    assert result["effect"][logit["id"]] == pytest.approx(effect)
    assert effect * edge["weight"] < 0  # setting a feature to zero removes its contribution
    assert abs(effect + edge["weight"]) < abs(edge["weight"])  # and by roughly that much
    assert set(result["baseline"]) == {n["id"] for n in graph.nodes if n["kind"] == "logit"}
    # Setting a feature to the value it already has changes nothing.
    same = graph.measure([edge["source"]], value=nodes[edge["source"]]["activation"])
    assert same["effect"][logit["id"]] == pytest.approx(0.0, abs=1e-5)


def test_inputs_are_the_tokens_the_graph_ran(lab_for, tmp_path):
    lab = lab_for("gpt2")
    graph = lab.attribution_graph(PROMPT, transcoders_for(lab, tmp_path), **EVERYTHING)
    assert graph.inputs["input_ids"][0].tolist() == graph.metadata["input_ids"]
    with torch.no_grad():
        logits = lab.model(**graph.inputs).logits[0, -1]
    top = next(n for n in graph.nodes if n["kind"] == "logit")
    assert float((logits - logits.mean())[top["token_id"]]) == pytest.approx(
        top["activation"], abs=1e-4
    )


def test_intervene_on_the_graphs_inputs_matches_measure(lab_for, tmp_path):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)
    graph = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING)
    logit, edge, _ = strongest_last_layer_parent(graph, transcoders)
    with graph.intervene([edge["source"]]), torch.no_grad():
        row = lab.model(**graph.inputs).logits[0, -1]
    after = float((row - row.mean())[logit["token_id"]])
    assert after == pytest.approx(graph.measure([edge["source"]])["intervened"][logit["id"]])


def test_intervene_refuses_a_forward_on_other_tokens(lab_for, tmp_path):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)
    graph = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING)
    _, edge, _ = strongest_last_layer_parent(graph, transcoders)
    before = snapshot(lab)
    for run in (
        lambda: lab.score(PROMPT, [" yes", " no"], batch_size=1),
        lambda: lab.score(PROMPT, ["yes"], batch_size=1),
        lambda: lab.model(input_ids=torch.tensor([[2, 3]])),
    ):
        with pytest.raises(ValueError, match="graph.measure"):
            with graph.intervene([edge["source"]]):
                run()
    assert_restored(lab, before)


def test_measure_refuses_what_intervene_refuses(lab_for, tmp_path):
    lab = lab_for("gpt2")
    graph = lab.attribution_graph(PROMPT, transcoders_for(lab, tmp_path), **EVERYTHING)
    node = next(n["id"] for n in graph.nodes if n["kind"] == "logit")
    before = lab.forward_passes
    with pytest.raises(ValueError, match="feature"):
        graph.measure([node])
    with pytest.raises(ValueError, match="unknown node"):
        graph.measure(["feat.9.9.9"])
    assert lab.forward_passes == before


def test_intervene_refuses_what_is_not_a_feature(lab_for, tmp_path):
    lab = lab_for("gpt2")
    graph = lab.attribution_graph(PROMPT, transcoders_for(lab, tmp_path), **EVERYTHING)
    for kind in ("embedding", "error", "logit"):
        node = next(n["id"] for n in graph.nodes if n["kind"] == kind)
        with pytest.raises(ValueError, match="feature"):
            with graph.intervene([node]):
                pass
    with pytest.raises(ValueError, match="unknown node"):
        with graph.intervene(["feat.9.9.9"]):
            pass


# Refusals


def test_a_missing_layer_is_refused(lab_for, tmp_path):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path, layers=(0,))
    with pytest.raises(ValueError, match="every MLP.*layer 1"):
        lab.attribution_graph(PROMPT, transcoders)


def test_a_plain_sae_is_refused(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(tmp_path / "plain", "blocks.0.hook_resid_post")
    plain = lab.load_sae(path=tmp_path / "plain")
    transcoders = transcoders_for(lab, tmp_path, layers=(1,))
    with pytest.raises(ValueError, match="not a transcoder"):
        lab.attribution_graph(PROMPT, {0: plain, 1: transcoders[1]})


def test_a_transcoder_that_does_not_replace_an_mlp_is_refused(lab_for, tmp_path):
    lab = lab_for("gpt2")
    make_sae(
        tmp_path / "wrong",
        "blocks.0.hook_resid_pre",
        architecture="transcoder",
        hook_name_out="blocks.0.hook_resid_post",
        d_out=D,
    )
    wrong = lab.load_sae(path=tmp_path / "wrong")
    other = transcoders_for(lab, tmp_path, layers=(1,))[1]
    with pytest.raises(ValueError, match="MLP's input and writes"):
        lab.attribution_graph(PROMPT, [wrong, other])


def test_two_transcoders_for_one_layer_are_refused(lab_for, tmp_path):
    lab = lab_for("gpt2")
    first = transcoders_for(lab, tmp_path / "a")
    second = transcoders_for(lab, tmp_path / "b", seed=7)
    with pytest.raises(ValueError, match="two transcoders for layer 0"):
        lab.attribution_graph(PROMPT, [first[0], second[0], first[1]])


def test_a_transcoder_from_another_session_is_refused(lab_for, tmp_path):
    lab, other = lab_for("gpt2"), lab_for("gpt2")
    foreign = transcoders_for(other, tmp_path)
    with pytest.raises(ValueError, match="another session"):
        lab.attribution_graph(PROMPT, foreign)


def test_an_unsupported_architecture_is_refused(lab_for, tmp_path, monkeypatch):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)

    def refuse(layers=None):
        raise NotImplementedError("a fused query_key_value (GPT-NeoX, Falcon, Bloom)")

    monkeypatch.setattr(lab, "heads", refuse)
    with pytest.raises(ValueError, match="do not support this architecture.*query_key_value"):
        lab.attribution_graph(PROMPT, transcoders)


def test_a_layer_without_softmax_attention_is_refused(tiny_model, make_lab, tmp_path):
    model, tokenizer = tiny_model("qwen3_5")
    lab = make_lab(model, tokenizer)
    make_sae(
        tmp_path / "tc",
        "blocks.1.ln2.hook_normalized",
        architecture="transcoder",
        hook_name_out="blocks.1.hook_mlp_out",
        d_out=D,
    )
    with pytest.raises(ValueError, match="attention"):
        lab.attribution_graph(PROMPT, [lab.load_sae(path=tmp_path / "tc")])


def test_a_half_precision_model_is_refused(lab_for, tmp_path):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)
    lab.model.to(torch.bfloat16)
    with pytest.raises(ValueError, match="float32"):
        lab.attribution_graph(PROMPT, transcoders)


def test_a_transcoder_with_runtime_normalization_is_refused(lab_for, tmp_path):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)
    transcoders[0].sae.cfg.normalize_activations = "layer_norm"
    with pytest.raises(ValueError, match="normalize_activations"):
        lab.attribution_graph(PROMPT, transcoders)


# Cleanup and budget


def test_nothing_is_left_behind_after_a_call(setup):
    lab, transcoders = setup
    lab.model.train()
    before = snapshot(lab)
    lab.attribution_graph(PROMPT, transcoders, max_features=6)
    assert_restored(lab, before)


def test_nothing_is_left_behind_after_a_failure(setup, monkeypatch):
    lab, transcoders = setup
    before = snapshot(lab)
    real = lab.model.forward
    calls = []

    def fail_on_the_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(lab.model, "forward", fail_on_the_second)
    with pytest.raises(RuntimeError, match="boom"):
        lab.attribution_graph(PROMPT, transcoders)
    assert len(calls) == 2
    monkeypatch.undo()
    assert_restored(lab, before)


def test_the_pass_budget_is_checked_before_anything_runs(lab_for, tmp_path):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)
    lab._budget.limit = lab.forward_passes + 2
    before, used = snapshot(lab), lab.forward_passes
    from oumi_weights_lab import BudgetExceeded

    with pytest.raises(BudgetExceeded):
        lab.attribution_graph(PROMPT, transcoders)
    assert lab.forward_passes == used
    assert_restored(lab, before)


def test_a_capped_graph_still_sums_and_absorbs_the_rest_into_errors(setup):
    lab, transcoders = setup
    graph = lab.attribution_graph(PROMPT, transcoders, max_features=7, **EVERYTHING)
    data = graph.to_dict()
    assert sum(1 for n in data["nodes"] if n["kind"] == "feature") == 7
    incoming = {}
    for edge in data["edges"]:
        incoming[edge["target"]] = incoming.get(edge["target"], 0.0) + edge["weight"]
    for node in data["nodes"]:
        if node["kind"] in ("feature", "logit"):
            total = incoming.get(node["id"], 0.0) + node["bias"]
            assert abs(total - node["activation"]) < 1e-4, node["id"]
    # The kept features are the largest ones.
    full = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING).to_dict()
    largest = sorted(
        (n["activation"] for n in full["nodes"] if n["kind"] == "feature"), reverse=True
    )[:7]
    kept = sorted((n["activation"] for n in data["nodes"] if n["kind"] == "feature"), reverse=True)
    assert kept == pytest.approx(largest)


def test_targets_without_a_batching_rule_fall_back_to_one_backward_each(
    lab_for, tmp_path, monkeypatch
):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)
    expected = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING).to_dict()["edges"]
    real = torch.autograd.grad

    def refuse_batches(*args, **kwargs):
        if kwargs.get("is_grads_batched"):
            raise RuntimeError("no batching rule")
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", refuse_batches)
    again = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING).to_dict()["edges"]
    assert len(again) == len(expected)
    for a, b in zip(sorted(again, key=str), sorted(expected, key=str), strict=True):
        assert (a["source"], a["target"]) == (b["source"], b["target"])
        assert a["weight"] == pytest.approx(b["weight"], abs=1e-5, rel=1e-5)


# Skip transcoders


def within_float32(graph):
    """The edge sums hold to 1e-4 of the size of the values, which float32 sums can reach."""
    size = max(abs(n["activation"]) for n in graph.nodes) + max(
        abs(n.get("bias", 0.0)) for n in graph.nodes
    )
    return graph.metadata["checks"]["edge_sum_max_error"] < 1e-4 * max(1.0, size)


def test_a_skip_connection_is_a_path_not_an_error(lab_for, tmp_path):
    for kind in ("gpt2", "llama_gqa", "gemma2"):
        lab = lab_for(kind)
        transcoders = transcoders_for(lab, tmp_path / kind, "skip_transcoder")
        assert all(hasattr(t.sae, "W_skip") for t in transcoders.values())
        graph = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING)
        edges = {(e["source"], e["target"]): e["weight"] for e in graph.edges}
        assert within_float32(graph)
        error = {n["id"]: n["activation"] for n in graph.nodes if n["kind"] == "error"}
        for transcoder in transcoders.values():
            transcoder.sae.W_skip.data.mul_(2.0)
        doubled = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING)
        assert within_float32(doubled)
        changed = {n["id"]: n["activation"] for n in doubled.nodes if n["kind"] == "error"}
        assert any(abs(changed[k] - error[k]) > 1e-3 for k in error), kind
        again = {(e["source"], e["target"]): e["weight"] for e in doubled.edges}
        assert any(abs(again[k] - edges[k]) > 1e-3 for k in edges.keys() & again.keys()), kind


# Memory


def test_a_graph_over_the_memory_limit_is_refused_naming_max_features(lab_for, tmp_path):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)
    before = snapshot(lab)
    with pytest.raises(ValueError, match="max_features") as raised:
        lab.attribution_graph(PROMPT, transcoders, max_memory_gb=1e-6)
    assert "max_memory_gb" in str(raised.value)
    assert_restored(lab, before)
    for bad in (0, -1, "8"):
        with pytest.raises(ValueError, match="max_memory_gb"):
            lab.attribution_graph(PROMPT, transcoders, max_memory_gb=bad)


def test_features_are_capped_by_default(lab_for):
    import inspect

    parameters = inspect.signature(type(lab_for("gpt2")).attribution_graph).parameters
    assert parameters["max_features"].default == 10_000
    assert parameters["max_memory_gb"].default == 8


def test_the_estimate_grows_with_the_square_of_the_nodes():
    small, large = graphs._estimate_gb(100, 120), graphs._estimate_gb(1000, 1200)
    assert large == pytest.approx(small * 100)
    assert 3 < graphs._estimate_gb(20_000, 21_000) < 8


# Pruning


def test_pruning_keeps_embeddings_and_drops_isolated_nodes(lab_for, tmp_path):
    lab = lab_for("llama_gqa")
    transcoders = transcoders_for(lab, tmp_path)
    for node_threshold, edge_threshold in [(0.3, 0.5), (0.8, 0.98)]:
        graph = lab.attribution_graph(
            PROMPT, transcoders, node_threshold=node_threshold, edge_threshold=edge_threshold
        )
        kinds = {n["id"]: n["kind"] for n in graph.nodes}
        assert sum(k == "embedding" for k in kinds.values()) == len(graph.metadata["tokens"])
        touched = {e["source"] for e in graph.edges} | {e["target"] for e in graph.edges}
        assert all(node in touched for node, kind in kinds.items() if kind != "embedding")
        sources = {e["source"] for e in graph.edges}
        assert all(node in sources for node, kind in kinds.items() if kind in ("feature", "error"))


def test_edges_are_scored_on_the_pruned_graph(lab_for, tmp_path):
    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)
    graph = lab.attribution_graph(PROMPT, transcoders, node_threshold=0.5, edge_threshold=0.9)
    full = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING)
    keep = {n["id"] for n in graph.nodes}
    weights = {
        (e["source"], e["target"]): abs(e["weight"])
        for e in full.edges
        if e["source"] in keep and e["target"] in keep
    }
    totals = {}
    for (_source, target), weight in weights.items():
        totals[target] = totals.get(target, 0.0) + weight
    shares = {k: w / totals[k[1]] for k, w in weights.items()}
    logits = [n["id"] for n in graph.nodes if n["kind"] == "logit"]
    # Influence on the logits over the kept nodes only, with each logit weighted by its
    # probability (the graph records the same weights in its node activations' softmax).
    probabilities = {n["id"]: n["probability"] for n in graph.nodes if n["kind"] == "logit"}
    total = dict(probabilities)
    frontier = dict(probabilities)
    for _ in range(10):
        step = {}
        for (source, target), share in shares.items():
            if target in frontier:
                step[source] = step.get(source, 0.0) + frontier[target] * share
        for node, value in step.items():
            total[node] = total.get(node, 0.0) + value
        frontier = step
        if not frontier:
            break
    assert logits
    scores = {k: share * total.get(k[1], 0.0) for k, share in shares.items()}
    kept = {(e["source"], e["target"]) for e in graph.edges}
    assert kept <= set(scores)
    covered = sum(v for k, v in scores.items() if k in kept)
    assert covered >= 0.9 * sum(scores.values()) - 1e-6


# Labels


def test_labels_are_opt_in_and_use_the_network_only_when_asked(lab_for, tmp_path, monkeypatch):
    from oumi_weights_lab import saes

    lab = lab_for("gpt2")
    transcoders = transcoders_for(lab, tmp_path)
    for transcoder in transcoders.values():
        transcoder.neuronpedia_id = "test/model"
    calls = []

    def fetch(neuronpedia_id, index, timeout):
        calls.append(index)
        return f"label {index}"

    monkeypatch.setattr(saes, "_fetch_label", fetch)
    plain = lab.attribution_graph(PROMPT, transcoders, **EVERYTHING)
    assert calls == [] and not any("label" in n for n in plain.nodes)
    labelled = lab.attribution_graph(PROMPT, transcoders, labels=True, **EVERYTHING)
    assert calls
    for node in labelled.nodes:
        if node["kind"] == "feature":
            assert node["label"] == f"label {node['feature']}"
        else:
            assert "label" not in node
    assert labelled.metadata["settings"]["labels"] is True
