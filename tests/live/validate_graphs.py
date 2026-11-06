"""Build an attribution graph from public per-layer transcoders and test a path in it.

SAELens has no public per-layer transcoder set for GPT-2 small. Gemma Scope 2 has one for
Gemma 3 270M (16k features, every layer), small enough for the CPU. `google/gemma-3-270m` is
gated, so this loads the ungated copy `unsloth/gemma-3-270m` and passes
`allow_model_mismatch=True` to the transcoders, which were trained on the Google model.

Gemma Scope 2 transcoders record their input as `blocks.L.hook_mlp_in` but read the MLP's own
input, after the pre-MLP norm. `load_sae` maps that hook name to the raw residual stream, so
this script loads each transcoder with `hook_name="blocks.L.mlp.hook_in"` instead, and checks
the result the way `validate_saes.py` does: the fraction of variance unexplained on short
prompts, BOS excluded, must be at most 0.30 for every layer.

The check then builds a graph for a short factual prompt and prints the replacement score,
the top next-token node's strongest parents, and the measured effect of `graph.measure` on the
strongest feature parent and on a control feature (the same layer and position, the weakest
edge to the top logit). It needs about 2 GB of memory and a few minutes:

    python tests/live/validate_graphs.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

PROMPTS = [
    "The capital of France is Paris, and the river Seine runs through it.",
    "She opened the old wooden door and stepped into the quiet kitchen.",
    "In 1969, astronauts landed on the Moon for the first time.",
    "To install the package, run the command in your terminal.",
]
CRITERIA = {
    "passes_when": (
        "every layer's transcoder has fvu <= 0.30 without the BOS token, the graph builds "
        "(its three exactness checks run inside the call), and measure changes the top "
        "token's logit and leaves the model unchanged afterward"
    ),
}


def log(message):
    print(message, file=sys.stderr, flush=True)


def centered(lab, input_ids, token):
    import torch

    with torch.no_grad():
        row = lab.model(input_ids=torch.tensor([input_ids], device=lab.device)).logits[0, -1]
    return float((row - row.mean())[token])


def load_transcoders(lab, release, width, l0):
    found, worst = {}, 0.0
    layers = sorted({entry["layer"] for entry in lab.heads()})
    for layer in layers:
        sae_id = f"layer_{layer}_width_{width}_l0_{l0}_affine"
        transcoder = lab.load_sae(
            release, sae_id, allow_model_mismatch=True, hook_name=f"blocks.{layer}.mlp.hook_in"
        )
        fvu = transcoder.features(PROMPTS, positions=slice(1, None))["fvu"]
        log(f"layer {layer}: fvu {fvu:.4f}")
        worst = max(worst, fvu)
        found[layer] = transcoder
    return found, worst


def main(argv=None):
    from oumi_weights_lab.lab import load_lab

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", default="unsloth/gemma-3-270m")
    parser.add_argument("--release", default="gemma-scope-2-270m-pt-transcoders-all")
    parser.add_argument("--width", default="16k")
    parser.add_argument("--l0", default="small", choices=["small", "big"])
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, help="session directory, default a temp folder")
    args = parser.parse_args(argv)
    session = args.output_dir or Path(tempfile.mkdtemp(prefix="validate-graphs-")) / "session"
    session = session.expanduser().resolve()
    session.mkdir(parents=True, exist_ok=True)

    lab = load_lab(
        {"model": args.model, "device": args.device, "dtype": "float32", "seed": 0}, session
    )
    transcoders, worst = load_transcoders(lab, args.release, args.width, args.l0)

    started = time.time()
    graph = lab.attribution_graph(args.prompt, transcoders)
    elapsed = time.time() - started
    data = graph.to_dict()
    graph.to_json(session / "graph.json")
    meta = data["metadata"]
    log(
        f"graph: {len(data['nodes'])} nodes, {len(data['edges'])} edges in {elapsed:.0f}s, "
        f"replacement score {meta['replacement_score']:.3f}, checks {meta['checks']}"
    )

    nodes = {n["id"]: n for n in data["nodes"]}
    top = next(n for n in data["nodes"] if n["kind"] == "logit")
    parents = sorted(
        (e for e in data["edges"] if e["target"] == top["id"]), key=lambda e: -abs(e["weight"])
    )
    strongest = [
        {**nodes[e["source"]], "weight": e["weight"]}
        for e in parents
        if nodes[e["source"]]["kind"] == "feature"
    ][:5]
    for item in strongest:
        log(
            f"  {item['id']} at {item['token']!r} activation {item['activation']:.2f} "
            f"edge {item['weight']:.3f} label {item.get('label')!r}"
        )

    result = {
        "model": args.model,
        "transcoders": f"{args.release} width {args.width} l0 {args.l0} (affine)",
        "prompt": args.prompt,
        "tokens": meta["tokens"],
        "worst_fvu": worst,
        "nodes": len(data["nodes"]),
        "edges": len(data["edges"]),
        "build_seconds": round(elapsed, 1),
        "replacement_score": meta["replacement_score"],
        "checks": meta["checks"],
        "forward_passes": meta["forward_passes"],
        "top_logit": {k: top[k] for k in ("id", "token", "activation")},
        "strongest_feature_parents": strongest,
        "criteria": CRITERIA,
    }
    ids = meta["input_ids"]
    baseline = centered(lab, ids, top["token_id"])
    result["top_logit"]["measured"] = baseline
    if strongest:
        feature = strongest[0]
        same = [
            e
            for e in parents
            if nodes[e["source"]]["kind"] == "feature"
            and nodes[e["source"]]["layer"] == feature["layer"]
            and nodes[e["source"]]["position"] == feature["position"]
        ]
        control = min(same, key=lambda e: abs(e["weight"])) if len(same) > 1 else None
        effects = {}
        for name, node_id, weight in [
            ("top feature", feature["id"], feature["weight"]),
            (
                "control feature",
                control["source"] if control else None,
                control and control["weight"],
            ),
        ]:
            if node_id is None:
                continue
            moved_by = graph.measure([node_id])["effect"][top["id"]]
            effects[name] = {
                "node": node_id,
                "edge_weight": weight,
                "measured_effect_of_zeroing": moved_by,
            }
            log(f"{name} {node_id}: edge {weight:.3f}, zeroing moved the logit by {moved_by:.3f}")
        result["intervention"] = effects
        after = centered(lab, ids, top["token_id"])
        moved = abs(effects["top feature"]["measured_effect_of_zeroing"]) > 0
        restored = after == baseline
    else:
        moved = restored = False
    state = worst <= 0.30 and moved and restored
    result["state"] = "passed" if state else "failed"
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if main()["state"] != "passed":
        raise SystemExit(1)
