"""Build the calibration tasks on real GPT-2 small and grade their keys, and sweep planted seeds.

Run ``python tests/live/validate_proving.py --output-dir /tmp/proving``. It downloads
``openai-community/gpt2`` (about 500 MB), builds ``gpt2-ioi`` and ``gpt2-induction``, prints
the baseline accuracy and the accuracy with the key heads zeroed, and checks that each key,
written as a claim with honest numbers, scores 1 and is solved. With ``--seeds N`` it also
builds the first N dev tasks of every tiny planted family and prints how often they validate
on the first attempt.

With ``--gpt2-seeds N`` it builds the first N dev tasks of each family planted inside GPT-2
small (or of each ``--gpt2-family`` given) on CPU, prints each build's validation numbers as
one JSON line and a summary per family, and grades the key of each family's first task as an
honest claim. Each build takes 6 to 35 minutes; task directories (about 500 MB each) are
deleted once graded, and the keys stay in ``<output-dir>/gpt2-keys``.

Each GPT-2 build is also attacked from its weights alone, with no forward pass and no copy
of stock GPT-2: the solver that read the key off the first version of these families (a
head with a quarter of its value singular values under 10% of the top is planted, the count
names the family, the lowest value-output rank is the decoy), and, for every statistic of
``ranking_statistics``, a ranking of all 144 heads by how unusual the head is within its
layer and among all heads, plus a ranking by the sum of each head's within-layer scores. It
prints how often each attack names the exact key and where the key heads fall.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from oumi_weights_lab.proving import build_task, grade_claim, load_key
from oumi_weights_lab.proving.gpt2_planted import GPT2_FAMILIES
from oumi_weights_lab.proving.planted import PLANTED_FAMILIES


def honest_claim(key):
    check = key["validation"]
    return {
        "version": 1,
        "task": key["task"],
        "components": key["components"],
        "ablation": {
            "accuracy_before": check["baseline_accuracy"],
            "accuracy_after": check["key_ablated_accuracy"],
            "rows": check["rows"],
        },
    }


def calibration(out: Path) -> None:
    for family in ("gpt2-ioi", "gpt2-induction"):
        info = build_task(family, "calibration", 0, out / "tasks", out / "keys")
        key = load_key(info["key_path"])
        check = key["validation"]
        print(
            family,
            "heads:",
            len(key["components"]),
            "baseline:",
            check["baseline_accuracy"],
            "key zeroed:",
            check["key_ablated_accuracy"],
        )
        result = grade_claim(info["task_dir"], key, honest_claim(key), out / "work")
        print(json.dumps({k: v for k, v in result.items() if k != "measured"}))
        print("measured:", result["measured"])
        assert result["component_f1"] == 1.0, result
        assert result["ablation_reproduced"] == 1, result
        assert result["solved"], result


def sweep(out: Path, seeds: int) -> None:
    for family in PLANTED_FAMILIES:
        attempts = []
        for index in range(seeds):
            info = build_task(family, "dev", index, out / "sweep", out / "sweep-keys")
            attempts.append(load_key(info["key_path"])["validation"]["attempt"])
        first = sum(a == 0 for a in attempts)
        print(f"{family}: {first}/{seeds} validated on the first attempt; attempts {attempts}")


GPT2_NUMBERS = (
    "attempt",
    "word_draw",
    "native_accuracy",
    "baseline_accuracy",
    "baseline_margin",
    "key_ablated_accuracy",
    "only_key_accuracy",
    "key_head_ablated_alone",
    "key_pair_ablated",
    "backup_attention_to_flag",
    "decoy_effect",
    "largest_other_effect",
    "write_size",
)


def solve_from_weights(model) -> dict:
    """The weights-only solver from the review of the first GPT-2 families, unchanged in
    substance: a head with a quarter or more of its value singular values under 10% of the
    top counts as planted, the count names the family, the planted head with the lowest
    value-output rank is the decoy, and the rest are the key."""
    import torch

    config = model.config
    width, size = config.n_embd, config.n_embd // config.n_head
    found = []
    for layer, block in enumerate(model.transformer.h):
        weight = block.attn.c_attn.weight.detach().double()
        out = block.attn.c_proj.weight.detach().double()
        for h in range(config.n_head):
            value = weight[:, 2 * width + h * size : 2 * width + (h + 1) * size]
            values = torch.linalg.svdvals(value)
            if int((values < 0.1 * values[0]).sum()) >= size // 4:
                circuit = torch.linalg.svdvals(value @ out[h * size : (h + 1) * size])
                found.append((f"{layer}.{h}", int((circuit > 0.1 * circuit[0]).sum())))
    family = {4: "gpt2-copy-redundant", 3: "gpt2-copy-selfrepair"}.get(len(found))
    decoy = min(found, key=lambda item: item[1])[0] if found else None
    return {
        "found": [name for name, _ in found],
        "family": family,
        "key": sorted(name for name, _ in found if name != decoy),
    }


def ranking_statistics(model) -> dict[str, dict[str, float]]:
    """Every head's statistics from ``head_statistics``, plus the ratio of the first to the
    second singular value of each product and each matrix's condition number."""
    import torch

    from oumi_weights_lab.proving.gpt2_planted import _spectrum, head_statistics

    stats = head_statistics(model)
    config = model.config
    width, size = config.n_embd, config.n_embd // config.n_head
    for layer, block in enumerate(model.transformer.h):
        weight = block.attn.c_attn.weight.detach().double()
        out = block.attn.c_proj.weight.detach().double()
        for h in range(config.n_head):
            parts = {
                name: weight[:, offset + h * size : offset + (h + 1) * size]
                for name, offset in (("q", 0), ("k", width), ("v", 2 * width))
            }
            parts["o"] = out[h * size : (h + 1) * size].T
            entry = stats[f"{layer}.{h}"]
            for name, (a, b) in (("qk", ("q", "k")), ("ov", ("v", "o"))):
                values = _spectrum(parts[a], parts[b])
                entry[f"{name}_ratio"] = float(values[0] / values[1])
            for name, part in parts.items():
                values = torch.linalg.svdvals(part)
                entry[f"{name}_cond"] = float(values[0] / values[-1])
    return stats


def _robust_z(values: dict[str, float], groups: dict[str, str]) -> dict[str, float]:
    """How far each value sits from its group's median, in robust standard deviations."""
    import statistics

    members: dict[str, list[float]] = {}
    for name, value in values.items():
        members.setdefault(groups[name], []).append(value)
    scores = {}
    for name, value in values.items():
        group = members[groups[name]]
        middle = statistics.median(group)
        spread = 1.4826 * statistics.median(abs(x - middle) for x in group)
        scores[name] = abs(value - middle) / max(spread, 1e-9 * (abs(middle) + 1))
    return scores


def rank_heads(model) -> dict[str, list[str]]:
    """For each statistic, every head ordered from most to least unusual, measured against
    its own layer (``layer:<stat>``) and against all heads (``all:<stat>``); and ``combined``,
    the sum of each head's within-layer scores (each capped at 10)."""
    stats = ranking_statistics(model)
    names = list(stats)
    by_layer = {name: name.split(".")[0] for name in names}
    everyone = dict.fromkeys(names, "all")
    rankings, combined = {}, dict.fromkeys(names, 0.0)
    for stat in next(iter(stats.values())):
        values = {name: stats[name][stat] for name in names}
        for scope, groups in (("layer", by_layer), ("all", everyone)):
            scores = _robust_z(values, groups)
            rankings[f"{scope}:{stat}"] = sorted(names, key=lambda n: -scores[n])
            if scope == "layer":
                for name in names:
                    combined[name] += min(scores[name], 10.0)
    rankings["combined"] = sorted(names, key=lambda n: -combined[n])
    return rankings


def attack(model, key: dict) -> dict:
    """What the weights alone say about this task: the solver's guess, and where the key
    heads and the decoys fall in each ranking (1 is the most unusual head)."""
    heads = sorted(f"{c['layer']}.{c['head']}" for c in key["components"])
    decoys = sorted(f"{c['layer']}.{c['head']}" for c in key["decoys"])
    solved = solve_from_weights(model)
    report = {
        "solver": {
            **solved,
            "exact_key": solved["key"] == heads,
            "family_right": solved["family"] == key["family"],
        },
        "rankings": {},
    }
    for name, order in rank_heads(model).items():
        report["rankings"][name] = {
            "exact_key": sorted(order[: len(heads)]) == heads,
            "key_ranks": [order.index(h) + 1 for h in heads],
            "decoy_ranks": [order.index(h) + 1 for h in decoys],
        }
    return report


def attack_summary(reports: list[dict]) -> None:
    """Print, per attack, how often it names the exact key, how often a key head is the
    most unusual head, and the median of each build's best and mean key-head rank."""
    import statistics

    print(
        f"weights-only solver: exact key {sum(r['solver']['exact_key'] for r in reports)}/"
        f"{len(reports)}, family {sum(r['solver']['family_right'] for r in reports)}/"
        f"{len(reports)}, heads flagged per build "
        f"{[len(r['solver']['found']) for r in reports]}"
    )
    rows = []
    for name in reports[0]["rankings"]:
        found = [r["rankings"][name] for r in reports]
        best = [min(f["key_ranks"]) for f in found]
        mean = [statistics.mean(f["key_ranks"]) for f in found]
        rows.append(
            (
                statistics.median(mean),
                name,
                sum(f["exact_key"] for f in found),
                sum(b == 1 for b in best),
                statistics.median(best),
            )
        )
    print("attack: exact key / a key head ranked 1st / median best rank / median mean rank")
    print("(144 heads; a ranking blind to the key gives a median mean rank near 72)")
    total = len(reports)
    for mean, name, exact, first, best in sorted(rows):
        print(f"  {name:18s} {exact:2d}/{total}  {first:2d}/{total}  {best:6.1f}  {mean:6.1f}")


def gpt2_sweep(out: Path, seeds: int, families: list[str]) -> None:
    from transformers import AutoModelForCausalLM

    every_report = []
    for family in families:
        checks, reports = [], []
        for index in range(seeds):
            start = time.time()
            info = build_task(family, "dev", index, out / "gpt2-tasks", out / "gpt2-keys")
            key = load_key(info["key_path"])
            check = key["validation"]
            if index == 0:
                result = grade_claim(info["task_dir"], key, honest_claim(key), out / "work")
                print(f"{family} graded: score {result['score']} solved {result['solved']}")
                print("measured:", result["measured"])
                assert result["solved"], result
            model = AutoModelForCausalLM.from_pretrained(Path(info["task_dir"]) / "model")
            report = attack(model, key)
            reports.append(report)
            del model
            shutil.rmtree(info["task_dir"])
            heads = [f"{c['layer']}.{c['head']}" for c in key["components"]]
            decoy = [f"{c['layer']}.{c['head']}" for c in key["decoys"]]
            numbers = {name: check[name] for name in GPT2_NUMBERS if name in check}
            # Where each planted statistic sits in its layer's range: 0 is the smallest, 1 the
            # largest (a range of one value counts as 0.5).
            places = [
                0.5 if high == low else (value - low) / (high - low)
                for parts in check["statistics"].values()
                for part in parts.values()
                if isinstance(part, list)
                for value, low, high in [part]
            ]
            numbers["statistic_place_in_layer"] = [round(min(places), 3), round(max(places), 3)]
            numbers["solver"] = report["solver"]
            numbers["combined_key_ranks"] = report["rankings"]["combined"]["key_ranks"]
            numbers["seconds"] = round(time.time() - start)
            print(
                json.dumps(
                    {"family": family, "index": index, "key": heads, "decoy": decoy, **numbers}
                ),
                flush=True,
            )
            checks.append(check)
        every_report += reports
        print(f"{family} weight attacks over {len(reports)} builds:")
        attack_summary(reports)
        alone = [min(c["key_head_ablated_alone"].values()) for c in checks]
        pairs = [min(c["key_pair_ablated"].values()) for c in checks if "key_pair_ablated" in c]
        print(
            f"{family}: {sum(c['attempt'] == 0 for c in checks)}/{seeds} on the first attempt; "
            f"baseline min {min(c['baseline_accuracy'] for c in checks):.3f}; "
            f"key zeroed max {max(c['key_ablated_accuracy'] for c in checks):.3f}; "
            f"only key kept max {max(c['only_key_accuracy'] for c in checks):.3f}; "
            f"one key head zeroed min {min(alone):.3f}; "
            + (f"two key heads zeroed min {min(pairs):.3f}; " if pairs else "")
            + f"largest other-head effect {max(c['largest_other_effect'] for c in checks):.3f}"
        )
    if len(families) > 1:
        print(f"weight attacks over all {len(every_report)} builds:")
        attack_summary(every_report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=0)
    parser.add_argument("--gpt2-seeds", type=int, default=0)
    parser.add_argument("--gpt2-family", action="append", choices=GPT2_FAMILIES)
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()
    if not args.skip_calibration:
        calibration(args.output_dir)
    if args.seeds:
        sweep(args.output_dir, args.seeds)
    if args.gpt2_seeds:
        gpt2_sweep(args.output_dir, args.gpt2_seeds, args.gpt2_family or list(GPT2_FAMILIES))


if __name__ == "__main__":
    main()
