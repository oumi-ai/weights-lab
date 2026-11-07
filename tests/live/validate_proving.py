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
honest claim. Each build takes a few minutes; task directories (about 500 MB each) are
deleted once graded, and the keys stay in ``<output-dir>/gpt2-keys``.
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


def gpt2_sweep(out: Path, seeds: int, families: list[str]) -> None:
    for family in families:
        checks = []
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
            shutil.rmtree(info["task_dir"])
            heads = [f"{c['layer']}.{c['head']}" for c in key["components"]]
            decoy = [f"{c['layer']}.{c['head']}" for c in key["decoys"]]
            numbers = {name: check[name] for name in GPT2_NUMBERS if name in check}
            # Where each planted norm sits in its layer's range: 0 is the smallest, 1 the largest.
            places = [
                (value - low) / (high - low)
                for parts in check["norms"].values()
                for value, low, high in parts.values()
            ]
            numbers["norm_place_in_layer"] = [round(min(places), 3), round(max(places), 3)]
            numbers["seconds"] = round(time.time() - start)
            print(
                json.dumps(
                    {"family": family, "index": index, "key": heads, "decoy": decoy, **numbers}
                ),
                flush=True,
            )
            checks.append(check)
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
