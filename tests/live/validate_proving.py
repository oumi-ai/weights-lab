"""Build the calibration tasks on real GPT-2 small and grade their keys, and sweep planted seeds.

Run ``python tests/live/validate_proving.py --output-dir /tmp/proving``. It downloads
``openai-community/gpt2`` (about 500 MB), builds ``gpt2-ioi`` and ``gpt2-induction``, prints
the baseline accuracy and the accuracy with the key heads zeroed, and checks that each key,
written as a claim with honest numbers, scores 1 and is solved. With ``--seeds N`` it also
builds the first N dev tasks of every planted family and prints how often they validate
on the first attempt.
"""

import argparse
import json
from pathlib import Path

from oumi_weights_lab.proving import build_task, grade_claim, load_key
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=0)
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()
    if not args.skip_calibration:
        calibration(args.output_dir)
    if args.seeds:
        sweep(args.output_dir, args.seeds)


if __name__ == "__main__":
    main()
