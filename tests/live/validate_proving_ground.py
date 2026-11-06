"""Run both arms of the proving ground on one dev task per planted family, with a real agent.

Needs Docker running, the two images built (see docs/proving-ground.md) and an API key in the
environment. The owner runs it:

    python tests/live/validate_proving_ground.py \\
        --image weights-lab-proving:skill --control-image weights-lab-proving:control \\
        --agent-cmd "claude -p --dangerously-skip-permissions --model sonnet" \\
        --agent-label claude-sonnet --env ANTHROPIC_API_KEY --repeats 1

It builds the tasks in a temporary directory, runs the skill arm and the control arm, prints
each run's score and the paired comparison, and deletes everything afterwards unless
``--keep`` is given. Dev tasks only: nothing held out is touched.
"""

import argparse
import json
import tempfile
from pathlib import Path

from oumi_weights_lab.proving import beats, build_task, compare
from oumi_weights_lab.proving.planted import PLANTED_FAMILIES
from oumi_weights_lab.proving.runner import read_results, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image", required=True, help="the skill-arm image")
    parser.add_argument("--control-image", required=True, help="the control-arm image")
    parser.add_argument("--agent-cmd", required=True)
    parser.add_argument("--agent-label", required=True)
    parser.add_argument("--env", action="append", default=[], help="environment variable names")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--keep", action="store_true", help="keep the working directory")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="proving-live-") as scratch:
        root = Path(scratch)
        for family in PLANTED_FAMILIES:
            build_task(family, "dev", 0, root / "tasks", root / "keys")
        results = root / "results.jsonl"
        for arm, image in (("skill", args.image), ("control", args.control_image)):
            run(
                root / "tasks",
                root / "keys",
                arm=arm,
                image=image,
                agent_cmd=args.agent_cmd,
                agent_label=args.agent_label,
                repeats=args.repeats,
                timeout=args.timeout,
                results=results,
                env=args.env,
            )
        records = read_results(results)
        for record in records:
            print(
                f"{record['arm']:8} {record['family']:18} repeat {record['repeat']} "
                f"score {record['score']['score']:.2f} solved {record['score']['solved']} "
                f"timed_out {record['timed_out']} {record['seconds']}s {record['problems']}"
            )
        left = [r for r in records if r["arm"] == "control"]
        right = [r for r in records if r["arm"] == "skill"]
        report = compare(left, right)
        print(json.dumps({**report, "beats": beats(report)}, indent=2))
        if args.keep:
            kept = Path(tempfile.mkdtemp(prefix="proving-live-kept-"))
            for item in root.iterdir():
                item.rename(kept / item.name)
            print("kept:", kept)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
