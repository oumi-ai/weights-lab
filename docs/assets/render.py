"""Render the README figure from the audited, recorded Qwen experiment.

Run without installing plotting dependencies in the project:
    uv run --no-project --with matplotlib==3.11.2 python docs/assets/render.py

This only plots checked-in data; it never loads a model or runs inference.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BLUE = "#034ADA"
INK = "#152039"
SLATE = "#53627A"
PALE = "#E7EDF6"
SCALE = [PALE, "#BCCFF6", "#82A7F1", "#4277E7", BLUE]


def validate(data: dict) -> None:
    """Keep the diagram, denominators, and paired grids tied to the evidence."""
    if data["errors"] or not data["benchmark"]["sample_counts_recounted"]:
        raise ValueError("The recorded experiment must pass its sample audit")
    edit = data["weight_edit"]
    if len(edit["changed_tensors"]) != 2 or edit["tensor_count_checked"] != 488:
        raise ValueError("Expected the two-tensor edit audited against all 488 tensors")
    if (
        sum(t["changed_entries"] for t in edit["tensors"].values()) != 130188
        or edit["changed_entries_total"] != 130188
    ):
        raise ValueError("Changed-entry total differs from the reference edit")
    for tensor in edit["tensors"].values():
        deltas = tensor["column_mean_absolute_delta"]
        columns = tensor["changed_columns"]
        if (
            tensor["shape"] != [1024, 3584]
            or len(deltas) != 3584
            or len(set(columns)) != 64
            or [i for i, value in enumerate(deltas) if value > 0] != columns
        ):
            raise ValueError("Column diagram disagrees with the audited tensor deltas")
    protocol = data["benchmark"]["protocol"]
    if protocol["sampling"]["max_tokens"] != 32768:
        raise ValueError("Expected the recorded 32,768-token generation cap")
    for arm in data["benchmark"]["arms"].values():
        if arm["problems"] != 500 or arm["samples_per_problem"] != 4 or arm["total"] != 2000:
            raise ValueError("Expected 500 problems and four generated samples per arm")
        for counts_key, total_key, proportion_key in (
            ("correct_counts", "correct", "accuracy"),
            ("cap_counts", "cut", "cut_share"),
        ):
            counts = arm["per_problem"][counts_key]
            if (
                len(counts) != 500
                or any(type(count) is not int or not 0 <= count <= 4 for count in counts)
                or sum(counts) != arm[total_key]
                or arm[proportion_key] != arm[total_key] / arm["total"]
            ):
                raise ValueError("Grid cells disagree with the recorded sample totals")


def render(data: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "path",
            "svg.hashsalt": "oumi-weights-lab-qwen-edit",
        }
    )
    fig = plt.figure(figsize=(14.4, 9.2), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1440), ylim=(920, 0))
    ax.axis("off")

    def text(x, y, content, size=14, color=INK, weight="normal", **kwargs):
        return ax.text(
            x, y, content, fontsize=size, color=color, fontweight=weight, va="top", **kwargs
        )

    text(64, 47, "Two layers. Different answers.", 33, weight="bold")
    text(67, 113, "A released Qwen3.5-0.8B edit, replayed in Oumi Weights Lab.", 15, SLATE)
    ax.plot([64, 1376], [166, 166], color=PALE, linewidth=1)

    text(64, 201, "Inside the weights", 18, weight="bold")
    text(64, 252, f"{data['weight_edit']['changed_entries_total']:,}", 33, BLUE, "bold")
    text(66, 309, "BF16 values changed", 14, SLATE)

    for index, tensor in enumerate(data["weight_edit"]["tensors"].values()):
        y = 383 + index * 142
        text(64, y - 31, f"Layer {tensor['layer']}  /  MLP down projection", 12, weight="bold")
        ax.add_patch(Rectangle((64, y), 332, 62, facecolor=PALE, linewidth=0))
        # Each mark occupies its true column coordinate on the shared 0..3583 scale.
        # A minimum stroke width keeps sparse columns visible at README size.
        for column in tensor["changed_columns"]:
            x = 64 + 332 * column / 3584
            ax.plot([x, x], [y, y + 62], color=BLUE, linewidth=0.6, solid_capstyle="butt")
        text(64, y + 69, "0", 10, SLATE)
        text(396, y + 69, "3,583", 10, SLATE, ha="right")

    text(64, 647, "64 of 3,584 columns per tensor.", 13, weight="bold")
    text(64, 678, "2 changed tensors out of 488 audited.", 12, SLATE)
    text(64, 742, "swap → evaluate → undo", 16, BLUE, "bold")
    text(64, 780, "Replay verified; undo restored exact bytes.", 11, SLATE)

    for label, key, x in (("Base model", "base", 475), ("Released edit", "edited", 951)):
        arm = data["benchmark"]["arms"][key]
        text(x, 201, label, 18, weight="bold")
        text(x, 249, f"{arm['accuracy']:.2%}", 38, BLUE if key == "edited" else INK, "bold")
        text(x + 2, 315, f"{arm['correct']:,} / {arm['total']:,} samples correct", 13, SLATE)
        counts = arm["per_problem"]["correct_counts"]
        patches = [Rectangle((x + (i % 25) * 17, 365 + (i // 25) * 15), 13, 11) for i in range(500)]
        ax.add_collection(
            PatchCollection(patches, facecolor=[SCALE[count] for count in counts], linewidth=0)
        )
        text(x, 724, "At the 32,768-token cap", 12, SLATE)
        ax.add_patch(Rectangle((x, 759), 421, 10, facecolor=PALE, linewidth=0))
        ax.add_patch(
            Rectangle((x, 759), 421 * arm["cut_share"], 10, facecolor="#A06418", linewidth=0)
        )
        text(x, 787, f"{arm['cut_share']:.2%}", 16, weight="bold")

    text(475, 687, "Each tile is one problem; shade shows correct samples out of four.", 11, SLATE)
    for index, color in enumerate(SCALE):
        x = 1108 + index * 49
        ax.add_patch(Rectangle((x, 684), 15, 12, facecolor=color, linewidth=0))
        text(x + 21, 682, str(index), 10, SLATE)

    ax.plot([64, 1376], [838, 838], color=PALE, linewidth=1)
    text(64, 861, "MATH-500. 500 problems × 4 samples per arm on an NVIDIA H200.", 12, SLATE)
    text(
        64, 890, "Recorded reference run. Reproduction of Takehi et al.’s released edit.", 10, SLATE
    )

    output.mkdir(parents=True, exist_ok=True)
    description = (
        "Qwen3.5-0.8B released edit: 130,188 changed BF16 entries in the MLP down "
        "projections of layers 15 and 19, with 64 of 3,584 columns changed in each. "
        "All 488 tensors were audited. Paired grids each show the same 500 MATH-500 "
        "problems in dataset order, with shade encoding 0–4 correct samples. Sample "
        "accuracy rises from 157/2,000 (7.85%) to 1,025/2,000 (51.25%); share at the "
        "32,768-token cap falls from 92.90% to 30.55%. Recorded H200 reference run, "
        "not a new benchmark or edit discovery. Separate replay restored exact bytes."
    )
    fig.savefig(
        output / "weight-space.svg",
        metadata={"Date": None, "Creator": "Oumi Weights Lab", "Description": description},
    )
    # Keep path-based typography while giving the standalone SVG an accessible name.
    svg = output / "weight-space.svg"
    document = svg.read_text()
    opening_end = document.index(">", document.index("<svg"))
    document = (
        document[:opening_end]
        + ' role="img" aria-labelledby="figure-title figure-description"'
        + document[opening_end : opening_end + 1]
        + '<title id="figure-title">Two layers. Different answers.</title>'
        + '<desc id="figure-description">'
        + description
        + "</desc>"
        + document[opening_end + 1 :]
    )
    document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
    ET.fromstring(document)
    svg.write_text(document)
    fig.savefig(output / "weight-space.png", dpi=150, metadata={"Description": description})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "docs/assets/qwen-edit.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    validate(data)
    render(data, args.output)
    print(f"Rendered SVG and PNG in {args.output}")


if __name__ == "__main__":
    main()
