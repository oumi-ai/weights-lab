# Inside the Qwen edit

The README figure connects the actual changed weight columns to the recorded
MATH-500 results. The edit is the released
[weight-space programming experiment](https://rikiyat.github.io/blog/weight-space-programming/)
by Takehi, Teoh, Zhang, and Khattab, replayed with Oumi Weights Lab.

## Explore it

Download [the interactive visualization](weight-space.html) and open the HTML file
in your browser. From a checkout, open `docs/weight-space.html` directly. It works
offline and needs no installation. Switch between accuracy and token-cap results;
select a problem to compare its four samples in both conditions.

The README uses a standalone [SVG](assets/weight-space.svg); a
[PNG](assets/weight-space.png) is also available for sharing.

## What the figure shows

The two column diagrams mark the real edited column positions in the MLP
`down_proj.weight` tensors of layers 15 and 19, using zero-based indices. Each
tensor has shape 1,024 × 3,584, and 64 columns change in each. Marks have a minimum
stroke width to remain visible when the figure is scaled down. A full raw-byte
comparison of all 488 stored checkpoint tensors found exactly two changed tensors
and 130,188 changed BF16 entries. The separate replay restored the original tensor
bytes with `lab.undo()`.

The paired grids each contain the same 500 problems in original dataset order,
read left to right, top to bottom. Each square's shade encodes how many of that
problem's four generated samples were correct. These are observed results, not
decorative or simulated patterns.

| Recorded metric | Base model | Released edit |
| --- | ---: | ---: |
| Correct samples | 157 / 2,000 (7.85%) | 1,025 / 2,000 (51.25%) |
| Samples reaching the 32,768-token cap | 1,858 / 2,000 (92.90%) | 611 / 2,000 (30.55%) |

Accuracy uses generated samples as its denominator; it is not pass@4. Samples
from the same problem are correlated. The token-cap metric counts samples whose
recorded finish reason was `length`; every such sample had 32,768 generated tokens.

## Data and reproduction

[qwen-edit.json](assets/qwen-edit.json) contains the audited column differences,
paired problem counts, model and dataset revisions, generation settings, grader
hash, and hashes of the original evidence files. It includes no question text or
generated answers. The recorded benchmark ran on an NVIDIA H200 with BF16,
thinking enabled, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, seed
0, and four samples per problem. Dataset revision:
`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.

The visualization was built by rechecking cached checkpoint bytes and recounting
saved sample records. It does not represent a new inference run, regrading, or
discovery of an edit. The byte-for-byte replay and the benchmark were separate
experiments; the full benchmark was not repeated after undo.

- [Pinned edit verification and replay](../examples/replay_qwen_edit.py)
- [Full benchmark protocol and runner](../tests/live/validate_qwen_benchmark.py)
- [Base checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)
- [Released edited checkpoint](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming/tree/d1671a178705bcab7644806d23c535517f35723b)

Regenerate the SVG and PNG from the checked-in data without loading a model:

```sh
uv run --no-project --with matplotlib==3.11.2 python docs/assets/render.py
```

Plotting dependencies are separate from the worker's runtime dependencies.
