# Published generation and continuation-scoring parity

`examples/parity.py` checks the public GPT-2 124M checkpoint against the exact
20-token greedy output in Hugging Face's
[GPT-2 integration test, Transformers v4.56.2](https://github.com/huggingface/transformers/blob/v4.56.2/tests/models/gpt2/test_modeling_gpt2.py#L763-L788).
The checkpoint is pinned to
[`607a30d783dfa663caf39e06633721c8d4cfcd7e`](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e).
The script also checks the continuation's summed log probability against
Hugging Face's teacher-forced loss with all prompt labels masked to `-100`.

Run from the repository root with an unused output directory:

```sh
uv run python -m examples.parity --output-dir /tmp/weights-lab-parity
```

The first run downloads roughly 500 MB of public model weights. Execution uses
CPU and float32. The script saves the source URL, pinned checkpoint, expected and
observed token IDs, numerical comparison, and pass count in `parity.json`.

Measured on 2026-11-03, macOS arm64, PyTorch 2.15.0, Transformers 5.22.1:

| Check | Result |
| --- | --- |
| Upstream greedy output | All 20 token IDs match exactly |
| Prompt | `The dog` |
| Continuation tokens | 18 |
| Weights Lab continuation score | -35.2241325378418 |
| HF mean continuation loss | 1.9568963050842285 |
| Negative HF loss × 18 | -35.22413349151611 |
| Absolute score difference | 0.00000095367431640625 |
| Allowed absolute score difference | 0.0001 |
| Counted model forwards | 20: 18 generation, one scorer, one HF loss |

The published reference is the generation token sequence. The likelihood number
above was measured locally against HF's loss; it is not a published benchmark
score. This checks checkpoint loading, eager generation, causal score alignment,
prompt masking, sum reduction, and forward accounting. It does not reproduce a
dataset-level accuracy or perplexity result.

The scorer tokenizes prompt and continuation separately without special tokens,
concatenates their IDs, and scores only continuation targets. For this fixture
the concatenation exactly equals the published token sequence. The policy avoids
accidentally scoring prompt suffixes at a merged tokenizer boundary; it can differ
from tokenizing arbitrary concatenated text. Include desired leading whitespace
in each completion. The unit suite additionally checks this boundary analytically
and compares a randomly initialized tiny HF model against masked teacher-forced
loss without downloading any model.
