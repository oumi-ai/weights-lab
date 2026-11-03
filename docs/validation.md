# Validation evidence

Measured on November 3, 2026. The [result summary](validation-results.json) records
checkpoint and dataset pins, runtime versions, exact counts, and report hashes.
The Qwen experiment evaluates a released edit; it does not test autonomous discovery.

## Runtime and package checks

| Environment | Result |
| --- | --- |
| macOS, Python 3.12, Torch 2.15.0, Transformers 5.22.1 | 231 tests passed, including Apple MPS checks |
| Linux, Python 3.11.15, Torch 2.14.0, Transformers 5.20.2 | 225 passed, five MPS-only cases skipped |
| NVIDIA H200, Torch 2.14.0+cu130, Transformers 5.20.2 | CUDA worker, inspection, editing, scoring, and export checks passed |
| Upstream SkyPilot 0.12.3.post1 and Slurm 23.11.4 | Relative-path CPU worker, inference, backup, and normal teardown passed |
| Source distribution, wheel, installed CLI and API | Build and installation checks passed |

The Linux and H200 reports cover `5995bb0`. The later Slurm transport fix and its
regression are covered by `eb5a1e5` and the live check below. The unit suite creates
tiny random models locally; public-model examples download their pinned checkpoints.
These recorded engine-validation counts precede the added plugin-packaging checks
in the current suite.

The CUDA harness exercised a random GPT-2 with a nonzero LoRA adapter. All 15
accepted forwards were accounted for. Edit/undo restored weights and scores exactly;
the largest export/reload logit difference was `7.450580596923828e-09`. All 33 backup
files matched their source digests, installed source fingerprints matched the
validation checkout, and the harness confirmed its worker had stopped.

For a short, measured baseline/edit/undo/export walkthrough, use the
[CPU demo](dogfood.md).

## Full MATH-500 comparison

Both pinned checkpoints ran on the same NVIDIA H200 with all 500 problems and four
samples per problem, for 4,000 generations total.

| Metric | Base | Released edit |
| --- | ---: | ---: |
| Correct samples | 157 / 2,000 | 1,025 / 2,000 |
| Accuracy | 7.85% | 51.25% |
| Samples reaching the 32,768-token cap | 1,858 (92.90%) | 611 (30.55%) |
| Mean generated tokens | 30,845.27 | 21,699.80 |
| Arm duration, including initialization and grading | 88.03 min | 55.98 min |

The predeclared criteria passed: both arms completed, edited accuracy was higher,
and edited token-cap share was lower. Accuracy increased by 43.4 percentage points;
token-cap share decreased by 62.35 points. These are sample-level metrics, not pass@4.

The runner uses the upstream `think35` protocol: thinking-enabled prompts, BF16,
temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, seed 0, and a
32,768-token generation cap. It starts a fresh vLLM process for each checkpoint.
Model pins, dataset revision, frozen input hashes, grader hash, and exact dependency
versions are recorded in the [JSON summary](validation-results.json). See the
[pinned upstream evaluator](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/evaluate.py)
and [benchmark runner](../scripts/validate_qwen_benchmark.py).

The upstream dataset revision was unpinned. This run froze the resolved revision,
so identical published percentages are not a validation requirement. The unchanged
upstream math grader imports an unavailable loop diagnostic; an explicit shim
raises if that unused diagnostic is called. Prompt construction, answer extraction,
and mathematical grading use the pinned source unchanged.

An independent raw-file audit checked four unique samples per problem in each arm,
frozen input hashes, valid finish reasons and token counts, and agreement with all
reported aggregates. Diagnostic runs with fewer than 500 problems cannot report
full validation.

## Checkpoint replay

The [Qwen replay](qwen-replay.md) audited all 488 tensors across both real BF16
checkpoints. Exactly the two published MLP tensors differed: 130,188 entries across
64 columns in each. Released raw-byte hashes matched. Applying both edits through
`lab.swap` produced the expected live tensor hashes; undo restored the base exactly.
The replay used zero forward calls and stopped its worker.

## Remote lifecycle

Live SkyPilot provisioning and SSH execution passed on the H200. The independent
Slurm check used a dedicated CPU host and `infra="slurm/fixture/validation"`.
Its worker reported the same job ID as the live scheduler allocation. A relative
session path became ready, a model forward ran on CPU, and send, peek, stop,
backup, and verified download all succeeded.

The downloaded archive contained seven files and matched its remote SHA-256.
Normal `sky down` succeeded; the scheduler confirmed the allocation was gone
without forced cancellation. The full lifecycle took 77.55 seconds. This validates
a single-node CPU Slurm lifecycle; GPU execution was tested separately on the H200.
Modal is unsupported. Remote sessions use upstream SkyPilot without a fork.

## Repeat checks and inspect evidence

Run the reusable runtime harness in an environment with the adapters extra:

```sh
python scripts/validate_runtime.py --device cpu --output-dir /tmp/runtime-cpu
python scripts/validate_runtime.py --device mps --output-dir /tmp/runtime-mps
python scripts/validate_runtime.py --device cuda --output-dir /tmp/runtime-cuda
```

Each output directory must be new. The harness records results, exports and reloads
the checkpoint, verifies its backup, and stops its worker even after an assertion
failure. The full MATH-500 runner requires CUDA and the pinned benchmark dependencies.

The original validation workspace retains raw samples, reports, source fingerprints,
and audit files under `sessions/validation-20261103/`. Those files are ignored by Git;
a fresh checkout includes the compact JSON summary rather than the raw artifacts.
See [validation notes](validation-notes.md) for regression reproductions, environment
details, report provenance, and resource cleanup.
