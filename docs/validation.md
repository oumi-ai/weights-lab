# Validation evidence

Validation was performed on November 3, 2026. Public model checkpoints are pinned;
the test suite creates its own small models and does not download weights.

## Runtime and package checks

| Environment | Result |
| --- | --- |
| macOS, Python 3.12, Torch 2.15.0, Transformers 5.22.1 | 230 tests passed, including actual Apple MPS checks |
| Linux, Python 3.11.15, Torch 2.14.0, Transformers 5.20.2 | 225 passed, five MPS-only cases skipped; uploaded source hashes verified |
| NVIDIA H200, Torch 2.14.0+cu130, Transformers 5.20.2 | Full worker runtime harness passed on CUDA |
| Source distribution, wheel, installed CLI and API | Build and installation checks passed |

The CUDA harness used a local random GPT-2 and a nonzero LoRA adapter. It checked
actual input/output device placement, generation accounting, activation recording
and patch cleanup, per-row gradients, dev and held-out scoring, budget exhaustion,
bit-exact edit/undo, and merged checkpoint reload. All 15 accepted forward calls
were accounted for. The largest export/reload logit difference was
`7.450580596923828e-09`; all 33 backup files matched their source SHA-256 digests.
The harness stopped its worker and confirmed that it was dead.

The downloaded CUDA report includes SHA-256 fingerprints of every installed
`weights_lab` Python source file. All matched the local sources used for validation.
The Linux compatibility run used GPUs hidden and one CPU thread; it did not change
the benchmark environment's packages or consume GPU compute.

The branch review also found an Apple MPS random-state leak in sampled scoring.
All four CPU/MPS success/failure regression cases failed before the fix and passed
afterward. Sampled evaluation now preserves the caller's random state on all three
backends. A separate regression executes the generated SkyPilot bootstrap with a
`Path` model argument, ensuring that its configuration is serializable before
provisioning.

Run the reusable harness in an installed package environment with the adapters extra:

```sh
python scripts/validate_runtime.py --device cpu --output-dir /tmp/runtime-cpu
python scripts/validate_runtime.py --device mps --output-dir /tmp/runtime-mps
python scripts/validate_runtime.py --device cuda --output-dir /tmp/runtime-cuda
```

Each output directory must be new. The harness creates model assets, records CLI
calls and results, exports and reloads the checkpoint, verifies the backup, and
stops its worker even if an assertion fails.

## Real Qwen checkpoint replay

The actual pinned checkpoints were downloaded and all 488 tensors audited on CPU
in BF16. Exactly the two published MLP tensors differed. Their released raw-byte
hashes matched, with 130,188 changed entries across 64 columns in each tensor.
Both slice edits were applied through `Lab`, the resulting live tensor hashes
were checked, and undo restored the pinned base hashes exactly. The replay used
zero forward calls. Its worker was stopped and confirmed dead.

The full replay took 63.86 seconds, including the edited-checkpoint download wait.
See [the pinned recipe and provenance](qwen-replay.md). The replay's raw report
SHA-256 was `f639da9ad7a24e3e4e779d13a05bb298894f3d9f3424b7e838eeccb855cd7fe4`.

## Full MATH-500 protocol

`scripts/validate_qwen_benchmark.py` freezes all 500 MATH-500 test problems and
their thinking-enabled prompts, dataset revision, checkpoint revisions, and grader
hash before running either arm. It uses the pinned upstream grader, vLLM 0.33.0,
Transformers 5.20.2, BF16, and four samples per problem per arm. The protocol is
`think35`: temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, a 32,768-token
cap, and seed 0. See the [upstream evaluator](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/evaluate.py).

The runner starts separate child processes for the base and edited checkpoints on
the same GPU, preserving raw samples and sample-level grades. Full validation
requires both complete 2,000-sample arms, higher edited accuracy, and a lower
edited token-cap share. These criteria are saved before scoring. Diagnostic runs
with fewer than 500 problems cannot report full validation.

The upstream dataset revision was not pinned. This validation resolves and freezes
the current revision and records it, so it does not promise identical scores to
the published run. The upstream math module imports a missing loop-diagnostic
helper; an explicit shim raises if that unused diagnostic is called. The prompt,
answer extraction, and mathematical grading functions are loaded unchanged and
verified against the pinned source hash.

## Measured MATH-500 result

Both full 500-problem arms completed with four samples per problem on the same
NVIDIA H200. The predeclared directional criteria passed (`full_validated=true`).

| Metric | Base | Edited |
| --- | ---: | ---: |
| Correct samples | 157 / 2,000 | 1,025 / 2,000 |
| Accuracy | 7.85% | 51.25% |
| Samples reaching the 32,768-token cap | 1,858 (92.90%) | 611 (30.55%) |
| Mean generated tokens | 30,845.27 | 21,699.80 |
| Arm duration, including initialization and grading | 88.03 min | 55.98 min |

Accuracy increased by 43.4 percentage points and token-cap share decreased by
62.35 percentage points. These are sample-level metrics, rather than pass@4.

The dataset revision was `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`. Checkpoints
were pinned to base `2fc06364715b967f1860aea9cf38778875588b17` and edited
`d1671a178705bcab7644806d23c535517f35723b`. The grader SHA-256 was
`f0d576a1adc625dede7bff749b3bf8c9a5a8f6bae43d51e2d0ee36d694e89843`.

An independent audit of the raw files verified exactly four unique samples for
every problem in each arm, unchanged frozen inputs, valid finish reasons and token
counts, and agreement with every reported aggregate. The comparison report SHA-256
is `34ff6a8d5e182b382f5858e05945d3efd8e36928d70f26fe6934e6d0ce837c50`.
Raw outputs, runtime reports, source fingerprints, and the audit are retained in
this checkout under `sessions/validation-20261103/`, which is ignored by Git.

## Remote coverage and limits

Remote sessions now use upstream SkyPilot exclusively. The CLI and Python API
forward `infra` selectors, including `slurm/CLUSTER/PARTITION`, to `sky launch`.
Direct SSH, Slurm, and Modal launchers and their validation scripts were removed;
SSH remains a private transport for SkyPilot's cluster alias. Modal is unsupported.
SkyPilot's CLI is installed separately from the model worker.

Live SkyPilot provisioning and SSH CUDA execution succeeded on a dedicated Nebius
H200 instance. Validation exposed and fixed a detached-startup race: the launcher
now waits for the remote environment and session files with bounded readiness
checks. Relative paths are anchored to the remote work directory. Command routing,
Slurm infrastructure selection, transfers, and teardown failure paths have local
regression coverage.

Slurm through SkyPilot has not been live-validated. Its upstream bootstrap invokes
`ray stop`, which would disrupt the active SkyPilot workload on the same disposable
VM. That test needs a separate cluster or isolated host. Earlier single-node tests
of the removed direct Slurm launcher passed, but do not validate the new route.
The original cloud-controller report retains its first direct-Slurm harness failure;
the benchmark comparison and final runtime reports are separate results.

The benchmark's first attempt exposed a virtualenv issue: invoking Python by
absolute path does not expose compiler executables on PATH. The benchmark child now
adds its executable directory before starting vLLM. Failed cloud attempts were
torn down. Each cloud validation run has an independent teardown watchdog, a finite
worker lifetime, and SkyPilot autodown as additional cleanup mechanisms.

All three validation instances were terminated. A final provider-side inventory
found no remaining validation instances, disks, or GPU clusters. Estimated compute
across all attempts was $12.68 at the launch-time $4.50/hour quote, below the $50
cap. This is an estimate from controller durations, excludes storage, and is not a
billing statement.
