# Validation evidence

Validation was performed on November 3, 2026. Public model checkpoints are pinned;
the test suite creates its own small models and does not download weights.

## Runtime and package checks

| Environment | Result |
| --- | --- |
| macOS, Python 3.12, Torch 2.15.0, Transformers 5.22.1 | 248 tests passed, including actual Apple MPS checks |
| Linux, Python 3.11.15, Torch 2.14.0, Transformers 5.20.2 | 246 passed, one MPS-only test skipped; uploaded source hashes verified |
| NVIDIA H200, Torch 2.15.0+cu132, Transformers 5.22.1 | Full worker runtime harness passed on CUDA |
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

## Live-provider findings

Live SkyPilot provisioning and SSH CUDA worker execution succeeded on a dedicated
Nebius H200 instance. Real validation exposed and fixed a startup race: SkyPilot's
detached launch can return before setup creates Python and session files. The
launcher now waits for those files with a bounded, diagnostic readiness check.
Relative Sky paths are consistently anchored to the remote work directory.

The initial benchmark attempt exposed a separate environment issue: invoking a
virtualenv's Python by absolute path does not put its compiler executables on PATH.
The benchmark child now adds its own executable directory before starting vLLM.
Failed cloud attempts were explicitly torn down. Each validation run also has an
independent teardown watchdog and a finite worker lifetime with SkyPilot autodown.

A real Slurm 23.11.4 scheduler was also exercised on that disposable VM. A CPU
worker ran as scheduler job 1, and its `SLURM_JOB_ID` matched the submitted ID.
Send, peek, cancellation through `scancel`, and checksum verification of all seven
backup files passed. A login-process guard rejected compute-node PID signalling
during cancellation. Both control-plane listeners were verified on loopback, and
all temporary daemons exited after the check.

The first Slurm harness attempt exposed a setup detail rather than a package
failure: that Slurm version resolves daemon bind addresses from the hostname.
The corrected harness gives only its daemon processes a private UTS hostname;
the VM's global hostname and network remain unchanged. Its keys and spool state
are removed during cleanup. This validates a real single-node scheduler lifecycle,
not a production multi-node cluster deployment.

Modal authentication and SDK compatibility were checked using Modal 1.6.3. The
actual image build was rejected by the workspace's billing-cycle spend limit
before any GPU sandbox was allocated. The account's billing settings were not
changed. Live Modal execution therefore remains blocked by that account limit.
