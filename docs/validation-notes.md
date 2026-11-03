# Validation notes

Supplement to the [measured results](validation.md) and
[machine-readable summary](validation-results.json).

## Runtime regressions

Sampled scoring previously restored CPU and CUDA random state while leaving Apple
MPS state changed, including for CPU models. Four CPU/MPS success/failure cases
failed before the fix and passed afterward. Generation now preserves the caller's
random state on all three backends.

A `Path` model argument was accepted but embedded as `PosixPath(...)` in a generated
bootstrap without its constructor. Model paths are now serialized as strings; the
regression executes generated session creation in a child process.

## SkyPilot startup and Slurm paths

Detached SkyPilot launch can return before Python and session files exist. The
launcher waits with bounded readiness checks, retains failure diagnostics, and
handles terminal worker and SSH authentication failures separately from setup delay.

The live Slurm baseline reproduced a different failure: the worker was ready under
its per-cluster home, while noninteractive SSH retained the login home. A stock
`.bashrc` returned before SkyPilot's appended `.sky_ssh_rc` hook. Readiness timed
out in the wrong session directory. The fix loads the upstream hook explicitly
for both probes and normal commands. A local shell regression also checks stdin,
quoted paths, stop, and binary backup transfer; the live rerun completed normally.

The fixture used a real Slurm 23.11.4 scheduler, private loopback SSH, and private UTS
hostnames for its daemons. Fixture-only callback and node-resolution failures were
corrected before the product test passed. Failed attempts are retained separately.
The successful rerun required no forced scheduler cancellation.

Slurm's upstream backend skips Ray setup. The isolated test used SkyPilot
0.12.3.post1, FastAPI 0.135.3, and Starlette 0.47.2; a fresh FastAPI 0.162.0 install
rejected the older SkyPilot server constructor. The model environment was separate
from the SkyPilot client. No fork or vendor patch was used.

## Benchmark environment

Invoking a virtualenv's Python by absolute path does not expose compiler executables
on PATH. The first benchmark attempt therefore failed to find Ninja. Benchmark
children now prepend their executable directory and record Ninja and driver metadata.

The upstream dataset was unpinned, and an unused upstream loop diagnostic was
unavailable. The run froze the resolved dataset commit and recorded a grader shim
that raises if called. These qualifications are preserved in the public result
summary; they do not change the prompt, extraction, or mathematical grading code.

## Artifact provenance

The ignored `sessions/validation-20261103/` directory in the original workspace
contains the full MATH-500 samples and raw audit, Linux and CUDA reports, the real
Qwen replay, and the Slurm failure/passing reports. `sky-slurm/launchers-tested.py`
has the exact launcher bytes used in the passing test.

The initial cloud-controller report retains a failure from an earlier direct-Slurm
harness that was later removed. The full benchmark comparison, current Slurm report,
and runtime reports are independent results. Their digests are recorded in
[validation-results.json](validation-results.json).

## Resource cleanup

All four disposable instances were terminated. Final provider inventory found no
validation instances, disks, GPU clusters, or snapshots; unrelated resources were
left untouched. Local fixture daemons and the isolated API server also exited.

Estimated compute was $12.72, including $0.04 for the Slurm CPU fixture. The H200 and
CPU quotes were $4.50/hour and $0.0992/hour. These estimates use controller durations,
exclude storage, and are not billing statements. Cloud runs had an independent
teardown watchdog, a finite worker lifetime, and SkyPilot autodown.
