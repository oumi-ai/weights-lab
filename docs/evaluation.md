# Paired evaluation and dev screens

Evaluation runs preserve every JSON row field, including nested `metadata`, source
labels, and document IDs. `prompt`, `options`, and `answer` define semantic dataset
identity; changing IDs, metadata, filenames, option labels, or row order cannot
reset a held-out reservation. Per-row semantic hashes also prevent moving overlapping
subsets between dev and held-out roles, including after a session resume that
restores only the registry. Result fields (`index`, `prediction`, `scores`,
`correct`, `generated`, `error`) are reserved. Use `metadata` for arbitrary details.

```python
data = [
    {
        "id": "capital-1",
        "prompt": "France's capital is",
        "options": {"paris": " Paris", "rome": " Rome"},
        "answer": "paris",
        "metadata": {"source": "geography", "document_id": "atlas-1"},
    }
]
before = lab.evaluate(data, name="baseline")
# Apply an edit, then run the same rows again.
after = lab.evaluate(data, name="candidate")
changes = lab.diff(before, after, include_scores=True)
report = lab.compare_runs(before, after, group_by="metadata.source", seed=7)
```

`diff` still returns only changed predictions/correctness by default, pairing by
ordered task content while ignoring row IDs for compatibility. It requires
matching metadata and scoring settings, and refuses incomplete successful runs.
`include_scores=True` also returns `score_changed` items whose likelihoods changed
without changing the winner. `compare_runs` reads saved files without model calls.
It returns `aggregate`, `groups`, `top_probability_movers`, and `top_score_movers`.
Movers rank absolute changes to the gold answer's probability or summed log score.

Each aggregate/group has `before`, `after`, `delta` (after minus before), `fixed`,
`broken`, an exact two-sided `mcnemar_p`, and a deterministic 95% percentile
bootstrap interval for accuracy and NLL changes. `bootstrap_samples=1000`, `seed=0`,
and `top=10` are the defaults. The bootstrap resamples entire source documents
when rows provide `source_document`, `source_doc`, or `document_id`, at the top
level or under `metadata`. Set `bootstrap_group_by="your.document.field"` to use
another document key. Without document metadata, it resamples rows. Missing or
null document IDs and missing requested group fields raise an error.
Group names preserve JSON types: non-string values use `@json:` plus JSON, and
strings beginning with `@` are escaped with `@str:`.

Likelihood metrics normalize `exp(score)` across the supplied candidates. NLL is
negative log probability of the gold candidate, Brier is the sum of squared errors
across candidates, and ECE uses ten equal-width confidence bins. These are
**candidate-set probabilities**, not calibrated probabilities over every possible
answer. Changing the candidate set changes the question these metrics answer.
Sampled runs have accuracy/paired statistics; probability metrics are null.
No calibration is fitted. Bootstrap intervals and p-values describe a fixed pair;
repeatedly choosing edits on the same dev data does not establish held-out gains.

`compare_runs` pairing requires the same session, mode, tokenizer scoring policy,
seed, generation
settings, ordered rows, IDs, and metadata. IDs must be unique and present on every
row if supplied; rows without IDs pair by their exact position and content. Rows
are never intersected or silently reordered. Custom sampled checkers need an
explicit, matching `checker_id="grading-v1"` on `evaluate` to compare runs: update
that ID whenever grading behavior changes.

Failed runs remain inspectable. `compare_runs` reconstructs every unfinished row
from the pinned dataset, counts it wrong, and reports failures. For likelihood
runs, missing/error scores contribute NLL `-log(1e-12)`, Brier 2, and confidence 1
with incorrectness for ECE. This explicit failure penalty retains the full
denominator; it is not an estimated probability. A suite with any failed run
cannot pass its gate, even with permissive metric tolerances.

## Named regression suites

```python
baseline = lab.evaluate_suite(
    {
        "geography": "/data/geography-dev.jsonl",
        "formatting": {"data": "/data/format-dev.jsonl", "mode": "sampled"},
    },
    name="baseline-suite",
    max_passes=500,
)
candidate = lab.evaluate_suite(
    {
        "geography": "/data/geography-dev.jsonl",
        "formatting": {"data": "/data/format-dev.jsonl", "mode": "sampled"},
    },
    name="candidate-suite",
    max_passes=500,
)
gate = lab.compare_suites(
    baseline,
    candidate,
    accuracy_tolerance={"default": 0.01, "geography": 0},
    nll_tolerance=0.02,
)
assert gate["pass"]
```

A suite accepts a name-to-data mapping. A dataset specification with `data` may
also override any `evaluate` option. Other keyword arguments become defaults for
each dataset; a pass cap applies separately to each dataset. Suite-generated run
names avoid collisions. All datasets are attempted after ordinary evaluation
failures; errors and failed runs remain in the returned and saved manifest at
`artifacts/suites/NAME/suite.json`. The manifest pins the expected dataset names
before running any of them. Interruptions still propagate, and an interrupted or
incomplete suite cannot pass comparison.

`compare_suites` accepts manifests, manifest paths, or suite directories. Its
offline gate requires every named dataset on both sides, strict paired data and
protocol, successful runs, accuracy loss no greater than `accuracy_tolerance`, and
NLL increase no greater than `nll_tolerance` (both default to zero). A tolerance
may be a scalar or mapping of dataset names with optional `default`. Sampled data
has no NLL check. Use `group_by="metadata.source"` to apply the same tolerances
both overall and to every group; `required_groups=["geography", "language"]`
ensures expected groups exist in every dataset. Missing datasets, groups, rows,
or mismatched protocols fail the gate. Check results include the full paired
report and per-surface verdicts (`aggregate` and `group:NAME`).

## Repeatable dev screens

```python
screen = lab.create_screen(
    "/data/dev.jsonl",
    name="screen-v1",
    size=120,
    stratify_by=["metadata.source", "metadata.difficulty"],
    min_per_cell=3,
    seed=17,
)
screen_rows = lab.verify_screen(screen)
run = lab.evaluate(screen_rows, name="candidate-screen")
```

Screens are dev-only filters for deciding whether an edit warrants a larger run.
They do not establish a full-suite or held-out improvement. Choose the size and
strata before comparing candidates, and reuse the same pins for every candidate.
A fixed seed samples at least `min_per_cell` rows from every nonempty observed
cell (or all its rows if smaller), then samples the remaining quota from remaining
rows. Selection is independent of input row order when IDs/content are unchanged.
An impossible size raises an error. `size=None` pins the entire source.

`create_screen` saves `screen.json`, `rows.json`, and an inline source snapshot
when given a list, under `artifacts/screens/NAME/`. The manifest hashes the source
bytes, selected rows, and selection settings. `verify_screen` checks all of them,
requires the original source file/snapshot, and returns verified rows for
`evaluate`. A changed, missing, or tampered file raises an error; hashes detect
accidental changes, not an adversary who can rewrite every artifact.

Rows require unique IDs or unique semantic content. Screens reject
`held_out=True`, top-level or metadata `held_out` flags, and `split` values `test`,
`heldout`, or `held_out`. They also reject any source row overlapping a previously
reserved held-out run, including failed runs. Screening reserves the entire source
as dev in the persistent dataset registry, preventing any overlapping subset from
later becoming held-out. The registry stores semantic row hashes before model
calls, so role checks survive resumes without old run files. Legacy entries without
row hashes fall back to saved datasets; unavailable legacy datasets fail closed.
Keep independently authored held-out data separate; these guards do not sandbox unrestricted filesystem/model access.

All five helpers (`compare_runs`, `evaluate_suite`, `compare_suites`,
`create_screen`, `verify_screen`) are also exposed through `Scorer(lab)`.
