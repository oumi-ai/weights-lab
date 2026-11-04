# Evaluation workflow

Use `lab.evaluate` on explicit dev rows, retaining IDs, source labels, and source
**document** IDs in JSON metadata. Preserve candidate completions and tokenization
policy across comparisons. Run names are immutable; failed runs save partial
results and consume held-out reservations. A run that reaches its pass cap
stops between batches, so no batch is partly scored.

```python
baseline = lab.evaluate(dev, name="baseline")
# Make one reversible edit.
candidate = lab.evaluate(dev, name="candidate")
report = lab.compare_runs(baseline, candidate, group_by="metadata.source")
print(report["aggregate"])
print(report["groups"])
print(report["top_probability_movers"])
```

`diff` pairs two runs of the same dataset by position and reports
prediction/correctness changes; use `include_scores=True` to see score movement
with unchanged winners. It ignores row IDs but checks ordered content, metadata,
settings, and complete results. `compare_runs` and suite pairing are strict: same
ordered tasks, IDs, metadata, session, mode, seed, scoring policy, scoring
config (including `batch_size`), and generation settings. Explicit IDs must be unique and complete. Custom sampled
checkers require a matching explicit `checker_id` for comparison.

Inspect accuracy **and** candidate-set NLL/Brier/ECE overall and per source. For
sampled runs, also inspect `truncation_share` and `mean_generated_tokens`: an edit
that stops answers running to the token limit can matter as much as accuracy.
Probabilities are softmax over supplied continuation log scores, not calibrated
open-answer confidence. `fixed`, `broken`, and exact McNemar describe paired flips.
Bootstrap intervals use source documents (`document_id`, `source_doc`, or
`source_document`, top-level or nested under `metadata`) when present; override
with `bootstrap_group_by`. Missing requested groups/documents must be corrected,
not dropped. Failed/unfinished rows remain in denominators with an explicit
failure penalty and are listed in `failed_rows`, never ranked as movers. Exact
score ties count as incorrect (`tie: true`). A failed suite cannot pass.

For recurring checks, use `lab.evaluate_suite({"surface": data, ...}, name=...)`.
Each value may instead be `{"data": data, "mode": ..., "max_passes": ...}`.
Compare manifests offline with `lab.compare_suites(before, after,
accuracy_tolerance=.01, nll_tolerance=.02, group_by="metadata.source")`.
Tolerances apply to the aggregate and every source; add `required_groups=[...]`
when groups must exist. Tolerances can be dataset-name mappings with `default`.
Missing datasets, mismatched protocols, and incomplete runs fail the gate.

For cheap repeatable iteration, pin a dev screen once:

```python
pins = lab.create_screen(
    dev, name="dev-screen-v1", size=100, stratify_by=["metadata.source"], min_per_cell=3, seed=0
)
rows = lab.verify_screen(pins)
run = lab.evaluate(rows, name="edit-screen")
```

Verification requires unchanged source, manifest, and selected content; missing or
tampered files raise. Never select from declared or reserved held-out data. Screens
reserve their whole source as dev; do not report a screen result as a full-suite or
held-out improvement. Reuse fixed pins and run the full regression suite before
promoting an edit. Repeated dev selection invalidates a naive interpretation of
p-values as evidence of unseen-data gains. No calibration fitting is provided.
