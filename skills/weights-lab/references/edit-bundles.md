# Save an edit and continue in a fresh worker

Save a bundle after establishing a candidate with the normal baseline/edit/undo
workflow. Bundles preserve exact API edit snapshots and undo history, forward
passes, the saved cap, and dev/held-out reservations.

```python
bundle = lab.save_edits("candidate-v1")
print(bundle)
```

Stop and archive the source session. Load the same original model, tokenizer,
adapter and precision in a new session, pointing at the saved bundle:

```sh
weights-lab start --session sessions/continued --model /path/to/original \
  --device cpu --dtype float32 --resume /path/to/candidate-v1
weights-lab send --session sessions/continued \
  --code 'print(lab.edits(), lab.forward_passes); lab.undo()' --follow
```

Alternatively, `lab.load_edits("/path/to/candidate-v1")` loads into a fresh model
workspace before any forwards, edits, or evaluations. The saved pass cap is
inherited and cannot be raised; reservations include failed held-out attempts.
Model IDs alone do not establish compatibility: loading verifies all original
tensor contents, configuration, tokenizer, and adapter settings.

A bundle is an explicit save point. Work done afterward and arbitrary Python
variables are not restored. Only edits through the API replay; direct writes
and arbitrary hooks are outside the bundle contract. Remove active patches and
attachments and leave evaluation/pass-limit contexts before saving. Hashing the
base and copying edited parameters can take time and CPU memory on large models.
Saving does not run model forwards. Keep logs and evaluation artifacts separately.

For SkyPilot, the bundle must already exist at the remote path passed to
`--resume`; the flag does not upload it. Download saved bundles before teardown.
To share a complete model instead, use `lab.export(...)` while it is alive.
