# Test a method on known answers

Use this when you build or change a method for finding how a model works, and want to know whether it finds the real mechanism. The proving ground builds small models with a planted mechanism and a hidden answer key, grades a claim against the key, and compares methods on the same tasks. Held-out tasks also exist. Their keys and seeds are kept private, so never tune a recipe on them, and never ask for them.

## 1. Build dev tasks

Dev seeds are public, so a dev task's key is yours to read. Keep the keys directory outside the tasks directory.

```sh
oumi-weights-lab proving-ground build --family copy-head --family lookup-mlp \
  --split dev --count 2 --out tasks --keys keys
```

The families are `copy-head`, `copy-backup`, `lookup-mlp` and `feature-direction`. Each task directory holds `task.json` (the question), `model/` and `rows.jsonl` (rows for `lab.evaluate`). The key is `keys/<task id>.json`; do not open it while you test a method on a task, or you only test your reading.

## 2. Run the method

Start a session on `tasks/<id>/model`, and use the method as you would on any model: locate candidates, then zero them together and measure accuracy on the rows before and after. Undo every edit. Name only parts you tested. A part you name that the model does not use lowers your score, and some parts look right and are not used, so check each one by ablating it alone.

## 3. Write the claim

A claim is one JSON file of data. Layers and heads count from 0.

```json
{"version": 1, "task": "<task id>",
 "components": [{"kind": "head", "layer": 2, "head": 1, "role": "...", "evidence": "..."},
                {"kind": "mlp", "layer": 1, "role": "...", "evidence": "..."},
                {"kind": "neurons", "layer": 1, "indices": [3, 77]}],
 "directions": [{"after_layer": 1, "vector": [64 numbers]}],
 "ablation": {"accuracy_before": 1.0, "accuracy_after": 0.12, "rows": 200},
 "direction_ablation": {"accuracy_before": 1.0, "accuracy_after": 0.5, "rows": 200},
 "summary": "...", "failed_hypotheses": ["..."]}
```

`ablation` is required: accuracy with every claimed component zeroed at once. The grader reruns it on fresh rows, so write the numbers you measured. `directions` and `direction_ablation` are optional. `from oumi_weights_lab.proving import CLAIM_FORMAT` holds the full description.

## 4. Grade it

```sh
oumi-weights-lab proving-ground grade --task tasks/<id> --key keys/<id>.json --claim claim.json
```

The result has `score`, which is `identification` times 1.0 if your ablation numbers reproduced and 0.5 if not, times 1.0 if your claimed parts really break the behavior (`faithful`) and 0.5 if not. `identification` comes from how well your components, and any directions, match the key; naming a direction that is not used costs precision. A claim that names only a decoy scores 0. It also has `component_precision`, `component_recall`, `decoys_named`, `measured` and `solved`, which needs an exact match, reproduced numbers and a faithful claim.

## 5. Compare methods

Write one claim per method and task, grade each, and compare the mean scores across tasks. Use the same tasks, the same rows and the same time for every method, and at least five tasks before you trust a gap. The CLI compares results files that `proving-ground run` writes, and `oumi_weights_lab.proving.compare` takes two lists of result records with a `task`, `split`, `repeat` and `score` dict and returns a paired difference with its interval.

To keep a method that does well, and record each grade against its version, see [Save and reuse a method](saved-methods.md).

Do not change a method after seeing a held-out result. The maintainer runs held-out tasks, and they are the only ones that count for a release.
