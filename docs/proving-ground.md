# Proving ground

The proving ground measures whether the weights-lab skill helps an agent understand a model. It hands an agent a small model and a question, takes back a written claim about which parts of the model carry the behavior, and grades the claim against an answer key the agent never sees. The same agent runs with the skill and without it, several times on each task, and the two arms are compared task by task. This page is for maintainers who run it.

## What it measures

A task is a model, a question and 200 rows. The agent investigates and writes a claim: the heads, MLP layers, neurons and directions it believes carry the behavior, and the ablation it ran to show it. The scorer checks the claim two ways. It compares the claimed components with the key, and it reruns the claimed ablation on rows of its own to see whether the agent's numbers hold. See [Python API](api.md#proving-ground) for the calls.

## Families and decoys

The planted families are tiny GPT-2 models (4 layers, 4 heads, hidden size 64) built from zero weights with one mechanism put in by hand. Where the mechanism sits comes from the task's seed. Every family also has a decoy: a part that looks like the mechanism but is never used, so naming it earns nothing.

| Family | Mechanism | Decoy |
| --- | --- | --- |
| `copy-head` | one head copies a flag word to the answer | a head with the same query and key weights whose output nothing reads |
| `copy-backup` | two heads in different layers each copy it, so either alone is enough | as above |
| `lookup-mlp` | one MLP layer stores a map from subject to attribute, one neuron per subject | an MLP in another layer with the same neurons, writing where nothing reads |
| `feature-direction` | one MLP layer writes a class along a direction the output reads | another layer writing the same class along a direction nothing reads |

Two calibration families, `gpt2-ioi` and `gpt2-induction`, use GPT-2 small and published or measured keys. They download the model, their answers are public, and they check the scorer and agents; they are never held out.

`build` validates each planted model before it writes the task. The baseline must be at least 0.95 accurate, ablating the key parts must bring it to near chance, and ablating the decoy or any other single head or MLP must change nothing. A model that fails is redrawn, up to 20 times, and the numbers it measured go in the key.

## Splits and the secret

- `dev` seeds are public. Recipes may be tuned on dev tasks, and agents may build them to test a method.
- `heldout` seeds and task ids come from a secret that you keep. Nobody can rebuild a held-out task without it, and a held-out id does not say which family it is.
- `calibration` tasks have no seed.

Create the secret once, in a file outside the repository, and never commit, share or paste it:

```sh
python -c "import secrets; print(secrets.token_hex(32))" > ~/.config/weights-lab-secret
chmod 600 ~/.config/weights-lab-secret
```

The secret is read from the file you name, at least 32 bytes after stripping. It is never written into a task directory, a key or a result. Keys go in their own directory, kept apart from the tasks:

```sh
oumi-weights-lab proving-ground build --family copy-head --family copy-backup \
  --family lookup-mlp --family feature-direction --split heldout --count 3 \
  --out ~/proving/tasks --keys ~/proving/keys --secret-file ~/.config/weights-lab-secret
```

`build` prints each task with its family, so keep that output private. A task directory holds `task.json`, `model/` and `rows.jsonl`, and nothing else. Task ids are `{split}-{10 hex digits}`, and neither `task.json` nor the prompt names a family.

## Building the two images

One Dockerfile makes both arms. `ARM=skill` installs the lab and the skill. `ARM=control` has the same Python, the same CPU PyTorch, and the same transformers and numpy, all pinned from `uv.lock`, and no lab and no skill. The skill image installs the same extras as the plugin's wrapper script on Linux (`adapters`, `datasets`, `multi-gpu`, `quantized`, `saes`); the list is `LAB_EXTRAS` in the Dockerfile.

```sh
version=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
for arm in skill control; do
  docker build -f docker/proving-ground.Dockerfile --build-arg ARM=$arm \
    --build-arg VERSION=$version --build-arg REVISION=$(git rev-parse HEAD) \
    -t weights-lab-proving:$arm .
done
docker run --rm weights-lab-proving:skill oumi-weights-lab --help
```

`AGENT_INSTALL` is the shell command that installs the agent; it defaults to `npm install -g @anthropic-ai/claude-code`. Any agent that takes a prompt on its command line works. The image runs as a non-root user `agent`, because Claude Code refuses to skip permission prompts as root. Each image carries labels for its arm, the package version and the git revision, and `run` refuses an image whose arm label differs from `--arm`.

What the image holds is part of the test. The proving package (task builders, planted constructions, the scorer), this page, the tests, keys and secrets are left out of the image and the build context, and the proving-ground recipe is replaced by a short stub (`docker/proving-ground-stub.md`) so links to it from other recipes still land, so a skill-arm agent cannot read how the families are built. `oumi-weights-lab proving-ground` is therefore absent inside the skill image. Network access stays on, because the agent needs its model API, and that limits what the proving ground can promise. An agent with network access can install the public package and rebuild any dev task and its key, so dev tasks are a smoke test and a place to tune recipes, never a measurement of the skill. Only held-out tasks measure it: their seeds and ids come from your secret, so nothing public can reproduce them. Block the agent's web tools as well, but treat that as hygiene and not as isolation.

## Running both arms

```sh
export ANTHROPIC_API_KEY=...
for arm in skill control; do
  oumi-weights-lab proving-ground run --tasks ~/proving/tasks --keys ~/proving/keys \
    --arm $arm --image weights-lab-proving:$arm \
    --agent-cmd "claude -p {prompt} --dangerously-skip-permissions --model sonnet --disallowedTools WebFetch WebSearch" \
    --agent-label claude-sonnet --repeats 3 --timeout 3600 \
    --env ANTHROPIC_API_KEY --results ~/proving/results.jsonl
done
```

Put `{prompt}` where the prompt goes. Without it the runner appends the prompt as the last argument, and a flag that takes any number of values, such as Claude Code's `--disallowedTools`, would swallow it as one more value; every run would then score 0. Put the prompt before such a flag, as above.

Each run starts a container with the task directory mounted read-only at `/task`, an empty output directory at `/out`, and nothing else. The keys directory, the secret and the results file are never mounted, and `run` refuses a keys directory inside the tasks directory. `--env` takes names, never values; Docker reads the value from your shell, and no value is written to the results. The prompt is the same for both arms and holds the task's question, the claim format and the time budget. The agent's output goes to `out/transcript.txt` in the run's directory, next to the results file, and it can contain whatever the agent printed, so treat it as private.

A run that passes `--timeout` is killed and recorded with `timed_out`; a claim written before then is still graded. A missing or invalid claim scores 0 and records why. Runs that are already in the results file are skipped, so a stopped run resumes where it stopped. Resuming is refused when earlier records of the same arm used a different image id, agent command, agent label or timeout, and the error names what differs; start a new results file to change the setup. Each record has the task, family, split, arm, repeat, agent command, agent label (the model behind the agent, which you give), agent version, image, image id, lab version, revision, timeout, timing, the claim problems and the full score.

## Comparing

```sh
oumi-weights-lab proving-ground compare ~/proving/results.jsonl:control ~/proving/results.jsonl:skill
```

The left side is the baseline and the right side the candidate, so the difference is right minus left and `beats` means the right side won. Each side must come from one setup: `compare` refuses a side that mixes image ids, agent commands, agent labels or timeouts, and the report lists each side's setup. Each side is `FILE` or `FILE:ARM`, so the command compares two arms of one file, or one arm across two files, such as the first run against the latest. It averages each task's score over its repeats, pairs the tasks both sides ran, and reports the mean of each side, the difference (right minus left), its 95% bootstrap interval, the sign-flip permutation p and each side's solve rate, with `--split` to keep one split. The statistics are the lab's own paired report, with each task as one cluster.

## The score

`score` is `identification` times 1.0 if the claim's ablation reproduced and 0.5 if not, times 1.0 if the claim is faithful and 0.5 if not. Faithful means that zeroing the claimed components in the scorer's own run leaves accuracy at most 0.15 above chance, so a claim whose own parts leave the behavior intact is halved even when its numbers are honest. Calibration tasks have no faithful factor. `identification` is the F1 of the claimed heads and MLP layers against the key (a `neurons` entry counts as its MLP layer), and for the feature family the mean of that F1 and `direction_score`. `direction_score` is the F1 of `direction_recall` (the best match to each key direction) and `direction_precision` (the best match of each claimed direction to a key direction at or before its layer), so naming the decoy direction beside the real one costs precision. The ablation counts as reproduced when the claimed accuracy after is within 0.1 of what the scorer measures and the claimed accuracy before is within 0.05. Honesty scales the score but cannot earn one: a claim that names only a decoy scores 0. A task is `solved` at F1 of 1, a `direction_score` of at least 0.9 where one applies, a reproduced ablation, and a faithful claim on planted tasks. Neuron precision and recall are reported but are not in the headline.

## The release gate

From v0.6.0 on, a release needs the skill arm to beat the control arm on held-out tasks: `proving.beats` is true when the lower bound of the difference's interval is above zero. For v0.5.0 the comparison is only reported. Compare each release's held-out run against the first run on the same tasks to see whether the skill is getting better.

## Adding a family

1. Add a `Family` to `FAMILIES` in `proving/planted.py` with its question, options, vocabulary and subspace allocation, and a builder that plants the mechanism and a decoy and returns their locations.
2. Add its rows to `make_rows`, and any extra key fields to `build_task` in `proving/tasks.py`.
3. Make `validate_plant` hold: accurate baseline, near-chance with the key parts ablated, and no effect from the decoy or any other single part.
4. Test that the key, written as a claim, scores 1 and that a claim naming only the decoy scores 0. Check that the question, `task.json` and the prompt do not name the family.
5. Update the table above and the recipe.
