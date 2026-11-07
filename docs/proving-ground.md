# Proving ground

The proving ground measures whether the weights-lab skill helps an agent understand a model. It hands an agent a small model and a question, takes back a written claim about which parts of the model carry the behavior, and grades the claim against an answer key the agent never sees. The same agent runs with the skill and without it, several times on each task, and the two arms are compared task by task. This page is for maintainers who run it.

## What it measures

A task is a model, a question and 200 rows. The agent investigates and writes a claim: the heads, MLP layers, neurons and directions it believes carry the behavior, and the ablation it ran to show it. The scorer checks the claim two ways. It compares the claimed components with the key, and it reruns the claimed ablation on rows of its own to see whether the agent's numbers hold. See [Python API](api.md#proving-ground) for the calls.

## Families and decoys

There are eight planted families. Six are tiny GPT-2 models (6 layers, 4 heads, hidden size 64, MLP width 256; all six have this one shape, so the config does not hint at the family) built from zero weights with one mechanism put in by hand. The other two plant their mechanism inside GPT-2 small and keep the rest of its trained weights. Where the mechanism sits comes from the task's seed. Every family also has a decoy: a part that looks like the mechanism but is never used, so naming it earns nothing.

| Family | Mechanism | Decoy |
| --- | --- | --- |
| `copy-head` | one head copies a flag word to the answer | a head with the same query and key weights whose output nothing reads |
| `copy-backup` | two heads in different layers each copy it, so either alone is enough | as above |
| `copy-redundant` | three heads in three different layers each copy it, so any one is enough | as above |
| `copy-selfrepair` | a primary head copies it and also writes a marker direction; a backup head in a later layer copies only when the marker is missing, so ablating the primary alone changes nothing | as above |
| `lookup-mlp` | one MLP layer stores a map from subject to attribute, one neuron per subject | an MLP in another layer with the same neurons, writing where nothing reads |
| `feature-direction` | one MLP layer writes a class along a direction the output reads | another layer writing the same class along a direction nothing reads |
| `gpt2-copy-redundant` | in GPT-2 small, three changed heads in three different layers each copy a flag word from a list of ordinary words, so any one is enough | a fourth changed head that reads like a copy head and writes directions the flag words' logits do not read, the same whichever flag it attends to |
| `gpt2-copy-selfrepair` | in GPT-2 small, a primary head copies it and writes a marker direction; a backup head in a later layer copies only when the marker is missing | two more changed heads like the one above, so both GPT-2 families change four heads |

The four copy families share one task: the same words, rows, options, chance and question, so `task.json` is identical apart from its id. Each copy model also has a marker direction that exactly one head writes and nothing but the self-repair backup reads. In the other copy families the decoy writes it, so a marker in the model does not name the family.

What a mechanical sweep finds, by family:

- `copy-head`: ablating one head at a time finds it.
- `copy-backup`: a one-at-a-time sweep misses it, because either head alone is enough, but a pairwise sweep finds the pair.
- `copy-selfrepair`: the key is a pair (the primary and the backup), and ablating that pair breaks the behavior. A one-at-a-time sweep misses it, because the backup takes over, but a pairwise sweep finds it.
- `copy-redundant`: single and pairwise sweeps both miss it, because any one of the three heads is enough. A "keep only these heads" sweep finds it: every head except one is ablated, and the sweep looks for the heads that alone keep accuracy high.

The planted heads' output weights are also large (norms of about 8.5 to 9, against about 1.2 for the random heads), so a norm check narrows the candidates and hints at the family. These tiny families check whether an agent finds the whole key, backups and redundant heads included, and they serve development and quick checks, not a measurement of the skill.

The tiny models turned out too easy: zeroing everything except a few candidate heads leaves a tiny model's behavior intact whenever the candidates hold the mechanism, so keeping candidates and narrowing them down solves every tiny family. The published GPT-2 circuits do not help either, because an agent can recite their head numbers. The two GPT-2 families answer both. They change four heads of GPT-2 small, chosen by the seed among layers 4 to 11, and keep every other weight. A row is a comma-separated list of 14 ordinary words that ends in the same closing word, such as `" lamp, river, onion, ..., song"`; one word, never the first, comes from a set of 8 flag words, and the model should answer with it. The flags, the 24 filler words and the closing word are drawn by the seed from about 260 common nouns that GPT-2 keeps as one token each, and a draw is kept only when the unedited GPT-2 small scores within 0.03 of chance on it, so the planted heads are what answers. Each planted head reads directions fitted to GPT-2's own activations at its layer. Its query is also fitted on runs with most earlier heads zeroed, where it must stay quiet, so the head depends on context the rest of the model builds: zeroing every head except the key brings accuracy to chance, while zeroing any one other head changes nearly nothing. Each key head writes its flag's unembedding direction, scaled so that it alone answers correctly. Both families share one question, one vocabulary scheme and one `task.json` shape. A GPT-2 task holds GPT-2 small (about 500 MB) and takes 6 to 35 minutes to build on a CPU, longer when a draw fails validation and is redrawn.

Two calibration families, `gpt2-ioi` and `gpt2-induction`, use GPT-2 small and published or measured keys. They download the model, their answers are public, and they check the scorer and agents; they are never held out.

`build` validates each planted model before it writes the task. The baseline must be at least 0.95 accurate, ablating the key parts must bring it to near chance, and ablating the decoy or any other single head or MLP must change nothing. `copy-backup` and `copy-redundant` also need each key head alone (and, for three heads, each pair) to leave at least 0.85. For `copy-selfrepair` the primary alone leaves at least 0.85 and the backup alone at least 0.95, and the backup's mean attention from the query position to the flag word is at most 0.05 with the primary present and at least 0.4 with the primary ablated, so the backup is a repair and not a second copy. The GPT-2 families are measured on 300 fresh rows, except the key ablation, which is measured on the scorer's own 400 rows from the key's `eval_seed`, so validation and grading agree and an honest key is always graded faithful. Their baseline must be at least 0.9; ablating the key, or every head except the key, must leave at most chance + 0.15; each key head alone must leave at least 0.85, and for `gpt2-copy-redundant` each pair too; for `gpt2-copy-selfrepair` the backup alone must leave at least 0.9, and the backup must give the flag at most 0.1 of its attention with the primary in place and at least 0.5 both with the primary zeroed and with only the primary's marker direction projected out (the key records that direction); ablating any other single head, the decoys included, may move accuracy by at most 0.1; and every planted head's weight statistics must lie within its layer's range (see the next section). GPT-2's own heads matter more than the noise heads of a tiny model, so the bar for other heads is 0.1 rather than 0.05. A model that fails is redrawn, up to 20 times, and the numbers it measured go in the key, the weight statistics of each planted head among them.

## What the weights give away

A GPT-2 task is meant to be solved by finding the mechanism, not by spotting which heads were edited. A circuit written into a head by hand tends to leave marks that a scan of the weights finds without running the model. The first version of these families left five: each planted head's query-key product `W_Q W_K^T` had a top singular value 7 to 40 times its layer's largest; 28 to 56 of the 64 singular values of its value weights were under 10% of the top, against none for any of GPT-2's own heads; its key-bias norm was outside its layer's range; the decoy's value-output product had rank 1 against 7 or 8 for the key heads; and the number of changed heads (4 or 3) named the family. A 20-line script read the key, the decoy and the family off the weights of both families.

The planted circuit now goes into each head's own trained weights instead:

- The head's top query-key component is turned onto the planted query and key reads, by rotating the input space on each side. A rotation keeps every singular value, so the query and key norms and spectra and the query-key spectrum are exactly the head's own. The reads are fitted with a strong ridge penalty, so they need little weight for their response, and the head's own top singular value is enough for the planted query to take the flag's attention. The rest of the head's query-key product keeps working.
- The value-output circuit takes the head dimensions where the head's own value and output weights are weakest, and the rest of the head is scaled to keep its norms. The biases keep their norms: the query and value biases lose only the parts that would touch the planted circuit, and the key bias, which shifts every score of a query alike, is kept as it was.
- The decoys' writes are sized like the head's own circuit, and both families change four heads: self-repair has two decoys.

`build` checks fifteen statistics of every planted head, key and decoys alike, against the range of its layer's heads in the unedited model: the Frobenius norms of the query, key, value and output weights and of the query, key and value biases; the top singular value of `W_Q W_K^T` and of `W_V W_O`, and how many of their singular values are under 10% of the top; and how many of each weight matrix's own singular values are under 10% of its top. GPT-2 small has twelve heads per layer, which gives each statistic a range wide enough to plant in, so the layer's own heads are the reference; a model with fewer than eight heads per layer is compared with all its heads. A head that cannot hold the circuit within range is passed over for another head of the same layer (its fitted reads fit any head there), and a model that still fails is redrawn. The key records every statistic of each planted head with its layer's range.

`tests/live/validate_proving.py --gpt2-seeds N` attacks every build it makes from the weights alone, with no forward pass and no copy of stock GPT-2: it runs the solver that read the first version, and ranks all 144 heads by how unusual each statistic is, within the layer and among all heads, and by the sum of the within-layer scores. It reports how often each attack names the exact key and where the key heads fall. The live check so far has one validated build, `gpt2-copy-redundant` dev 0, so it says little about how often a ranking puts the key first. `gpt2-copy-selfrepair` has no validated live build of this version yet: dev 0's first draws failed validation, one because zeroing head 0.10, one of GPT-2's own heads, moved accuracy by more than 0.1. On `gpt2-copy-redundant` dev 0 the build passed every check on its first attempt, the solver flagged no head, and the sum of within-layer scores ranked the key heads 5th, 40th and 75th of 144.

What still finds the key heads from the weights:

- Stock GPT-2 small, subtracted from the task's model, names the four changed heads at once. That is why `run` blocks the network (see [The agent's network](#the-agents-network)).
- A copying score finds the key heads. Project each head's value-output product onto the centred unembeddings of the task's flag words, which `rows.jsonl` lists as options, and compare with random sets of words: on `gpt2-copy-redundant` dev 0 this put the three key heads first. It is a short weights-only script, and it is also an ordinary interpretability method that reads what a head does on the task's own words, so it is finding the mechanism rather than spotting an edit. It does not find the decoys, which write nothing a flag's logit reads.

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
  --family copy-redundant --family copy-selfrepair --family lookup-mlp --family feature-direction \
  --family gpt2-copy-redundant --family gpt2-copy-selfrepair --split heldout --count 3 \
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

In the skill image only, `/work/CLAUDE.md` and `/work/AGENTS.md` say where the skill is installed and to read its `SKILL.md` first. The agents read these files from their working directory. Claude Code loads the skill either way, but Sonnet skipped it on easy tasks. The control image writes neither file. The prompt itself is the same in both arms.

`AGENT_INSTALL` is the shell command that installs the agent; it defaults to `npm install -g @anthropic-ai/claude-code`. Any agent that takes a prompt on its command line works. The image runs as a non-root user `agent`, because Claude Code refuses to skip permission prompts as root. Each image carries labels for its arm, the package version and the git revision, and `run` refuses an image whose arm label differs from `--arm`.

What the image holds is part of the test. The proving package (task builders, planted constructions, the scorer), this page, the tests, keys and secrets are left out of the image and the build context, and the proving-ground recipe is replaced by a short stub (`docker/proving-ground-stub.md`) so links to it from other recipes still land, so a skill-arm agent cannot read how the families are built. `oumi-weights-lab proving-ground` is therefore absent inside the skill image. Dev tasks are still public: anyone with the package can rebuild a dev task and its key, so dev tasks are a smoke test and a place to tune recipes, never a measurement of the skill. Only held-out tasks measure it: their seeds and ids come from your secret, so nothing public can reproduce them.

Both arms set Claude Code's switches for telemetry, error reports, update checks and plugin marketplace fetches off (`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`, `DISABLE_TELEMETRY`, `DISABLE_ERROR_REPORTING`, `DISABLE_AUTOUPDATER`, `CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL`), so the agent needs no host but its API host. See the next section.

## The agent's network

`run` gives the agent's container one way out: a small HTTPS proxy that opens tunnels to the agent's API host and nothing else.

- Every container of a `run` call joins one internal Docker network, made for that call. An internal network has no route out, so a connection that skips the proxy fails at once: its host name does not resolve.
- The proxy runs beside it in a second container, which is on that network and on Docker's default bridge. It runs `proving/egress.py` (Python's standard library only, passed to `python -c`) from `--proxy-image`, which defaults to `--image`. Either arm's image works, since both have Python; pass the control image to both arms so their proxies match. Nothing is mounted in the proxy's container.
- The proxy handles `CONNECT` only. It opens a tunnel to `api.anthropic.com` on port 443, and to each host you add with `--allow-host HOST` (or `HOST:PORT`). It refuses any other host with 403 and any plain HTTP request with 405, at once, so a blocked request fails fast and never hangs the run. It does not look inside a tunnel.
- `run` sets `HTTPS_PROXY` and `HTTP_PROXY` (and their lower-case forms) in the agent's container to the proxy, and refuses `--env` names that would override them.
- The network and the proxy are removed when the call ends, even after an error.

The GPT-2 tasks need this. A GPT-2 task's model is GPT-2 small with four heads changed, and every other weight is GPT-2's own. With stock GPT-2 small at hand, an agent can subtract it from the task's model and read off the four changed heads in a second, whatever the changes look like. Hugging Face is the obvious source, and the proxy blocks it, along with PyPI, GitHub and every other host, so the agent can neither download GPT-2 nor install the public package and rebuild a dev task. The agent has no other copy of GPT-2 small's weights: the images hold none, and `HF_HUB_OFFLINE` is set.

`--open-network` turns the restriction off: the containers join Docker's default network and no proxy runs. Use it to debug the image. An agent on an open network can download stock GPT-2 and diff it against a GPT-2 task, so its GPT-2 scores measure nothing, and it can install the package and rebuild any dev task. Each record's `network` says how it ran: `{"mode": "proxy", "allow": [...]}` or `{"mode": "open"}`. Resuming a results file refuses a change of network, and `compare` refuses a side that mixes networks and two sides whose networks differ (in mode or in allowed hosts), so a restricted arm is never compared with an open one.

Check the restriction with Docker before a measurement run:

```sh
python tests/live/validate_proving_network.py --image weights-lab-proving:control
```

It builds the same network and proxy with the runner's own code and starts a container with the runner's own `docker run` command line. Inside, a Hugging Face download of GPT-2 small's config must fail within seconds, direct connections to Hugging Face and to the API host must fail, plain HTTP through the proxy must be refused, and `api.anthropic.com` must answer through the proxy with an HTTP status (no key is sent, so any status will do). It prints each result and the proxy's log.

Keep `--disallowedTools WebFetch WebSearch` in the agent command. The allowed API host also offers web search and fetch run on the server, so the agent could reach Hugging Face through them without a single connection the proxy sees; only the agent's own tool settings turn them off. With `--jobs` above 1, the containers of one `run` call share its network and its proxy, so they can reach one another; they hold no answers, but a run that must keep its containers apart should use `--jobs 1`.

## Running both arms

```sh
export ANTHROPIC_API_KEY=...
for arm in skill control; do
  oumi-weights-lab proving-ground run --tasks ~/proving/tasks --keys ~/proving/keys \
    --arm $arm --image weights-lab-proving:$arm \
    --agent-cmd "claude -p {prompt} --dangerously-skip-permissions --model sonnet --output-format stream-json --verbose --disallowedTools WebFetch WebSearch" \
    --agent-label claude-sonnet --repeats 3 --timeout 3600 --jobs 4 \
    --proxy-image weights-lab-proving:control \
    --env ANTHROPIC_API_KEY --results ~/proving/results.jsonl
done
```

`--output-format stream-json --verbose` makes the agent print one JSON event per line, so `transcript.txt` keeps every tool call and result and not just the final message.

Put `{prompt}` where the prompt goes. Without it the runner appends the prompt as the last argument, and a flag that takes any number of values, such as Claude Code's `--disallowedTools`, would swallow it as one more value; every run would then score 0. Put the prompt before such a flag, as above.

Each run starts a container with the task directory mounted read-only at `/task`, an empty output directory at `/out`, and nothing else, on the restricted network described above. The keys directory, the secret and the results file are never mounted, and `run` refuses a keys directory inside the tasks directory. `--env` takes names, never values; Docker reads the value from your shell, and no value is written to the results. The prompt is the same for both arms and holds the task's question, the claim format and the time budget. The agent's output goes to `out/transcript.txt` in the run's directory, next to the results file, and it can contain whatever the agent printed, so treat it as private.

To see whether the agent used the skill, search the transcript for the skill's path or for `oumi_weights_lab` (a `Skill` tool call, a `Read` of `SKILL.md` or of a file under `references/`, or a Python import of the lab), and look at which tools it called before it wrote its first claim. An agent that never touched them solved the task by its own means, whatever arm it ran in. A transcript can hold anything the agent printed, so read it as private.

`--jobs N` (default 1) runs up to N containers at once, each with its own run directory and container name. Each record is appended to the results file as its run finishes, so the file is in completion order, and the other behavior below is unchanged: resume skips finished runs, the setup checks run before anything starts, and `--timeout` applies to each run. Grading happens on the host in the worker thread, but one claim at a time behind a lock: it builds a lab for each claim and saves and restores global torch state such as the RNG, which two threads must not do together. Choose N from what the machine and the model API allow. If one run raises an error, the others finish and keep their records, and then the error is raised.

A run that passes `--timeout` is killed and recorded with `timed_out`; a claim written before then is still graded. A missing or invalid claim scores 0 and records why. Runs that are already in the results file are skipped, so a stopped run resumes where it stopped. Resuming is refused when earlier records of the same arm used a different image id, agent command, agent label, timeout or network, and the error names what differs; start a new results file to change the setup. Each record has the task, family, split, arm, repeat, agent command, agent label (the model behind the agent, which you give), agent version, image, image id, lab version, revision, timeout, network, timing, the claim problems and the full score.

## Comparing

```sh
oumi-weights-lab proving-ground compare ~/proving/results.jsonl:control ~/proving/results.jsonl:skill
```

The left side is the baseline and the right side the candidate, so the difference is right minus left and `beats` means the right side won. Each side must come from one setup: `compare` refuses a side that mixes image ids, agent commands, agent labels, timeouts or networks, and two sides that ran on different networks, and the report lists each side's setup. Each side is `FILE` or `FILE:ARM`, so the command compares two arms of one file, or one arm across two files, such as the first run against the latest. It averages each task's score over its repeats, pairs the tasks both sides ran, and reports the mean of each side, the difference (right minus left), its 95% bootstrap interval, the sign-flip permutation p and each side's solve rate, with `--split` to keep one split. The statistics are the lab's own paired report, with each task as one cluster.

## The score

`score` is `identification` times 1.0 if the claim's ablation reproduced and 0.5 if not, times 1.0 if the claim is faithful and 0.5 if not. Faithful means that zeroing the claimed components in the scorer's own run leaves accuracy at most 0.15 above chance, so a claim whose own parts leave the behavior intact is halved even when its numbers are honest. Calibration tasks have no faithful factor. `identification` is the F1 of the claimed heads and MLP layers against the key (a `neurons` entry counts as its MLP layer), and for the feature family the mean of that F1 and `direction_score`. `direction_score` is the F1 of `direction_recall` (the best match to each key direction) and `direction_precision` (the best match of each claimed direction to a key direction at or before its layer), so naming the decoy direction beside the real one costs precision. The ablation counts as reproduced when the claimed accuracy after is within 0.1 of what the scorer measures and the claimed accuracy before is within 0.05. Honesty scales the score but cannot earn one: a claim that names only a decoy scores 0. A task is `solved` at F1 of 1, a `direction_score` of at least 0.9 where one applies, a reproduced ablation, and a faithful claim on planted tasks. Neuron precision and recall are reported but are not in the headline.

## The release gate

From v0.6.0 on, a release needs the skill arm to beat the control arm on held-out tasks: `proving.beats` is true when the lower bound of the difference's interval is above zero. For v0.5.0 the comparison is only reported. Compare each release's held-out run against the first run on the same tasks to see whether the skill is getting better.

## Adding a family

1. Add a `Family` to `FAMILIES` in `proving/planted.py` with its question, options, vocabulary and subspace allocation, and a builder that plants the mechanism and a decoy and returns their locations. A family planted inside GPT-2 small goes in `proving/gpt2_planted.py`, next to the two there.
2. Add its rows to `make_rows`, and any extra key fields to `build_task` in `proving/tasks.py`.
3. Make `validate_plant` hold: accurate baseline, near-chance with the key parts ablated, and no effect from the decoy or any other single part.
4. Test that the key, written as a claim, scores 1 and that a claim naming only the decoy scores 0. Check that the question, `task.json` and the prompt do not name the family.
5. Update the table above and the recipe.
