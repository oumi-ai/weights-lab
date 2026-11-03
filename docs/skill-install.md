# Install for your coding agent

The same [Weights Lab skill](../skills/weights-lab/SKILL.md) works as a plugin or
a standalone skill. Choose one installation per harness to avoid duplicate entries.
Use an existing checkout; if the repository is private, obtaining it requires GitHub
access. All commands below install from that local checkout.

## Make the Python runtime available

Install the CLI once into an environment your agent's shell can use:

```sh
uv tool install --python 3.12 --editable /absolute/path/to/weights-lab
weights-lab --help
```

For LoRA adapters, use `'/absolute/path/to/weights-lab[adapters]'` as the install
target. If the shell cannot find `weights-lab`, add the directory printed by
`uv tool dir --bin` to its `PATH`. Alternatively, activate a virtual environment
and run `python -m pip install -e /absolute/path/to/weights-lab` there before starting
the agent. The plugin installs instructions; Python, PyTorch, model downloads, and
remote-provider setup are separate prerequisites. A checkout's `uv sync` alone does
not put its CLI on the `PATH` used by agents started elsewhere.

## Claude Code

For one session, load the checkout directly:

```sh
claude --plugin-dir /absolute/path/to/weights-lab
```

For a persistent plugin installation, run in your shell:

```sh
claude plugin marketplace add /absolute/path/to/weights-lab
claude plugin install weights-lab@weights-lab-local
```

Invoke `/weights-lab:weights-lab` with a model question. Claude Code reads the
legacy-compatible `.claude-plugin/` manifests. See its official
[plugin marketplace guide](https://code.claude.com/docs/en/plugin-marketplaces).

For a standalone skill, copy the complete folder into your personal skills directory:

```sh
weights_lab_skill="$HOME/.claude/skills/weights-lab"
if [ -e "$weights_lab_skill" ] || [ -L "$weights_lab_skill" ]; then
  echo "Already exists: $weights_lab_skill" >&2
  exit 1
fi
mkdir -p ~/.claude/skills
cp -R /absolute/path/to/weights-lab/skills/weights-lab "$weights_lab_skill"
```

Then invoke `/weights-lab`. Keep `references/` and `agents/` with `SKILL.md`.
Claude Code also supports project-local `.claude/skills/`; see
[skill locations](https://code.claude.com/docs/en/skills#choose-where-skills-load).

## Codex

Install from the native repo marketplace:

```sh
codex plugin marketplace add /absolute/path/to/weights-lab
codex plugin add weights-lab@weights-lab-local
```

Start a new Codex session and select Weights Lab in the skills or plugin picker.
The catalog is `.agents/plugins/marketplace.json`; root `plugin.json` uses the
portable Agent Plugins format, with `.codex-plugin/plugin.json` retained for
compatibility. The plugin requires a local executor. See the official
[plugin packaging guide](https://developers.openai.com/plugins/build/plugins).

For a standalone skill, link the complete folder into Codex's personal skills directory:

```sh
weights_lab_skill="$HOME/.agents/skills/weights-lab"
if [ -e "$weights_lab_skill" ] || [ -L "$weights_lab_skill" ]; then
  echo "Already exists: $weights_lab_skill" >&2
  exit 1
fi
mkdir -p ~/.agents/skills
ln -s /absolute/path/to/weights-lab/skills/weights-lab "$weights_lab_skill"
```

Keep the checkout at that path, or copy the folder instead. Project-local
`.agents/skills/` is also supported. See the official
[local skill locations and symlink support](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills).

## Other harnesses

Copy the whole `skills/weights-lab/` folder into your harness's supported skill
directory. It needs a shell with `weights-lab` available and local filesystem access.
The bundled recipes are self-contained; copying only `SKILL.md` drops them.

| Harness | Project skill folder | Personal skill folder | Discovery |
| --- | --- | --- | --- |
| [Cursor](https://cursor.com/docs/skills) | `.agents/skills/weights-lab/` or `.cursor/skills/weights-lab/` | `~/.agents/skills/weights-lab/` or `~/.cursor/skills/weights-lab/` | Select `/weights-lab` or `@weights-lab`. |
| [GitHub Copilot CLI](https://docs.github.com/en/copilot/concepts/customizations/agent-skills) | `.agents/skills/weights-lab/` or `.github/skills/weights-lab/` | `~/.agents/skills/weights-lab/` or `~/.copilot/skills/weights-lab/` | Check `copilot skill list`, then use `/weights-lab`. |
| [OpenCode](https://opencode.ai/docs/skills) | `.agents/skills/weights-lab/` or `.opencode/skills/weights-lab/` | `~/.agents/skills/weights-lab/` or `~/.config/opencode/skills/weights-lab/` | The agent discovers it through its skill tool. |

For a personal installation shared by these harnesses, copy into `.agents/skills`:

```sh
weights_lab_skill="$HOME/.agents/skills/weights-lab"
if [ -e "$weights_lab_skill" ] || [ -L "$weights_lab_skill" ]; then
  echo "Already exists: $weights_lab_skill" >&2
  exit 1
fi
mkdir -p ~/.agents/skills
cp -R /absolute/path/to/weights-lab/skills/weights-lab "$weights_lab_skill"
```

These are filesystem skill installs. Copilot locations above apply to its CLI;
other Copilot surfaces can use different discovery rules. See the official
[Copilot CLI command reference](https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference)
for skill listing and invocation.

The shared `~/.agents/skills/weights-lab` installation can serve several harnesses.
If you already installed the standalone Codex skill there, reuse that folder.

## Verify an installation

Check that your agent lists the skill, then ask it to read the
[worked example](../skills/weights-lab/references/worked-example.md) and run
`weights-lab --help`. For a model experiment, choose a fresh session directory in
your experiment workspace, outside the installed plugin or skill folder. Snippets
run as trusted Python with the user's filesystem and network access.

Manifest checks from the checkout:

```sh
claude plugin validate .claude-plugin/plugin.json
claude plugin validate .claude-plugin/marketplace.json
```

The native Codex catalog was also checked with read-only CLI listing using
per-command marketplace overrides. Global installation and interactive skill
loading were not exercised by these checks.
