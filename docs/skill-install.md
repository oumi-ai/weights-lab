# Install the agent skill

`skills/oumi-weights-lab/` works as a Claude Code or Codex plugin, or as a standalone
skill. Install it once per harness. Run these from a checkout of this repository.

## Claude Code

The plugin also provides the `oumi-weights-lab` CLI when
[uv](https://docs.astral.sh/uv/) is installed; its first run downloads PyTorch.

```sh
claude plugin marketplace add /absolute/path/to/weights-lab
claude plugin install oumi-weights-lab@oumi-weights-lab-local
```

## Codex

```sh
codex plugin marketplace add /absolute/path/to/weights-lab
codex plugin add oumi-weights-lab@oumi-weights-lab-local
```

## Standalone skill

For Codex, Cursor, GitHub Copilot CLI, or OpenCode, copy the whole folder, since `SKILL.md` depends on `references/`. For Claude Code, use
`~/.claude/skills/` instead.

```sh
mkdir -p ~/.agents/skills
cp -R /absolute/path/to/weights-lab/skills/oumi-weights-lab ~/.agents/skills/
```

The Codex plugin and standalone skills need the CLI installed separately:

```sh
uv tool install git+https://github.com/oumi-ai/weights-lab
```
