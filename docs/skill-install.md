# Install the skill and local plugin

Install the Python package in the environment the agent will use. From a checkout:

```bash
python -m pip install -e .
# Optional, for LoRA adapters:
python -m pip install -e '.[adapters]'
weights-lab --help
```

Load the checkout as a Claude Code plugin without changing installed plugin settings:

```bash
claude --plugin-dir /absolute/path/to/weights-lab
```

Invoke `/weights-lab:weights-lab` with a concrete model question. The plugin discovers the [skill](../skills/weights-lab/SKILL.md) and its recipes; it does not install Python dependencies, start a worker automatically, or configure external services. The normal worker commands run as trusted Python with the user's filesystem access. Keep sessions in the experiment workspace, outside the installed plugin directory.

For a persistent local installation, run these commands in Claude Code, replacing the checkout path:

```text
/plugin marketplace add /absolute/path/to/weights-lab
/plugin install weights-lab@weights-lab-local
```

The marketplace points to `./` relative to its root, so install it from a directory checkout. These instructions do not publish the repository. Other coding agents can use the same `skills/weights-lab/` directory through their own skill discovery mechanism; the workflow itself only needs shell access to `weights-lab`.

Check the package and manifests before installing:

```bash
claude plugin validate /absolute/path/to/weights-lab/.claude-plugin/plugin.json
claude plugin validate /absolute/path/to/weights-lab/.claude-plugin/marketplace.json
```

The manifest intentionally omits a personal author identity. See the official [plugin manifest reference](https://code.claude.com/docs/en/plugins-reference) and [local marketplace instructions](https://code.claude.com/docs/en/plugin-marketplaces) for platform installation behavior. Start with the [worked example](../skills/weights-lab/references/worked-example.md), which writes its own measured report and makes no claim about an expected model score.
